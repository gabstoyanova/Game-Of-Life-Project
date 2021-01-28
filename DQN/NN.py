from functools import partial
import itertools

from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Relu, LogSoftmax, GeneralConv, Flatten, elementwise
from jax import grad, jit, lax, tree_map, random, tree_util
import jax.numpy as jnp


def piecewise_constant(boundaries, values, t):
    """Piecewise constant.
    Helper function for stepped learning rate.
    Args:
    boundaries: boundaries at which value changes
    values: values start at corresponding boundary and finish at next
      boundary
    t: value to sample at
    Returns:
    value
    """
    index = jnp.sum(boundaries < t)
    return jnp.take(values, index)


def create_stepped_learning_rate_fn(base_learning_rate, steps_per_epoch,
                                    lr_sched_steps, warmup_length=0.0):
    """Create a stepped learning rate function.
    Args:
    base_learning_rate: base learning rate
    steps_per_epoch: number of steps per epoch
    lr_sched_steps: learning rate schedule as a list of pairs where each
      pair is `[step, lr_factor]`
    warmup_length: linear LR warmup length; 0 for no warmup
    Returns:
    function of the form f(step) -> learning_rate
    """
    boundaries = [step[0] for step in lr_sched_steps]
    decays = [step[1] for step in lr_sched_steps]
    boundaries = jnp.array(boundaries) * steps_per_epoch
    boundaries = jnp.round(boundaries).astype(jnp.int32)
    values = jnp.array([1.0] + decays) * base_learning_rate

    def step_fn(step):
        lr = piecewise_constant(boundaries, values, step)
        if warmup_length > 0.0:
            lr = lr * jnp.minimum(1., step / float(warmup_length) / steps_per_epoch)
        return lr
    return step_fn



def huber_loss(x, delta: float = 1.0):
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
    See "Robust Estimation of a Location Parameter" by Huber.
    (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
    Args:
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.
    Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
    a vector of same shape of `x`.
    """

    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic ** 2 + delta * linear





class Neural_Net(object):
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def __init__(self,  n_actions,
                        input_shape,
                        adam_params,
                        use_target=True,
                        seed=0
                ):

        ### set seed and JAX rng
        self.seed=seed
        rng = random.PRNGKey(self.seed)


        ### define attributes
        self.n_actions = n_actions
        self.input_shape = input_shape
        
        self.use_target = use_target
        self.itercount = itertools.count()


        ### intialize the network
        print( "\nDQN input shape: {}.".format(input_shape) )
        
        initialize_params, self._predict = self._create_network_architecture(n_actions)
        
        rng_1, rng_2 = random.split(rng)
        _, self.Q_net        = initialize_params(rng_1, input_shape)
        _, self.Q_net_target = initialize_params(rng_2, input_shape)

        self.predict = jit(self._predict)
        #self.predict = self._predict


        ### create optimizer
        self._create_optimizer(adam_params)


    def _create_network_architecture(self, action_dim):

        dim_nums=('NHWC', 'HWIO', 'NHWC')

        initialize_params, predict = stax.serial(
                                            # elementwise(lambda x: x/255.0),  # normalize
                                            ### convolutional NN (CNN)
                                            GeneralConv(dim_nums, 32, (8,8), strides=(4,4) ), 
                                            Relu,
                                            # GeneralConv(dim_nums, 64, (4,4), strides=(2,2) ), 
                                            # Relu,
                                            # GeneralConv(dim_nums, 64, (3,3), strides=(1,1) ), 
                                            # Relu,
                                            Flatten, # flatten output
                                            Dense(512), 
                                            Relu,
                                            Dense(action_dim)
                                        )

        return initialize_params, predict


    def _create_optimizer(self,params):

        stepsize_schedule = create_stepped_learning_rate_fn(
            base_learning_rate=params['step_size'],
            steps_per_epoch=1,
            lr_sched_steps=[[int(params['N_iterations'] / 8.0), 0.5]],
        )

        # build the optimizer
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(stepsize_schedule, 
                                                                            b1=params['b1'], 
                                                                            b2=params['b2'], 
                                                                            eps=params['eps'],
                                                                        )
        self.opt_state = self.opt_init(self.Q_net)



    @partial(jit, static_argnums=(0,))
    def Bellman_loss(self, Qnet_params, batch, actions):
        inputs, targets = batch
        preds = self.predict(Qnet_params, inputs)
        preds_select = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=1), axis=1)
        return jnp.mean(huber_loss(preds_select - targets))


    @partial(jit, static_argnums=(0,))
    def DQN_update(self, step, Qnet_params, opt_state, batch, actions):
        # update the Qnet
        gradients = grad(self.Bellman_loss)(Qnet_params, batch, actions)

        # clip the gradient
        clipped_gradients = tree_map(lambda g: jnp.clip(g, -10.0, 10.0), gradients)

        return self.opt_update(step, clipped_gradients, opt_state) 


    @partial(jit, static_argnums=(0,))
    def Qlearning_step(self, Q_net_target, gamma, rewards, next_states, is_terminal):
        print('ot function Qlearning_step: ',Q_net_target )
        ### evaluate Q(s',a) for all actions
        next_Q_values = self.predict(Q_net_target, next_states)

        ### Bellman update
        # (1-is_terminal) sets Q-value of terminal states to zero
        Q_values = rewards + gamma * jnp.max(next_Q_values, axis=1) * (1-is_terminal)  

        ### do not push gradients thru Q_values
        Q_values = lax.stop_gradient(Q_values)

        return Q_values

    



    def _fit_batch(self, gamma, states, actions, rewards, next_states, is_terminal):
        # evaluate Q(s',a) for all actions
        if self.use_target:
            Q_values = self.Qlearning_step(self.Q_net_target, gamma, rewards, next_states, is_terminal)
        else:
            Q_values = self.Qlearning_step(self.Q_net, gamma, rewards, next_states, is_terminal)

        # update model parameters; note how we are passing actions as a mask to multiply the targets
        self.opt_state = self.DQN_update(next(self.itercount), self.get_params(self.opt_state), self.opt_state, [states, Q_values[:, None]], actions )
        self.Q_net = self.get_params(self.opt_state)
        

    def update_Qnet(self, replay_buffer, minibatch_size, gamma):
        # sample a minibatch
        states, actions, rewards, next_states, is_terminals = replay_buffer.sample(minibatch_size)
        print('in updaste Qnet we take this from buffer', states, actions, rewards, next_states, is_terminals)
        # fit batch
        self._fit_batch(gamma, states, actions, rewards, next_states, is_terminals)


    def update_Qnet_target(self):
        
        self.Q_net_target = self.get_params(self.opt_state)




