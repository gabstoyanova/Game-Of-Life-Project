import os.path as osp
import sys, time

import gym
from gym import wrappers

import numpy as np
import random

from atari_wrappers import *
from buffer import ReplayBuffer

from NN import Neural_Net
from env_utils import episode_step
from video_utils import learning_logger


###############################################################
# hyperparameters

# seed
seed = 0
np.random.seed(seed)
random.seed(seed)
np.random.RandomState(seed)

# Q-learning & network
N_iterations = 3000000 # int(2e8) # 200 #   

# discount factor
gamma = 0.99

# Q network update frequency
update_frequency = 4

# frame history length
agent_history_length = 4


use_target = True
# target network update frequency
target_update = 10000 # 100 # 
minibatch_size = 32

# replay buffer parameters
replay_memory_size = 1000000 # 10000 # 

# buffer prefilling steps
replay_start_size = 50000 # 500 # 

# adam parameters
step_size = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 1e-4
adam_params=dict(N_iterations=N_iterations,
                step_size=step_size,
                b1=adam_beta1,
                b2=adam_beta2,
                eps=adam_eps,
                )

# exploration (epsilon-greedy) schedule
eps_schedule_step = [0, 1e6, int(N_iterations / 8)]
eps_schedule_val = [1.0, 0.1, 0.01]
#eps_schedule_val = [0.2, 0.1, 0.01]
eps_schedule_args = dict(
    eps_schedule_step=eps_schedule_step, eps_schedule_val=eps_schedule_val
)

# video logging: default is to not log video so that logs are small enough
video_log_freq = -1
###############################################################


def get_env(seed):
    env = gym.make("MsPacman-v0")
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    expt_dir = "./"

    # the video recorder only captures a sampling of episodes
    # (those with episodes numbers which are perfect cubes: 1, 8, 27, 64, ... and then every `video_log_freq`-th).
    def capped_cubic_video_schedule(episode_id):
        if episode_id < video_log_freq:
            return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
        else:
            return episode_id % video_log_freq == 0

    env = wrappers.Monitor(
        env,
        osp.join(expt_dir, "video"),
        force=True,
        video_callable=(capped_cubic_video_schedule if video_log_freq > 0 else False),
    )

    # configure environment for DeepMind-style Atari
    env = wrap_deepmind(env) 
    return env


##### Create a breakout environment
# fix env seeds
env = get_env(seed)
# reset environment to initial state
frame = env.reset()
# get the size of the action space
n_actions = env.action_space.n

# define logger
rl_logger = learning_logger(env, eps_schedule_args)

###############################################################


##### create buffer
frame_shape = (env.observation_space.shape[0], env.observation_space.shape[1])
replay_buffer = ReplayBuffer(replay_memory_size, agent_history_length, lander=False)
# channel last format of the input
input_shape = (1,) + frame_shape + (agent_history_length,)


###############################################################

#####
print("build the Q learning network.\n")
##### Create deep neural net
model = Neural_Net(
                    n_actions,
                    input_shape,
                    adam_params,
                    use_target=use_target,
                    seed=seed
                )

#####
print("Start prefilling the buffer.\n")

tot_time = time.time()

##### prefill buffer using the random policy
pre_iteration = 0
while pre_iteration < replay_start_size:
    # reset environment
    state = env.reset()
    is_terminal = False

    while not is_terminal:

        # store state in buffer
        buffer_index = replay_buffer.store_frame(state)
        last_obs_encode = replay_buffer.encode_recent_observation()
        state_enc = np.expand_dims(last_obs_encode, 0)

        # take environment step and overwrite state; reward is not used to prefill buffer
        state, reward, is_terminal = episode_step(
                                                    pre_iteration,
                                                    env,
                                                    model,
                                                    replay_buffer,
                                                    buffer_index,
                                                    state_enc,
                                                    prefill_buffer=True,
                                    )
        pre_iteration += 1

print("\nFinished prefilling the buffer.\n")


# reset environment
state = env.reset()


#####
print("Start learning.\n")
##### run DQN
for iteration in range(N_iterations):

    # store state in buffer and compute its encoding
    buffer_index = replay_buffer.store_frame(state)
    last_obs_encode = replay_buffer.encode_recent_observation()
    state_enc = np.expand_dims(last_obs_encode, 0)

    # take  one episode step 
    state, reward, is_terminal = episode_step(
                                                iteration,
                                                env,
                                                model,
                                                replay_buffer,
                                                buffer_index,
                                                state_enc,
                                                eps_schedule_args=eps_schedule_args,
                                            )

    # update deep Q-net
    if iteration % update_frequency == 0:
        model.update_Qnet(replay_buffer, minibatch_size, gamma)

    # update target Q-net
    if iteration % target_update == 0:
        model.update_Qnet_target()

    if is_terminal:
        # print stats
        rl_logger.stats(iteration)

        # reset environment
        state = env.reset()


print("\n\ntotal time: {}".format(time.time() - tot_time))



rl_logger.plot(env.spec._env_name)


