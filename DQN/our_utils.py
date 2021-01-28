def huber_loss(x, delta: float = 1.0):
    """
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.
    Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
    a vector of same shape of `x`.
    """
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    linear = abs_x - quadratic
    return 0.5 * quadratic ** 2 + delta * linear

def bellman_loss(Qnet_params, targets, actions, states):
    # for now we work only with a single trasition ... when implementing the buffer w'll revert bacj  to arrays and stuff
    predictions = predict(Qnet_params, states)
    # print(actions)
    preds_select = predictions[0][actions]

    # preds_select = jnp.take_along_axis(predictions, jnp.expand_dims(actions, axis=1), axis=1)

    return jnp.mean(huber_loss(preds_select - targets))