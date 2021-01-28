import numpy as np

def get_state_id(state):
    id = 0
    for index, value in enumerate(state.flatten()):
        id += 2**index * value
    return id


class Buffer():
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def __init__(self, size=1000):
        self.size = size
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        state_id = get_state_id(state)
        next_state_id = get_state_id(next_state)
        if len(self.states) >= self.size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)

        self.states.append(state_id)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state_id)
        self.dones.append(done)

    def sample(self, batch_size=32):
        if batch_size > len(self.states):
          raise NameError('buffer is not filled.')
        batch = np.empty((5, batch_size))


        for i, _ in enumerate(range(batch_size)):
          #  take random index
          j = np.random.choice(len(self.states))
          # add to batch
          batch[0][i] = self.states[j]
          batch[1][i] = self.actions[j]
          batch[2][i] = self.rewards[j]
          batch[3][i] = self.next_states[j]
          batch[4][i] = self.dones[j]

        return batch



# a = range(10)
# buffer = Buffer(5)
# for i in a:
#     buffer.add(i)
#     print(buffer.b)
