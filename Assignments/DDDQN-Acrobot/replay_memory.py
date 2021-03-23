from collections import deque
import numpy as np


class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_store = deque(maxlen = self.memory_size)

    def reset(self):
        self.memory_store = deque()

    def add_record(self, state, action, reward, next_state, done):
        record_obj = {'state' : state, 'action' : action, 'reward' : reward, 'next_state' : next_state, 'done' : done}
        self.memory_store.append(record_obj)

    def get_batch(self, batch_size):
        batch_size = min(batch_size, len(self.memory_store))
        batch = np.random.choice(self.memory_store, batch_size)
        return np.array(batch), batch_size