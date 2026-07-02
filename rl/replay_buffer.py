import random
from collections import deque

import numpy as np


class ReplayBuffer:

    def __init__(self, capacity):

        self.buffer = deque(maxlen=capacity)

    def __len__(self):

        return len(self.buffer)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states = np.array(
            [item[0] for item in batch],
            dtype=np.float32
        )

        actions = np.array(
            [item[1] for item in batch],
            dtype=np.int64
        )

        rewards = np.array(
            [item[2] for item in batch],
            dtype=np.float32
        )

        next_states = np.array(
            [item[3] for item in batch],
            dtype=np.float32
        )

        dones = np.array(
            [item[4] for item in batch],
            dtype=np.float32
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def clear(self):

        self.buffer.clear()