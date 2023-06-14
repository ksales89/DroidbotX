import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ExperienceReplay:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.memory = []

    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def sample_batch(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        states = torch.tensor([experience[0] for experience in batch], dtype=torch.float64)
        actions = torch.tensor([experience[1] for experience in batch], dtype=torch.int64)
        rewards = torch.tensor(np.array([experience[2] for experience in batch]), dtype=torch.float32)
        next_states = torch.tensor([experience[3] for experience in batch], dtype=torch.float32)
        dones = torch.tensor([experience[4] for experience in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones