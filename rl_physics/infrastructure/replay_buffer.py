import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, reward, next_state, done):
        """ 
        Allows both single data point and batch data to be added to the replay buffer.

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
        """
        # Convert inputs to numpy arrays
        state = np.asarray(state)
        action = np.asarray(action)
        next_state = np.asarray(next_state)
        reward = np.asarray(reward)
        done = np.asarray(done)

        # Check if single data point or batch
        if state.ndim == 1:
            state = state[np.newaxis, :]
            action = action[np.newaxis, :]
            next_state = next_state[np.newaxis, :]
            reward = reward[np.newaxis]
            done = done[np.newaxis]

        batch_size = state.shape[0]

        idxs = np.arange(self.ptr, self.ptr + batch_size) % self.max_size

        self.state[idxs] = state
        self.action[idxs] = action
        self.next_state[idxs] = next_state
        self.reward[idxs] = reward
        self.done[idxs] = done.astype(float)

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)


    def sample(self, batch_size):
        """_summary_

        Args:
            batch_size (_type_): _description_

        Returns:
            tuple[torch Tensors]: state, action, reward, next_state, done
        """
        ind = np.random.randint(0, self.size, size=batch_size)
 
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
       ) 
    
    def save(self, filename):
        np.savez(filename, 
                 state=self.state, 
                 action=self.action, 
                 next_state=self.next_state, 
                 reward=self.reward, 
                 done=self.done,
                 ptr =self.ptr,
								 size=self.size)

    def load(self, filename):
        data = np.load(filename)
        self.state = data['state']
        self.action = data['action']
        self.next_state = data['next_state']
        self.reward = data['reward']
        self.done = data['done']
        self.ptr = data['ptr']
        self.size = data['size']
    
    def debug(self):
        # Print out some states
        print("Replay Buffer Debug Info:")
        print("State shape: ", self.state)