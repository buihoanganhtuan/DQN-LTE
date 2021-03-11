from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np
from rl2.util import SumSegmentTree, MinSegmentTree

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high
        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick
        # Returns
            A list of samples of length size, with values between low and high
            As usual, the sample indexes would be in the half-open interval of
            [low, high) i.e., low can be included, but high will not
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')              
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs

def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation
    # Argument
        observation (list): List of observation
    
    # Return
        A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.

class RingBuffer(object):
    # This is the implementation of a buffer for fast insertion/deletion/access
    # buffer that will be used in our Sequential memory implementation
    def __init__(self, maxlen):
        self.maxlen = maxlen
        # Note that when a maxlen is defined, deque can be extremely efficient in both
        # appending and random accessing. Also, when having maxlen and the dequeue has
        # been completely filled, appending a new element will automatically removes
        # the first one at O(1) complexity => deque with maxlen is the most suitable
        # solution for implementing a ring queue
        # https://stackoverflow.com/questions/4151320/efficient-circular-buffer
        self.data = deque(maxlen=maxlen)

    def __len__(self):
        # we will write the method length() later
        return self.length()

    # A python object will have this default __getitem__() method unimplemented
    # By implementing this __getitem__() method, we are creating a class whose instance
    # can be indexed
    # https://stackoverflow.com/questions/5359679/python-class-accessible-by-iterator-and-index
    def __getitem__(self, idx):
        """Return element of buffer at specific index
        # Argument
            idx (int): Index wanted
        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx > self.length():
            raise KeyError()
        return self.data[idx]

    def append(self, v):
        """Append an element to the buffer
        # Argument
            v (object): Element to append
        """
        return self.data.append(v)

    def length(self):
        """Return the length of Deque
        # Argument
            None
        # Returns
            The lenght of deque element
        """
        return len(self.data)                        

class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        """Return list of last observations
        # Argument
            current_observation (object): Last observation
        # Returns
            A list of the last observations
        """ 
        # When implementing this, remember to avoid the records corresponding to when
        # the agent is at the terminal state but does not know it yet
        # Also, a state here corresponds to a window of previous observations up to and
        # including this new observation. If you want the state to just contain a single 
        # last observation, then set the window_length of memory to 1
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            # If it is not far enough from the episode's beginning to obtain
            # enough consecutive recent observations to construct the state,
            # fill the remaining needed observations with zero-observation
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory
        
        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        # remember to include window_length as a kwarg when init
        # the sequential memory
        super().__init__(**kwargs)
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences
        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        # check if the minimum idx is beyond the no-sampling range (the first window_length+1 entries
        # in the memory). The plus one is because we also ignore the very first entry (as stated above)
        # alongside the first window_length entries
        assert np.min(batch_idxs) >= self.window_length + 1
        # check if the maximum idx is still within permittable range (<= nb_entries - 1)
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences (basically by setting state1 at an index as state0 of the very next entry
        # if the current entry does not correspond to a terminal state)
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice. Note that the sampled idx will be for state1
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            # Remember that a state is a contiguous window of observations
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            # Below, "Experience" is the named tuple define in the very top of this script
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
            
        assert len(experiences) == batch_size
        return experiences
                                                 
    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory
        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        super().append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and wether the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
        
    @property
    def nb_entries(self):
        """Return number of observations
        In other words, the number of entries currently in the memory
        # Returns
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory
        # Returns
            Dict of config
        """
        config = super().get_config()
        config['limit'] = self.limit              
        return config

class PartitionedRingBuffer(object):
    """
    Buffer with a section that can be sampled from but never overwritten.
    Used for demonstration data. Can be used without a partition,
    where it would function as a fixed-idxs variant of RingBuffer.
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.length = 0
        self.data = [None for _ in range(maxlen)]
        self.permanent_idx = 0
        self.next_idx = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0:
            raise KeyError()
        return self.data[idx % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        self.data[(self.permanent_idx + self.next_idx)] = v
        self.next_idx = (self.next_idx + 1) % (self.maxlen - self.permanent_idx)

    def load(self, load_data):
        assert len(load_data) < self.maxlen, "Must leave space to write new data."
        for idx, data in enumerate(load_data):
            self.length += 1
            self.data[idx] = data
            self.permanent_idx += 1

class PrioritizedMemory(Memory):
    def __init__(self, limit, alpha=.4, start_beta=1., end_beta=1., steps_annealed=1, reset=False, 
                    reset_max=False, reset_max_factor=1, reset_max_period=200000, **kwargs):
        super(PrioritizedMemory, self).__init__(**kwargs)

        #The capacity of the replay buffer
        self.limit = limit

        #Transitions are stored in individual RingBuffers, similar to the SequentialMemory.
        self.actions = PartitionedRingBuffer(limit)
        self.rewards = PartitionedRingBuffer(limit)
        self.terminals = PartitionedRingBuffer(limit)
        self.observations = PartitionedRingBuffer(limit)

        assert alpha >= 0
        #how aggressively to sample based on TD error
        self.alpha = alpha
        #how aggressively to compensate for that sampling. This value is typically annealed
        #to stabilize training as the model converges (beta of 1.0 fully compensates for TD-prioritized sampling).
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.steps_annealed = steps_annealed

        #SegmentTrees need a leaf count that is a power of 2
        tree_capacity = 1
        while tree_capacity < self.limit:
            tree_capacity *= 2

        #Create SegmentTrees with this capacity
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        # The initial max_priority of 1. is actually not safe for our system, because in some system e.g., ours, we will always have TD errors
        # much smaller than 1. => the max_priority will always remain 1. the new transition, on which most of the mass is concentrated on is gonna be 
        # sampled multiple times (causing a minibatch to sometime contain only the new transition
        # repeated many time) => high variance
        #self.max_priority = 1.
        self.max_priority = 1e-3

        #wrapping index for interacting with the trees
        self.next_index = 0

        # Whether or not to completely use uniform sampling (not prioritized sampling w/ IS correction)
        self.reset = reset

        # Whether or not to reset the max priority into (a multiple of) the second largest priority in store to avoid max locking
        self.reset_max = reset_max
        self.reset_max_factor = reset_max_factor
        self.reset_max_period = reset_max_period
        self.append_count = 0
        assert reset_max_factor >= 1., 'reset_max_factor cannot be smaller than 1'

    def append(self, observation, action, reward, terminal, training=True):\
        #super() call adds to the deques that hold the most recent info, which is fed to the agent
        #on agent.forward()
        super(PrioritizedMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.control_max()
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            #The priority of each new transition is set to the maximum
            self.sum_tree[self.next_index] = self.max_priority ** self.alpha
            self.min_tree[self.next_index] = self.max_priority ** self.alpha

            #shift tree pointer index to keep it in sync with RingBuffers
            self.next_index = (self.next_index + 1) % self.limit

            self.append_count += 1

    def _sample_proportional(self, low, high, batch_size):
        #outputs a list of idxs to sample, based on their priorities.
        #in this function, the sampled index is in the half open interval [low, high)
        assert high - low >= batch_size, 'Not enough samples'
        idxs = list()
        sum_priority = self.sum_tree.sum(0, self.limit - 1)
        for _ in range(batch_size):
            mass = random.random() * sum_priority
            idx = self.sum_tree.find_prefixsum_idx(mass)
            idx = min(high - 1, max(low, idx))
            idxs.append(idx)
        return idxs

    def sample(self, batch_size, step, beta=1., nstep=1, gamma=1.):
        if (step < self.steps_annealed) or not self.reset:
            idxs = self._sample_proportional(self.window_length + 1, self.nb_entries - nstep + 1, batch_size)
        else:
            idxs = sample_batch_indexes(self.window_length + 1, self.nb_entries - nstep + 1, size=batch_size)

        #importance sampling weights are a stability measure
        importance_weights = list()

        #The lowest-priority experience defines the maximum importance sampling weight
        # So the probability of selecting an experience can be either increased/decreased depending on it TD difference
        # However, once selected, their values used in calculating the loss is then scaled back by the importance weight so that the 
        # estimated loss resembles sampling from a hybrid distribution (between the one specified by TD differences and the uniform one)
        sum_priority = self.sum_tree.sum()
        #prob_min = self.min_tree.min() / self.sum_tree.sum()
        prob_min = self.min_tree.min() / sum_priority
        max_importance_weight = (prob_min * self.nb_entries)  ** (-beta)

        experiences = list()
        for idx in idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                # The below line is wrong for Prioritized Memory as that line implies uniform re-sampling
                #idx = sample_batch_indexes(self.window_length + 1, self.nb_entries - nstep, size=1)[0]
                if (step < self.steps_annealed) or not self.reset:
                    idx = self._sample_proportional(self.window_length + 1, self.nb_entries - nstep + 1, 1)[0]
                else:
                    # idx, NOT idxs!!! Previously you carelessly type idxs instead of idx so terminal0 is never updated !!!
                    # => infinite while loop !!!
                    idx = sample_batch_indexes(self.window_length + 1, self.nb_entries - nstep + 1, size=1)[0]
                terminal0 = self.terminals[idx - 2]

            assert self.window_length + 1 <= idx < self.nb_entries

            #probability of sampling transition is the priority of the transition over the sum of all priorities
            #prob_sample = self.sum_tree[idx] / self.sum_tree.sum()
            prob_sample = self.sum_tree[idx] / sum_priority
            importance_weight = (prob_sample * self.nb_entries) ** (-beta)
            #normalize weights according to the maximum value
            importance_weights.append(importance_weight/max_importance_weight)

            #assemble the initial state from the ringbuffer.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))

            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            # N-step TD
            """reward = 0
            nstep = nstep
            for i in range(nstep):
                reward += (gamma**i) * self.rewards[idx + i - 1]
                if self.terminals[idx + i - 1]:
                    #episode terminated before length of n-step rollout.
                    nstep = i
                    break

            terminal1 = self.terminals[idx + nstep - 1]"""

            
            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])
            # The original implementation is wrong since if we have nstep=1, then nstep may be rewritten to 0
            # by the "nstep=i" above. In that case, state1 would be exactly the same as state0
            """state1 = [self.observations[idx + nstep - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx + nstep - 1 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state1.insert(0, self.observations[current_idx])
            while len(state1) < self.window_length:
                state1.insert(0, zeroed_observation(state0[0]))"""

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size

        # Return a tuple whre the first batch_size items are the transititions
        # while -2 is the importance weights of those transitions and -1 is
        # the idxs of the buffer (so that we can update priorities later)
        return tuple(list(experiences)+ [importance_weights, idxs])

    def update_priorities(self, idxs, priorities):
        #adjust priorities based on new TD error
        for i, idx in enumerate(idxs):
            assert 0 <= idx < self.limit
            priority = priorities[i] ** self.alpha
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
            # Update the new max priority. This should be priority BEFORE APPLYING ^ alpha.
            # The original implementation is wrong
            #self.max_priority = max(self.max_priority, priority)
            self.max_priority = max(self.max_priority, priorities[i])

    def calculate_beta(self, current_step):
        a = float(self.end_beta - self.start_beta) / float(self.steps_annealed)
        b = float(self.start_beta)
        current_beta = min(self.end_beta, a * float(current_step) + b)
        return current_beta

    def control_max(self):
        if self.reset_max and (self.append_count > 0) and (self.append_count % self.reset_max_period == 0):
            # Note: the sum_tree store is priority after ^ alpha, not normal priority (i.e., normal TD)
            memory = self.sum_tree[:]
            # That's why we need to reverse ^ alpha to find the true priority
            second_largest = sorted(set(memory))[-2] ** (1 / self.alpha)
            if second_largest != 0.:
                print('Old max priority: {:.6f}, second highest priority: {:.6f}, new max priority: {:.6f}'.format(self.max_priority, second_largest, min(self.max_priority, second_largest * self.reset_max_factor)))
                self.max_priority = min(self.max_priority, second_largest * self.reset_max_factor)

    def get_config(self):
        config = super(PrioritizedMemory, self).get_config()
        config['alpha'] = self.alpha
        config['start_beta'] = self.start_beta
        config['end_beta'] = self.end_beta
        config['beta_steps_annealed'] = self.steps_annealed

    @property
    def nb_entries(self):
        """Return number of observations
        # Returns
            Number of observations
        """
        return len(self.observations)
