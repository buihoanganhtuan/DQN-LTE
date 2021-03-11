import numpy as np
import operator

from tensorflow.keras.models import model_from_config, Sequential, Model, model_from_config

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.util.tf_export import keras_export

import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.backend as K
import tensorflow as tf

#from keras.models import model_from_config, Sequential, Model
#import keras.optimizers as optimizers
#import tensorflow as tf

def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone

def clone_optimizer(optimizer):
    if type(optimizer) is str:
        print(optimizer)
        return optimizers.get(optimizer)
    # Requires Keras 1.0.7 since get_config has breaking changes.
    params = dict([(k, v) for k, v in optimizer.get_config().items()])
    config = {
        'class_name': optimizer.__class__.__name__,
        'config': params,
    }
    if hasattr(optimizers, 'optimizer_from_config'):
        # COMPATIBILITY: Keras < 2.0
        clone = optimizers.optimizer_from_config(config)
    else:
        clone = optimizers.deserialize(config)
    return clone

def get_object_config(o):
    if o is None:
        return None
    
    config = {
        'class_name': o.__class__.__name__,
        'config': o.get_config()
    }
    return config

def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights), 'target network and online network have different number of weights'

    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates

def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0., 'gradient clipping value must be higher than zero'

    x = y_true - y_pred
    if np.isinf(clip_value):
        # We need to use tf here to return a tensor object as the loss
        # for compatibility with tensorflow later
        # Return the usual squared loss
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    # below is huber loss with delta = 1
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    # if above clip_value, return linear loss. Otherwise return squared loss
    if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)
    else:
        return tf.where(condition, squared_loss, linear_loss)

class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    # We defined a modified version of the passed-in optimizer that performs, 
    # in addition to the usual weight update on the online network, a soft update 
    # on the weights of the target network
    def __init__(self, optimizer, additional_updates):
        super().__init__(optimizer._name)
        #super().__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    # override the get_update method of the base keras Optimizer class
    def get_updates(self, params, loss):
        # Get the list of update ops (which only applies to the online network) of the
        # passed-in optimizer and additionally append another update op to the list.
        # The additional op is the op to apply a soft update to the target network
        # Note that here we are overwriting the base get_updates of base kears optimizer class
        updates = self.optimizer.get_updates(params=params, loss=loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()

@keras_export("keras.optimizers.schedules.OffsetExponentialDecay")
class OffsetExponentialDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses an exponential decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      decay_rate,
      decay_start,
      min_learning_rate,
      staircase=False,
      name=None):
    """
    """
    super(OffsetExponentialDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.min_learning_rate = min_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.decay_start = decay_start
    self.staircase = staircase
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "OffsetExponentialDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2(
          self.initial_learning_rate, name="initial_learning_rate")
      min_learning_rate = ops.convert_to_tensor_v2(
          self.min_learning_rate, name="min_learning_rate")          
      dtype = initial_learning_rate.dtype

      decay_steps = math_ops.cast(self.decay_steps, dtype)
      decay_rate = math_ops.cast(self.decay_rate, dtype)
      decay_start = ops.convert_to_tensor_v2(self.decay_start, dtype)

      global_step_recomp = math_ops.cast(math_ops.maximum(step - decay_start, ops.convert_to_tensor_v2(0.)), dtype)
      p = global_step_recomp / decay_steps
      if self.staircase:
          p = math_ops.floor(p)
      return math_ops.maximum(min_learning_rate, math_ops.multiply(
          initial_learning_rate, math_ops.pow(decay_rate, p)), name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "min_learning_rate": self.min_learning_rate,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "decay_start": self.decay_start,
        "staircase": self.staircase,
        "name": self.name
    }

# Based on https://github.com/openai/baselines/blob/master/baselines/common/mpi_running_mean_std.py
class WhiteningNormalizer(object):
    def __init__(self, shape, eps=1e-2, dtype=np.float64):
        self.eps = eps
        self.shape = shape
        self.dtype = dtype

        self._sum = np.zeros(shape, dtype=dtype)
        self._sumsq = np.zeros(shape, dtype=dtype)
        self._count = 0

        self.mean = np.zeros(shape, dtype=dtype)
        self.std = np.ones(shape, dtype=dtype)

    def normalize(self, x):
        return (x - self.mean)/self.std

    def denormalize(self, x):
        return self.std * x + self.mean

    def update(self, x):
        if x.ndim == len(self.shape):
            # The asterisk means that "expand self.shape into a tuple"
            # The -1 here means "unknown dimension,  numpy please figure it out"
            # Of course we cannot have more than one unknown dimension
            # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
            x = x.reshape(-1, *self.shape)
        # check if the dimensions (from 2nd dim onward) agree
        assert x.shape[1:] == self.shape

        self._count += x.shape[0]
        self._sum += np.sum(x, axis=0)
        self._sumsq += np.sum(np.square(x), axis=0)

        self.mean = self._sum / float(self._count)
        self.std = np.sqrt(np.maximum(np.square(self.eps), self._sumsq / float(self._count) - np.square(self.mean)))

class SegmentTree(object):
    """
    Abstract SegmentTree data structure used to create PrioritizedMemory.
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """
    def __init__(self, capacity, operation, neutral_element):
        #powers of two always have no bits in common with the previous integer => use that prop for checking
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be positive and a power of 2"
        self._capacity = capacity

        #a segment tree has (2*n)-1 total nodes
        self._value = [neutral_element for _ in range(2 * capacity)]

        # operation that shall be performed on the tree
        self._operation = operation

        self.next_index = 0

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        if isinstance(idx, int):
            assert 0 <= idx < self._capacity
            return self._value[self._capacity + idx]
        else:
            assert idx.start is None and idx.stop is None and idx.step is None, 'only support whole array slicing'
            return self._value[self._capacity:]

class SumSegmentTree(SegmentTree):
    """
    SumTree allows us to sum priorities of transitions in order to assign each a probability of being sampled.
    """
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        prefixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity

class MinSegmentTree(SegmentTree):
    """
    In PrioritizedMemory, we normalize importance weights according to the maximum weight in the buffer.
    This is determined by the minimum transition priority. This MinTree provides an efficient way to
    calculate that.
    """
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)