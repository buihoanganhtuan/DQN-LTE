import warnings
import timeit
import json

import numpy as np
import wandb
import tensorflow as tf
import tensorflow.keras.backend as K
#from tensorflow.keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
# See this issue with importing CallbackList directly from tensorflow.keras.callbacks
# https://github.com/tensorflow/tensorflow/pull/23880#issuecomment-514821825\
# The below setting is based on https://github.com/wau/keras-rl2/blob/master/rl/callbacks.py
from tensorflow.keras import __version__ as KERAS_VERSION
from tensorflow.python.keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
from tensorflow.python.keras.utils.generic_utils import Progbar
from collections import deque # newly added
from tensorflow.python.platform import tf_logging as logging

#from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
#from keras.utils import Progbar

class Callback(KerasCallback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        pass

    def on_step_begin(self, episode, logs={}):
        """Called at beginning of each step"""
        pass

    def on_step_end(self, episode, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, episode, logs={}):
        """Called at beginning of each action"""
        pass

    def on_action_end(self, episode, logs={}):
        """Called at end of each action"""
        pass

# Now we are implementing specific callbacks
# Most of the callbacks will be to log data
# but other-purpose callbacks are also possible
class ModelIntervalCheckpoint(Callback):
    """ This specific callback type periodically save
        the weights during training """
    def __init__(self, filepath, interval, verbose=0):
        super().__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """ Save weights every interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            return
        
        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))
        # self.model is an attribute of base Keras callback class,
        # and is set using the set_model() method (see core.py)
        # where we set an Agent object as the model (although the)
        # Agent object itself is not a keras Model subclass
        self.model.save_weights(filepath, overwrite=True)

class TrainIntervalLogger(Callback):
    """ This callback type logs the process every interval steps. Initialize it with interval=1
        to log on every step (may slow down performance) """
    def __init__(self, interval=10000):
        self.interval = interval
        self.step = 0
        """ The reset step also predefined several necessary attributes e.g., self.metrics """
        self.reset()
    
    def reset(self):
        """ Mainly used to clear up results of the previous logging interval
            when we start a new logging interval """
        self.interval_start = timeit.default_timer()
        self.progbar = Progbar(target=self.interval)
        self.metrics = []
        """ Infos to be printed out """
        self.infos = []
        self.info_names = None
        self.episode_rewards = []

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        """ self.metrics_names attribute is defined here"""
        self.metrics_names = self.model.metrics_names
        """ The base class keras.callback does have a property named params
            which we will set later, in the core body """
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('Finished. Took {:.3f} seconds'.format(duration))

    def on_step_begin(self, step, logs):
        """ Check if another interval has passed and printout the metric if it did
            logs here is the accumulated logs """
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                """ self.metrics should have already contained recorded metrics for
                    all steps in the current interval """
                metrics = np.array(self.metrics)
                assert metrics.shape == (self.interval, len(self.metrics_names))
                formatted_metrics = ''
                if not np.isnan(metrics).all():
                    means = np.nanmean(self.metrics, axis=0)
                    assert means.shape == (len(self.metrics_names),)
                    for name, mean in zip(self.metrics_names, means):
                        formatted_metrics += ' - {}: {:,3f}'.format(name, mean)

                formatted_infos = ''
                if len(self.infos) > 0:
                    infos = np.array(self.infos)
                    if not np.isnan(infos).all():
                        means = np.nanmean(self.infos, axis=0)
                        assert means.shape == (len(self.info_names),)
                        for name, mean in zip(self.info_names, means):
                            formatted_infos += ' - {}: {:,.3f}'.format(name, mean)

                print('{} episodes - avg episode reward: {:,.3f} [min={:,.3f}, max={:,.3f}]{}{}'.format(len(self.episode_rewards), np.mean(self.episode_rewards),
                     np.min(self.episode_rewards), np.max(self.episode_rewards), formatted_metrics, formatted_infos))
                print('')
            self.reset()
            print('Interval {} ({} steps performed)'.format(self.step // self.interval + 1, self.step))

    def on_step_end(self, step, logs):
        """ Update the progression bar
            logs here is the instant log of the step"""
        if self.info_names is None:
            self.info_names = logs['info'].keys()
        values = [('reward', logs['reward'])]
        self.progbar.update(self.step % self.interval + 1, values=values)
        self.step += 1
        self.metrics.append(logs['metrics'])
        if len(self.info_names) > 0:
            self.infos.append([logs['info'][k] for k in self.info_names])

    def on_episode_end(self, episode, logs):
        self.episode_rewards.append(logs['episode_reward'])

class TrainEpisodeLogger(Callback):
    def __init__(self, moving_avg_winlen=20):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        # Newly added
        self.infos = {}
        self.step = 0
        # Newly added 2
        self.start = timeit.default_timer()
        self.moving_avg_winlen = moving_avg_winlen
        self.episode_rewards_record = deque(maxlen=self.moving_avg_winlen)
        self.total_rewards = 0.
        self.last_moving_avg = 0.

    def on_train_begin(self, logs):
        """ Print training values at the beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        if np.isinf(self.params['nb_steps']):
            self.params['nb_steps'] = 1000
        print('Training for {} steps ...'.format(self.params['nb_steps']))
        # Newly added
        self.infos_names = None

    def on_train_end(self, logs):
        """ Print total training time when training ends """
        duration = timeit.default_timer() - self.train_start
        print('Done, took {:,.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at the beginning of each episode"""
        # Note that if x is a dictionary, then x[number] = y means that
        # we are creating a NEW KEY number (that's right, the key is not
        # necessarily a string and can be a number, too), and cram the value y
        # in the entry keyed by number
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []
        # Newly added
        self.infos[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])
        episode_reward = np.sum(self.rewards[episode])
        # Newly added
        if len(self.episode_rewards_record) < self.moving_avg_winlen:
            self.total_rewards = self.total_rewards + episode_reward
            moving_average = self.total_rewards / max(1, len(self.episode_rewards_record))
        else:
            self.total_rewards = self.total_rewards - self.episode_rewards_record.popleft() + episode_reward
            moving_average = self.total_rewards / self.moving_avg_winlen
        self.episode_rewards_record.append(episode_reward)

        # Formatting the metrics
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)

        # Formatting the infos (newly added)
        infos = np.array(self.infos[episode])
        infos_template = ''
        infos_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.infos_names):
                if idx > 0:
                    infos_template += ', '
                try:
                    value = np.nanmean(infos[:, idx])
                    infos_template += '{}: {:.2f}'
                except Warning:
                    value = '--'
                    infos_template += '{}: {}'
                infos_variables += [name, value]
        infos_text = infos_template.format(*infos_variables)

        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        """template = '{step: ' + nb_step_digits + \
            'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f}, [{reward_min:.3f}, {reward_max:.3f}], rewards moving average: {rewards_mov_avg:.3f}, mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}, {infos}'
                
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'rewards_mov_avg': moving_average,            
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': metrics_text,
            'infos': infos_text,
        }"""

        template = '{step: ' + nb_step_digits + \
            'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, wallclock time: {wc_time:.3f}h, episode reward: {episode_reward:.3f} [{diff_vs_curr_avg:+.1f}], rewards moving average: {rewards_mov_avg:.3f} [{diff_vs_prev_avg:+.1f}], grads (x 1e6): {avg_grad:.2f} [max: {max_grad:.2f}], {metrics}, {infos}'

        gradstats = logs['episode_gradstats'] * 1e6
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'wc_time': (timeit.default_timer() - self.start) / 3600,
            'episode_reward': episode_reward,
            'diff_vs_curr_avg': episode_reward - moving_average,
            'rewards_mov_avg': moving_average,
            'diff_vs_prev_avg': moving_average - self.last_moving_avg,
            'avg_grad': gradstats[0],
            'max_grad': gradstats[2],            
            'metrics': metrics_text,
            'infos': infos_text,
        }

        print(template.format(**variables))

        self.last_moving_avg = moving_average
        # Free up resources
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]
        # Newly added
        del self.infos[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after every step
        The log passed in is a step log """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1
        # Newly added, just like in train interval logger
        # info = info from environment
        if self.infos_names is None:
            self.infos_names = list(logs['info'].keys())
        if len(self.infos_names) > 0:
            self.infos[episode].append([logs['info'][k] for k in self.infos_names])
        

class TestLogger(Callback):
    """ Logger Class for logging when agent is in testing mode """
    def on_train_begin(self, logs):
        """ Print logs at beginning of training"""
        print('Testing for {} episodes ...'.format(self.params['nb_episodes']))

    def on_episode_end(self, episode, logs):
        """ Print logs at end of each episode """
        template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
        variables = [
            episode + 1,
            logs['episode_reward'],
            logs['nb_steps'],
        ]
        print(template.format(*variables))

class FileLogger(Callback):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval
        
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        self.metrics = {}
        self.starts = {}
        self.data = {}

    def on_train_begin(self, logs):
        """ Initialize model metrics before training """
        self.metrics_names = self.model.metrics_names        

class WandbLogger(Callback):
    """ This callback type is pretty similar to TrainEpisodeLogger but will send the logs 
        back to weights & bias (wandb) for user to visualize the simulation result. 
    """
    def __init__(self, **kwargs):
        kwargs = {
            'project': 'keras-rl',
            'anonymous': 'allow',
            **kwargs
        }
        wandb.init(**kwargs)
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
    
    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        wandb.config.update({
            'params': self.params,
            'env': self.env.__dict__,
            'agent': self.model.__dict__,
        })

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """        
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and log training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        metrics = np.array(self.metrics[episode])
        metrics_dict = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    metrics_dict[name] = np.nanmean(metrics[:, idx])
                except Warning:
                    metrics_dict[name] = float('nan')
        
        wandb.log({
            'step': self.step,
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            **metrics_dict
        })

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1

""" Since some specific types of callbacks defined above do not actually implement
    all methods of the base callback class, it will cause an implementation error when
    we try to call a non-implemented method during the simulation e.g., ModelIntervalCheckpoint
    callback does not implement the on_episode_end() method and will give an error if we
    try to call the on_episode_end() of a ModelIntervalCheckpoint instance at the end of the episode. 
    Instead of having to manually make sure that we don't call the non-implemented method for a specific
    callback types, it is better to call all methods (at appropriate times) but fallback to certain 
    default behaviors if a method is not yet implemented. This is handled via the "CallbackList" class"""

class CallbackList(KerasCallbackList):
    # Note: callbacks (a list of callbacks) is a property of the base Keras CallbackList class
    def _set_env(self, env):
        """ Set environment (if possible) for each callback in the list """
        for callback in self.callbacks:
            if callable(getattr(callback, '_set_env', None)):
                callback._set_env(env)
    
    def on_episode_begin(self, episode, logs={}):
        """ At the beginning of each episode, call every callbacks in the list """
        for callback in self.callbacks:
            """ Important: Check if the current callback in the list support the
            on_episode_begin() method. If not, fallback to the on_epoch_begin() method
            of the base Keras callbacks to be compatible with Keras callbacks"""
            if callable(getattr(callback, 'on_episode_begin', None)):
                callback.on_episode_begin(episode, logs=logs)
            else:
                callback.on_epoch_begin(episode, logs=logs)
    
    def on_episode_end(self, episode, logs={}):
        """ At the end of each episode, call every callbacks in the list """
        for callback in self.callbacks:
            """ Similar as above, if the current callback in the list does not support
            the on_episode_end() method, fallback to on_epoch_end() of Keras base callback"""
            if callable(getattr(callback, 'on_episode_end', None)):
                callback.on_episode_end(episode, logs=logs)
            else:
                callback.on_epoch_end(episode, logs=logs)
    
    def on_step_begin(self, step, logs={}):
        """ At the begining of each step, call every callbacks in the list """
        for callback in self.callbacks:
            """ If on_step_begin() is not supported, fallback to base on_batch_begin() """
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_step_begin(step, logs=logs)
            else:
                callback.on_batch_begin(step, logs=logs)

    def on_step_end(self, step, logs={}):
        """ At the end of each step, call every callbacks in the list """
        for callback in self.callbacks:
            """ If on_step_end() is not supported, fallback to base on_batch_end() """
            if callable(getattr(callback, 'on_step_end', None)):
                callback.on_step_end(step, logs=logs)
            else:
                callback.on_batch_end(step, logs=logs)
    
    def on_action_begin(self, action, logs={}):
        """ Before performing an action, call every callbacks in the list """
        for callback in self.callbacks:
            """ In Keras, there is no method that on_action_begin() can fallback onto
            because on_action_begin() is an exclusive method defined for RL, so the
            best strategy here is no fallback and only execute if the method exists """
            if callable(getattr(callback, 'on_action_begin', None)):
                callback.on_action_begin(action, logs=logs)

    def on_action_end(self, action, logs={}):
        """ After performing an action, call every callbacks in the list """
        for callback in self.callbacks:
            """ Similarly, for this exclusive method, we only executes if exists """
            if callable(getattr(callback, 'on_action_end', None)):
                callback.on_action_end(action, logs=logs)

class ReduceLROnPlateau(Callback):
  """Reduce learning rate when a metric has stopped improving.

  Models often benefit from reducing the learning rate by a factor
  of 2-10 once learning stagnates. This callback monitors a
  quantity and if no improvement is seen for a 'patience' number
  of epochs, the learning rate is reduced.

  Example:

  ```python
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
  model.fit(X_train, Y_train, callbacks=[reduce_lr])
  ```

  Arguments:
      monitor: quantity to be monitored.
      factor: factor by which the learning rate will be reduced. new_lr = lr *
        factor
      patience: number of epochs with no improvement after which learning rate
        will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
        quantity monitored has stopped decreasing; in `max` mode it will be
        reduced when the quantity monitored has stopped increasing; in `auto`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
      cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
      min_lr: lower bound on the learning rate.
  """

  def __init__(self,
               monitor='val_loss',
               factor=0.1,
               patience=10,
               verbose=0,
               mode='max',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
               **kwargs):
    super(ReduceLROnPlateau, self).__init__()

    self.monitor = monitor
    if factor >= 1.0:
      raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.best_ep = 0
    self.mode = mode
    self.monitor_op = None
    self._reset()

  def _reset(self):
    """Resets wait counter and cooldown counter.
    """
    assert self.mode in ['min', 'max'], 'Mode must be either min or max'

    if self.mode == 'min':
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.Inf
    self.cooldown_counter = 0
    self.wait = 0

  def on_train_begin(self, logs=None):
    self._reset()

  def on_episode_end(self, episode, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.trainable_model.optimizer.lr)
    current = logs.get(self.monitor)
    if current is None:
      logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
    else:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

      if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
        self.best_ep = episode
      elif not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
          old_lr = float(K.get_value(self.model.trainable_model.optimizer.lr))
          if old_lr > self.min_lr:
            new_lr = old_lr * self.factor
            new_lr = max(new_lr, self.min_lr)
            K.set_value(self.model.trainable_model.optimizer.lr, new_lr)
            if self.verbose > 0:
              print('\n*****Episode %5d: ReduceLROnPlateau reducing learning '
                    'rate to %s. Current best: %f at episode %5d *****' % (episode + 1, new_lr, self.best, self.best_ep + 1))
            self.cooldown_counter = self.cooldown
            self.wait = 0

  def in_cooldown(self):
    return self.cooldown_counter > 0

class SelectiveCheckpoint(Callback):
    """ This specific callback type save the weights
    when a new record is detected during training """
    def __init__(self, monitor, filepath, start=0, mode='max', delta=1e-4, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.start = start
        self.verbose = verbose
        self.total_steps = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.check_op = None
        self.monitor = monitor
        self.delta = delta
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        assert self.mode in ['min', 'max'], 'Mode must be either min or max'

        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b + self.delta)
            self.check_op = lambda a, b: np.less(a, b)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b - self.delta)
            self.check_op = lambda a, b: np.greater(a, b)
            self.best = -np.Inf

    def on_train_begin(self, logs=None):
        self._reset()

    def on_step_end(self, step, logs={}):
        """ Save weights every interval steps during training """
        self.total_steps += 1

    def on_episode_end(self, episode, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            if self.check_op(current, self.best):
                self.best = current
                
            if self.total_steps >= self.start:
                filepath = self.filepath.format(episode=episode, step=self.total_steps, record=current//1, **logs)
                if self.verbose > 0:
                    print('Episode {}, Step {}: saving model to {}'.format(episode, self.total_steps, filepath))
                self.model.save_weights(filepath, overwrite=True)