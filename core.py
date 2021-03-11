import warnings
import numpy as np

from copy import deepcopy
from tensorflow.keras.callbacks import History
#from keras.callbacks import History
from rl2.callbacks import (
    CallbackList, 
    TrainIntervalLogger, 
    TrainEpisodeLogger, 
    ModelIntervalCheckpoint, 
    TestLogger
)

class Agent(object):
    """Abstract base class for all implemented agents.
    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.
    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.
    To implement your own agent, you have to implement the following methods:
    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`
    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """

    def __init__(self, processor=None):
        self.processor = processor
        self.training = False
        self.step = 0

    def get_config(self):
        """Configuration of the agent for serialization.
        # Returns
            Dictionnary with agent configuration
        """
        return {}

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            nb_max_start_steps=0, start_step_policy=None, log_interval=10000, 
            nb_max_episode_steps=None):
        #When training, speficy a limit on the number of steps
        """Trains the agent on the given environment.
        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        
        if not self.compiled:
            raise RuntimeError('Compile your agent first by call compile() before fit()')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, got {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]

        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)

        # set_model() is a method of the base class CallbackList of Keras
        # although self is not a Keras Model subclass, it is still ok since
        # the set_model() method simply assign the "model" attribute of the
        # base class Keras callback to an instance of another class not necessarily
        # Keras model. We just need to remember that IF we try to access self.model
        # when defining a new callback, we are not actually referring to a Keras Model
        # object, but the Agent object, and we should define necessary method of the
        # Agent object for calling when defining a new callback. Remember that in Python
        # everything is an object and thus, the function can take instances of ANY classes
        # It is your job to make sure that you know what you are dealing with as an input
        callbacks.set_model(self)

        # _set_env() is our newly implemented method i.e., specifically for RL
        callbacks._set_env(env)

        params = {
            'nb_steps': nb_steps,
        }
        # set_params() is a method of the base class CallbackList of Keras
        # we need to pass this for the default fallback to Keras base callback 
        # (if we do fallback)
        callbacks.set_params(params)

        # the self._on_train_begin() refer's to the Agent's behavior (not the callbacks) when training begin
        # this is to be implemented later in a concrete class subclassing this abstract Agent class
        self._on_train_begin()

        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        # observation is the observed state of the agent. The same observation can either correspond to a terminal
        # state or a non-terminal state. Whether it is the terminal state is decided by the environment
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False

        # newly added
        episode_gradstats = []

        # the try here is not because we are expecting an error, but just because we want to use the try as 
        # a middle step for the "exception" on Keyboard interruption (in case you want to abort midway)
        try:
            while self.step < nb_steps:
                if observation is None:     # start of a new episode
                    callbacks.on_episode_begin(episode)     # we are not passing any logs here because usually only on_..._end() needs the logs
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None
                    
                    # If you want, you can make the agent perform a random number of random steps at beginning of each episode 
                    # and do not record them into the experience, if doing so helps randomizing the pattern
                    # This slightly changes the start position between episodes.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        # deciding which action to take during this phase
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn('Episode ended before {} random steps could be performed from episode start, try lowering nb_max_start_steps parameter'.format(
                                nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(observation)
                            break

                # At this point, we expect to be initialized, the below is to assert that we are initialized
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step (or action_repetition steps) in the environment. First thing first, make sure that we call on_step_begin of all callbacks
                callbacks.on_step_begin(episode_step)
                # (continue) this is where all the works happen. We first perceive and decide the action to be performed
                # (forward step) and then use the reward to improve (backward step)
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                # The info may be accumulated in the sense that when action_repetition > 1, one
                # action is repeated in more than one steps, and the info recorded is accumulated during
                # those repeating steps. If action_repetition = 1, info is recorded every step so basically
                # no accumulation, but the accumulated word is put here for generalization. Note that
                # info here is info coming from the environment, mainly used for debuggin purpose and
                # should not have meaningful impacts on agent's behaviors
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(
                            observation, r, done, info)
                    # recording the info
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            #accumulated_info[key] = np.zeros_like([value])
                            accumulated_info[key] = 0.
                        #accumulated_info[key] += value
                        # Newly added
                        try:
                            accumulated_info[key] += value
                        except:
                            accumulated_info[key] = value                        
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                
                # If we want to let the episode run as long as it needs, we don't need
                # to pass anything to the fit function since the default of nb_max_episode_steps is None
                # or if you want to make sure, pass nb_max_episode_steps=None to the fit function
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state
                    done = True
                # Newly added "if" for alleviating sparse reward problem
                """if observation[:-1] == [0.0] * (len(observation) - 1):
                    if env.action_space.actions[action][0] == 1.:
                        #pseudo_r = 1e-7 * (1 + (1 - env.action_space.actions[action][1] / env.action_space.max_actions[1]) ** 3)
                        pseudo_r = 1e-7
                        metrics, g_stats = self.backward(pseudo_r, terminal=done) 
                    else:
                        pair = env.action_space.actions[action]
                        pseudo_r = 1e-7 * pair[0] * (1 - pair[1] / env.action_space.max_actions[1]) ** 3
                        metrics, g_stats = self.backward(pseudo_r, terminal=done)
                        #metrics, g_stats = self.backward(reward, terminal=done)
                else:
                    metrics, g_stats = self.backward(reward, terminal=done)"""
                metrics, g_stats = self.backward(reward, terminal=done)
                episode_reward += reward
                #newly added
                episode_gradstats.append(g_stats)

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(step=episode_step, logs=step_logs)
                episode_step += 1
                # the total number of steps performed is stored in self.step
                # episode_step only stores the number of steps performed in current episode
                # and will be reset when opening a new episode
                self.step += 1

                if done:
                    # the environment has reach the terminal state, but agent does not know yet
                    # so the action for this state must be taken as if it is not terminal i.e.,
                    # when evaluating an action, agent still consider reward + gamma*Q. In this
                    # case, we simply perform a forward-backward call and simply choose an action but does not apply it
                    # since we are resetting the env. We need to pass in terminal=False to the backward
                    # step since any action it perform at the terminal state will result in 1) no reward
                    # and 2) the environment transiting to a  (the agent should not know this when sampling from memory
                    # and we will take extra care when writing the memory class to avoid state1 of an experience belonging
                    # to another episode i.e., the initial state of a new episode). 
                    # Technically, we don't need to store the state at this point if our memory format is (s0,a,r,s1,ter)
                    # but since they implement the memory as (s0,a,r,ter) and takes s1 as s0 of the next entry, we do still
                    # need to save this weird state in the memory (only to get its s0 as s1 for the previous entry in the memory)
                    self.forward(observation)
                    self.backward(0., terminal=False)

                    # This episode has finished, reset to a new one
                    # If you want to see the average reward per step, then the class TrainEpisodeLogger
                    # does have it displayed as 'reward_mean'
                    # Also, we pass the same logs to every callbacks' on_episode_end() method.
                    # What they do with the logs is their business
                    if 'end_info' in step_logs['info']:
                        step_logs['info']['end_info']['episode_reward'] = episode_reward # newly added for plotting episode reward
                        episode_logs = {
                            'episode_reward': episode_reward,
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                            'episode_gradstats': np.mean(episode_gradstats, axis=0),
                            'end_info': step_logs['info']['end_info']
                        }
                    else:                       
                        episode_logs = {
                            'episode_reward': episode_reward,
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                            'episode_gradstats': np.mean(episode_gradstats, axis=1)
                        }
                    callbacks.on_episode_end(episode, logs=episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_gradstats = []
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        # self._on_train_end() describes what THE AGENT CALLBACK (NOT THE PASSED-IN CALLBACKS) will do on training's end
        self._on_train_end()

        return history   

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, nb_max_episode_steps=None,
                nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.
        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError('You try to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`') 
        if action_repetition < 1:
            raise ValueError('action_repetition must be >=1, got {} instead',format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        callbacks.set_params(params)

        self._on_test_begin()
        # with non-agent callbacks, we don't distinguish between testing and training
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0.

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None
            # Perform random starts at beginning of episode (if specified) 
            # and do not record them into the experience.
            # This slightly changes the start position between episodes.            
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Episode ended before {} random steps could be performed at the start. Try lowering the `nb_max_start_steps` param'.format(nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break
            
            # Run the episode until we're done.
            done = False
            while not done:
                # The "on step" callback when testing actually operates on episode step, not total step
                # since we do not care about improvement over time when testing
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = 0.
                        #accumulated_info[key] += value
                        # Newly added
                        try:
                            accumulated_info[key] += value
                        except:
                            accumulated_info[key] = value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, logs=step_logs)
                episode_step += 1
                self.step += 1
            
            # Similar to when training, at this point we have reach a terminal state
            # but agent has not yet know
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report the episode's end
            if 'end_info' in step_logs['info']:
                step_logs['info']['end_info']['episode_reward'] = episode_reward # newly added for plotting episode reward
                episode_logs = {
                    'episode_reward': episode_reward,
                    'nb_steps': episode_step,
                    'end_info': step_logs['info']['end_info']
                }
            else:
                episode_logs = {
                    'episode_reward': episode_reward,
                    'nb_steps': episode_step,
                }
            callbacks.on_episode_end(episode, logs=episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history

        # When testing, specify a limit on the number of episodes.
    def reset_states(self):
        """ Resets all internally kept states after an episode is completed. """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.
        # Argument
            observation (object): The current observation from the environment.
        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.
        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        # Returns
            List of metrics values
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics={}):
        """Compiles an agent and the underlaying models to be used for training and testing.
        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.
        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.
        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    def _on_train_begin(self):
        """ Callback that is called before training begins. """        
        pass

    def _on_train_end(self):
        """ Callback that is called after training ends. """
        pass        

    def _on_test_begin(self):
        """ Callback that is called before testing begins. """        
        pass

    def _on_test_end(self):
        """ Callback that is called after testing ends. """
        pass

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        # Returns
            A list of metric's names (string)
        """
        return []       
    



class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    To implement your own environment, you need to define the following methods:
    - `step`
    - `reset`
    - `render`
    - `close`
    Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
    """
    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # If the step() method of this abstract class is called, then inform user that the subclass did not implement this method
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def __del__(self):
        self.close()

    def __str__(self):
        # This is the method to return a string containing information about the env type/address, etc.
        # that python prints onto screen when you plainly type in the instance of Env
        return '<{} instance>'.format(type(self).__name__)

class Space(object):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    Please refer to [Gym Documentation](https://gym.openai.com/docs/#spaces)
    """

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError()

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        raise NotImplementedError()

class Processor(object):
    """Abstract base class for implementing processors.
    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.
    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own. If you don't want to process anything, then just pass processor=None or don't
    pass any processor into the agent, in that case it will default to processor=None
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.
        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.
        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            observation (object): An observation as obtained by the environment
        # Returns
            Observation obtained by the environment processed
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            reward (float): A reward as obtained by the environment
        # Returns
            Reward obtained by the environment processed
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            info (dict): An info as obtained by the environment
        # Returns
            Info obtained by the environment processed
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        # Arguments
            action (int): Action given to the environment
        # Returns
            Processed action given to the environment
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        # Arguments
            batch (list): List of states
        # Returns
            Processed list of states
        """
        return batch                        

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.
        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []                