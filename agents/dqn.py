# https://github.com/wau/keras-rl2/blob/master/rl/agents/dqn.py
from __future__ import division
import warnings

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Layer, Dense

from rl2.core import Agent
from rl2.util import *
from rl2.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl2.memory import PrioritizedMemory

import tensorflow as tf
import itertools
import timeit
tf.compat.v1.disable_eager_execution()

def mean_q(y_true, y_pred):
    # Despite this being a relatively straightforward computation,
    # we still use Keras backend to get a tensor instead of simple lists
    return K.mean(K.max(y_pred, axis=-1))

class AbstractDQNAgent(Agent):
    def __init__(self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=25000,
                 train_interval=1, memory_interval=1, target_model_update=0.2,
                 delta_clip=np.inf, custom_model_object={}, **kwargs):
        # If you have a processor to process state/reward, ..., remember to pass it in
        super().__init__(**kwargs)

        # Soft vs hard target model update
        if target_model_update < 0:
            raise ValueError('target_model_update must be >= 0.')
        elif target_model_update >= 1:
            # If target_model_update is higher than one, we interpret that user wants to use
            # hard update every target_model_update steps
            target_model_update = int(target_model_update)
        else:
            # If 0 <= target_model_update < 1, we interpret that user wants to use soft update
            # with rate = target_model_update
            target_model_update = float(target_model_update)

        # Parameters
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_object = custom_model_object

        # Related objects:
        self.memory = memory

        # State
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        # the "process_state_batch" method below is of
        # the processor, not the agent. Remember that
        # we are writing this method for the agent itself
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        # compute the q_values of a whole batch of states
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        # In python, len() return the size of FIRST dimension
        # that is, if you have a 2-dim array i.e., a matrix,
        # then len() return the number of ROWS, not COLUMNS as in MATLAB
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_value(self, state):
        # compute the q_values for a single state
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions, )
        return q_values
    
    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }

# We then implement a concrete DQN Agent model below, which contains both Vanilla DQN and DDQN
class DQNAgent(AbstractDQNAgent):
    """
    # Arguments
        model__: A Keras model.
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enable target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn__: A boolean which enable dueling architecture proposed by Mnih et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581).
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)
    """    
    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
                 dueling_type='avg', n_step=1, log_grads=True, reset_mem=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Validate important input
        #if hasattr(model.output, '__len__') and len(model.output) > 1:
        if isinstance(model.output, list):
            # This means that the (underlying) network (not the trainable network that we will construct later)
            # has more than one output, which is usable for DQN as the underlying DQN network can only have one
            # output (the predicted Q(s,a))
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output'.format(model))
        if list(model.output.shape) != list((None, self.nb_actions)):
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))
        
        # Parameters
        self.log_grads = log_grads
        self.doubling_batch_interval = kwargs.get('doubling_batch_interval', None)
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            # get the second last layer of the model, abandon the last layer
            layer = model.layers[-2]
            nb_action = model.output.shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"
            model = Model(inputs=model.input, outputs=outputlayer)

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy
        self.lr = None
        self.n_step = n_step

        # State
        self.reset_states()

        #flag for changes to algorithm that come from dealing with importance sampling weights and priorities
        self.prioritized = True if isinstance(self.memory, PrioritizedMemory) else False
        self.reset_mem = reset_mem

    def get_config(self):
        config = super().get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics):
        # a custom metric compatible with keras must be a function that takes exactly two argument
        # of name "y_true" and "y_pred" and output a number (whether based on that two inputs or not). 
        # Our mean_q function defined up top this file is indeed one of such function
        # Here, besides keras default metrics i.e., [total_loss, out0_loss, out1_loss], we can have
        # several custom metrics, and mean_q is one default custom metric
        metrics += [mean_q]

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        # We also never DIRECTLY train the online model (but indirectly via our later constructed
        # trainable model) so the loss property of the underlying model itself can also be set arbitrarily
        self.target_model = clone_model(self.model, self.custom_model_object)
        # use standard keras compile method of class Model to compile the target and the online network
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Now we will start writing the code to construct and compile a trainable custom model for Q-learning
        if self.target_model_update < 1.:
            # Use the modified optimizer to add an update op (on top of the ones produced by the passed-in
            # optimr) that soizeft-update the weight of the target network
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        # Why does sometimes we pass arguments in with asterisk (*) and sometime not?
        # https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/
        def clipped_masked_error(args):
            y_true, y_pred, importance_weights, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask    # apply the mask since we only want to get the difference in Q value for a specific action
            #adjust updates by importance weights. Note that importance weights are just 1.0
            #(and have no effect) if not using a prioritized memory            
            return K.sum(loss * importance_weights, axis=-1)

        # At this part, we will construct the trainable model. We define a model with two outputs: One output is the 
        # masked & clipped loss of the online network and the other output is the prediction on all actions of the online model
        # (which we will use for enabling computations of custom RL-exclusive metrics). As such, the keras "loss" on the first output
        # should be the output itself (since it is already the loss of the online model) and the keras "loss" on the second ourput
        # should be zero since we are only using the second output to compute the custom metrics and the second output should NOT
        # affect learning in anyway. To define the custom loss for the two outputs of the trainable model, we use a list of two python
        # lambda functions. As "loss", each function in the list must take exactly two arguments i.e., y_true and y_pred 
        # (just like metrics) and output a single number. The first function in the Lambda layer (i.e., the custom Keras "loss" 
        # for the 1st output) should return the output itself i.e., y_pred, while the second function should return 0.
        # The Keras Lambda layer (which is basically a layer with no trainable weights and is only used for computing custom keras "losses",
        # and is NOT python lambda functionality) will be in charge of computing the true loss of the online network to output as the
        # 1st output in the trainable model. The input of the trainable models, on the other hand, should be both the state, the action
        # (inform of a mask), and the target Q (produced by the target net) for all actions
        # Note that the y_pred below represents the output of the online model
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        importance_weights = Input(name='importance_weights',shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, importance_weights, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs = ins + [y_true, importance_weights, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        # Define a list of custom metrics to be calculated from the second output i.e., the prediction of the online model
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred : y_pred,
            lambda y_true, y_pred : K.zeros_like(y_pred),
        ]
        # If you use tensorflow.keras instead of keras, there will be an error caused by the AdditionalUpdateOptimizer class
        # which was written for keras (tensorflow.keras and keras have quite different implementations of optimizers)
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        #trainable_model.compile(optimizer='adam', loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        # Newly added
        self.grads_tensor = self.trainable_model.optimizer.get_gradients(self.trainable_model.total_loss, self.trainable_model.trainable_weights)
        self.grads_tensor = [K.flatten(w) for w in self.grads_tensor]
        self.grads_tensor = K.abs(K.concatenate(self.grads_tensor))
        self.get_grads = K.function((self.trainable_model._feed_inputs + self.trainable_model._feed_targets), self.grads_tensor)           

        self.compiled = True

    # Implement the load_weights method of the bas Agent model
    # Note that the online model itself (which is a Keras model)
    # also has a built-in load_weights() method. We only need to
    # call this method and additionally copy the weights to the target_net
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            # Remember that the online and target nets (both are Keras models)
            # also have built-in reset_states() method
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def forward(self, observation):
        # Select an action based on the current observation
        # A state here is a window of recent observations including
        # the latest one. If you want the state to just corresponds to the
        # last observation, then either set the window_length of the memory to 1
        # or just takes the first element of the list i.e., state = ...[0]
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_value(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        self.recent_observation = observation
        self.recent_action = action

        return action
    
    def backward(self, reward, terminal):
        # Store most recent experience in memory
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal, training=self.training)
        
        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # update batch size
        if self.doubling_batch_interval is not None and self.step > self.nb_steps_warmup and self.step % round(self.doubling_batch_interval) == 0:
            self.batch_size = min(self.batch_size*2, 4096)

        # Train the network on a single stochastic batch.
        grads = 0.
        #a = timeit.default_timer()
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:

            if self.prioritized:
                # Calculations for current beta value based on a linear schedule.
                current_beta = self.memory.calculate_beta(self.step)
                # Sample from the memory.
                experiences = self.memory.sample(self.batch_size, self.step, current_beta, self.n_step, self.gamma)
            else:
                #SequentialMemory
                experiences = self.memory.sample(self.batch_size)
                assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            importance_weights = []
            # We will be updating the idxs of the priority trees with new priorities
            pr_idxs = []
            if self.prioritized:
                for e in experiences[:-2]: # Prioritized Replay returns Experience tuple + weights and idxs.
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)
                #importance_weights = experiences[-2]
                # Newly added
                if (self.step < self.memory.steps_annealed) or not self.reset_mem:
                    importance_weights = experiences[-2]
                else:
                    importance_weights = [1. for _ in range(self.batch_size)]
                pr_idxs = experiences[-1]
            else:   
                #SequentialMemory            
                for e in experiences:
                    # e is a named tuple (see the memory implementation)
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)
            
            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)                              
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,)) 
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = (self.gamma**self.n_step) * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch

            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            if not self.prioritized:
                importance_weights = [1. for _ in range(self.batch_size)]
            #Make importance_weights the same shape as the other tensors that are passed into the trainable model
            assert len(importance_weights) == self.batch_size
            importance_weights = np.array(importance_weights)
            importance_weights = np.vstack([importance_weights]*self.nb_actions)
            importance_weights = np.reshape(importance_weights, (self.batch_size, self.nb_actions))

            # Finally, perform a single gradient update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            # Note that below, the [dummy_targets, targets] Keras "targets" are not used for our
            # custom metrics (mean_q), but may be used for other default keras metrics so we need
            # to properly pass [dummy_targets, targets] as y_true of the trainable network for the
            # two outputs.
            # Remember: the OUTPUT (y_pred) of the model and the LOSS of the model are not the same. LOSS
            # can be seen as an additional layer that take y_pred and y_target in and produce a loss value out
            # The output0 of the trainable_model is the precomputed loss (a scalar) and we have also defined a 
            # custom loss func for this output. This custom loss function will always output the precomputed loss
            # as the true loss so the so-called "target computed loss" a.k.a., "dummy_target" is not used at all
            # when computing the custom loss for this output0
            x, y, _ = self.trainable_model._standardize_user_data(ins + [targets, importance_weights, masks], [dummy_targets, targets])
            #metrics = self.trainable_model.train_on_batch(ins + [targets, importance_weights, masks], [dummy_targets, targets])
            metrics = self.trainable_model.train_on_batch(x, y)

            if self.prioritized:
                # The if condition is to prevent the initially large TD differences from locking the max priorty in the memory at too high value
                if self.step >= 2 * self.nb_steps_warmup:                
                    assert len(pr_idxs) == self.batch_size
                    #Calculate new priorities. The original implementation below is wrong since it sum over the TD errors for ALL action, not the recorded action only
                    #y_true = targets
                    #y_pred = self.model.predict_on_batch(ins)
                    y_true = targets[range(self.batch_size), action_batch]
                    y_pred = self.model.predict_on_batch(ins)[range(self.batch_size), action_batch]
                    #Proportional method. Priorities are the abs TD error with a small positive constant to keep them from being 0.
                    #new_priorities = (abs(np.sum(y_true - y_pred, axis=-1))) + 1e-5
                    new_priorities = (abs(y_true - y_pred)) + 1e-7
                    assert len(new_priorities) == self.batch_size
                    #update priorities
                    self.memory.update_priorities(pr_idxs, new_priorities)

            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)] # throw away individual default losses of the two outputs (see metrics_names property)
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics
            
            # Newly added
            if self.log_grads:
                grads = self.get_grads(x + y)
        
        #print(timeit.default_timer() - a)
        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()
        
        return metrics, [np.mean(grads), np.min(grads), np.max(grads)]

    # The decorator "@property" implicitly means "@property.getter", which means that we are actually
    # decorating the function "layers" with the property.getter decorator i.e., after the decoration,
    # the new "layers" will point to property.getter(old "layers" func i.e., the function we define below)
    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # In Keras, the default metrics_names are ["total loss", "loss of output 0", "loss of output1", 
        # ..., "loss of output n-1"]. These default metrics are always present i.e., if user specify any 
        # additional metrics e.g., 'mse', the metrics will be APPENDED to the list of metrics (not replacing)
        # and the list of metrics will look like this ["total loss", "loss of output 0", ..., "loss of output n-1",
        # "mse of output 0", ..., "mse of output n-1"]. Since in our trainble_model design principle, we have two outputs, the first
        # output is already the loss and the second output is the predicted Q (with a custom 0 loss since we are only
        # outputtting the prediction Q for calculating our custom metrics, not for imposing loss as that is already
        # done by the first output), the Keras metrics will be ["total loss", "loss of output0", "loss of output1",
        # any_additional_metrics_for_each_of_the_outputs (we can even speficy different metrics for different outputs as in
        # https://stackoverflow.com/questions/54750890/multiple-metrics-to-specific-inputs)]. Since our custom loss functions
        # for output0 and output1 ensures that "loss of output0" = output0 (as output0 is already the loss calculated via Lambda layer)
        # and that "loss of output1" = 0, the "total loss" = loss_of_out0 + loss_of_out1 = out0 => we don't need to keep the
        # default individual loss for each output (index 1 and 2 in the metrics list). Just keeping the total loss and
        # additional metrics are enough

        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy
    
    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)
    
    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
