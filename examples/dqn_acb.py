import sys
sys.path.insert(0, "c:\\Users\\zzzcr\\Desktop\\keras-project-2")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay

#from keras.models import Model, Sequential
#from keras.layers import Dense, Activation, Flatten, Input
#from keras.optimizers import Adam
#import keras.backend as K

# Note that importing from dqn is very important since the file dqn has
#       import tensorflow as tf
#       tf.compat.v1.disable_eager_execution()
# which disable eager execution and let us get a symbolic "+" when adding
# two tensor in AdditionalOptimizer instead of directly carry out the
# "+"" operator to get a float tensor. Try adding, says 
# model.trainable_weights[0]+model.trainable_weights[0] and see
# If you get <tf.Tensor 'add:0' shape=(something) dtype=float32>
# then you correctly get the symbolic + (note the "add" type)
from rl2.environments.lte_env import ACB_env
from rl2.agents.dqn import DQNAgent
from rl2.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl2.memory import SequentialMemory, PrioritizedMemory
from rl2.core import Processor
from rl2.callbacks import ModelIntervalCheckpoint, ReduceLROnPlateau, SelectiveCheckpoint
from rl2.specialized_callbacks.ltecallbacks import lte_visualizer, lte_episode_logger
from rl2.util import OffsetExponentialDecay
from rl2.specialized_processor.lte_processors import *

# Note: You don't need to import TrainEpisodeLogger and TrainIntervalLogger
# you can set verbose=1 as input to the fit function to enable IntervalLogger
# or verbose=2 to enable EpisodeLogger
    
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
delay_thresh = 1000
nb_attempts = 2
nb_acb_update = 2
SI_win_len = 1
nb_idle_thresh = delay_thresh - nb_attempts*(1 + 5) - nb_acb_update*SI_win_len
nb_tx_thresh = nb_attempts
nb_rx_thresh = nb_attempts*5 + nb_acb_update*SI_win_len
energy_thresh = (nb_idle_thresh * 0.025e-3 + nb_tx_thresh * 50e-3 + nb_rx_thresh * 50e-3)*1e-3
env_config = {
    'mode': 'episodic',
    'backlog_log': False,
    'n': 30000,
    'R': 54,
    'T_RAO': 5,
    'T': 10,
    'traffic_mode': 'beta',
    'acb_mode': True,
    'T_bar': 0,
    'bo_mode': True,
    'bi': 20,
    'control_backoff': False,
    'control_tbar': True,
    'delay_thresh': delay_thresh,
    'energy_thresh': energy_thresh,
    'SI_win_len': SI_win_len,
    'start_offset': 0,
    'end_trail': 0,    
}
# Why ** below? This is basically an "unpacking" sign
# that inform python to decompose the dictionary into
# keyword-value pair
# https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist
# https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters
env = ACB_env(**env_config)
nb_actions = env.action_space.nb_actions

#if env_config['control_backoff'] | env_config['control_tbar']:
#    INPUT_SHAPE = (6,) # observation is of the form [Ns, Nc, Pacb, bo/tbar] or [Ns, Nc, Davg, Evag, Pacb, bo/tbar]
#else:
#    INPUT_SHAPE = (5,) # observation is of the form [Ns, Nc, Pacb] or [Ns, Nc, Davg, Evag, Pacb]
INPUT_SHAPE = (5,) # processed observation is of the form [Ns, Nc, Davg, Evag, action_index]
WINDOW_LENGTH = 20

# Next, we build our model. We need the Flatten layer to prevent Keras from
# assigning one prediction to each of the observation inside the same state
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model_input = Input(shape=input_shape, name="input_state")
x = Flatten()(model_input)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
model_output = Dense(units=nb_actions, activation='linear')(x)
model = Model(inputs=model_input, outputs=model_output, name='online_network')
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
#memory = PrioritizedMemory(limit=2**16, alpha=.4, start_beta=.2, end_beta=1., steps_annealed=1000000, window_length=WINDOW_LENGTH)
#processor = LteProcessor()
# number of rx subframes = due to W_RAR and listening of ACB factors
processor = LteProcessorConstraintNew2(delay_thresh=delay_thresh, energy_thresh=energy_thresh, ratio=1., 
                                        action_space_size=nb_actions)

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
nb_steps = np.inf
nb_annealing_steps = 1000000
nb_steps_warmup = 25000
nb_test_eps = 25
offset = 0
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.0, nb_steps=nb_annealing_steps)

dqn = DQNAgent(model, nb_actions=nb_actions, policy=policy, test_policy=policy, memory=memory, processor=processor, 
               enable_double_dqn=False, enable_dueling_network=True, nb_steps_warmup=nb_steps_warmup, gamma=.9, target_model_update=25000, train_interval=1, delta_clip=10.)
lr = OffsetExponentialDecay(initial_learning_rate=1e-4, min_learning_rate=1e-6, decay_start=nb_annealing_steps//2, decay_steps=nb_annealing_steps//2, decay_rate=.5, staircase=True)
#dqn.compile(optimizer=Adam(learning_rate=lr), metrics=['mae'])
#lr = ExponentialDecay(initial_learning_rate=.0001, decay_steps=nb_annealing_steps/2, decay_rate=.5, staircase=True)
#dqn.compile(optimizer=Adam(learning_rate=1e-4, beta_1=.99, beta_2=.99), metrics=['mae'])   
dqn.compile(optimizer=Adam(learning_rate=1e-4), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_acb_weights.h5f'
    checkpoint_weights_filename = 'dqn_acb_weights_{step}.h5f'
    selective_checkpoint_weights_filename = 'selective_weights_{episode}_{step}_{record}.h5f'
    log_filename = 'dqn_acb_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
    callbacks += [lte_visualizer(vis_interval_train=100, vis_window_train=20, mode=True)]
    callbacks += [lte_episode_logger(period=20, mode=True)]
    callbacks += [ReduceLROnPlateau(monitor='episode_reward', factor=.5, patience=50, verbose=1, mode='max', min_delta=0.5, cooldown=0, min_lr=2.5e-6)]
    #callbacks += [SelectiveCheckpoint(monitor='episode_reward', filepath=selective_checkpoint_weights_filename, start=nb_annealing_steps, mode='max')]
    dqn.fit(env, nb_steps=nb_steps, callbacks=callbacks, verbose=2)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    callbacks = [lte_visualizer(vis_interval_test=nb_test_eps, vis_window_test=nb_test_eps, mode=False, offset=offset)]
    callbacks += [lte_episode_logger(period=nb_test_eps, mode=False, offset=offset)]
    #dqn.test(env, callbacks=callbacks, nb_episodes=nb_test_eps+offset)
elif args.mode == 'test':
    weights_filename = 'dqn_acb_weights_1700000.h5f'        
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    callbacks = [lte_visualizer(vis_interval_test=nb_test_eps, vis_window_test=nb_test_eps, mode=False, offset=offset)]
    callbacks += [lte_episode_logger(period=nb_test_eps, mode=False, offset=offset)]
    dqn.test(env, callbacks=callbacks ,nb_episodes=nb_test_eps+offset)