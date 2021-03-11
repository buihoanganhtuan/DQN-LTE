import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pickle
import timeit

# Enable interactive environment so that showing the pyplot figures 
# won't block the flow of the program until the figure is closed
plt.ion()
from rl2.callbacks import Callback

class lte_visualizer(Callback):
    """ Callback class for visualizing agent's behavior during training/testing
        __init__ agruments:
            vis_interval_train: the average agent's behavior over a certain number of episodes will be plotted once
                every vis_interval_train episodes during training
            vis_window_train: the number of episodes to average the agent's behavior during training for plotting purpose.
                Naturally, vis_window_train must be  <= vis_interval_train
            vis_interval_test: similar to vis_interval_train but for testing case
            vis_window_test: similar to vis_window_train but for testing case
            mode: the mode you want this callback to operate, either True (train) or False (test)
    """
    def __init__(self, vis_interval_train=1000, vis_window_train=10, 
                    vis_interval_test=2, vis_window_test=1 , mode=True, offset=0):
        assert vis_interval_train >= vis_window_train
        assert vis_interval_test >= vis_window_test
        self.vis_interval_train = vis_interval_train
        self.vis_window_train = vis_window_train
        self.vis_interval_test = vis_interval_test
        self.vis_window_test = vis_window_test
        self.infos_names = None
        self.count = 0
        self.step = 0
        self.mode = mode
        self.x_lim = None
        self.offset = offset
        #https://stackoverflow.com/questions/30218802/get-parent-of-current-directory-from-python-script/30218825
        self.fig_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '\\figs'

    def reset(self):
        self.count = 0
        if self.mode:
            self.infos = {key : [ [] for _ in range(self.vis_window_train) ] for key in self.infos_names}
        else:
            self.infos = {key : [ [] for _ in range(self.vis_window_test) ] for key in self.infos_names}
    
    def on_train_begin(self, logs):
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        if self.mode:
            print('Plot agent\'s behavior once every {} episodes during training'.format(self.vis_interval_train))
        else:
            print('Plot agent\'s behavior once every {} episodes during testing'.format(self.vis_interval_test))

    def on_episode_end(self, episode, logs):
        if episode < self.offset:
            return
        else:
            episode = episode - self.offset
        condition1 = self.mode & ((episode - self.vis_window_train + 1) % self.vis_interval_train == 0)
        condition2 = ~self.mode & ((episode - self.vis_window_test + 1) % self.vis_interval_test == 0)
        if condition1 | condition2:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            if condition1:
                fig_name = '\\training_averages_episodes_{}_to_{}.png'.format(np.max([0, episode + self.offset - self.vis_window_train + 1]), episode + self.offset)
            else:
                fig_name = '\\testing_averages_episodes_{}_to_{}.png'.format(np.max([0, episode + self.offset - self.vis_window_test + 1]), episode + self.offset)
                with open('test_infos.pkl', 'wb') as f:
                    pickle.dump(self.infos, f)                            
            lines = []
            # Padding so that all episode records have equal length
            l = [len(x) for x in self.infos[self.infos_names[0]]]
            l = max(l)
            for name in self.infos_names:
                for x, _ in enumerate(self.infos[name]): 
                    self.infos[name][x] += [np.nan]*(l - len(self.infos[name][x]))
                    # Average the (padded) episodes and do the plotting
                lines.append(np.nanmean(self.infos[name], axis=0))
                if name == 'p_bar' or name == 'bo' or name == 't_bar' or name == 't_bar_0' or name == 't_bar_1':
                    if name == 'p_bar':
                        ax2.plot(lines[-1], 'r--*', label=name, alpha=.5) 
                    elif name == 't_bar' or name == 't_bar_0':
                        ax2.plot(lines[-1], 'k--x', label=name, alpha=.5)
                    elif name == 't_bar_1':
                        ax2.plot(lines[-1], 'c--x', label=name, alpha=.5)
                    else:
                        raise NotImplementedError('invalid control name')
                    ax2.set_ylim([0, 1])
                    ax2.set_yticks(np.arange(0, 1.1, .1))
                else:
                    ax1.plot(lines[-1], label=name)
            # https://stackoverflow.com/questions/26752464/how-do-i-align-gridlines-for-two-y-axis-scales-using-matplotlib
            up_lim = ax1.dataLim.intervaly[-1]
            nb_ticks = len(ax2.get_yticks())
            up_lim = np.ceil(up_lim / (nb_ticks - 1)) * (nb_ticks - 1)
            low_lim = 0.
            ax1.set_ylim([low_lim, up_lim])
            ax1.set_yticks(np.append(np.linspace(low_lim, up_lim, nb_ticks), [20, 54]))
            if self.x_lim is None:
                temp = ax1.dataLim.intervalx[-1]
                x_digit = np.floor(np.log10(temp))
                first_digit = temp // 10**x_digit
                if temp - first_digit * 10**x_digit > 5*10**(x_digit - 1):
                    self.x_lim = (first_digit + 1) * 10**x_digit
                else:
                    self.x_lim = first_digit * 10**x_digit + 5 * 10**(x_digit - 1)
            ax1.set_xlim([0, self.x_lim])
            ax1.grid()
            # https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
            handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            fig.legend(handles, labels, loc='upper center', ncol=int(np.ceil(len(ax1.lines + ax2.lines) / 2)), prop={'size': 10})        
            plt.savefig(self.fig_dir + fig_name)
            plt.close(fig)
            self.reset()
            return
        condition3 = episode % self.vis_interval_train < self.vis_window_train - 1
        condition4 = episode % self.vis_interval_test < self.vis_window_test - 1
        if condition3 | condition4:
            self.count += 1       
    
    def on_step_end(self, step, logs):
        if self.infos_names is None:
            self.infos_names = list(logs['info'].keys())
            if self.mode:
                self.infos = {key : [ [] for _ in range(self.vis_window_train) ] for key in self.infos_names}
            else:
                self.infos = {key : [ [] for _ in range(self.vis_window_test) ] for key in self.infos_names}            
        episode = logs['episode']
        # Episode count starts from 0
        if episode < self.offset:
            return
        else:
            episode = episode - self.offset
        condition1 = self.mode & (episode % self.vis_interval_train < self.vis_window_train)
        condition2 = ~self.mode & (episode % self.vis_interval_test < self.vis_window_test)
        if condition1 | condition2:
            for name in self.infos_names: self.infos[name][self.count].append(logs['info'][name])
        self.step += 1

class lte_episode_logger(Callback):
    def __init__(self, period, mode, moving_avg_winlen=20, offset=0):
        assert period > 0, 'plotting period cannot be smaller than 1'
        self.period = period
        self.end_infos_names = None
        self.nb_keys = 0
        self.mode = mode # True = training, False = testing
        self.offset = offset
        self.fig_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '\\figs_metrics'
        self.moving_avg_winlen = moving_avg_winlen
        self.steps_list = []
        self.gradstats = []
        self.wallclock_duration = []

    def on_train_begin(self, logs):
        self.start = timeit.default_timer()

    def on_episode_end(self, episode, logs):
        if self.end_infos_names is None:
            self.end_infos_names = list(logs['end_info'].keys())
            self.end_infos = {key: [] for key in self.end_infos_names}
            self.moving_avgs = {key: [] for key in self.end_infos_names}
            self.nb_keys = len(self.end_infos_names) + 1
            self.nb_rows = 2
            self.nb_cols = int(math.ceil(self.nb_keys / self.nb_rows))
        
        self.steps_list.append(logs['nb_steps'])
        if self.mode:
            self.gradstats.append((logs['episode_gradstats']))
        self.wallclock_duration.append((timeit.default_timer() - self.start) / 3600)

        for name in self.end_infos_names:
            # the logs[][name] can be either scalar or numpy array
            val = logs['end_info'][name] if np.ndim(logs['end_info'][name]) == 0 | (not self.mode) else np.mean(logs['end_info'][name])
            winlen = min(self.moving_avg_winlen, len(self.end_infos[name]))
            if len(self.end_infos[name]) < self.moving_avg_winlen:
                if len(self.end_infos[name]) == 0:
                    self.moving_avgs[name].append(val)
                else:
                    self.moving_avgs[name].append((self.moving_avgs[name][-1] * winlen + val) / (winlen + 1))
            else:
                self.moving_avgs[name].append((self.moving_avgs[name][-1] * winlen - self.end_infos[name][-self.moving_avg_winlen] + val) / winlen)
            self.end_infos[name].append(val)

        if episode < self.offset:
            return
        else:
            if (episode - self.offset + 1) % self.period == 0:            
                fig, axs = plt.subplots(self.nb_rows, self.nb_cols, figsize=[13.66, 7.68])
                if self.mode:
                    fig_name = '\\metrics_evolution_episode_0_to_{}_({}_steps).png'.format(episode, logs['nb_steps']) # modified
                    for (index, name) in enumerate(self.end_infos_names):
                        index = np.unravel_index(index, [self.nb_rows, self.nb_cols])
                        if np.ndim(self.end_infos[name][0]) == 0:
                            axs[index].plot(self.end_infos[name])
                        else:
                            temp = [np.mean(self.end_infos[name][i]) for i in range(len(self.end_infos[name]))]
                            axs[index].plot(temp, axis=-1)
                        axs[index].plot(self.moving_avgs[name], 'r')
                        axs[index].set_title(name)
                        axs[index].grid()

                    index = np.unravel_index(self.nb_keys - 1, [self.nb_rows, self.nb_cols])
                    g = np.array(self.gradstats)
                    axs[index].set_yscale('log')
                    axs[index].plot(g[:, 0], 'b')
                    axs[index].plot(g[:, 1], 'k')
                    axs[index].plot(g[:, 2], 'r')
                    axs[index].set_title('gradient')
                    axs[index].grid()

                else:
                    fig_name = '\\metrics_ecdf_of_{}_episodes.png'.format(self.period)
                    # https://stackoverflow.com/questions/3209362/how-to-plot-empirical-cdf-in-matplotlib-in-python
                    for (index, name) in enumerate(self.end_infos_names):
                        if np.ndim(self.end_infos[name][0]) == 0:
                            data = self.end_infos[name]
                        else:
                            data = np.array(self.end_infos[name])
                            data.flatten()
                        mean = np.mean(data)
                        index = np.unravel_index(index, [self.nb_rows, self.nb_cols])
                        axs[index].plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False))
                        axs[index].set_title(name + ', mean = {:.4f}'.format(mean))
                        axs[index].grid()
                # https://matplotlib.org/3.2.2/tutorials/intermediate/tight_layout_guide.html
                plt.tight_layout()
                plt.savefig(self.fig_dir + fig_name)
                plt.close(fig)
    
    def on_train_end(self, logs):
        if self.mode:
            with open('train_record.pkl', 'wb') as f:
                pickle.dump([self.end_infos, self.steps_list, self.gradstats], f)
        else:
            with open('test_record.pkl', 'wb') as f:
                pickle.dump(self.end_infos, f)