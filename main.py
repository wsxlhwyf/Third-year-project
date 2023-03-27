import numpy as np
import env
import gym
import stable_baselines3
import math
from math import log
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from IPython.display import clear_output
import os
import matplotlib.pyplot as plt


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True


# topology A
log_dir = "./tmp/network_Env-v0/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
env_args = dict(episode_length=100, load=0.1,
                mean_service_holding_time=10, k_paths=2, topology_num=None)
the_env = gym.make('network_Env-v0', **env_args)
the_env = Monitor(the_env, log_dir + 'training', info_keywords=('P_accepted', 'topology_num', 'load','reason','NF','MF','SF'))

"""# topology B
env_args_2 = dict(episode_length=5000, load=3,
                  mean_service_holding_time=10, k_paths=2, topology_num=None)
the_env_2 = gym.make('network_Env-v0', **env_args_2)
the_env_2 = Monitor(the_env_2, log_dir + 'training', info_keywords=('P_accepted', 'topology_num', 'load',))"""

# create agent
model = A2C("MultiInputPolicy", the_env, verbose=1, device='cuda', gamma=0.63, n_steps=10) #, learning_rate=0.0007
model.learn(total_timesteps=100000)
print("xxxxxxxxxxxxxxxxxxxx")
eva = evaluate_policy(model, the_env, n_eval_episodes=1000000)
the_env.close()

#eva_2 = evaluate_policy(model, the_env_2, n_eval_episodes=10)
#the_env_2.close()

'''
import optuna
def objective(trial):
    log_dir = "./tmp/network_Env-v0/"
    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
    env_args = dict(episode_length=200, load=6,
                mean_service_holding_time=10, k_paths=2, topology_num=4)
    the_env = gym.make('network_Env-v0', **env_args)
    the_env = Monitor(the_env, log_dir + 'training', info_keywords=('P_accepted', 'topology_num', 'load',))
    #model = A2C("MultiInputPolicy", the_env, learning_rate=trial.suggest_float('learning_rate',0.00007,0.007,log=True), verbose=1, device='cuda', gamma=trial.suggest_float('gamma', 0.3, 0.99), n_steps=10) #, learning_rate=0.0007
    model = A2C("MultiInputPolicy", the_env, verbose=1, device='cuda', gamma=trial.suggest_float('gamma', 0.3, 0.99), n_steps=10) #, learning_rate=0.0007
    model.learn(total_timesteps=200000)
    eva = evaluate_policy(model, the_env, n_eval_episodes=10)
    the_env.close()
    return eva[0]
study = optuna.create_study(direction = 'maximize', storage = 'sqlite:///db.sqlite3')
study.optimize(objective, n_trials=10)
study.best_params
print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
fig1 = optuna.visualization.plot_slice(study, params=["gamma"])
fig1.show()
fig2 = optuna.visualization.plot_optimization_history(study)
fig2.show()
#fig3 = optuna.visualization.plot_parallel_coordinate(study, params=["learning_rate", "gamma"])
#fig3.show()
#fig4 = optuna.visualization.plot_slice(study, params=["learning_rate", "gamma"])
#fig4.show()'''


loop = False
if loop is True:
    logs = ["./tmp/network_Env_0/", "./tmp/network_Env_1/", "./tmp/network_Env_2/", "./tmp/network_Env_3/",
            "./tmp/network_Env_4/", "./tmp/network_Env_5/", "./tmp/network_Env_6/", "./tmp/network_Env_7/",
            "./tmp/network_Env_8/", "./tmp/network_Env_9/", "./tmp/network_Env_10/", "./tmp/network_Env_11/"]
    load = [0.1, 0.5, 1, 2, 3, 4, 5, 6]
    for i in range(len(load)):
        os.makedirs(logs[i], exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=logs[i])

        env_args = dict(episode_length=100, load=load[i],
                        mean_service_holding_time=10, k_paths=2, topology_num=None)
        """env_args_2 = dict(episode_length=5000, load=load[i],
                          mean_service_holding_time=10, k_paths=2, topology_num=None)"""
        the_env = gym.make('network_Env-v0', **env_args)
        the_env = Monitor(the_env, logs[i] + 'training', info_keywords=('P_accepted', 'topology_num', 'load','reason','NF','MF','SF','BP','D'))
        """the_env_2 = gym.make('network_Env-v0', **env_args_2)
        the_env_2 = Monitor(the_env_2, logs[i] + 'training', info_keywords=('P_accepted', 'topology_num', 'load',))"""

        model = A2C("MultiInputPolicy", the_env, verbose=1, device='cuda', gamma=0.6, learning_rate=0.0007, n_steps=10)
        model.learn(total_timesteps=100000, callback=callback)
        #results_plotter.plot_results([log_dir], 200000, results_plotter.X_TIMESTEPS, 'network_Env-v0')
        eva = evaluate_policy(model, the_env, n_eval_episodes=1000)
        #eva_2 = evaluate_policy(model, the_env_2, n_eval_episodes=5)
        the_env.close()
        #the_env_2.close()




