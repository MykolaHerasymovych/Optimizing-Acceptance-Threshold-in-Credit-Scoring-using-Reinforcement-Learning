'''
Manager class provides functionality to go through experiment, such as functions
to initialize the agent and the environment, functions to run train, test, 
distorted and real episodes, functions to run train and test experiments, 
functions to visualize episodes and experiments.
'''

# import external packages
import os
import numpy as np
import pandas as pd
import datetime as dt
import joblib
import matplotlib.pyplot as plt

# import internal classes
from simulation import SimulationEnv
from agent import Agent
from model import FeatureTransformer, Model, EnvironmentModel
from policy import Policy

class Manager():
    # initialize the Manager instance
    def __init__(self, agent):
        self.agent = self.initAgent() if agent is None else agent
        
    # initialize the agent instance
    def initAgent(self):
        # micro-loan business simulation environment instance
        env = SimulationEnv()
        # feature transformer instance to convert numerous outputs of environment into simple numeric variables understood by the RL agent
        ft = FeatureTransformer(env)
        # value function model instance - the brain of the RL agent. Approximates value of each action in every state of environment
        lr = 0.0001                                                   # learning rate defines how adaptive the value function is to new input
        model = Model(env, ft, lr) 
        # environment model instance - the planning center of the agent. Predicts future environment states based on the current one
        env_model = EnvironmentModel(env, lr)
        # policy instance - includes different kinds of behaviors the agent can use to interact with the environment
        policy = Policy(env)
        # the agent instance - the guy that uses all of the above in order to optimize whatever you need
        eps = 1                                                       # exploration rate defines how much randomness to use to explore the environment
        gamma = 0.95                                                  # discounting rate defines how quick the agent forgets his previous experience
        agent = Agent(env, model, env_model, policy, eps, gamma, gamma)
        
        return agent
    
    # initialize experiment variables
    def initExperiment(self, train_episodes = 100, test_episodes = 5, test_frequency = 2, distorted_episodes = 100, experiment_name = 'baseline', bookkeeping_directory = os.getcwd(), bookkeeping_frequency = 1):
        # define train and test episode numbers
        self.train_episodes = train_episodes + 1                      # number of train episodes, where agent learns the environment and value function
        self.test_episodes = test_episodes                            # number of test episodes in a row to evaluate the current agent
        self.test_frequency = test_frequency                          # frequency of testing to track the progress of the agent
        self.distorted_episodes = distorted_episodes + 1
        
        # define variables to store the experiment history
        self.experiment_name = experiment_name                        # name of experiment
        self.bookkeeping_directory = bookkeeping_directory            # directory to store history
        self.bookkeeping_frequency = bookkeeping_frequency            # frequency of storing
        self.path = ''
        
        self.progress = pd.DataFrame(data = [])          
        self.weekly_progress = pd.DataFrame(data = [])
        self.distorted_progress = pd.DataFrame(data = [])
        self.run_agents = []
        self.run_progress = []
        self.run_test_progress = []
        self.train_episode = 0
        self.episode = 0                                              # variable to track entries to the main dataframe
        
        # define variables to track the timing of experiment
        self.start_time = dt.datetime.now()
        self.time = dt.datetime.now()
    
    # run an example set of test episodes
    def runTestEpisode(self):
        for test_episode in range(self.test_episodes):
            test_episode_progress = self.agent.play_one(sample_action = self.agent.policy.boltzmann_q_sample_action, visualize_learning = 2, train = False)
            self.run_test_progress.append(test_episode_progress)
        
            # timing
            current_time = dt.datetime.now()
            episode_time_delta = current_time - self.time
            run_time_delta = current_time - self.start_time
            self.time = current_time
            
            # extract episode info
            max_reward = self.agent.env.env.true_rewards.loc[54:114, :].sum().max()
            max_action = self.agent.env.env.convert_to_real_action(np.argmax(self.agent.env.env.true_rewards.loc[54:114, :].sum()))
            true_optimal_reward = test_episode_progress['True optimal reward'].sum()
            true_optimal_action = test_episode_progress.loc[0, 'True optimal action']
            optimized_action_avg = test_episode_progress['Optimized action'].mean()
            optimized_state_avg = test_episode_progress['Optimized state'].mean()
            optimized_value_avg = test_episode_progress['Optimized value'].mean()
            actual_reward = test_episode_progress['Actual reward'].sum()
            actual_action_avg = test_episode_progress['Actual action'].mean()
            episode_time = test_episode_progress.loc[test_episode_progress.shape[0] - 1, 'Episode time']
            
            # store episode info
            self.progress.loc[self.episode, 'lr'] = self.agent.model.learning_rate
            self.progress.loc[self.episode, 'gamma'] = self.agent.gamma1
            self.progress.loc[self.episode, 'episode'] = self.train_episode
            self.progress.loc[self.episode, 'test'] = 1
            self.progress.loc[self.episode, 'test_episode'] = test_episode
            self.progress.loc[self.episode, 'true_optimal_action'] = true_optimal_action
            self.progress.loc[self.episode, 'true_optimal_reward'] = true_optimal_reward
            self.progress.loc[self.episode, 'optimized_action_avg'] = optimized_action_avg
            self.progress.loc[self.episode, 'optimized_state_avg'] = optimized_state_avg
            self.progress.loc[self.episode, 'optimized_value_avg'] = optimized_value_avg
            self.progress.loc[self.episode, 'actual_reward'] = actual_reward
            self.progress.loc[self.episode, 'actual_action_avg'] = actual_action_avg
            self.progress.loc[self.episode, 'optimal_reward_dif'] = actual_reward - true_optimal_reward
            self.progress.loc[self.episode, 'optimal_action_dif'] = actual_action_avg - true_optimal_action
            self.progress.loc[self.episode, 'optimized_action_dif'] = optimized_action_avg - true_optimal_action
            self.progress.loc[self.episode, 'max_reward'] = max_reward
            self.progress.loc[self.episode, 'max_reward_dif'] = actual_reward - max_reward
            self.progress.loc[self.episode, 'max_action'] = max_action
            self.progress.loc[self.episode, 'max_action_dif'] = actual_action_avg - max_action
            self.progress.loc[self.episode, 'episode_time'] = episode_time_delta
            self.progress.loc[self.episode, 'episode_time_full'] = episode_time
            self.progress.loc[self.episode, 'run_time'] = run_time_delta
            self.episode += 1
            
            # bookkeeping
            self.path = self.bookkeeping_directory + '/bookkeeping/' + self.experiment_name + '/episode_' + str(self.train_episode) + '/test_episode_' + str(test_episode) + '/'
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            joblib.dump(self.progress, self.path + 'test_episode_progress.pkl')
            
            return test_episode_progress
    
    # run example train episode
    def runTrainEpisode(self):
        episode_progress = self.agent.play_one(sample_action = self.agent.policy.boltzmann_q_sample_action, visualize_learning = 2, save = True, path = 'E:/bookkeeping/baseline_final/episode_0/')
        episode_progress['episode'] = self.train_episode
        self.run_agents.append(self.agent)
        self.run_progress.append(episode_progress)
        self.weekly_progress = pd.concat([self.weekly_progress, episode_progress], axis = 0)
        
        # timing
        current_time = dt.datetime.now()
        episode_time_delta = current_time - self.time
        run_time_delta = current_time - self.start_time
        self.time = current_time
        
        # extract episode info
        max_reward = self.agent.env.env.true_rewards.loc[54:114, :].sum().max()
        max_action = self.agent.env.env.convert_to_real_action(np.argmax(self.agent.env.env.true_rewards.loc[54:114, :].sum()))
        true_optimal_reward = episode_progress['True optimal reward'].sum()
        true_optimal_action = episode_progress.loc[0, 'True optimal action']
        optimized_action_avg = episode_progress['Optimized action'].mean()
        optimized_state_avg = episode_progress['Optimized state'].mean()
        optimized_value_avg = episode_progress['Optimized value'].mean()
        actual_reward = episode_progress['Actual reward'].sum()
        actual_action_avg = episode_progress['Actual action'].mean()
        episode_time = episode_progress.loc[episode_progress.shape[0] - 1, 'Episode time']
        
        # store episode info
        self.progress.loc[self.episode, 'lr'] = self.agent.model.learning_rate
        self.progress.loc[self.episode, 'gamma'] = self.agent.gamma1
        self.progress.loc[self.episode, 'episode'] = self.train_episode
        self.progress.loc[self.episode, 'test'] = 0
        self.progress.loc[self.episode, 'test_episode'] = None
        self.progress.loc[self.episode, 'true_optimal_action'] = true_optimal_action
        self.progress.loc[self.episode, 'true_optimal_reward'] = true_optimal_reward
        self.progress.loc[self.episode, 'optimized_action_avg'] = optimized_action_avg
        self.progress.loc[self.episode, 'optimized_state_avg'] = optimized_state_avg
        self.progress.loc[self.episode, 'optimized_value_avg'] = optimized_value_avg
        self.progress.loc[self.episode, 'actual_reward'] = actual_reward
        self.progress.loc[self.episode, 'actual_action_avg'] = actual_action_avg
        self.progress.loc[self.episode, 'optimal_reward_dif'] = actual_reward - true_optimal_reward
        self.progress.loc[self.episode, 'optimal_action_dif'] = actual_action_avg - true_optimal_action
        self.progress.loc[self.episode, 'max_reward'] = max_reward
        self.progress.loc[self.episode, 'max_reward_dif'] = actual_reward - max_reward
        self.progress.loc[self.episode, 'max_action'] = max_action
        self.progress.loc[self.episode, 'max_action_dif'] = actual_action_avg - max_action
        self.progress.loc[self.episode, 'episode_time'] = episode_time_delta
        self.progress.loc[self.episode, 'episode_time_full'] = episode_time
        self.progress.loc[self.episode, 'run_time'] = run_time_delta
        self.episode += 1
        
        # bookkeeping
        self.path = self.bookkeeping_directory + '/bookkeeping/' + self.experiment_name + '/episode_' + str(self.train_episode) + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        joblib.dump(self.agent, self.path + 'agent.pkl')
        joblib.dump(self.agent.model, self.path + 'model.pkl')
        joblib.dump(self.progress, self.path + 'progress.pkl')
        joblib.dump(episode_progress, self.path + 'episode_progress.pkl')
        joblib.dump(self.weekly_progress, self.path + 'weekly_progress.pkl')
        
        return episode_progress
    
    # run example distorted episode
    def runDistortedEpisode(self):
        self.train_episode = 'distorted'
        distorted_episode = 0
        
        distorted_episode_progress = self.agent.play_one(sample_action = self.agent.policy.boltzmann_q_greedy_sample_action, visualize_learning = 2)#, save = True, path = 'E:/bookkeeping/baseline_final/episode_distorted/')
        
        distorted_episode_progress['episode'] = distorted_episode
        self.distorted_progress = pd.concat([self.distorted_progress, distorted_episode_progress])
        
        # timing
        current_time = dt.datetime.now()
        episode_time_delta = current_time - self.time
        run_time_delta = current_time - self.start_time
        self.time = current_time
        
        # extract episode info
        max_reward = self.agent.env.env.true_rewards.loc[54:114, :].sum().max()
        max_action = self.agent.env.env.convert_to_real_action(np.argmax(self.agent.env.env.true_rewards.loc[54:114, :].sum()))
        true_optimal_reward = distorted_episode_progress['True optimal reward'].sum()
        true_optimal_action = distorted_episode_progress.loc[0, 'True optimal action']
        optimized_action_avg = distorted_episode_progress['Optimized action'].mean()
        optimized_state_avg = distorted_episode_progress['Optimized state'].mean()
        optimized_value_avg = distorted_episode_progress['Optimized value'].mean()
        actual_reward = distorted_episode_progress['Actual reward'].sum()
        actual_action_avg = distorted_episode_progress['Actual action'].mean()
        episode_time = distorted_episode_progress.loc[distorted_episode_progress.shape[0] - 1, 'Episode time']
        
        # store episode info
        self.progress.loc[self.episode, 'lr'] = self.agent.model.learning_rate
        self.progress.loc[self.episode, 'gamma'] = self.agent.gamma1
        self.progress.loc[self.episode, 'episode'] = self.train_episode
        self.progress.loc[self.episode, 'test'] = 2
        self.progress.loc[self.episode, 'test_episode'] = distorted_episode
        self.progress.loc[self.episode, 'true_optimal_action'] = true_optimal_action
        self.progress.loc[self.episode, 'true_optimal_reward'] = true_optimal_reward
        self.progress.loc[self.episode, 'optimized_action_avg'] = optimized_action_avg
        self.progress.loc[self.episode, 'optimized_state_avg'] = optimized_state_avg
        self.progress.loc[self.episode, 'optimized_value_avg'] = optimized_value_avg
        self.progress.loc[self.episode, 'actual_reward'] = actual_reward
        self.progress.loc[self.episode, 'actual_action_avg'] = actual_action_avg
        self.progress.loc[self.episode, 'optimal_reward_dif'] = actual_reward - true_optimal_reward
        self.progress.loc[self.episode, 'optimal_action_dif'] = actual_action_avg - true_optimal_action
        self.progress.loc[self.episode, 'optimized_action_dif'] = optimized_action_avg - true_optimal_action
        self.progress.loc[self.episode, 'max_reward'] = max_reward
        self.progress.loc[self.episode, 'max_reward_dif'] = actual_reward - max_reward
        self.progress.loc[self.episode, 'max_action'] = max_action
        self.progress.loc[self.episode, 'max_action_dif'] = actual_action_avg - max_action
        self.progress.loc[self.episode, 'episode_time'] = episode_time_delta
        self.progress.loc[self.episode, 'episode_time_full'] = episode_time
        self.progress.loc[self.episode, 'run_time'] = run_time_delta
        self.episode += 1
        
        # bookkeeping
        self.path = self.bookkeeping_directory + '/bookkeeping/' + self.experiment_name + '/episode_' + str(self.train_episode) + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        joblib.dump(self.agent, self.path + 'agent.pkl')
        joblib.dump(self.agent.model, self.path + 'model.pkl')
        joblib.dump(distorted_episode_progress, self.path + 'episode_progress.pkl')
        
        return distorted_episode_progress
    
    # run real episode
    def runRealEpisode(self, path = 'E:/My/Reinforcement Learning Algorithm/bookkeeping/baseline_final/episode_real/'):
        self.train_episode = 'real'
        distorted_episode = 0
        
        distorted_episode_progress = self.agent.play_one(sample_action = self.agent.policy.boltzmann_q_greedy_sample_action, visualize_learning = 2, save = True, path = path)
        
        distorted_episode_progress['episode'] = distorted_episode
        self.distorted_progress = pd.concat([self.distorted_progress, distorted_episode_progress])
        
        # timing
        current_time = dt.datetime.now()
        episode_time_delta = current_time - self.time
        run_time_delta = current_time - self.start_time
        self.time = current_time
        
        # extract episode info
        max_reward = self.agent.env.env.true_rewards.loc[54:114, :].sum().max()
        max_action = self.agent.env.env.convert_to_real_action(np.argmax(self.agent.env.env.true_rewards.loc[54:114, :].sum()))
        true_optimal_reward = distorted_episode_progress['True optimal reward'].sum()
        true_optimal_action = distorted_episode_progress.loc[0, 'True optimal action']
        optimized_action_avg = distorted_episode_progress['Optimized action'].mean()
        optimized_state_avg = distorted_episode_progress['Optimized state'].mean()
        optimized_value_avg = distorted_episode_progress['Optimized value'].mean()
        actual_reward = distorted_episode_progress['Actual reward'].sum()
        actual_action_avg = distorted_episode_progress['Actual action'].mean()
        episode_time = distorted_episode_progress.loc[distorted_episode_progress.shape[0] - 1, 'Episode time']
        
        # store episode info
        self.progress.loc[self.episode, 'lr'] = self.agent.model.learning_rate
        self.progress.loc[self.episode, 'gamma'] = self.agent.gamma1
        self.progress.loc[self.episode, 'episode'] = self.train_episode
        self.progress.loc[self.episode, 'test'] = 3
        self.progress.loc[self.episode, 'test_episode'] = distorted_episode
        self.progress.loc[self.episode, 'true_optimal_action'] = true_optimal_action
        self.progress.loc[self.episode, 'true_optimal_reward'] = true_optimal_reward
        self.progress.loc[self.episode, 'optimized_action_avg'] = optimized_action_avg
        self.progress.loc[self.episode, 'optimized_state_avg'] = optimized_state_avg
        self.progress.loc[self.episode, 'optimized_value_avg'] = optimized_value_avg
        self.progress.loc[self.episode, 'actual_reward'] = actual_reward
        self.progress.loc[self.episode, 'actual_action_avg'] = actual_action_avg
        self.progress.loc[self.episode, 'optimal_reward_dif'] = actual_reward - true_optimal_reward
        self.progress.loc[self.episode, 'optimal_action_dif'] = actual_action_avg - true_optimal_action
        self.progress.loc[self.episode, 'optimized_action_dif'] = optimized_action_avg - true_optimal_action
        self.progress.loc[self.episode, 'max_reward'] = max_reward
        self.progress.loc[self.episode, 'max_reward_dif'] = actual_reward - max_reward
        self.progress.loc[self.episode, 'max_action'] = max_action
        self.progress.loc[self.episode, 'max_action_dif'] = actual_action_avg - max_action
        self.progress.loc[self.episode, 'episode_time'] = episode_time_delta
        self.progress.loc[self.episode, 'episode_time_full'] = episode_time
        self.progress.loc[self.episode, 'run_time'] = run_time_delta
        self.episode += 1
        
        # bookkeeping
        self.path = self.bookkeeping_directory + '/bookkeeping/' + self.experiment_name + '/episode_' + str(self.train_episode) + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        joblib.dump(self.agent, self.path + 'agent.pkl')
        joblib.dump(self.agent.model, self.path + 'model.pkl')
        joblib.dump(distorted_episode_progress, self.path + 'episode_progress.pkl')
        
        return distorted_episode_progress
    
    # run training experiment
    def train(self):
        # run a train episode
        for train_episode in range(1, self.train_episodes):
            self.train_episode = train_episode  
            episode_progress = self.agent.play_one(sample_action = self.agent.policy.boltzmann_q_sample_action)
            episode_progress['episode'] = self.train_episode
            self.run_agents.append(self.agent)
            self.run_progress.append(episode_progress)
            self.weekly_progress = pd.concat([self.weekly_progress, episode_progress], axis = 0)
            
            # timing
            current_time = dt.datetime.now()
            episode_time_delta = current_time - self.time
            run_time_delta = current_time - self.start_time
            self.time = current_time
            
            # extract episode info
            max_reward = self.agent.env.env.true_rewards.loc[54:114, :].sum().max()
            max_action = self.agent.env.env.convert_to_real_action(np.argmax(self.agent.env.env.true_rewards.loc[54:114, :].sum()))
            true_optimal_reward = episode_progress['True optimal reward'].sum()
            true_optimal_action = episode_progress.loc[0, 'True optimal action']
            optimized_action_avg = episode_progress['Optimized action'].mean()
            optimized_state_avg = episode_progress['Optimized state'].mean()
            optimized_value_avg = episode_progress['Optimized value'].mean()
            actual_reward = episode_progress['Actual reward'].sum()
            actual_action_avg = episode_progress['Actual action'].mean()
            episode_time = episode_progress.loc[episode_progress.shape[0] - 1, 'Episode time']
            
            # store episode info
            self.progress.loc[self.episode, 'lr'] = self.agent.model.learning_rate
            self.progress.loc[self.episode, 'gamma'] = self.agent.gamma1
            self.progress.loc[self.episode, 'episode'] = self.train_episode
            self.progress.loc[self.episode, 'test'] = 0
            self.progress.loc[self.episode, 'test_episode'] = None
            self.progress.loc[self.episode, 'true_optimal_action'] = true_optimal_action
            self.progress.loc[self.episode, 'true_optimal_reward'] = true_optimal_reward
            self.progress.loc[self.episode, 'optimized_action_avg'] = optimized_action_avg
            self.progress.loc[self.episode, 'optimized_state_avg'] = optimized_state_avg
            self.progress.loc[self.episode, 'optimized_value_avg'] = optimized_value_avg
            self.progress.loc[self.episode, 'actual_reward'] = actual_reward
            self.progress.loc[self.episode, 'actual_action_avg'] = actual_action_avg
            self.progress.loc[self.episode, 'optimal_reward_dif'] = actual_reward - true_optimal_reward
            self.progress.loc[self.episode, 'optimal_action_dif'] = actual_action_avg - true_optimal_action
            self.progress.loc[self.episode, 'optimized_action_dif'] = optimized_action_avg - true_optimal_action
            self.progress.loc[self.episode, 'max_reward'] = max_reward
            self.progress.loc[self.episode, 'max_reward_dif'] = actual_reward - max_reward
            self.progress.loc[self.episode, 'max_action'] = max_action
            self.progress.loc[self.episode, 'max_action_dif'] = actual_action_avg - max_action
            self.progress.loc[self.episode, 'episode_time'] = episode_time_delta
            self.progress.loc[self.episode, 'episode_time_full'] = episode_time
            self.progress.loc[self.episode, 'run_time'] = run_time_delta
            self.episode += 1
            
            # visualize value function and print out episode results
            self.agent.plot_q_values()
            print(self.progress[self.progress['episode'] == self.train_episode][['episode', 'optimized_action_dif', 'optimal_action_dif', 'optimal_reward_dif']].rename(columns = {'optimized_action_dif' : 'Optimized action difference', 'optimal_action_dif' : 'Action difference', 'optimal_reward_dif' : 'Reward difference'}).round(0).to_string(index = False))
            print('------------------------------------------------------------------------')
            
            if self.train_episode in range(0, self.train_episodes, self.test_frequency):
                # run set of test episodes
                for test_episode in range(self.test_episodes):
                    test_episode_progress = self.agent.play_one(sample_action = self.agent.policy.greedy_sample_action, train = False)
                    self.run_test_progress.append(test_episode_progress)
                    
                    # timing
                    current_time = dt.datetime.now()
                    episode_time_delta = current_time - self.time
                    run_time_delta = current_time - self.start_time
                    self.time = current_time
                    
                    # extract episode info
                    max_reward = self.agent.env.env.true_rewards.loc[54:114, :].sum().max()
                    max_action = self.agent.env.env.convert_to_real_action(np.argmax(self.agent.env.env.true_rewards.loc[54:114, :].sum()))
                    true_optimal_reward = test_episode_progress['True optimal reward'].sum()
                    true_optimal_action = test_episode_progress.loc[0, 'True optimal action']
                    optimized_action_avg = test_episode_progress['Optimized action'].mean()
                    optimized_state_avg = test_episode_progress['Optimized state'].mean()
                    optimized_value_avg = test_episode_progress['Optimized value'].mean()
                    actual_reward = test_episode_progress['Actual reward'].sum()
                    actual_action_avg = test_episode_progress['Actual action'].mean()
                    episode_time = test_episode_progress.loc[episode_progress.shape[0] - 1, 'Episode time']
                    
                    # store episode info
                    self.progress.loc[self.episode, 'lr'] = self.agent.model.learning_rate
                    self.progress.loc[self.episode, 'gamma'] = self.agent.gamma1
                    self.progress.loc[self.episode, 'episode'] = self.train_episode
                    self.progress.loc[self.episode, 'test'] = 1
                    self.progress.loc[self.episode, 'test_episode'] = test_episode
                    self.progress.loc[self.episode, 'true_optimal_action'] = true_optimal_action
                    self.progress.loc[self.episode, 'true_optimal_reward'] = true_optimal_reward
                    self.progress.loc[self.episode, 'optimized_action_avg'] = optimized_action_avg
                    self.progress.loc[self.episode, 'optimized_state_avg'] = optimized_state_avg
                    self.progress.loc[self.episode, 'optimized_value_avg'] = optimized_value_avg
                    self.progress.loc[self.episode, 'actual_reward'] = actual_reward
                    self.progress.loc[self.episode, 'actual_action_avg'] = actual_action_avg
                    self.progress.loc[self.episode, 'optimal_reward_dif'] = actual_reward - true_optimal_reward
                    self.progress.loc[self.episode, 'optimal_action_dif'] = actual_action_avg - true_optimal_action
                    self.progress.loc[self.episode, 'optimized_action_dif'] = optimized_action_avg - true_optimal_action
                    self.progress.loc[self.episode, 'max_reward'] = max_reward
                    self.progress.loc[self.episode, 'max_reward_dif'] = actual_reward - max_reward
                    self.progress.loc[self.episode, 'max_action'] = max_action
                    self.progress.loc[self.episode, 'max_action_dif'] = actual_action_avg - max_action
                    self.progress.loc[self.episode, 'episode_time'] = episode_time_delta
                    self.progress.loc[self.episode, 'episode_time_full'] = episode_time
                    self.progress.loc[self.episode, 'run_time'] = run_time_delta
                    self.episode += 1
                
                # print out test results
                print('..............................Testing...................................')
                print(self.progress[(self.progress['episode'] == self.train_episode) & (self.progress['test'] == 1)][['episode', 'optimized_action_dif', 'optimal_action_dif', 'optimal_reward_dif']].mean().rename({'optimized_action_dif' : 'Optimized action difference', 'optimal_action_dif' : 'Action difference', 'optimal_reward_dif' : 'Reward difference'}).round(0).to_string())
                print('------------------------------------------------------------------------')
                    
            # bookkeeping
            if self.train_episode in range(0, self.train_episodes, self.bookkeeping_frequency):
                self.path = self.bookkeeping_directory + '/bookkeeping/' + self.experiment_name + '/episode_' + str(self.train_episode) + '/'
                if not os.path.exists(self.path):
                    os.makedirs(self.path)
                joblib.dump(self.agent, self.path + 'agent.pkl')
                joblib.dump(self.agent.model, self.path + 'model.pkl')
                joblib.dump(self.progress, self.path + 'progress.pkl')
                joblib.dump(episode_progress, self.path + 'episode_progress.pkl')
                joblib.dump(self.weekly_progress, self.path + 'weekly_progress.pkl')
                if self.train_episode in range(0, self.train_episode, self.test_frequency):
                    joblib.dump(self.progress, self.path + 'test_episode_progress.pkl')
        
        return self.weekly_progress, self.progress

    # run distorted test experiment    
    def simulateDistortedEpisodes(self, distortions = None, lr = 0.001, gamma = 0.5):
        for distorted_episode in range(1, self.distorted_episodes): 
            # if no ditortions specified, generate random ones
            if distortions is None:
                
                np.random.seed(int(dt.datetime.now().second + dt.datetime.now().microsecond))
                
                news_positives_score_bias = np.random.rand() * 2 - 1
                repeats_positives_score_bias = np.random.rand() * 2 - 1
                news_negatives_score_bias = np.random.rand() * 2 - 1
                repeats_negatives_score_bias = np.random.rand() * 2 - 1
                news_default_rate_bias = np.random.rand() - 0.5
                repeats_default_rate_bias = np.random.rand() - 0.5
                late_payment_rate_bias = np.random.rand() * 2 - 1
                
                distortions = {'e': 1, 
                               'news_positives_score_bias': news_positives_score_bias, 
                               'repeats_positives_score_bias': repeats_positives_score_bias, 
                               'news_negatives_score_bias': news_negatives_score_bias, 
                               'repeats_negatives_score_bias': repeats_negatives_score_bias, 
                               'news_default_rate_bias': news_default_rate_bias, 
                               'repeats_default_rate_bias': repeats_default_rate_bias, 
                               'late_payment_rate_bias': late_payment_rate_bias, 
                               'ar_effect': 0}
            print(distortions)
            
            # pass distortions to the environment instance and adjust agent's parameters
            env = SimulationEnv(distortions = distortions)
            model = joblib.load(self.bookkeeping_directory + '/bookkeeping/' + self.experiment_name + '/episode_100/model.pkl')
            model.set_learning_rate(lr)
            self.agent.env = env
            self.agent.model = model
            self.agent.gamma1 = self.agent.gamma2 = gamma
            
            # run distorted episode
            distorted_episode_progress = self.agent.play_one(sample_action = self.agent.policy.boltzmann_q_greedy_sample_action)
            distorted_episode_progress['episode'] = distorted_episode
            self.distorted_progress = pd.concat([self.distorted_progress, distorted_episode_progress])
            
            # timing
            current_time = dt.datetime.now()
            episode_time_delta = current_time - self.time
            run_time_delta = current_time - self.start_time
            self.time = current_time
            
            # extract episode info
            max_reward = self.agent.env.env.true_rewards.loc[54:114, :].sum().max()
            max_action = self.agent.env.env.convert_to_real_action(np.argmax(self.agent.env.env.true_rewards.loc[54:114, :].sum()))
            true_optimal_reward = distorted_episode_progress['True optimal reward'].sum()
            true_optimal_action = distorted_episode_progress.loc[0, 'True optimal action']
            optimized_action_avg = distorted_episode_progress['Optimized action'].mean()
            optimized_state_avg = distorted_episode_progress['Optimized state'].mean()
            optimized_value_avg = distorted_episode_progress['Optimized value'].mean()
            actual_reward = distorted_episode_progress['Actual reward'].sum()
            actual_action_avg = distorted_episode_progress['Actual action'].mean()
            episode_time = distorted_episode_progress.loc[distorted_episode_progress.shape[0] - 1, 'Episode time']
            
            # store episode info
            self.progress.loc[self.episode, 'lr'] = self.agent.model.learning_rate
            self.progress.loc[self.episode, 'gamma'] = self.agent.gamma1
            self.progress.loc[self.episode, 'episode'] = self.train_episode
            self.progress.loc[self.episode, 'test'] = 2
            self.progress.loc[self.episode, 'test_episode'] = distorted_episode
            self.progress.loc[self.episode, 'true_optimal_action'] = true_optimal_action
            self.progress.loc[self.episode, 'true_optimal_reward'] = true_optimal_reward
            self.progress.loc[self.episode, 'optimized_action_avg'] = optimized_action_avg
            self.progress.loc[self.episode, 'optimized_state_avg'] = optimized_state_avg
            self.progress.loc[self.episode, 'optimized_value_avg'] = optimized_value_avg
            self.progress.loc[self.episode, 'actual_reward'] = actual_reward
            self.progress.loc[self.episode, 'actual_action_avg'] = actual_action_avg
            self.progress.loc[self.episode, 'optimal_reward_dif'] = actual_reward - true_optimal_reward
            self.progress.loc[self.episode, 'optimal_action_dif'] = actual_action_avg - true_optimal_action
            self.progress.loc[self.episode, 'optimized_action_dif'] = optimized_action_avg - true_optimal_action
            self.progress.loc[self.episode, 'max_reward'] = max_reward
            self.progress.loc[self.episode, 'max_reward_dif'] = actual_reward - max_reward
            self.progress.loc[self.episode, 'max_action'] = max_action
            self.progress.loc[self.episode, 'max_action_dif'] = actual_action_avg - max_action
            self.progress.loc[self.episode, 'episode_time'] = episode_time_delta
            self.progress.loc[self.episode, 'episode_time_full'] = episode_time
            self.progress.loc[self.episode, 'run_time'] = run_time_delta
            self.episode += 1
            
            # visualize value function and print out episode results
            self.agent.plot_q_values()
            print(self.progress[(self.progress['test_episode'] == distorted_episode) & (self.progress['test'] == 2)][['test_episode', 'max_action_dif', 'optimized_action_dif', 'optimal_action_dif', 'optimal_reward_dif']].rename({'max_action_dif' : 'Max action difference', 'optimized_action_dif' : 'Optimized action difference', 'optimal_action_dif' : 'Action difference', 'optimal_reward_dif' : 'Reward difference'}).round(0).to_string(index = False))
            print('------------------------------------------------------------------------')
        
        # bookkeeping
        self.path = self.bookkeeping_directory + '/bookkeeping/' + self.experiment_name + '/episode_' + str(self.train_episode) + '/distorted_episode_' + str(distorted_episode) + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        joblib.dump(self.agent, self.path + 'agent.pkl')
        joblib.dump(self.agent.model, self.path + 'model.pkl')
        joblib.dump(self.progress, self.path + 'progress.pkl')
        joblib.dump(self.distorted_progress, self.path + 'distorted_progress.pkl')
        joblib.dump(distorted_episode_progress, self.path + 'episode_progress.pkl')
        
        return self.distorted_progress, self.progress
    
    # visualize episode results
    def plotEpisode(self, episode_progress):
        # plot difference between the optimized action and the true optimal action
        plt.figure(figsize = (12, 5))
        plt.plot(episode_progress['Optimized action'] - episode_progress['True optimal action'], label = 'optimized')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title('Difference between the optimized action and the true optimal action', fontsize = 16)
        plt.xlabel('Week', fontsize = 16)
        plt.ylabel('Action difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot difference between the actual action and the true optimal action
        plt.figure(figsize = (12, 5))
        plt.plot(episode_progress['Action difference'], label = 'actual')
        plt.plot(episode_progress['Action difference'].rolling(window = 12).mean(), ls = '--', label = '3-months moving average')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title('Difference between the actual action and the true optimal action', fontsize = 16)
        plt.xlabel('Week', fontsize = 16)
        plt.ylabel('Action difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot difference between the actual reward recieved and the true optimal reward
        plt.figure(figsize = (12, 5))
        plt.plot(episode_progress['Reward difference'], label = 'actual')
        plt.plot(episode_progress['Reward difference'].rolling(window = 12).mean(), ls = '--', label = '3-months moving average')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title('Difference between the actual reward recieved and the true optimal reward', fontsize = 16)
        plt.xlabel('Week', fontsize = 16)
        plt.ylabel('Reward difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot difference between the actual cummulative reward recieved and the true optimal cummulative reward
        plt.figure(figsize = (12, 5))
        plt.plot(episode_progress['Reward difference'].cumsum(), label = 'actual')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title('Difference between the actual cummulative reward recieved and the true optimal cummulative reward', fontsize = 16)
        plt.xlabel('Week', fontsize = 16)
        plt.ylabel('Cumulative reward difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        plt.show()
    
    # visualize train experiment results 
    def plotRun(self, weekly_progress = None, progress = None):
        weekly_progress = self.weekly_progress if weekly_progress is None else weekly_progress
        progress = self.progress if progress is None else progress
        
        # plot difference between the optimized action and true optimal action
        plt.figure(figsize = (12, 5))
        plt.plot(weekly_progress.reset_index().loc[0:1000, 'Optimized action'] - weekly_progress.reset_index().loc[0:1000, 'True optimal action'], label = 'actual')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title("Difference between the optimized action and true optimal action", fontsize = 16)
        plt.xlabel('Week', fontsize = 16)
        plt.ylabel('Action difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot difference between the actual action and true optimal action
        plt.figure(figsize = (12, 5))
        plt.plot(weekly_progress.reset_index().loc[0:1000, 'Action difference'], ls = '--', label = 'actual')
        plt.plot(weekly_progress.reset_index().loc[0:1000, 'Action difference'].rolling(window = 100, min_periods = 20).mean(), lw = 3, label = '100-week moving average')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title("Difference between the actual action and true optimal action", fontsize = 16)
        plt.xlabel('Week', fontsize = 16)
        plt.ylabel('Action difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot difference between actual action average and true optimal action
        plt.figure(figsize = (12, 5))
        plt.plot(progress[progress['test'] == 0]['episode'], progress[progress['test'] == 0]['optimal_action_dif'], ls = '--', label = 'actual')
        plt.plot(progress[progress['test'] == 0]['episode'], progress[progress['test'] == 0]['optimal_action_dif'].rolling(window = 10, min_periods = 1).mean(), lw = 3, label = '10-episode moving average')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title("Difference between actual action average and true optimal action", fontsize = 16)
        plt.xlabel('Episode', fontsize = 16)
        plt.ylabel('Action difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot difference between actual reward and true optimal reward for train policy
        plt.figure(figsize = (12, 5))
        plt.plot(progress[progress['test'] == 0]['episode'], progress[progress['test'] == 0]['optimal_reward_dif'], ls = '--', label = 'actual')
        plt.plot(progress[progress['test'] == 0]['episode'], progress[progress['test'] == 0]['optimal_reward_dif'].rolling(window = 10, min_periods = 1).mean(), lw = 3, label = '10-episode moving average')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title("Difference between actual reward and true optimal reward for train policy", fontsize = 16)
        plt.xlabel('Episode', fontsize = 16)
        plt.ylabel('Reward difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot difference between actual reward and true optimal reward for test policy
        plt.figure(figsize = (12, 5))
        plt.plot(progress[progress['test'] == 1].groupby('episode').mean()['optimal_reward_dif'], ls = '--', label = 'actual')
        plt.plot(progress[progress['test'] == 1].groupby('episode').mean()['optimal_reward_dif'].rolling(window = 10, min_periods = 1).mean(), lw = 3, label = '10-episode moving average')
        plt.axhline(0, c = 'black', label = 'true optimal')
        plt.title("Difference between actual reward and true optimal reward for test policy", fontsize = 16)
        plt.xlabel('Episode', fontsize = 16)
        plt.ylabel('Reward difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        plt.show()
    
    # visualize distorted test experiment results
    def plotDistortedEpisodes(self, distorted_progress = None, progress = None):
        distorted_progress = self.distorted_progress if distorted_progress is None else distorted_progress
        progress = self.progress if progress is None else progress
        
        # plot difference between actual weekly reward and true optimal weekly reward in distorted evnvironment
        plt.figure(figsize = (12, 5))
        for e in range(28):
            plt.plot(self.distorted_progress[self.distorted_progress['episode'] == e].loc[0:60, 'Reward difference'].cumsum(), alpha = 0.4, label = '')
        plt.plot(self.distorted_progress.groupby('Week').mean().loc[0:60, 'Reward difference'].cumsum(), lw = 5, c = 'black', label = 'average')
        plt.axhline(0, ls = '--', c = 'black', label = 'true optimal')
        plt.title('Difference between actual weekly reward and true optimal weekly reward in distorted evnvironment', fontsize = 16)
        plt.xlabel('Week', fontsize = 16)
        plt.ylabel('Reward difference', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot actual episode reward and true optimal episode reward distributions in distorted evnvironment
        plt.figure(figsize = (12, 5))
        plt.hist(self.progress[self.progress['episode'] == 'distorted']['actual_reward'], color = 'orange', alpha = 0.5, bins = 10, label = 'actual')
        plt.hist(self.progress[self.progress['episode'] == 'distorted']['true_optimal_reward'], color = 'blue', alpha = 0.3, bins = 10, label = 'true optimal')
        plt.axvline(self.progress[self.progress['episode'] == 'distorted']['actual_reward'].mean(), ls = '--', lw = 2, c = 'orange', label = 'actual mean')
        plt.axvline(self.progress[self.progress['episode'] == 'distorted']['true_optimal_reward'].mean(), ls = '--', lw = 2, c = 'blue', label = 'true optimal mean')
        plt.title('Actual episode reward and true optimal episode reward distributions in distorted evnvironment', fontsize = 16)
        plt.xlabel('Total reward', fontsize = 16)
        plt.ylabel('Episodes count', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        # plot distribution of difference between actual and true optimal episode rewards in distorted evnvironment
        plt.figure(figsize = (12, 5))
        plt.hist(self.progress[self.progress['episode'] == 'distorted']['optimal_reward_dif'])
        plt.axvline(self.progress[self.progress['episode'] == 'distorted']['optimal_reward_dif'].mean(), ls = '--', c = 'red', lw = 2, label = 'mean')
        plt.title('Distribution of difference between actual and true optimal episode rewards in distorted evnvironment', fontsize = 16)
        plt.xlabel('Reward difference', fontsize = 16)
        plt.ylabel('Episodes count', fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize = 16)
        
        plt.show()
    
    # visualize value function
    def plot_q_values(self, episode = 0):
        path = self.bookkeeping_directory + '/bookkeeping/' + self.experiment_name + '/episode_' + str(episode) + '/model.pkl'
        model_to_plot = joblib.load(path)
        self.agent.plot_q_values(model_to_plot)