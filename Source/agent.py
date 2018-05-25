'''
Agent class incorporates the reinforcement learning algorithm. It provides the
functionality to interact with the environment passing actions sampled using
a value function model instance and policy instance, to update value function 
model parameters based on observations received from environment.
'''

# import external packages
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.externals import joblib

class Agent():
    # initialize agent parameters
    def __init__(self, env, model, env_model, policy, eps, gamma1, gamma2, target_model = None):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.env_model = env_model
        self.policy = policy
        self.eps = eps
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
        # timing
        self.start_time = dt.datetime.now()
        self.time = dt.datetime.now()
    
    # get (previous state-action-state-reward) tuples and update the value function parameters
    def learn_value(self):
        # get the environment history
        iteration = self.env.env.iteration
        rewards = self.env.env.rewards
        states = self.env.env.stateFeatures
        predicted_states = self.env.env.statePrediction
        action_set = self.env.env.action_set
        
        # update model for each iteration
        for i in range(1, iteration + 1):
            # for each action
            for a in action_set:
                # if the reward is defined
                if not rewards.empty and not np.isnan(rewards.loc[i, a]):
                    prev_observation = states.loc[i-1, :].as_matrix()
                    action = a
                    observation = predicted_states.loc[i, action] #self.env_model.predict(action) #states.loc[i, :].as_matrix()
                    reward = rewards.loc[i, a]
                    
                    self.update_model(prev_observation, action, observation, reward)
    
    # perform value function update based on (previous state-action-state-reward) tuples
    def update_model(self, prev_observation, action, observation, reward):
        if self.target_model is None:
            # Q-learning update
            G = reward + self.gamma1*np.max(self.model.predict(observation))
            self.model.update(prev_observation, action, G)
        else:
            # double-Q-learning update
            if np.random.binomial(1, 0.5) == 1:
                predicted_q = self.model.predict(observation)
                G = reward + self.gamma1 * np.max(predicted_q)
                self.target_model.update(prev_observation, action, G)
            else:
                predicted_q = self.target_model.predict(observation)
                G = reward + self.gamma1 * np.max(predicted_q)
                self.model.update(prev_observation, action, G)
    
    # update the environment model based on (action, state) tuple            
    def learn_environment(self, a, s):
        self.env_model.update(a, s)
    
    # run one episode
    def play_one(self, sample_action, train = True, visualize_learning = 0, save = False, path = 'E:/bookkeeping/baseline_final/episode_0/'):
        observation = self.env.reset()
        done = False
        totalreward = 0
        iters = 0
        progress = pd.DataFrame(data = [])
        
        # timing
        self.start_time = dt.datetime.now()
        self.time = dt.datetime.now()
        
        # repeat until the end of the episode        
        while not done:
            action = sample_action(self.model, observation, self.eps, self.target_model)
            observation, reward, done, info = self.env.step(action)
            
            # update the value function model
            if train:
                self.learn_value()
            # update the environment model
            #self.learn_environment(action, observation)
            
            # timing
            current_time = dt.datetime.now()
            step_time_delta = current_time - self.time
            episode_time_delta = current_time - self.start_time
            self.time = current_time
            
            # extract iteration info
            max_action = self.env.env.convert_to_real_action(np.argmax(self.env.env.true_rewards.loc[self.env.env.iteration, :]))
            max_reward = self.env.env.true_rewards.loc[self.env.env.iteration, :].max()
            optimized_action = np.argmax(self.get_q_table().max(axis = 1))
            optimized_state = np.argmax(self.get_q_table().max(axis = 0))
            optimized_value = self.get_q_table().max().max()
            actual_action = self.env.env.convert_to_real_action(action)
            actual_reward = round(self.env.env.true_rewards.loc[self.env.env.iteration, action])
            true_optimal_action = self.env.env.convert_to_real_action(12)
            true_optimal_reward = round(self.env.env.true_rewards.loc[self.env.env.iteration, 12])
            
            # store iteration info
            progress.loc[iters, 'Week'] = int(iters)
            progress.loc[iters, 'Optimized action'] = optimized_action
            progress.loc[iters, 'Optimized state'] = optimized_state
            progress.loc[iters, 'Optimized value'] = optimized_value
            progress.loc[iters, 'Actual action'] = actual_action
            progress.loc[iters, 'Actual reward'] = actual_reward
            progress.loc[iters, 'True optimal action'] = true_optimal_action
            progress.loc[iters, 'True optimal reward'] = true_optimal_reward
            progress.loc[iters, 'Optimized action difference'] = optimized_action - true_optimal_action
            progress.loc[iters, 'Action difference'] = actual_action - true_optimal_action
            progress.loc[iters, 'Reward difference'] = actual_reward - true_optimal_reward
            progress.loc[iters, 'Max action'] = max_action
            progress.loc[iters, 'Max reward'] = max_reward
            progress.loc[iters, 'Step time'] = step_time_delta
            progress.loc[iters, 'Episode time'] = episode_time_delta
            
            # visualize iteration
            if visualize_learning > 0:
                self.plot_q_values()
                if visualize_learning > 2 and self.target_model is not None:
                    print('Target model')
                    self.plot_q_values(self.target_model)
                    
                if visualize_learning > 1:

                    if not np.isnan(actual_reward):
                        print(progress.loc[iters, ['Week', 'Optimized action difference', 'Action difference', 'Reward difference']].to_string())
                    else:
                        print(progress.loc[iters, ['Week', 'Optimized action difference', 'Action difference']].to_string())
                        print('learning delayed rewards...')
                print('---------------------------------------------------------------------')
            
            # bookkeeping
            if save:
                if not os.path.exists(path):
                    os.makedirs(path)
                joblib.dump(self, path + 'agent_' + str(iters) + '.pkl')
            
            totalreward += reward
            iters += 1
        
        return progress
    
    # get value function in the form of a table of Q-values
    def get_q_table(self, model = None):
        model = self.model if model is None else model
        q_values_df = pd.DataFrame(data = [], index = range(5, 105, 5))
        for obs in range(0, 21):
            observation = [obs * 0.05]
            q_values = model.predict(np.array(observation))
            q_values_df.loc[:, observation[0]] = pd.Series(data = q_values, index = range(5, 105, 5))
            
        return q_values_df
    
    # visualize value function
    def plot_q_values(self, model = None):
        model = self.model if model is None else model
        q_values_df = pd.DataFrame(data = [], index = range(5, 105, 5))
        for obs in range(0, 21):
            observation = [obs * 0.05]
            q_values = model.predict(np.array(observation))
            q_values_df.loc[:, observation[0]] = pd.Series(data = q_values, index = range(5, 105, 5))
        #q_values_df
        
        x = q_values_df.columns
        y = q_values_df.index
        
        xs, ys = np.meshgrid(x, y)
        
        zs = q_values_df.as_matrix()
        
        fig = plt.figure(figsize = (8, 5))
        ax = Axes3D(fig)
        ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
        
        if q_values_df.columns[((q_values_df == q_values_df.max().max()).sum(axis = 0) == 1).tolist()] != 0 and q_values_df.index[((q_values_df == q_values_df.max().max()).sum(axis = 1) == 1).tolist()] != 0:
            
            max_value = q_values_df.max().max()
            min_value = q_values_df.min().min()
            opt_state = q_values_df.columns[((q_values_df == q_values_df.max().max()).sum(axis = 0) == 1).tolist()][0]
            opt_action = q_values_df.index[((q_values_df == q_values_df.max().max()).sum(axis = 1) == 1).tolist()][0]
            
            x = [opt_state, opt_state]
            y = [opt_action, opt_action]
            z = [min_value, max_value]
            ax.plot(x,y,z,'--',alpha=0.8, c = 'r', linewidth = 1, label = 'optimum')
            
            x = [0, 1]
            y = [opt_action, opt_action]
            z = [max_value, max_value]
            ax.plot(x,y,z,'--',alpha=0.8, c = 'r', linewidth = 1)
            
            x = [opt_state, opt_state]
            y = [0, 100]
            z = [max_value, max_value]
            ax.plot(x,y,z,'--',alpha=0.8, c = 'r', linewidth = 1)
        
            ax.legend()
                    
        ax.set_title('Estimated values of state-action pairs')
        ax.set_xlabel('State: acceptance rate')
        ax.set_ylabel('Action: acceptance threshold')
        ax.set_zlabel('Value')

        plt.show()
    
    # plot average rewards    
    def plot_running_avg(self, totalrewards):
        N = len(totalrewards)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
        plt.plot(running_avg)
        plt.title("Running Average")
        plt.show()  
    
    # save agent istance to a file   
    def save(self, path = '/agent_dump'):
        wd = os.getcwd().replace('\\', '/')
        path = path if wd in path else wd + path
        if not os.path.exists(path.replace('.pkl', '')):
            os.makedirs(path)
        path = path if '.pkl' in path else path + '.pkl'
        '''    
        for model_index in range(len(self.models)):
            file_name = path + '/model_' + str(model_index) + '.pkl'
            joblib.dump(self.models[model_index], file_name)
        file_name = path + '/feature_transformer.pkl'
        joblib.dump(self.feature_transformer, file_name)
        '''
        joblib.dump(self, path)
    
    # load agent instance from a file 
    def load(self, path = '/agent_dump'):
        wd = os.getcwd().replace('\\', '/')
        path = path if wd in path else wd + path
        path = path if '.pkl' in path else path + '.pkl'
        if not os.path.exists(path):
            print('the path doesn\'t exist')
        '''    
        for model_index in range(len(self.models)):
            file_name = path + '/model_' + str(model_index) + '.pkl'
            self.models[model_index] = joblib.load(file_name)
        file_name = path + '/feature_transformer.pkl'
        self.feature_transformer = joblib.load(file_name)
        '''
        model = joblib.load(path)
        self.__dict__.update(model.__dict__)
        