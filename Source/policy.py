'''
Policy class provides a choice of action sample policies for the RL agent,
including greedy, epsilon-greedy, random, default, boltzmann-Q and derivatives.
'''

# import external packages
import numpy as np
import matplotlib.pyplot as plt

# import internal classes
from utils import argmax

class Policy():
    # relate policy to environment
    def __init__(self, env):
        self.env = env
    
    # greedy policy - take action with maximum value
    def greedy_sample_action(self, model, s, eps, target_model = None):
        optimal_a = argmax(model.predict(s)) if target_model is None else argmax(model.predict(s) + target_model.predict(s))
        return optimal_a
    
    # with a certain probability take action right below the one with the maximum value
    # otherwise, follow greedy policy
    def one_lower_epsGreedy_sample_action(self, model, s, eps, target_model = None):
        optimal_a = argmax(model.predict(s)) if target_model is None else argmax(model.predict(s) + target_model.predict(s))
        if np.random.random() <= eps:
            suboptimal_a = optimal_a - 1 if optimal_a != 0 else optimal_a
            return suboptimal_a
        return optimal_a
    
    # boltzmann-Q policy with a low tau
    def boltzmann_q_greedy_sample_action(self, model, s, eps, target_model = None, clip = (-500, 500)):
        tau = eps # 0.1
        print(self.env.env.iteration)
        if self.env.env.iteration == 1:
            print('here')
            return 12
        
        q_values = model.predict(s) if target_model is None else ((model.predict(s) + target_model.predict(s)) / 2)
        nb_actions = q_values.shape[0]
        
        q_values_normed = (q_values - q_values.mean()) / q_values.std() #q_values / q_values.max()
        exp_values = np.exp(np.clip(q_values_normed / tau, clip[0], clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action
    
    # boltzmann-Q policy with a low tau defined only for actions lower or equal to the action with maximum value
    def lower_boltzmann_q_greedy_sample_action(self, model, s, eps, target_model = None, clip = (-500, 500)):
        
        tau = eps # 0.5
        
        optimal_a = self.greedy_sample_action(model, s, eps, target_model)
        
        q_values = model.predict(s) if target_model is None else ((model.predict(s) + target_model.predict(s)) / 2)
        suboptimal_a_limtis = [optimal_a - 3, optimal_a + 1]
        lower_q_values = q_values[suboptimal_a_limtis[0] : suboptimal_a_limtis[1]]
        nb_actions = lower_q_values.shape[0] if lower_q_values.size != 0 else 1
        
        lower_q_values_normed = (lower_q_values - lower_q_values.mean()) / lower_q_values.std() #lower_q_values / lower_q_values.max()
        exp_values = np.exp(np.clip(lower_q_values_normed / tau, clip[0], clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs) + suboptimal_a_limtis[0]
        return action
    
    # with a certain probability randomly take one of actions below the one with the maximum value
    # otherwise, follow greedy policy
    def lower_epsGreedy_sample_action(self, model, s, eps, target_model = None):
        optimal_a = self.greedy_sample_action(model, s, eps, target_model)
        if np.random.random() <= eps:
            random_a = np.random.randint(0, optimal_a + 1)
            return random_a
        else:
          return optimal_a
    
    # take a random action
    def random_sample_action(self, model, s, eps, target_model = None):
        nb_actions = self.env.action_space.n
        random_a = np.random.randint(0, nb_actions)
        return random_a
    
    # take the lowest action
    def zero_sample_action(self, model, s, eps, target_model = None):
        return 0
    
    # take default action
    def default_sample_action(self, model, s, eps, target_model = None):
        return self.env.env.convert_to_simple_action(50)
    
    # take true optimal action
    def true_optimal_sample_action(self, model, s, eps, target_model = None):
        return self.env.env.convert_to_simple_action(65)
    
    # with a certain probability randomly take one of actions below the one with the maximum value
    # otherwise, take action right below the one with maximum value
    def lower_epsSubGreedy_sample_action(self, model, s, eps, target_model = None):
        optimal_a = self.greedy_sample_action(model, s, eps, target_model)
        suboptimal_a = optimal_a - 1 if optimal_a != 0 else optimal_a
        if np.random.random() <= eps:
            random_a = np.random.randint(0, optimal_a + 1)
            return random_a
        else:
          return suboptimal_a
    
    # boltzmann-Q policy
    def boltzmann_q_sample_action(self, model, s, eps, target_model = None, clip = (-500, 500)):
        tau = eps # 1
        q_values = model.predict(s) if target_model is None else ((model.predict(s) + target_model.predict(s)) / 2)
        nb_actions = q_values.shape[0]
        
        q_values_normed = (q_values - q_values.mean()) / q_values.std() if q_values.std() != 0 else q_values - q_values.mean()#q_values / q_values.max()#
        exp_values = np.exp(np.clip(q_values_normed / tau, clip[0], clip[1]))
        probs = exp_values / np.sum(exp_values)
        #plt.plot([round(x, 5) for x in probs])
        action = np.random.choice(range(nb_actions), p=probs)
        return action
    
    # boltzmann-Q policy defined only for actions lower or equal to the action with maximum value
    def lower_boltzmann_q_sample_action(self, model, s, eps, target_model = None, clip = (-500, 500)):
        tau = eps # 1
        
        optimal_a = self.greedy_sample_action(model, s, eps, target_model)
        
        q_values = model.predict(s) if target_model is None else ((model.predict(s) + target_model.predict(s)) / 2)
        lower_q_values = q_values[0 : optimal_a + 1]
        nb_actions = lower_q_values.shape[0]
        
        lower_q_values_normed = (lower_q_values - lower_q_values.mean()) / lower_q_values.std() if lower_q_values.std() != 0 else lower_q_values - lower_q_values.mean()#lower_q_values / lower_q_values.max()
        exp_values = np.exp(np.clip(lower_q_values_normed / tau, clip[0], clip[1]))
        probs = exp_values / np.sum(exp_values)
        #plt.plot([round(x, 5) for x in probs])
        action = np.random.choice(range(nb_actions), p=probs)
        return action
