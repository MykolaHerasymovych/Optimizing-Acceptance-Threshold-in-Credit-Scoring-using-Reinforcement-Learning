'''
SimulationEnv class is built on top of gym environment class to define
state and action spaces and set up the state-action-reward exchange between
the agent and the environment classes.
'''

# import external packages
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# import internal classes
from environment import Environment 

class SimulationEnv(gym.Env):
    # initialize environment instance and define state and action spaces
    def __init__(self, action_type = 'discrete_action', reward_type = 'real', window = 4, cheating = False, reward_scaler = 1, distortions = {'e': 1, 'news_positives_score_bias': 0, 'repeats_positives_score_bias': 0, 'news_negatives_score_bias': 0, 'repeats_negatives_score_bias': 0, 'news_default_rate_bias': 0, 'repeats_default_rate_bias': 0, 'late_payment_rate_bias': 0, 'ar_effect': 0}):
        self.action_type = action_type
        self.reward_type = reward_type
        self.window = window
        self.cheating = cheating
        self.reward_scaler = reward_scaler
        self.distortions = distortions
        self.env = Environment(action_type = self.action_type, reward_type = self.reward_type, window = self.window, cheating = self.cheating, reward_scaler = self.reward_scaler, distortions = self.distortions)
        #['Moving acceptance rate', 'Moving default to paid ratio']
        high = np.array([1])
        low = np.array([0])
    
        self.observation_space = spaces.Box(low, high)
        
        if(self.action_type == 'discrete_change'):
            self.action_space = spaces.Discrete(5)
        elif(self.action_type == 'discrete_action'):
            self.action_space = spaces.Discrete(20)
        elif(self.action_type == 'continuous_change'):
            self.min_action = -10
            self.max_action = 10
            self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        elif(self.action_type == 'continuous_action'):
            self.min_action = 5
            self.max_action = 100
            self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        elif(self.action_type == 'discrete_action_separate'):
            self.action_space = spaces.Discrete(100)
        elif(self.action_type == 'discrete_change_separate'):
            self.action_space = spaces.Discrete(25)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        
    # set random seed
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # step through the episode
    def _step(self, action):

        if(self.action_type == 'discrete_change'):
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
            action = self.env.convert_to_real_action(action)
        elif(self.action_type == 'discrete_action'):
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
            action = self.env.convert_to_real_action(action)
            
        self.env.take_action(action)
        self.state = self.env.state
        reward = self.env.reward
        done = 1 if self.env.iteration == 135 else 0                # finish the episode when all the delayed rewards are learned

        return np.array(self.state), reward, done, {}

    # reset the environment
    def _reset(self):
        self.env.run_iterations(iterations = 53, output = False)    # skip the warming-up phase of the simulation
        self.state = self.env.state
        return np.array(self.state)
