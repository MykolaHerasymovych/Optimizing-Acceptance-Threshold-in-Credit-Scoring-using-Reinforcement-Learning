'''
Environment class works as a connection between the sim class (the simulation
itself)and the RL agent. It gets actions from the agent, calls the simulation 
class to generate the loan applications based on the actions, dynamicaly 
calculates and stores the characteristics of loan portfolio and rewards 
received, provides a set of supplementary functions.
'''

# import external packages
import pandas as pd
import os 
import datetime

# import internal classes
from sim import Sim

class Environment:
    # initialize the environment
    def __init__(self, action_type = 'discrete_action', reward_type = 'real', lag = False, window = 4, cheating = False, reward_scaler = 1, distortions = {'e': 1, 'news_positives_score_bias': 0, 'repeats_positives_score_bias': 0, 'news_negatives_score_bias': 0, 'repeats_negatives_score_bias': 0, 'news_default_rate_bias': 0, 'repeats_default_rate_bias': 0, 'late_payment_rate_bias': 0, 'ar_effect': 0}):
        
        self.action_type = action_type
        self.reward_type = reward_type
        self.lag = lag
        self.window = window
        self.cheating = cheating
        self.reward_scaler = reward_scaler
        self.distortions = distortions
        
        self.single_threshold_action_dict = {0: 5, 1: 10, 2: 15, 3: 20, 4: 25, 5:30, 6: 35, 7: 40, 8: 45, 9: 50,
                                      10: 55, 11: 60, 12: 65, 13: 70, 14: 75, 15: 80, 16: 85, 17: 90, 18: 95, 19: 100}
        self.single_threshold_change_dict = {0: -10, 1: -5, 2: 0, 3: 5, 4: 10}
        self.separate_threshold_action_dict = {0:[5, 5], 1:[15, 5], 2:[25, 5], 3:[35, 5], 4:[45, 5], 5:[55, 5], 6:[65, 5], 7:[75, 5], 8:[85, 5], 9:[95, 5], 
                               10:[5, 15], 11:[15, 15], 12:[25, 15], 13:[35, 15], 14:[45, 15], 15:[55, 15], 16:[65, 15], 17:[75, 15], 18:[85, 15], 19:[95, 15], 
                               20:[5, 25], 21:[15, 25], 22:[25, 25], 23:[35, 25], 24:[45, 25], 25:[55, 25], 26:[65, 25], 27:[75, 25], 28:[85, 25], 29:[95, 25],
                               30:[5, 35], 31:[15, 35], 32:[25, 35], 33:[35, 35], 34:[45, 35], 35:[55, 35], 36:[65, 35], 37:[75, 35], 38:[85, 35], 39:[95, 35],
                               40:[5, 45], 41:[15, 45], 42:[25, 45], 43:[35, 45], 44:[45, 45], 45:[55, 45], 46:[65, 45], 47:[75, 45], 48:[85, 45], 49:[95, 45],
                               50:[5, 55], 51:[15, 55], 52:[25, 55], 53:[35, 55], 54:[45, 55], 55:[55, 55], 56:[65, 55], 57:[75, 55], 58:[85, 55], 59:[95, 55],
                               60:[5, 65], 61:[15, 65], 62:[25, 65], 63:[35, 65], 64:[45, 65], 65:[55, 65], 66:[65, 65], 67:[75, 65], 68:[85, 65], 69:[95, 65],
                               70:[5, 75], 71:[15, 75], 72:[25, 75], 73:[35, 75], 74:[45, 75], 75:[55, 75], 76:[65, 75], 77:[75, 75], 78:[85, 75], 79:[95, 75],
                               80:[5, 85], 81:[15, 85], 82:[25, 85], 83:[35, 85], 84:[45, 85], 85:[55, 85], 86:[65, 85], 87:[75, 85], 88:[85, 85], 89:[95, 85],
                               90:[5, 95], 91:[15, 95], 92:[25, 95], 93:[35, 95], 94:[45, 95], 95:[55, 95], 96:[65, 95], 97:[75, 95], 98:[85, 95], 99:[95, 95],
                               }
        self.separate_threshold_change_dict = {0:[-10, -10], 1:[-5, -10], 2:[0, -10], 3:[5, -10], 4:[10, -10], 
                                      5:[-10, -5], 6:[-5, -5], 7:[0, -5], 8:[5, -5], 9:[10, -5], 
                                      10:[-10, 0], 11:[-5, 0], 12:[0, 0], 13:[5, 0], 14:[10, 0], 
                                      15:[-10, 5], 16:[-5, 5], 17:[0, 5], 18:[5, 5], 19:[10, 5], 
                                      20:[-10, 10], 21:[-5, 10], 22:[0, 10], 23:[5, 10], 24:[10, 10]
                                      }
        self.choose_action_set()
        
        self.reset()
        
    # reset all the environment variables to default values
    def reset(self):
    
        self.sim = Sim(self.distortions)
        
        # define history dataframes
        self.result = pd.DataFrame(data = []) # data for each client
        
        self.scoreInfo = pd.DataFrame(data = 0, index = ['Default rate', 'Default paid rate'], columns = range(0, 105, 5)) # inference into score bins 
        self.result_predicted = self.result.copy() # result dataframe extrapolated with the learnt knowledge about score bins
        
        self.states = pd.DataFrame(data = []) # environment data for each state
        self.stateParameters = pd.DataFrame(data = []) # parameter values for each state
        self.stateFeatures = pd.DataFrame(data = []) # feature values for each state
        self.actions = pd.DataFrame(data = []) # action values for each state
        self.rewards = pd.DataFrame(data = 0, index = [], columns = list(self.action_set.keys())) # reward values for each action in each state
        self.true_rewards = pd.DataFrame(data = 0, index = [], columns = list(self.action_set.keys())) # true reward values for each action in each state
        self.statePrediction = pd.DataFrame(data = 0, index = [], columns = list(self.action_set.keys())) # next state predictions for each action
        self.history = {} # state, action, reward history
        
        self.iteration = 0 # iteration
        
        # timing
        self.start_time = datetime.datetime.now()
        self.time = datetime.datetime.now()
        
        # defining the thresholds for the scoring model
        # raise them to make the scoring model more liberal, lower them to make it more conservative
        # default threshold for repeat clients: 50
        # must be between 1 and 100
        THRESHOLD_REPEAT = 50
        # default threshold for new clients: 50
        # must be between 1 and 100
        THRESHOLD_NEW = 50
        
        # default policy
        self.default_policy = {'threshold_repeat': THRESHOLD_REPEAT, 'threshold_new': THRESHOLD_NEW}
        self.policy = self.default_policy
        
        # feature list
        #self.features = ['Moving profit growth rate', 'Moving applications growth rate', 'Moving repeat applications share', 'Moving acceptance rate', 'Moving default rate']
        #self.features = ['State profit', 'Total profit', 'State applications', 'Total new applications', 'Total repeat applications share', 'State acceptance rate', 'Total default rate']
        self.features = ['State acceptance rate'] #, 'Moving acceptance rate squared'
        
        # moving values list and growth variables list
        self.moving_variables = ['State profit', 'State applications', 'State new applications', 'State repeat applications', 'State accepted', 'State new accepted', 'State repeat accepted', 'State defaulted', 'State paid', 'State defaulted paid']
        self.growth_variables = ['Moving profit', 'Moving applications', 'Moving repeat applications share', 'Moving repeat loans share', 'Moving acceptance rate', 'Moving default rate', 'Moving paid rate', 'Moving defaulted paid rate']
        
        # default state and reward
        self.state = self.get_state_features()
        self.reward = 0
        
        
    # generate new environment state based on current policy
    def update_environment(self, policy):
        '''
        # timing
        current_time = datetime.datetime.now()
        iteration_time_delta = current_time - self.time
        episode_time_delta = current_time - self.start_time
        self.time = current_time
        
        print('iteration: ' + str(self.iteration) + '. Took ' + str(iteration_time_delta.seconds) + ' sec., ' + 'overall ' + str(episode_time_delta.seconds) + ' sec.')
        '''
        
        self.iteration += 1
        
#        try:
#            total_accepted = self.result['accept'].sum()
#            print(total_accepted)
#        except:
#            total_accepted = 1
        
        # generate new state of environment
        out, state_paid, state_defaulted, state_defaulted_paid = self.sim.simulate(self.iteration, self.sim.generateInput(self.iteration), policy['threshold_repeat']) # do not change this line
        self.result = pd.concat([self.result, out])                                                               

        return state_defaulted, state_paid, state_defaulted_paid
    
    # update state variables based on new environment state
    def update_state_history(self, policy, state_defaulted, state_paid, state_defaulted_paid):
        
        result_copy = self.result.copy()
        
        # store parameter values
        self.stateParameters.loc[self.iteration, 'Threshold repeat'] = policy['threshold_repeat']
        self.stateParameters.loc[self.iteration, 'Threshold new'] = policy['threshold_new']
        if self.lag:
            self.save_action_history()
            self.save_state_history()
        
        # calculate and store state variables values
        # calculate profits for each state
        state_profit = 0        
        
        # calculate rewards for each previous state
        if not self.cheating:
            self.predict_rewards(state_defaulted, state_paid, state_defaulted_paid)
        self.predict_rewards_immediate_cheating()
        
        # add loss for each defaulted loan    
        for id in state_defaulted:
            state_loss_defaulted = result_copy.loc[id, 'sum']
            state_profit -= state_loss_defaulted
        # add profit for each paid loan
        for id in state_paid:
            state_profit_paid = result_copy.loc[id, 'profit']
            state_profit += state_profit_paid
        # add profit for each defaulted paid loan
        for id in state_defaulted_paid:
            state_profit_defaulted_paid = result_copy.loc[id, 'profit'] + result_copy.loc[id, 'sum']
            state_profit += state_profit_defaulted_paid
        
        # add profit value to the states dataframe
        self.states.loc[self.iteration, 'State profit'] = state_profit
        # update total profit
        if(self.iteration == 1):
            self.states.loc[self.iteration, 'Total profit'] = self.states.loc[self.iteration, 'State profit']
        else:
            self.states.loc[self.iteration, 'Total profit'] = self.states.loc[self.iteration - 1, 'Total profit'] + self.states.loc[self.iteration, 'State profit']
        
        # calculate number of applications for each state
        # find total applications
        self.states.loc[self.iteration, 'Total applications'] = result_copy.shape[0]
        # find state applications
        if(self.iteration == 1):
            self.states.loc[self.iteration, 'State applications'] = result_copy.shape[0]
        else:
            self.states.loc[self.iteration, 'State applications'] = self.states.loc[self.iteration, 'Total applications'] - self.states.loc[self.iteration - 1, 'Total applications']
       
        # calculate number of new applications for each state
        # find total new applications
        self.states.loc[self.iteration, 'Total new applications'] = result_copy[result_copy['repeat'] == False].shape[0] if 'repeat' in result_copy.columns else 0
        # find state new applications
        if(self.iteration == 1):
            self.states.loc[self.iteration, 'State new applications'] = result_copy[result_copy['repeat'] == False].shape[0] if 'repeat' in result_copy.columns else 0
        else:
            self.states.loc[self.iteration, 'State new applications'] = self.states.loc[self.iteration, 'Total new applications'] - self.states.loc[self.iteration - 1, 'Total new applications']
        
        # calculate number of repeat applications for each state
        # find total repeat applications
        self.states.loc[self.iteration, 'Total repeat applications'] = result_copy[result_copy['repeat'] == True].shape[0] if 'repeat' in result_copy.columns else 0 
        # find state repeat applications
        if(self.iteration == 1):
            self.states.loc[self.iteration, 'State repeat applications'] = result_copy[result_copy['repeat'] == True].shape[0] if 'repeat' in result_copy.columns else 0
        else:
            self.states.loc[self.iteration, 'State repeat applications'] = self.states.loc[self.iteration, 'Total repeat applications'] - self.states.loc[self.iteration - 1, 'Total repeat applications']
        
        # calculate the share of repeat applications in the total number of applications
        # find total share of repeat applications
        self.states.loc[self.iteration, 'Total repeat applications share'] = self.states.loc[self.iteration, 'Total repeat applications'] / self.states.loc[self.iteration, 'Total applications'] if self.states.loc[self.iteration, 'Total applications'] != 0 else 0
        # find state share of repeat applications
        self.states.loc[self.iteration, 'State repeat applications share'] = self.states.loc[self.iteration, 'State repeat applications'] / self.states.loc[self.iteration, 'State applications'] if self.states.loc[self.iteration, 'State applications'] != 0 else 0
        
        # calculate number of accepted clients for each state
        # find total accepted clients
        self.states.loc[self.iteration, 'Total accepted'] = result_copy[result_copy['accept'] == True].shape[0] if 'accept' in result_copy.columns else 0
        # find state accepted clients
        if(self.iteration == 1):
            self.states.loc[self.iteration, 'State accepted'] = result_copy[result_copy['accept'] == True].shape[0] if 'accept' in result_copy.columns else 0
        else:
            self.states.loc[self.iteration, 'State accepted'] = self.states.loc[self.iteration, 'Total accepted'] - self.states.loc[self.iteration - 1, 'Total accepted']
        
        # calculate acceptance rate for each state
        # calculate state acceptance rate for each state
        self.states.loc[self.iteration, 'State acceptance rate'] = self.states.loc[self.iteration, 'State accepted'] / self.states.loc[self.iteration, 'State applications'] if self.states.loc[self.iteration, 'State accepted'] != 0 else 0
        # calculate total acceptance rate for each state
        self.states.loc[self.iteration, 'Total acceptance rate'] = self.states.loc[self.iteration, 'Total accepted'] / self.states.loc[self.iteration, 'Total applications'] if self.states.loc[self.iteration, 'Total accepted'] != 0 else 0
        
        # add state profit per loan
        self.states.loc[self.iteration, 'State profit per loan'] = self.states.loc[self.iteration, 'State profit'] / self.states.loc[self.iteration, 'State accepted'] if self.states.loc[self.iteration, 'State accepted'] != 0 else 0
        # find overall profit per loan
        self.states.loc[self.iteration, 'Total profit per loan'] = self.states.loc[self.iteration, 'Total profit'] / self.states.loc[self.iteration, 'Total accepted'] if self.states.loc[self.iteration, 'Total accepted'] != 0 else 0
        
        # calculate number of new accepted for each state
        # find total new accepted
        self.states.loc[self.iteration, 'Total new accepted'] = result_copy[(result_copy['accept'] == True) & (result_copy['repeat'] == False)].shape[0] if 'accept' in result_copy.columns else 0
        # find state new accepted
        if(self.iteration == 1):
            self.states.loc[self.iteration, 'State new accepted'] = result_copy[(result_copy['accept'] == True) & (result_copy['repeat'] == False)].shape[0] if 'accept' in result_copy.columns else 0
        else:
            self.states.loc[self.iteration, 'State new accepted'] = self.states.loc[self.iteration, 'Total new accepted'] - self.states.loc[self.iteration - 1, 'Total new accepted']
        
        # calculate new clients acceptance rate for each state
        # calculate state new clients acceptance rate for each state
        self.states.loc[self.iteration, 'State new acceptance rate'] = self.states.loc[self.iteration, 'State new accepted'] / self.states.loc[self.iteration, 'State new applications'] if self.states.loc[self.iteration, 'State new applications'] != 0 else 0
        # calculate total new clients acceptance rate for each state
        self.states.loc[self.iteration, 'Total new acceptance rate'] = self.states.loc[self.iteration, 'Total new accepted'] / self.states.loc[self.iteration, 'Total new applications'] if self.states.loc[self.iteration, 'Total new applications'] != 0 else 0
        
        # calculate number of repeat accepted for each state
        # find total repeat accepted
        self.states.loc[self.iteration, 'Total repeat accepted'] = result_copy[(result_copy['accept'] == True) & (result_copy['repeat'] == True)].shape[0] if 'accept' in result_copy.columns else 0
        # find state repeat accepted
        if(self.iteration == 1):
            self.states.loc[self.iteration, 'State repeat accepted'] = result_copy[(result_copy['accept'] == True) & (result_copy['repeat'] == True)].shape[0] if 'accept' in result_copy.columns else 0
        else:
            self.states.loc[self.iteration, 'State repeat accepted'] = self.states.loc[self.iteration, 'Total repeat accepted'] - self.states.loc[self.iteration - 1, 'Total repeat accepted']
        
        # calculate repeat clients acceptance rate for each state
        # calculate state repeat clients acceptance rate for each state
        self.states.loc[self.iteration, 'State repeat acceptance rate'] = self.states.loc[self.iteration, 'State repeat accepted'] / self.states.loc[self.iteration, 'State repeat applications'] if self.states.loc[self.iteration, 'State repeat applications'] != 0 else 0
        # calculate total repeat clients acceptance rate for each state
        self.states.loc[self.iteration, 'Total repeat acceptance rate'] = self.states.loc[self.iteration, 'Total repeat accepted'] / self.states.loc[self.iteration, 'Total repeat applications'] if self.states.loc[self.iteration, 'Total repeat applications'] != 0 else 0
        
        # calculate the share of repeat loans in the total number of loans
        # find total share of repeat loans
        self.states.loc[self.iteration, 'Total repeat loans share'] = self.states.loc[self.iteration, 'Total repeat accepted'] / self.states.loc[self.iteration, 'Total accepted'] if self.states.loc[self.iteration, 'Total accepted'] != 0 else 0
        # find state share of repeat loans
        self.states.loc[self.iteration, 'State repeat loans share'] = self.states.loc[self.iteration, 'State repeat accepted'] / self.states.loc[self.iteration, 'State accepted'] if self.states.loc[self.iteration, 'State accepted'] != 0 else 0
        
        # calculate number of defaulted clients for each state
        # calculate state number of defaulted clients for each state
        self.states.loc[self.iteration, 'State defaulted'] = len(state_defaulted)
        # calculate total number of defaulted clients for each state
        self.states.loc[self.iteration, 'Total defaulted'] = self.states['State defaulted'].sum()
    
        # calculate default rate for each state
        # calculate state default rate for each state
        if(self.states.loc[self.iteration, 'State accepted'] == 0):
            self.states.loc[self.iteration, 'State default rate'] = 0
        else:
            self.states.loc[self.iteration, 'State default rate'] = self.states.loc[self.iteration, 'State defaulted'] / self.states.loc[self.iteration, 'State accepted'] if self.states.loc[self.iteration, 'State accepted'] != 0 else 0
        # calculate total default rate for each state
        if(self.states.loc[self.iteration, 'Total accepted'] == 0):
            self.states.loc[self.iteration, 'Total default rate'] = 0
        else:
            self.states.loc[self.iteration, 'Total default rate'] = self.states.loc[self.iteration, 'Total defaulted'] / self.states.loc[self.iteration, 'Total accepted'] if self.states.loc[self.iteration, 'Total accepted'] != 0 else 0
    
        # calculate number of paid clients for each state
        # calculate state number of paid clients for each state
        self.states.loc[self.iteration, 'State paid'] = len(state_paid)
        # calculate total number of paid clients for each state
        self.states.loc[self.iteration, 'Total paid'] = self.states['State paid'].sum()
        
        # calculate paid rate for each state
        # calculate state paid rate for each state
        if(self.states.loc[self.iteration, 'State accepted'] == 0):
            self.states.loc[self.iteration, 'State paid rate'] = 0
        else:
            self.states.loc[self.iteration, 'State paid rate'] = self.states.loc[self.iteration, 'State paid'] / self.states.loc[self.iteration, 'State accepted'] if self.states.loc[self.iteration, 'State accepted'] != 0 else 0
        # calculate total paid rate for each state
        if(self.states.loc[self.iteration, 'Total accepted'] == 0):
            self.states.loc[self.iteration, 'total paid rate'] = 0
        else:
            self.states.loc[self.iteration, 'total paid rate'] = self.states.loc[self.iteration, 'Total paid'] / self.states.loc[self.iteration, 'Total accepted'] if self.states.loc[self.iteration, 'Total accepted'] != 0 else 0
        
        # calculate number of defaulted paid clients for each state
        # calculate state number of defaulted paid clients for each state
        self.states.loc[self.iteration, 'State defaulted paid'] = len(state_defaulted_paid)
        # calculate total number of defaulted paid clients for each state
        self.states.loc[self.iteration, 'Total defaulted paid'] = self.states['State defaulted paid'].sum()
        
        # calculate defaulted paid rate for each state
        # calculate state defaulted paid rate for each state
        if(self.states.loc[self.iteration, 'State accepted'] == 0):
        	self.states.loc[self.iteration, 'State defaulted paid rate'] = 0
        else:
        	self.states.loc[self.iteration, 'State defaulted paid rate'] = self.states.loc[self.iteration, 'State defaulted paid'] / self.states.loc[self.iteration, 'State accepted'] if self.states.loc[self.iteration, 'State accepted'] != 0 else 0
        # calculate total defaulted paid rate for each state
        if(self.states.loc[self.iteration, 'Total accepted'] == 0):
        	self.states.loc[self.iteration, 'total defaulted paid rate'] = 0
        else:
        	self.states.loc[self.iteration, 'total defaulted paid rate'] = self.states.loc[self.iteration, 'Total defaulted paid'] / self.states.loc[self.iteration, 'Total accepted'] if self.states.loc[self.iteration, 'Total accepted'] != 0 else 0
        
        # calculate defaulted to paid ratio for each state
        # calculate state defaulted to paid ratio for each state
        if(self.states.loc[self.iteration, 'State paid'] == 0):
        	self.states.loc[self.iteration, 'State defaulted to paid ratio'] = 0
        else:
        	self.states.loc[self.iteration, 'State defaulted to paid ratio'] = self.states.loc[self.iteration, 'State defaulted'] / self.states.loc[self.iteration, 'State paid'] if self.states.loc[self.iteration, 'State accepted'] != 0 else 0
        # calculate total defaulted to paid ratio for each state
        if(self.states.loc[self.iteration, 'Total paid'] == 0):
        	self.states.loc[self.iteration, 'total defaulted to paid ratio'] = 0
        else:
        	self.states.loc[self.iteration, 'total defaulted to paid ratio'] = self.states.loc[self.iteration, 'Total defaulted'] / self.states.loc[self.iteration, 'Total paid'] if self.states.loc[self.iteration, 'Total paid'] != 0 else 0
        
        
            
        # get moving variables
        if self.iteration > self.window:
            for var in self.moving_variables:
                self.states.loc[self.iteration, var.replace('State', 'Moving')] = self.states.loc[(self.iteration - self.window) : self.iteration, var].sum()
            
            # calculate ratios
            self.states.loc[self.iteration, 'Moving repeat applications share'] = self.states.loc[self.iteration, 'Moving repeat applications'] / self.states.loc[self.iteration, 'Moving applications'] if self.states.loc[self.iteration, 'Moving applications'] != 0 else 0
            self.states.loc[self.iteration, 'Moving repeat loans share'] = self.states.loc[self.iteration, 'Moving repeat accepted'] / self.states.loc[self.iteration, 'Moving accepted'] if self.states.loc[self.iteration, 'Moving accepted'] != 0 else 0
            self.states.loc[self.iteration, 'Moving acceptance rate'] = self.states.loc[self.iteration, 'Moving accepted'] / self.states.loc[self.iteration, 'Moving applications'] if self.states.loc[self.iteration, 'Moving applications'] != 0 else 0
            self.states.loc[self.iteration, 'Moving acceptance rate squared'] = self.states.loc[self.iteration, 'Moving acceptance rate'] ** 2
            self.states.loc[self.iteration, 'Moving default rate'] = self.states.loc[self.iteration, 'Moving defaulted'] / self.states.loc[self.iteration, 'Moving accepted'] if self.states.loc[self.iteration, 'Moving accepted'] != 0 else 0
            self.states.loc[self.iteration, 'Moving paid rate'] = self.states.loc[self.iteration, 'Moving paid'] / self.states.loc[self.iteration, 'Moving accepted'] if self.states.loc[self.iteration, 'Moving accepted'] != 0 else 0
            self.states.loc[self.iteration, 'Moving defaulted paid rate'] = self.states.loc[self.iteration, 'Moving defaulted paid'] / self.states.loc[self.iteration, 'Moving accepted'] if self.states.loc[self.iteration, 'Moving accepted'] != 0 else 0
            self.states.loc[self.iteration, 'Moving profit per loan'] = self.states.loc[self.iteration, 'Moving profit'] / self.states.loc[self.iteration, 'Moving accepted'] if self.states.loc[self.iteration, 'Moving accepted'] != 0 else 0
            self.states.loc[self.iteration, 'Moving defaulted to paid ratio'] = self.states.loc[self.iteration, 'Moving defaulted'] / self.states.loc[self.iteration, 'Moving paid'] if self.states.loc[self.iteration, 'Moving paid'] != 0 else 0
            
        else:
            for var in self.moving_variables:
                self.states.loc[self.iteration, var.replace('State', 'Moving')] = 0
            
            # calculate ratios
            self.states.loc[self.iteration, 'Moving repeat applications share'] = 0
            self.states.loc[self.iteration, 'Moving repeat loans share'] = 0
            self.states.loc[self.iteration, 'Moving acceptance rate'] = 0
            self.states.loc[self.iteration, 'Moving default rate'] = 0
            self.states.loc[self.iteration, 'Moving paid rate'] = 0
            self.states.loc[self.iteration, 'Moving defaulted paid rate'] = 0
            self.states.loc[self.iteration, 'Moving profit per loan'] = 0
            self.states.loc[self.iteration, 'Moving defaulted to paid ratio'] = 0
                           
#        # get growth variables
#        if self.iteration > self.window:
#            for var in self.growth_variables:
#                self.states.loc[self.iteration, (var + ' growth rate')] = (self.states.loc[self.iteration, var] / self.states.loc[self.iteration - 1, var]) - 1 if self.states.loc[self.iteration - 1, var] != 0 else 0
#        else:
#            for var in self.growth_variables:
#                self.states.loc[self.iteration, (var + ' growth rate')] = 0
        
    # convert state variables to features recognized by RL agent
    def get_state_features(self):
               
        if(self.states.empty):
            self.state = pd.Series(data = 0, index = self.features)
        else:
            self.state = self.states.loc[self.iteration, self.features]
        self.update_features_history(self.features)
    
    # add the latest state features to the feature history dataframe
    def update_features_history(self, features):
        
        for feature in features:
            self.stateFeatures.loc[self.iteration, feature] = self.state[feature]        
    
    # extract reward from the current state recognized by the RL agent
    def get_state_reward(self):
        
        # turn off the reward. The reward is taken by the agent straight from the observed reward dataframe
        if self.reward_type == 'real':
            if (self.iteration < 135):
                self.reward = 0
            else:
                self.reward = self.rewards.apply(lambda x: x[self.actions.loc[x.name - 1, 'action']], axis = 1).sum()
        # observe moving average profit each week
        elif self.reward_type == 'moving':
            reward = 'Moving profit'
            self.reward = self.states.loc[self.iteration, reward]
        # observe profit each week
        elif self.reward_type == 'state':
            reward = 'State profit'
            self.reward = self.states.loc[self.iteration, reward] if self.iteration <= 113 else self.states.loc[113, reward]
        # observe only total profit in the end of the episode
        elif self.reward_type == 'total':
            if (self.iteration < 62):
                self.reward = 0
            else:
                reward = 'Total profit'
                self.reward = self.states.loc[self.iteration, reward]
    
    # convert RL agent's action to a complete policy recognized by the environment
    def action_to_policy(self, action):
        
        policy = self.policy.copy()
        
        # if None, use default policy
        if(action is None):
            return policy
        
        if(self.action_type == 'discrete_change' or self.action_type == 'continuous_change'):
            policy['threshold_repeat'] = max(min(policy['threshold_repeat'] + action, 100), 1)
            policy['threshold_new'] = max(min(policy['threshold_new'] + action, 100), 1)
        elif(self.action_type == 'continuous_action'):
            policy['threshold_repeat'] = max(min(action, 100), 1)
            policy['threshold_new'] = max(min(action, 100), 1)
        elif(self.action_type == 'discrete_change_separate'):
            policy['threshold_repeat'] = max(min(policy['threshold_repeat'] + self.threshold_change_dict[action][0], 100), 1)
            policy['threshold_new'] = max(min(policy['threshold_new'] + self.threshold_change_dict[action][1], 100), 1)
        elif(self.action_type == 'discrete_action_separate'):
            policy['threshold_repeat'] = self.threshold_dict[action][0]
            policy['threshold_new'] = self.threshold_dict[action][1]
        else:
            policy['threshold_repeat'] = action
            policy['threshold_new'] = action
        
        self.policy = policy
        return policy
    
    # generate new state based on previous state and actions taken
    def take_action(self, action = None):
        self.actions.loc[self.iteration, 'action'] = self.convert_to_simple_action(action)
        
        policy = self.action_to_policy(action)
        
        state_defaulted, state_paid, state_profit_defaulted_paid = self.update_environment(policy)
        self.update_state_history(policy, state_defaulted, state_paid, state_profit_defaulted_paid)
        self.get_state_features()
        self.get_state_reward()

        self.predict_states()
        
    # visualize the results
    def plot_states(self):    
        self.stateParameters.plot(subplots = True, sharex = False, sharey = False, figsize = (12, 5 * self.stateParameters.shape[1]))
        self.states.plot(subplots = True, sharex = False, sharey = False, figsize = (12, 5 * self.states.shape[1]))
        
    # run a number of iterations
    def run_iterations(self, iterations = 114, plot = False, output = True):
        
        self.reset()
        
        for iteration in range(iterations):
            self.take_action()
            
        if plot:
            self.plot_states()
        
        # track the following values after the end of episode
        if output:
            results = {'result' : self.result, 'states' : self.states, 'stateParameters' : self.stateParameters, 'stateFeatures' : self.stateFeatures}
            return results
    
    # run a number of episodes
    def run_episodes(self, episodes = 1):
    
        results = []
                
        for episode in range(episodes):  
            print('episode ' + str(episode))
            results.append(self.run_iterations())
        
        return results
    
    # generate features for state space definition
    def generate_features(self, episodes = 100):
        
        features = pd.DataFrame(data = [])
                
        for episode in range(episodes):  
            self.run_iterations()
            features = pd.concat([features, self.stateFeatures])
        
        return features
    
    # get descriptive statistics of the state space
    def describe_features(self, features):
        
        feature_description = features.describe()
        
        return feature_description
    
    # load stored descriptive statistics of the state space
    def load_feature_description(self, update = False):
        
        if not update and os.path.isfile('./feature_description.txt'):
            feature_description = pd.read_csv('feature_description.txt', sep = '\t', index_col = 0)
            return feature_description
        
        features = self.generate_features()
        feature_description = self.describe_features(features)
        feature_description.to_csv('feature_description.txt', sep = '\t')
        return feature_description
    
    # get optimal threshold for the current episode
    def get_optimal_threshold(self):
        total_profits = pd.Series(data = [])
        
        total_profits = self.true_rewards.loc[53:113, :].sum()
        highest_profit = total_profits.max()
        optimal_threshold = total_profits.argmax()
        
        return optimal_threshold, highest_profit, total_profits
    
    # get optimal thresholds for a number of episodes                    
    def get_optimal_distribution(self, iterations):
        optimal_thresholds = pd.DataFrame(data = [])
        for i in range(iterations):
            print('iteration: {}'.format(i), end = "\r")
            self.run_iterations()
            optimal_threshold, highest_profit, total_profits = self.get_optimal_threshold()
            optimal_thresholds.loc[i, 'Optimal threshold'] = optimal_threshold
            optimal_thresholds.loc[i, 'Optimal profit'] = highest_profit  
                                  
        return optimal_thresholds
    
    # convert action generated by the RL agent to the actual acceptance threshold                              
    def convert_to_real_action(self, action):
        if(self.action_type == 'discrete_change'):
            real_action = (action - 2) * 5
        elif(self.action_type == 'discrete_action'):
            real_action = (action + 1) * 5        
        return real_action
    
    # convert actual acceptance threshold to an action understandable by the RL agent
    def convert_to_simple_action(self, action):
        if action is None:
            action = self.default_policy['threshold_repeat']
        if(self.action_type == 'discrete_change'):
            simple_action = action / 5 + 2
        elif(self.action_type == 'discrete_action'):
            simple_action = action / 5 - 1       
        return simple_action
    
    # save action history to a csv file
    def save_action_history(self):
        stateParameters = self.stateParameters.copy()
        stateParameters['Threshold repeat'] = self.stateParameters['Threshold repeat'].apply(self.convert_to_simple_action).astype(int)
        stateParameters['Threshold new'] = self.stateParameters['Threshold new'].apply(self.convert_to_simple_action).astype(int)
        stateParameters.to_csv('C:/My/Reinforcement Learning Algorithm/Example/keras-rl-master/stateParameters.csv') 
    
    # save state history to a csv file    
    def save_state_history(self):
        stateFeatures = self.stateFeatures.copy()
        stateFeatures.to_csv('C:/My/Reinforcement Learning Algorithm/Example/keras-rl-master/stateFeatures.csv') 
    
    # define the action space based on the action type    
    def choose_action_set(self, action_type = None):
        if action_type is None:
            action_type = self.action_type
            
        if(action_type == 'discrete_change' or action_type == 'continuous_change'):
            self.action_set = self.single_threshold_change_dict
        elif(self.action_type == 'discrete_action' or self.action_type == 'continuous_action'):
            self.action_set = self.single_threshold_action_dict
        elif(self.action_type == 'discrete_change_separate' or self.action_type == 'continuous_change_separate'):
            self.action_set = self.separate_threshold_change_dict
        elif(self.action_type == 'discrete_action_separate' or self.action_type == 'continuous_action_separate'):
            self.action_set = self.separate_threshold_action_dict
    
    # predict rewards for higher acceptance thresholds
    def predict_rewards(self, state_defaulted, state_paid, state_defaulted_paid):
        self.rewards.loc[self.iteration, list(self.action_set.keys())] = 0 if 53 <= self.iteration <= 113 else None
        result_copy = self.result.copy()
        if not result_copy.empty:                
            iteration = self.iteration if self.iteration <= 114 else 114
            for i in range(53, iteration):
                # for each possible threshold
                for action in self.action_set.keys():
                    threshold = self.action_set[action]
                    # if the threshold would be higher than the actual one and we actually have the outcome data for the calculations
                    if threshold < self.convert_to_real_action(self.actions.loc[i, 'action']):
                        self.rewards.loc[i, action] = None
                        continue
                    # reset the variables and tables
                    iteration_state_profit = iteration_state_loss_defaulted = iteration_state_profit_paid = iteration_state_profit_defaulted_paid = 0
                    result_state_defaulted = result_state_profit_paid = result_state_profit_defaulted_paid = pd.DataFrame(data = [])
                    # add loss for each defaulted loan    
                    result_state_defaulted = result_copy.loc[state_defaulted, :]         
                    result_state_defaulted = result_state_defaulted[(result_state_defaulted['iteration'] == i) & (result_state_defaulted['score'] >= threshold)]
                    iteration_state_loss_defaulted = result_state_defaulted['sum'].sum()
                    iteration_state_profit -= iteration_state_loss_defaulted
                    # add profit for each paid loan
                    result_state_profit_paid = result_copy.loc[state_paid, :]
                    result_state_profit_paid = result_state_profit_paid[(result_state_profit_paid['iteration'] == i) & (result_state_profit_paid['score'] >= threshold)]
                    iteration_state_profit_paid = result_state_profit_paid['profit'].sum()
                    iteration_state_profit += iteration_state_profit_paid
                    # add profit for each defaulted paid loan
                    result_state_profit_defaulted_paid = result_copy.loc[state_defaulted_paid, :]
                    result_state_profit_defaulted_paid = result_state_profit_defaulted_paid[(result_state_profit_defaulted_paid['iteration'] == i) & (result_state_profit_defaulted_paid['score'] >= threshold)]
                    iteration_state_profit_defaulted_paid = result_state_profit_defaulted_paid['profit'].sum() + result_state_profit_defaulted_paid['sum'].sum()
                    iteration_state_profit += iteration_state_profit_defaulted_paid
                    # add profit to rewards dataframe
                    self.rewards.loc[i, action] += iteration_state_profit * self.reward_scaler
    
    # calculate rewards for all the acceptance thresholds
    def predict_rewards_cheating(self):
        self.true_rewards.loc[self.iteration, list(self.action_set.keys())] = 0 if self.iteration <= 113 else None
        result_copy = self.result.copy()
        if not result_copy.empty:
            iteration = self.iteration if self.iteration <= 114 else 114
            for i in range(1, iteration):
                # for each possible threshold
                for action in self.action_set.keys():
                    threshold = self.action_set[action]
                    # if the threshold would be higher than the actual one and we actually have the outcome data for the calculations
    #                if threshold < self.convert_to_real_action(self.actions.loc[i, 'action']):
    #                    self.rewards.loc[i, action] = None
    #                    continue
                    # reset the variables and tables
                    iteration_state_profit = iteration_state_loss_defaulted = iteration_state_profit_paid = iteration_state_profit_defaulted_paid = 0
                    result_state_defaulted = result_state_profit_paid = result_state_profit_defaulted_paid = pd.DataFrame(data = [])
                    # add loss for each defaulted loan    
                    result_state_defaulted = result_copy[result_copy['dca_at'] == self.iteration]    
                    result_state_defaulted = result_state_defaulted[(result_state_defaulted['iteration'] == i) & (result_state_defaulted['score'] >= threshold)]
                    iteration_state_loss_defaulted = result_state_defaulted['sum'].sum()
                    iteration_state_profit -= iteration_state_loss_defaulted
                    # add profit for each paid loan
                    result_state_profit_paid = result_copy[(result_copy['maturation_at'] == self.iteration) & (result_copy['dca'] == False)]
                    result_state_profit_paid = result_state_profit_paid[(result_state_profit_paid['iteration'] == i) & (result_state_profit_paid['score'] >= threshold)]
                    iteration_state_profit_paid = result_state_profit_paid['profit'].sum()
                    iteration_state_profit += iteration_state_profit_paid
                    # add profit for each defaulted paid loan
                    result_state_profit_defaulted_paid = result_copy[result_copy['late_payment_at'] == self.iteration]
                    result_state_profit_defaulted_paid = result_state_profit_defaulted_paid[(result_state_profit_defaulted_paid['iteration'] == i) & (result_state_profit_defaulted_paid['score'] >= threshold)]
                    iteration_state_profit_defaulted_paid = result_state_profit_defaulted_paid['profit'].sum() + result_state_profit_defaulted_paid['sum'].sum()
                    iteration_state_profit += iteration_state_profit_defaulted_paid
                    # add profit to rewards dataframe
                    self.true_rewards.loc[i, action] += iteration_state_profit * self.reward_scaler
    
    # calculate rewards for all the acceptance thresholds faster
    def predict_rewards_immediate_cheating(self):
        self.true_rewards.loc[self.iteration, list(self.action_set.keys())] = 0 if self.iteration <= 113 else None

        iteration = self.iteration if self.iteration <= 114 else 114
        applications_data = self.result.copy()
        if not applications_data.empty and 'iteration' in applications_data.columns:
            applications_data = applications_data[applications_data['iteration'] == iteration]
            if not applications_data.empty:
                for action in range(20):
                    t = self.convert_to_real_action(action)
                    applications_data['accept'] = applications_data.apply(lambda x: 1 if x['score'] >= t else 0, axis = 1)
                    applications_data['realized_profit'] = applications_data.apply(lambda x: 0 if x['accept'] == 0 else (-x['sum'] if (x['dca'] and (not x['late_payment'])) else x['profit']), axis = 1)
                    self.true_rewards.loc[iteration, action] = applications_data['realized_profit'].sum()        
                    
    # predict states for various acceptance thresholds
    def predict_states(self):
        for action in self.action_set.keys():
            if self.result.empty:
                self.statePrediction.loc[self.iteration, action] = 0
            else:
                threshold = self.action_set[action]
                iteration_result = self.result[self.result['iteration'] == self.iteration]
                iteration_applications = iteration_result.shape[0]
                iteration_accepted = iteration_result[iteration_result['score'] >= threshold].shape[0]
                acceptance_rate = iteration_accepted / iteration_applications if iteration_applications != 0 else 0
                
                self.statePrediction.loc[self.iteration, action] = acceptance_rate
    
    # generate average rewards based on a number of episodes            
    def simulate_rewards(self, iterations = 100):
        rewards = pd.DataFrame(data = 0, index = range(114), columns = self.action_set)
        self.cheating = True
        for i in range(iterations):
            print(i, end = "\r")
            self.run_iterations(134)
            rewards += self.true_rewards
        rewards /= iterations
        
        return rewards
        