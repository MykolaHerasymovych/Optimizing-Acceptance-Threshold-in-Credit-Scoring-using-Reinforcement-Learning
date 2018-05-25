'''
Sim class is responsible for the generation of loan applications and their 
characteristics. Simulation parameters are substituted with ''s and 'np.nan's 
for confidentiality reasons.
'''

# import external packages
import numpy as np
import pandas as pd
import gc

class Sim:
    # initialize simulation parameters
    def __init__(self, distortions = {'e': 1, 'news_positives_score_bias': 0, 'repeats_positives_score_bias': 0, 'news_negatives_score_bias': 0, 'repeats_negatives_score_bias': 0, 'news_default_rate_bias': 0, 'repeats_default_rate_bias': 0, 'late_payment_rate_bias': 0, 'ar_effect': 0}):
        
        # distortion parameters
        self.distortions = distortions
        self.e = distortions['e']  # noize parameter
        self.news_positives_score_bias = distortions['news_positives_score_bias'] # factor increase in mean positives scores for new clients
        self.repeats_positives_score_bias = distortions['repeats_positives_score_bias'] # factor increase in mean positives scores for repeat clients
        self.news_negatives_score_bias = distortions['news_negatives_score_bias'] # factor decrease in mean negatives scores for new clients
        self.repeats_negatives_score_bias = distortions['repeats_negatives_score_bias'] # factor decrease in mean negatives scores for repeat clients
        self.news_default_rate_bias = distortions['news_default_rate_bias'] # factor change in segment default rates for new clients
        self.repeats_default_rate_bias = distortions['repeats_default_rate_bias'] # factor change in segment default rates for repeat clients
        self.late_payment_rate_bias = distortions['late_payment_rate_bias'] # factor change in segment late payment rates
        self.ar_effect = distortions['ar_effect'] # coefficient for acceptance rate dependence for applications number
        
        # constants
        self.ar_historical = np.nan # historical acceptance rate
        self.c_ar_new = np.nan # acceptance rate effect coefficient for new clients
        self.c_ar_repeat = np.nan # acceptance rate effect coefficient for repeat clients
        
        # credit scoring model performance
        self.new_negative_score_mean = np.nan # score mean for new good applications
        self.new_negative_score_std = np.nan # score std for new good applications
        self.new_positive_score_mean = np.nan # score mean for new bad applications
        self.new_positive_score_std = np.nan # score std for new bad applications
        self.repeat_negative_score_mean = np.nan # score mean for repeat good applications
        self.repeat_negative_score_std = np.nan # score std for repeat good applications
        self.repeat_positive_score_mean = np.nan # score mean for repeat bad applications
        self.repeat_positive_score_std = np.nan # score std for repeat bad applications
        
        # distort model performance
        new_score_dif = self.new_negative_score_mean - self.new_positive_score_mean
        repeat_score_dif = self.repeat_negative_score_mean - self.repeat_positive_score_mean
        
        # adjust model performance according to assumptions
        self.new_positive_score_mean += self.news_positives_score_bias * new_score_dif
        self.repeat_positive_score_mean += self.repeats_positives_score_bias * repeat_score_dif
        self.new_negative_score_mean -= self.news_negatives_score_bias * new_score_dif
        self.repeat_negative_score_mean -= self.repeats_negatives_score_bias * repeat_score_dif
        
        # loan application segment estimates
        # 0 - customer segment, 
        # 1 - frequency among all applications, 
        # 2 - loan sum proportion among all applications, 
        # 3 - probability of going 60 days overdue, 
        # 4 - average score with current credit scoring model, 
        # 5 - average loan sum, 
        # 6 - average loan duration, 
        # 7 - average number of loans, 
        # 8 - probability of paying when overdue, 
        # 9 - average profit value
        
        self.new_loans = {
        1:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        2:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        3:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        4:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        5:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        6:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        }
        
        self.repeat_loans = {
        7:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        8:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        9:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        10:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        11:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        12:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        13:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        14:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        15:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        16:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        17:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        18:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        19:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        20:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        21:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        22:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        23:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        24:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        25:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        26:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        27:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        28:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        29:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        30:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        31:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        32:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        33:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        34:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        35:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        36:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        37:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        38:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        39:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        40:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        41:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        42:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        43:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        44:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        45:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        46:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        47:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        48:['',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        }
            
        
        # distort segment estimates
        for x in self.new_loans:
            self.new_loans[x][3] *= 1 + self.news_default_rate_bias
            self.new_loans[x][3] = self.new_loans[x][3] if self.new_loans[x][3] <= 1 else 1
            self.new_loans[x][8] *= 1 + self.late_payment_rate_bias
            self.new_loans[x][8] = self.new_loans[x][8] if self.new_loans[x][8] <= 1 else 1
            
        for x in self.repeat_loans:
            self.repeat_loans[x][3] *= 1 + self.repeats_default_rate_bias
            self.repeat_loans[x][3] = self.repeat_loans[x][3] if self.repeat_loans[x][3] <= 1 else 1
            self.repeat_loans[x][8] *= 1 + self.late_payment_rate_bias
            self.repeat_loans[x][8] = self.repeat_loans[x][8] if self.repeat_loans[x][8] <= 1 else 1
            
        # debt segments
        self.debt_ranges = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        self.debt_probabilities = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        
        # accepted applications and accepted rate
        self.all_accepted = pd.DataFrame(data=[])
        self.ar = 0
    
    # generate dataframe of weekly loan applications
    def generateInput(self, iteration = 1):
        
        # received applications simulation parameters
        trend_change = np.nan ### week of trend change
        
        # simulate number of weekly applications
        if (iteration <= trend_change):
            # data generating process before the trend change
            
            # generating the total number of weekly new applications
            no_of_weekly_new_applications = (np.nan*(iteration)) - (np.nan*((iteration)**2))
            no_of_weekly_new_applications = no_of_weekly_new_applications + np.random.normal(np.nan,np.nan) * self.e
            no_of_weekly_new_applications = no_of_weekly_new_applications if no_of_weekly_new_applications > 0 else np.nan
            # generating the total number of weekly repeat applications
            no_of_weekly_repeat_applications = (np.nan*(iteration)) - (np.nan*((iteration)**2)) - (np.nan*((iteration)**3)) + (np.nan*no_of_weekly_new_applications)
            no_of_weekly_repeat_applications = no_of_weekly_repeat_applications + np.random.normal(np.nan,np.nan) * self.e
            no_of_weekly_repeat_applications = no_of_weekly_repeat_applications if no_of_weekly_repeat_applications > 0 else 0
        else:                                                                                          
            # data generating process after the trend change
            
            # generating the total number of weekly new applications
            no_of_weekly_new_applications = np.nan + (np.nan*(iteration)) - (np.nan*((iteration)**2))
            no_of_weekly_new_applications += (self.ar - self.ar_historical) * self.c_ar_new * (iteration - trend_change) * self.ar_effect
            no_of_weekly_new_applications += np.random.normal(np.nan,np.nan) * self.e
            no_of_weekly_new_applications = no_of_weekly_new_applications if no_of_weekly_new_applications > 0 else np.nan
            # generating the total number of weekly repeat applications
            no_of_weekly_repeat_applications = np.nan + (np.nan*(iteration)) - (np.nan*((iteration)**2)) + (np.nan*((iteration)**3)) + (np.nan*no_of_weekly_new_applications)
            no_of_weekly_repeat_applications += (self.ar - self.ar_historical) * self.c_ar_repeat * (iteration - trend_change) * self.ar_effect
            no_of_weekly_repeat_applications += np.random.normal(np.nan,np.nan) * self.e
        
        # scale volumes
        no_of_weekly_new_applications *= np.nan
        no_of_weekly_repeat_applications *= np.nan
        
        
        # generate application characteristics
        weekly_applications = pd.DataFrame(data=[])
        
        # generate new client loan application characteristics
        for i in range(1, (int(no_of_weekly_new_applications)+1)):
            loantype = round((np.random.choice(np.arange(1, len(self.new_loans) + 1), p=[round(self.new_loans[x][1], 3) for x in range(1, len(self.new_loans) + 1)]))) # segment
            id = 'new_' + str(iteration) + '_' + str(i) # unique id
            sum = round(self.new_loans[loantype][5], 0) # loan sum
            duration = round((self.new_loans[loantype][6])/7, 0) # loan duration
            debt = self.debt_ranges[np.random.choice(np.arange(0, len(self.debt_probabilities)), p=self.debt_probabilities)] # outstanding debt
            try:
                dca_probability = (self.new_loans[loantype][3]) #+ ((np.nan*debt**2) - (np.nan*debt)) # probability of going overdue
                dca = np.random.binomial(1, dca_probability) # if goes overdue
            except:
                dca = np.random.binomial(1, np.nan) # if the probability > 1
            late_payment = 1 if dca * np.random.binomial(1, self.new_loans[loantype][8]) == 1 else 0 # if repays after going overdue
            loan_value = self.new_loans[loantype][9] # profit value
            score = np.random.normal(self.new_negative_score_mean, self.new_negative_score_std) if dca == 0 else np.random.normal(self.new_positive_score_mean, self.new_positive_score_std) # credit score
            #score -= debt/17 # adjust for debt
            
            # store characteristics
            weekly_applications.loc[id, 'iteration'] = iteration
            weekly_applications.loc[id, 'maturation_at'] = iteration + duration
            weekly_applications.loc[id, 'repeat'] = False
            weekly_applications.loc[id, 'sum'] = int(sum)
            weekly_applications.loc[id, 'duration'] = int(round(self.new_loans[loantype][6], 0))
            weekly_applications.loc[id, 'debt'] = debt
            weekly_applications.loc[id, 'score'] = score
            weekly_applications.loc[id, 'dca'] = bool(dca)
            weekly_applications.loc[id, 'dca_at'] = iteration + duration + np.nan if dca == 1 else 'NA'
            weekly_applications.loc[id, 'late_payment'] = late_payment
            weekly_applications.loc[id, 'late_payment_at'] = iteration + duration + int(np.random.uniform(np.nan, np.nan)) if late_payment == 1 else 'NA'
            weekly_applications.loc[id, 'profit'] =  loan_value
        
        # generate repeat client loan application characteristics                           
        for i in range(1, (int(no_of_weekly_repeat_applications)+1)):
            loantype = round((np.random.choice(np.arange(len(self.new_loans) + 1, len(self.new_loans) + len(self.repeat_loans) + 1), p=[round(self.repeat_loans[x][1], 4) for x in range(len(self.new_loans) + 1, len(self.new_loans) + len(self.repeat_loans) + 1)]))) # segment
            id = 'repeat_' + str(iteration) + '_' + str(i) # unique id
            sum = round(self.repeat_loans[loantype][5], 0) # loan sum
            duration = round((self.repeat_loans[loantype][6])/7, 0) # loan duration
            debt = self.debt_ranges[np.random.choice(np.arange(0, len(self.debt_probabilities)), p=self.debt_probabilities)] # outstanding debt
            try:
                dca_probability = self.repeat_loans[loantype][3] #+ ((np.nan*debt**2) - (np.nan*debt)) # probability of going overdue
                dca = np.random.binomial(1, dca_probability) # if goes overdue
            except:
                dca = np.random.binomial(1, np.nan) # if the probability > 1
            late_payment = 1 if dca * np.random.binomial(1, self.repeat_loans[loantype][8]) == 1 else 0 # if repays after going overdue
            loan_value = self.repeat_loans[loantype][9] # profit value                                           
            score = np.random.normal(self.repeat_negative_score_mean, self.repeat_negative_score_std) if dca == 0 else np.random.normal(self.repeat_positive_score_mean, self.repeat_positive_score_std) # credit score
            #score -= debt/17 # adjust for debt
            
            # store characteristics
            weekly_applications.loc[id, 'iteration'] = iteration
            weekly_applications.loc[id, 'maturation_at'] = iteration + duration
            weekly_applications.loc[id, 'repeat'] = True
            weekly_applications.loc[id, 'sum'] = int(sum)
            weekly_applications.loc[id, 'duration'] = int(round(self.repeat_loans[loantype][6], 0))
            weekly_applications.loc[id, 'debt'] = debt
            weekly_applications.loc[id, 'score'] = score
            weekly_applications.loc[id, 'dca'] = bool(dca)
            weekly_applications.loc[id, 'dca_at'] = iteration + duration + np.nan if dca == 1 else 'NA'
            weekly_applications.loc[id, 'late_payment'] = late_payment
            weekly_applications.loc[id, 'late_payment_at'] = iteration + duration + int(np.random.uniform(np.nan, np.nan)) if late_payment == 1 else 'NA'
            weekly_applications.loc[id, 'profit'] =  loan_value                       
    
        return weekly_applications
    
    # performs the loan application acceptance decision
    def accept(self, app, threshold = 50):
        if app['score'] < threshold:
            app['accept'] = False
            return app
        else: 
            app['accept'] = True
            return app
    
    # generates dataframe of loan applications and ids of paid, overdue and paid after overdue loans for current week
    def simulate(self, i, weekly_applications, threshold = 50):
        
        weekly_applications = weekly_applications.apply(self.accept, axis = 1, args = [threshold])
        
        if 'accept' in weekly_applications.columns:
            accepted = weekly_applications.loc[weekly_applications['accept'] == True]
            self.ar = weekly_applications['accept'].mean()
            self.all_accepted = pd.concat([self.all_accepted, accepted])
            
            del accepted
                
        if not self.all_accepted.empty:    
            matured = self.all_accepted.loc[self.all_accepted['maturation_at'] == i]
            matured = matured.index
            dca = self.all_accepted.loc[self.all_accepted['dca_at'] == i]
            dca = dca.index
            paid_dca = self.all_accepted.loc[self.all_accepted['late_payment_at'] == i]
            paid_dca = paid_dca.index
            paid = self.all_accepted.loc[(self.all_accepted['maturation_at'] == i) & (self.all_accepted['dca'] == False)]
            paid = paid.index
            
            del matured
            
        else:
            self.ar = 0
            matured, dca, paid_dca, paid = [], [], [], []
            
        output = weekly_applications#[['iteration', 'sum', 'duration', 'score', 'repeat', 'accept', 'dca', 'profit']]
        
        gc.collect()
        del weekly_applications
        
        return output, paid, dca, paid_dca