'''
FeatureTransformer class provides functionality to transform a state object into
a set of features using the Gausian Radial Basis Functions transformation.
Model class incorporates the value function model providing functionality to
predict action values and update model parameters.
EnvironmentModel class provides the functionality for the environment model,
able to predict following states based on actions.
'''

# import external packages
import os
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.externals import joblib

class FeatureTransformer:
    def __init__(self, env):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)
    
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=500))
                ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))
    
        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, observations):
        scaled = self.scaler.transform(np.array(observations).reshape(1, -1))
        return self.featurizer.transform(scaled)

class Model:
    # initialize value function model as a set of SGD regressors for each action
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        self.learning_rate = learning_rate
        for i in range(env.action_space.n):
          #model = SGDRegressor(learning_rate='constant', eta0 = self.learning_rate)
          model = SGDRegressor(alpha = 0, l1_ratio = 0, max_iter = 1, shuffle = False, epsilon = 0, learning_rate = 'invscaling', eta0 = self.learning_rate, power_t = 0.25)
          model.partial_fit(feature_transformer.transform( env.observation_space.sample() ), [0])
          self.models.append(model)
    
    # reset the learning rate of the value function model
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for model in self.models:
            model.eta0 = self.learning_rate
            model.t_ = 2.0
            model.learning_rate = 'constant'
    
    # predict action values for a state
    def predict(self, s):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X)[0] for m in self.models])
    
    # perform SGD update
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])
        
    # save model instance to a file
    def save(self, path = '/model_dump'):
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
    
    # load model instance from a file
    def load(self, path = '/model_dump'):
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
        
class EnvironmentModel:
    # initialize environment model as an SGD regressor
    def __init__(self, env, learning_rate):
        self.env = env
        self.model = SGDRegressor(learning_rate='constant', eta0 = learning_rate)
        self.model.partial_fit(env.action_space.sample(), [0])
    
    # predict next state based on the action
    def predict(self, a):
        return self.model.predict(a)[0]
    
    # perform SGD update
    def update(self, a, s):
        self.model.partial_fit(a, [s])