# Structure of the folder

**Manager class** provides functionality to go through experiment, such as functions to initialize the agent and the environment, functions to run train, test, distorted and real episodes, functions to run train and test experiments, functions to visualize episodes and experiments.

**SimulationEnv class** is built on top of gym environment class to define state and action spaces and set up the state-action-reward exchange between the agent and the environment classes.

**Environment class** works as a connection between the sim class (the simulation itself)and the RL agent. It gets actions from the agent, calls the simulation class to generate the loan applications based on the actions, dynamicaly calculates and stores the characteristics of loan portfolio and rewards received, provides a set of supplementary functions.

**Sim class** is responsible for the generation of loan applications and their characteristics. Simulation parameters are substituted with ''s and 'np.nan's for confidentiality reasons.

**Agent class** incorporates the reinforcement learning algorithm. It provides the functionality to interact with the environment passing actions sampled using a value function model instance and policy instance, to update value function model parameters based on observations received from environment.

**FeatureTransformer class** (inside the Model script) provides functionality to transform a state object into a set of features using the Gausian Radial Basis Functions transformation.

**Model class** incorporates the value function model providing functionality to predict action values and update model parameters.

**EnvironmentModel class** (inside the Model script) provides the functionality for the environment model, able to predict following states based on actions.

**Policy class** provides a choice of action sample policies for the RL agent, including greedy, epsilon-greedy, random, default, boltzmann-Q and derivatives.
