# Optimizing Acceptance Threshold in Credit Scoring using Reinforcement Learning
<p align = "justify">
  This is a repository for my master thesis submitted as a part of the Master of Science in Quantitative Economics degree at the University of Tartu. The research was conducted in 2017-2018 at Creditstar Group, Estonia under the supervision of Karl Märka, Head of Data Science at Creditstar Group, and Oliver Lukason, PhD at the University of Tartu.

The source code for the project can be found in the Source directory.

A complete description of the project to be added.
</p>

## Background
<p align = "justify">
  The problem environment considered is a trivial credit business process. It starts when the loan provider receives a loan application with the data about the application and the potential borrower. The data is then passed to a credit scoring model that outputs a credit score which reflects the risk level of the loan application. Next, if the score is too low, the lender rejects the loan application. In case the score is high enough, the lender issues the loan. Eventually, if the loan applicant doesn't repay or defaults on their loan, the lender loses the money. In case the loan applicant repays, the lender gains extra money (from interest and fees). In the end, any of those cases affect the lender's profits.
</p>
<p align = "center">
  <img width = "600" alt = "Credit business process" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Credit_business_process.png">
</p>
<p align = "justify">
  In order to make the final accept / reject decision about a loan application, its credit score is compared to an acceptance threshold or cutoff point. The latter thus regulates the acceptance rates of a credit company and resulting default rates. The problem investigated in the master thesis is the optimization of an acceptance threshold to maximize credit company's profits. The traditional approach to the problem is to simply optimize the cutoff based on the credit score distribution of an independent test dataset of loan applications. Knowing the outcomes of all the loans in the dataset each possible acceptance threshold is considered and the corresponding profits are calculated. The optimal cutoff point is the one that corresponds to the maximum profit. In case the train / test split of loan application dataset is random, the acceptance threshold optimized for the test dataset is going to be optimal for the train dataset.
</p>
<p align = "center">
  <img width = "430" alt = "Acceptance threshold optimization: traditional approach" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Acceptance_threshold_optimization_1.png"><img width = "430" alt = "Acceptance threshold optimization: traditional approach" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Acceptance_threshold_optimization_2.png">
</p>

## Problem
<p align = "justify">
  The main flaw in the traditional approach is that it is static and backward looking. It fully focuses on the test dataset which consists only of historical prefiltered accepted loan applications for which the outcome is known. The actual population of loan applications that the credit scoring model is applied to consists of completely new both potentially rejeted and accepted loan applications. The inconsistency of the loan applications sample used to train the model and optimize the acceptance threshold and the general population of loan applications the model and cutoff point are applied to leads to the selection bias and population drift issues. Those distort the optimal acceptance threshold making the traditional solution incorrect.
</p>
<p align = "center">
  <img width="430" alt="Selection bias issue" src="https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Selection_bias.png"><img width="430" alt="Population drift issue" src="https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Population_drift.png">
</p>

## Solution
<p align = "justify">
  We solve the problem by developing and applying a reinforcement learning (RL) agent: a dynamic forward-looking system that adapts to the live data feedback (incoming loan applications) and adjusts acceptance threshold to maximize credit company's profits. The reinforcement learning problem is described by the interaction between the credit business environment and the RL agent. The interaction frequency is 1 week. The state space of environment consists of a continuous acceptance rate (in the previous week) variable that spans from 0 to 1. The action space consists of the discrete credit score acceptance threshold (for the following week) variable  that spans from 5 to 100 by the step of 5 (20 discrete actions). The reward variable is the company's profits. A more detailed RL problem specification can be found in the work.
</p>
<p align = "center">
  <img width = "800" alt = "RL scheme" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/RL_scheme_complex.png">
</p>
<p align = "justify">
  We solve the reinforcement learning problem with a simple Q-learning algorithm. Thus, the main goal is to approximate the Q-value function that is the mapping from state to expected discounted reward for each action. We preprocess state variable with Gaussian Radial Basis functions (RBFs) transformation to account for the non-linear nature of the value function. We use a set of 4 RBFs with 500 outputs each that gives us the output of 2000 preprocessed features. We then apply a separate Stochastic Gradient Descent (SGD) linear model for each action to approximate its Q-value, which results in 20 Q-values. Based on the predicted Q-values we use either greedy or Boltzmann-Q policy to choose an action in testing or training regime respectively. During the training only the SGD weights are learned.
</p>
<p align = "center">
  <img width = "600" alt = "Value function architecture" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Value_function_architecture.png">
</p>
<p align = "justify">
  The RL agent was trained using a Monte Carlo simulation of the credit business process. For training 100 simulation episodes were used. Each simulation episode cosists of 114 simulated weeks, 52 of which are warm-up with no interaction, 60 are interactive when the RL agent observes the state, chooses an action and learns the value function from the reward it gets later and 22 are weeks of delayed learning when the agent doesn't interact with the environment but observes delayed rewards and learns the value function. On average one training simulation episode takes around 5 minutes.
</p>
<p align = "center">
  <img width = "600" alt = "Training simulation episode structure" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Training_simulation_episode_structure.png">
</p>

## Results
### Baseline results
<p align = "justify">
  First, we apply the traditional approach to the cutoff point optimization based on the test dataset of loan applications for the credit scoring model. We compute the potential profit for the same range of acceptance threshold values as the action space of the reinforcement learning agent: from 5 to 100 by step of 5. One can notice that the lower the acceptance threshold, the more loan applications get accepted, the more issued loans default, the bigger the final loss. On the other hand, the higher the acceptance threshold, the less loan applications get accepted, the closer to zero the profit gets. The optimal acceptance threshold is found by maximizing profit and is equal to a credit score of 65.
</p>
<p align = "center">
  <img width = "600" alt = "Baseline results" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Baseline_results.png">
</p>

### Reinforcement learning results
<p align = "justify">
  Next, by performing the simulation-based training of the reinforcement learning agent as described above we approximate the Q-value function. The shape of the latter is very similar to the potential profit curve computed with the traditional approach: low acceptance thresholds have the lowest value and high acceptance thresholds have higher but still suboptimal value. The optimum is found by maximizing the value and corresponds to the same credit score of 65. Thus, the reinforcement learning approach can do as well as the traditional one in a static environment.
</p>
<p align = "center">
  <img width = "600" alt = "Value function shape" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/Value_function_shape.png">
</p>

### Performance comparison: simulated environments
<p align = "justify">
  To test the performance of the proposed approach in dynamic conditions we adjust the simulation parameters to mimic certain changes in the credit business environment. We simulate downwards and upwards shifts in the credit score distribution mimicing the population drift and downwards and upwards shift in default rates mimicing the selection bias issue. Based on 100 simulation runs for each scenario, the reinforcement learning algorithm manages to adapt to the new environments very quickly significantly outperforming the traditional approach in terms of profits according to the one-tailed t-test.
</p>
<p align = "center">
  <img width = "600" alt = "Performance comperison: simulated environments" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/RL_results_simulation.png">
</p>

<p align = "center">
  
  |Scenario | t-statistic | p-value|
  |:--------|-------------|--------|
  |Scenario 1: downwards shift in score distribution	| 29.56631	| 1.55E-51 |
  |Scenario 2: upwards shift in score distribution	| 42.72066 |	2.45E-66 |
  |Scenario 3: downwards shift in default rates	| 5.172688	| 5.95E-07 |
  |Scenario 4: upwards shift in default rates	| 4.600158	| 6.20E-06 |

</p>

### Performance comparison: real environment
<p align = "justify">
  Finally, we compare performance of the developed reinforcement learning system to the baseline approach on the more recent 24 weeks of real loan applications data. On the figure one can see the difference between the baseline acceptance threshold and the one chosen with the proposed algorithm. After exploring near-optimal acceptance thresholds for around 12 weeks the RL agent decides that it’s more profitable to be stricter in the current environment, shifting the threshold up by 10 points.
</p>
<p align = "center">
  <img width = "800" alt = "Performance comparison: real environment actions" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/RL_results_action.png">
</p>

<p align = "justify">
  And if one looks at the profits received by the agent, one can see that they oscillate around the baseline during the initial exploration phase, but once the agent adapts to the new environment, they tend to be higher than the baseline profits leading to a significantly higher total profit in the end of the 24th week.
</p>
<p align = "center">
  <img width = "800" alt = "Performance comparison: real environment rewards" src = "https://github.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/blob/master/Pics/RL_results_reward.png">
</p>
