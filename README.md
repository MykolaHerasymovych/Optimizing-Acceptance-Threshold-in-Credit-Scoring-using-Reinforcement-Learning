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
  <img width = "600" alt = "Credit business process" src = "https://raw.githubusercontent.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/master/Pics/Credit_business_process.png">
</p>
<p align = "justify">
  In order to make the final accept / reject decision about a loan application, its credit score is compared to an acceptance threshold or cutoff point. The latter thus regulates the acceptance rates of a credit company and resulting default rates. The problem investigated in the master thesis is the optimization of an acceptance threshold to maximize credit company's profits. The traditional approach to the problem is to simply optimize the cutoff based on the credit score distribution of an independent test dataset of loan applications. Knowing the outcomes of all the loans in the dataset each possible acceptance threshold is considered and the corresponding profits are calculated. The optimal cutoff point is the one that corresponds to the maximum profit. In case the train / test split of loan application dataset is random, the acceptance threshold optimized for the test dataset is going to be optimal for the train dataset.
</p>
<p align = "center">
  <img width = "430" alt = "Acceptance threshold optimization: traditional approach" src = "https://raw.githubusercontent.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/master/Pics/Acceptance_threshold_optimization_1.png"><img width = "430" alt = "Acceptance threshold optimization: traditional approach" src = "https://raw.githubusercontent.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/master/Pics/Acceptance_threshold_optimization_2.png">
</p>

## Problem
<p align = "justify">
  The main flaw in the traditional approach is that it is static and backward looking. It fully focuses on the test dataset which consists only of historical prefiltered accepted loan applications for which the outcome is known. The actual population of loan applications that the credit scoring model is applied to consists of completely new both potentially rejeted and accepted loan applications. The inconsistency of the loan applications sample used to train the model and optimize the acceptance threshold and the general population of loan applications the model and cutoff point are applied to leads to the selection bias and population drift issues. Those distort the optimal acceptance threshold making the traditional solution incorrect.
</p>
<p align = "center">
  <img width="430" alt="Selection bias issue" src="https://raw.githubusercontent.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/master/Pics/Selection_bias.png"><img width="430" alt="Population drift issue" src="https://raw.githubusercontent.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/master/Pics/Population_drift.png">
</p>


## Solution
<p align = "justify">
  We solve the problem by developing and applying a reinforcement learning (RL) agent: a dynamic forward-looking system that adapts to the live data feedback (incoming loan applications) and adjusts acceptance threshold to maximize credit company's profits. The reinforcement learning problem is described by the interaction between the credit business environment and the RL agent. The interaction frequency is 1 week. The state space of environment consists of a continuous acceptance rate (in the previous week) variable that spans from 0 to 1. The action space consists of the discrete credit score acceptance threshold (for the following week) variable  that spans from 5 to 100 by the step of 5 (20 discrete actions). The reward variable is the company's profits. A more detailed RL problem specification can be found in the work.
</p>
<p align = "center">
  <img width = "800" alt = "RL scheme" src = "https://raw.githubusercontent.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/master/Pics/RL_scheme_complex.png">
</p>
<p align = "justify">
  We solve the reinforcement learning problem with a simple Q-learning algorithm. Thus, the main goal is to approximate the Q-value function that is the mapping from state to expected discounted reward for each action. We preprocess state variable with Gaussian Radial Basis functions (RBFs) transformation to account for the non-linear nature of the value function. We use a set of 4 RBFs with 500 outputs each that gives us the output of 2000 preprocessed features. We then apply a separate Stochastic Gradient Descent (SGD) linear model for each action to approximate its Q-value, which results in 20 Q-values. Based on the predicted Q-values we use either greedy or Boltzmann-Q policy to choose an action in testing or training regime respectively. During the training only the SGD weights are learned.
</p>
<p align = "center">
  <img width = "600" alt = "Value function architecture" src = "https://raw.githubusercontent.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/master/Pics/Value_function_architecture.png">
</p>
