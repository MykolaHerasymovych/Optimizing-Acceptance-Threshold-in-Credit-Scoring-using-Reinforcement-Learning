# Optimizing Acceptance Threshold in Credit Scoring using Reinforcement Learning
<p align = "justify">
  This is a repository for a research project conducted in 2017-2018 at Creditstar Group, Estonia under the supervision of Karl Märka, Head of Data Science at Creditstar Group, and Oliver Lukason, PhD at the University of Tartu.

The research article can be found [here](../master/Article.pdf).

The source code for the project can be found in the [Source](../master/Source) directory.

WARNING: the project has many dependencies on outdated packages. So I wouldn't try to revive it but instead, use and adapt updated external packages used there, such as https://www.gymlibrary.ml/, to the developed theoretical concept using the source code as an example.

</p>

## Table of Contents
- **[Abstract](#abstract)**<br>
- **[Background](#background)**<br>
- **[Problem](#problem)**<br>
- **[Solution](#solution)**<br>
- **[Results](#results)**<br>
  - **[Baseline results](#baseline-results)**<br>
  - **[Reinforcement learning results](#reinforcement-learning-results)**<br>
  - **[Performance comparison: simulated environments](#performance-comparison-simulated-environments)**<br>
  - **[Performance comparison: real environment](#performance-comparison-real-environment)**<br>
- **[Summary](#summary)**<br>

## Abstract
<p align = "justify">
  The paper aims to study, whether using reinforcement learning to optimize acceptance threshold in credit scoring leads to higher profits of the lender compared to using a traditional optimization approach. We show that traditional static methods based on cost sensitive optimization do not ensure the optimality of the acceptance threshold, which might lead to biased conclusions and significant losses to the firm. We develop a dynamic reinforcement learning system that constantly adapts the threshold in response to the live data feedback, maximizing company’s profits. The developed algorithm is shown to outperform the traditional approach in terms of profits both in various simulated scenarios and on the real data of an international consumer credit company
</p>
  
## Background
<p align = "justify">
  The problem environment considered is a trivial credit business process. It starts when the loan provider receives a loan application with the data about the application and the potential borrower. The data is then passed to a credit scoring model that outputs a credit score which reflects the risk level of the loan application. Next, if the score is too low, the lender rejects the loan application. In case the score is high enough, the lender issues the loan. Eventually, if the loan applicant doesn't repay or defaults on their loan, the lender loses the money. In case the loan applicant repays, the lender gains extra money (from interest and fees). In the end, any of those cases affect the lender's profits.
</p>
<p align = "center">
  <img width = "600" alt = "Credit business process" src = "../master/Pics/Credit_business_process.png">
</p>
<p align = "center">
  <b>Figure 1.</b> Credit business process illustration based on the Credistar Group practice.
</p>
<p align = "justify">
  In order to make the final accept / reject decision about a loan application, its credit score is compared to an acceptance threshold or cutoff point. The latter thus regulates the acceptance rates of a credit company and resulting default rates. The problem investigated is the optimization of an acceptance threshold to maximize credit company's profits. The traditional approach to the problem is to simply optimize the cutoff based on the credit score distribution of an independent test dataset of loan applications. Knowing the outcomes of all the loans in the dataset each possible acceptance threshold is considered and the corresponding profits are calculated. The optimal cutoff point is the one that corresponds to the maximum profit. In case the train / test split of loan application dataset is random, the acceptance threshold optimized for the test dataset is going to be optimal for the train dataset.
</p>
<p align = "center">
  <img width = "430" alt = "Acceptance threshold optimization: traditional approach" src = "../master/Pics/Acceptance_threshold_optimization_1.png"><img width = "430" alt = "Acceptance threshold optimization: traditional approach" src = "../master/Pics/Acceptance_threshold_optimization_2.png">
</p>
<p align = "center">
  <b>Figure 2.</b> Traditional cost-sensitive optimization illustration.
</p>

## Problem
<p align = "justify">
  The main flaw in the traditional approach is that it is static and backward looking. It fully focuses on the test dataset which consists only of historical prefiltered accepted loan applications for which the outcome is known. The actual population of loan applications that the credit scoring model is applied to consists of completely new both potentially rejeted and accepted loan applications. The inconsistency of the loan applications sample used to train the model and optimize the acceptance threshold and the general population of loan applications the model and cutoff point are applied to leads to the selection bias and population drift issues. Those distort the optimal acceptance threshold making the traditional solution incorrect.
</p>
<p align = "center">
  <img width="430" alt="Selection bias issue" src="../master/Pics/Selection_bias.png"><img width="430" alt="Population drift issue" src="../master/Pics/Population_drift.png">
</p>
<p align = "center">
  <b>Figure 3.</b> Issues of the traditional approach: selection bias (left) and population drift (right).
</p>

## Solution
<p align = "justify">
  We solve the problem by developing and applying a reinforcement learning (RL) agent: a dynamic forward-looking system that adapts to the live data feedback (incoming loan applications) and adjusts acceptance threshold to maximize credit company's profits. The reinforcement learning problem is described by the interaction between the credit business environment and the RL agent. The interaction frequency is 1 week. The state space of environment consists of a continuous acceptance rate (in the previous week) variable that spans from 0 to 1. The action space consists of the discrete credit score acceptance threshold (for the following week) variable  that spans from 5 to 100 by the step of 5 (20 discrete actions). The reward variable is the company's profits. A more detailed RL problem specification can be found in the work.
</p>
<p align = "center">
  <img width = "800" alt = "RL scheme" src = "../master/Pics/RL_scheme_complex.png">
</p>
<p align = "center">
  <b>Figure 3.</b> The scheme of reinforcement learning algorithm for a credit scoring acceptance threshold optimization in a consumer credit company.
</p>
<p align = "justify">
  <b>Notes:</b> The figure shows the scheme of the interaction loop between the environment (the simulation) and reinforcement learning agent. The outer loop reflects state – action – reward exchange between the two. The inner loop shows the state – action evaluation and learning from reward by the RL agent. <i>S</i> – state or weekly acceptance rate, <i>A</i> – action or acceptance threshold, <i>Q</i> – action state value prediction, <i>R</i> – reward or profits, <i>α</i> – learning rate, <i>γ</i> – discount parameter, <i>S'</i> - following state.
</p>
<p align = "justify">
  We solve the reinforcement learning problem with a simple Q-learning algorithm. Thus, the main goal is to approximate the Q-value function that is the mapping from state to expected discounted reward for each action. We preprocess state variable with Gaussian Radial Basis functions (RBFs) transformation to account for the non-linear nature of the value function. We use a set of 4 RBFs with 500 outputs each that gives us the output of 2000 preprocessed features. We then apply a separate Stochastic Gradient Descent (SGD) linear model for each action to approximate its Q-value, which results in 20 Q-values. Based on the predicted Q-values we use either greedy or Boltzmann-Q policy to choose an action in testing or training regime respectively. During the training only the SGD weights are learned.
</p>
<p align = "center">
  <img width = "600" alt = "Value function architecture" src = "../master/Pics/Value_function_architecture.png">
</p>
<p align = "center">
  <b>Figure 4.</b> Value function model and policy architecture of the reinforcement learning agent.
</p>
<p align = "justify">
  The RL agent was trained using a Monte Carlo simulation of the credit business process. For training 100 simulation episodes were used. Each simulation episode cosists of 114 simulated weeks, 52 of which are warm-up with no interaction, 60 are interactive when the RL agent observes the state, chooses an action and learns the value function from the reward it gets later and 22 are weeks of delayed learning when the agent doesn't interact with the environment but observes delayed rewards and learns the value function. On average one training simulation episode takes around 5 minutes.
</p>
<p align = "center">
  <img width = "600" alt = "Training simulated episode structure" src = "../master/Pics/Training_simulation_episode_structure.png">
</p>
<p align = "center">
  <b>Figure 5.</b> Training simulated episode structure of the reinforcement learning system.
</p>

## Results
### Baseline results
<p align = "justify">
  First, we apply the traditional approach to the cutoff point optimization based on the test dataset of loan applications for the credit scoring model. We compute the potential profit for the same range of acceptance threshold values as the action space of the reinforcement learning agent: from 5 to 100 by step of 5. One can notice that the lower the acceptance threshold, the more loan applications get accepted, the more issued loans default, the bigger the final loss. On the other hand, the higher the acceptance threshold, the less loan applications get accepted, the closer to zero the profit gets. The optimal acceptance threshold is found by maximizing profit and is equal to a credit score of 65.
</p>
<p align = "center">
  <img width = "600" alt = "Baseline results" src = "../master/Pics/Baseline_results.png">
</p>
<p align = "center">
  <b>Figure 6.</b> Baseline threshold optimization results: potential profit and corresponding acceptance rate for every acceptance threshold based on the test dataset.
</p>
<p align = "justify">
  <b>Notes:</b> Profit is measured in thousands of euros.
</p>

### Reinforcement learning results
<p align = "justify">
  Next, by performing the simulation-based training of the reinforcement learning agent as described above we approximate the Q-value function. The shape of the latter is very similar to the potential profit curve computed with the traditional approach: low acceptance thresholds have the lowest value and high acceptance thresholds have higher but still suboptimal value. The optimum is found by maximizing the value and corresponds to the same credit score of 65. Thus, the reinforcement learning approach can do as well as the traditional one in a static environment.
</p>
<p align = "center">
  <img width = "600" alt = "Value function shape" src = "../master/Pics/Value_function_shape.png">
</p>
<p align = "center">
  <b>Figure 7.</b> Value function model shape .
</p>
<p align = "justify">
  <b>Notes:</b> State denotes the application acceptance rate during the previous week, action denotes the acceptance threshold for the following week, value is the prediction of the value function model for a particular state action pair, optimum shows the state action pair that corresponds to the highest value in the state action space.
</p>

### Performance comparison: simulated environments
<p align = "justify">
  To test the performance of the proposed approach in dynamic conditions we adjust the simulation parameters to mimic certain changes in the credit business environment. We simulate downwards and upwards shifts in the credit score distribution mimicing the population drift and downwards and upwards shift in default rates mimicing the selection bias issue. Based on 100 simulation runs for each scenario, the reinforcement learning algorithm manages to adapt to the new environments very quickly significantly outperforming the traditional approach in terms of profits according to the one-tailed t-test.
</p>
<p align = "center">
  <img width = "600" alt = "Performance comperison: simulated environments" src = "../master/Pics/RL_results_simulation.png">
</p>
<p align = "center">
  <b>Figure 8.</b> Performance comperison: simulated environments.
</p>
<p align = "justify">
  <b>Notes:</b> the figure shows the difference between reward variables and the baseline reward during interaction phase. Baseline denotes the profits received with the acceptance threshold optimized using the traditional approach. The upper figures for each scenario show cumulative profits for each out of 100 simulated episodes and their mean. For easier perception rewards are calculated for the week the corresponding loans were issued in (for instance, the reward for week 0 shows profits generated by loan applications issued in week 0). The lower figures show the distribution of episode profits for each out of 100 simulated episodes and their mean. Profit is measured in thousands of euros.
</p>
<p align = "center">
  <b>Table 1.</b> Results of the t-test for various distortion scenarios.
</p>

<p align = "center">
  
  |Scenario | t-statistic | p-value|
  |:--------|-------------|--------|
  |Scenario 1: downwards shift in score distribution	| 29.56631	| 1.55E-51 |
  |Scenario 2: upwards shift in score distribution	| 42.72066 |	2.45E-66 |
  |Scenario 3: downwards shift in default rates	| 5.172688	| 5.95E-07 |
  |Scenario 4: upwards shift in default rates	| 4.600158	| 6.20E-06 |

</p>
<p align = "justify">
  <b>Notes:</b> the t-test null hypothesis is that the mean difference between the episode reward received by the RL agent and the episode reward received using the traditional approach throughout 100 episodes is equal to or lower than zero.
</p>

### Performance comparison: real environment
<p align = "justify">
  Finally, we compare performance of the developed reinforcement learning system to the baseline approach on the more recent 24 weeks of real loan applications data. On the figure one can see the difference between the baseline acceptance threshold and the one chosen with the proposed algorithm. After exploring near-optimal acceptance thresholds for around 12 weeks the RL agent decides that it’s more profitable to be stricter in the current environment, shifting the threshold up by 10 points.
</p>
<p align = "center">
  <img width = "800" alt = "Performance comparison: real environment actions" src = "../master/Pics/RL_results_action.png">
</p>
<p align = "center">
  <b>Figure 9.</b> The dynamics of the action difference between the actual action taken and the baseline action, value function optimized action and the baseline action during the real episode.
</p>
<p align = "justify">
  <b>Notes:</b> the figure shows the difference between action variables and the baseline action during interaction and delayed learning phases. Baseline denotes the acceptance threshold optimized using the traditional approach, actual denotes the one used by the RL agent, value function optimal denotes the one optimal according to the value function model.
</p>

<p align = "justify">
  And if one looks at the profits received by the agent, one can see that they oscillate around the baseline during the initial exploration phase, but once the agent adapts to the new environment, they tend to be higher than the baseline profits leading to a significantly higher total profit in the end of the 24th week.
</p>
<p align = "center">
  <img width = "800" alt = "Performance comparison: real environment rewards" src = "../master/Pics/RL_results_reward.png">
</p>
<p align = "center">
  <b>Figure 10.</b> The dynamics of the (cumulative) reward difference between the actual reward received and the baseline reward during the real episode.
</p>
<p align = "justify">
  <b>Notes:</b> the figure shows the difference between reward variables and the baseline reward during interaction phase. Baseline denotes the profits received with the acceptance threshold optimized using the traditional approach, actual and cumulative denote profits received by the RL agent. For easier perception rewards are calculated for the week the corresponding loans were issued in (for instance, the reward for week 0 shows profits generated by loan applications issued in week 0). Profit is measured in thousands of euros.
</p>

## Summary
<p align = "justify">
  The results show that the traditional cutoff optimization approach does not ensure the optimality of the acceptance threshold, which might lead to biased conclusions and significant losses. The proposed dynamic reinforcement learning system manages to outperform the traditional method both in simulated and real credit business environments leading to significantly higher total profits of the credit company. The main advantages of the developed approach are: 
</p>

  1. <p align = "justify">its constant adaptation to and learning from actual data generating process, which removes the need for theoretical simplifications and keeps the algorithm up to date;</p> 
  2. <p align = "justify">flexible objective function definition that makes it easy to accurately specify the decision maker’s preferences and adjust them on the go if needed;</p>
  3. <p align = "justify">ability to train and test it in a simulated environment that lets the company avoid costly poor initial performance and stress test various scenarios.</p>

<p align = "justify">
  Overall, the developed algorithm can be immediately put into practice to accompany lender's decisions and is currently used by the company as a decision support system.
</p>
