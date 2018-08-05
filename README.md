# Optimizing Acceptance Threshold in Credit Scoring using Reinforcement Learning
This is a repository for my master thesis submitted as a part of the Master of Science in Quantitative Economics degree at the University of Tartu. The research was conducted in 2017-2018 at Creditstar Group, Estonia under the supervision of Karl Märka, Head of Data Science at Creditstar Group, and Oliver Lukason, PhD at the University of Tartu.

The source code for the project can be found in the Source directory.

A complete description of the project to be added.

## Background
The problem environment considered is a trivial credit business process. It starts when the loan provider receives a loan application with the data about the application and the potential borrower. The data is then passed to a credit scoring model that outputs a credit score which reflects the risk level of the loan application. Next, if the score is too low, the lender rejects the loan application. In case the score is high enough, the lender issues the loan. Eventually, if the loan applicant doesn't repay or defaults on their loan, the lender loses the money. In case the loan applicant repays, the lender gains extra money (from interest and fees). All those cases affect the lender's profits.

![Credit business process](https://raw.githubusercontent.com/MykolaGerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning/master/Pics/Credit_business_process.png)
