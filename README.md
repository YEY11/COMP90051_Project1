### COMP90051 Project1: Authorship Attribution with Limited Text on Twitter

This is the Project 1 for COMP90051 (Statistical Machine Learning) from the University of Melbourne.

#### 1. What is the task? 
Authorship attribution is a common task in Natural Language Processing (NLP) applications, such as academic plagiarism detection and potential terrorist suspects identification on social media. As for the traditional author classification task, the training dataset usually includes the entire corpus of the author’s published work, which contains a large number of examples of standard sentences that might reflect the writing style of the author. However, when it comes to the limited text on social media like Twitter, it brings some challenging problems, such as informal expressions, a huge number of labels, unbalanced dataset and extremely limited information related to identity.

<img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Result/kaggle.png" alt="Kaggle" width="70%">

In this project, the task is to predict authors of test tweets from among a very large number of authors found in training tweets, which comes from an in-class [Kaggle Competition](https://www.kaggle.com/c/whodunnit/leaderboard). Our works include data preprocessing, feature engineering, model selection and ensemble models etc. For more details, please check the [project specifications](https://github.com/Andy-TK/COMP90051_Project1/blob/master/Project%20specifications.pdf) and [project report](https://github.com/Andy-TK/COMP90051_Project1/blob/master/Project%20Report%20Team%2052.pdf).

#### 2. Data
The `Data` folder contains both original data and processed data.
#### 2.1. Original Data
`train_tweets.txt`
> _The original training dataset which contains 328932 tweets posted by 9297 users._

<img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Data/01_original_train.png" alt="original training data" width="70%">

