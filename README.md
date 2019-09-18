### COMP90051 Project1: Authorship Attribution with Limited Text on Twitter

This is the Project 1 for COMP90051 (Statistical Machine Learning) from the University of Melbourne.

#### 1. What is the task? 
Authorship attribution is a common task in Natural Language Processing (NLP) applications, such as academic plagiarism detection and potential terrorist suspects identification on social media. As for the traditional author classification task, the training dataset usually includes the entire corpus of the author’s published work, which contains a large number of examples of standard sentences that might reflect the writing style of the author. However, when it comes to the limited text on social media like Twitter, it brings some challenging problems, such as informal expressions, a huge number of labels, unbalanced dataset and extremely limited information related to identity.

<img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Result/kaggle.png" alt="Kaggle" width="70%">

The task here is to predict authors of test tweets from among a very large number of authors found in training tweets. In this project, we attempt to find an appropriate solution for such a massively multiclass classification task from a [Kaggle Competition](https://www.kaggle.com/c/whodunnit/leaderboard) based on various machine learning techniques.

#### 2. 