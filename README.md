# BankMarketing
Predicting the outcome of bank marketing calls

## Data Source

I used the UCI bank marketing database, which logs 45,000 bank calls, and sees whether the bank was successful selling the customer a term deposit product. I downloaded this dataset from Kaggle. 

## Data Cleaning 
### See: 'Bank Project EDA.ipynb' 

I looked at each of the columns individually, and tried to convert the categoricals into numerical columns in a sensible way. There is column specific information on how I did this in the associated notebook. Of the numerical columns, I found that many of the columns containing financial data were highly correlated, so I dropped two of them. I dropped the 'duration' column, which was very good at predicting the target variable, but only something you find out after the fact. Dropping it is also suggested by the creator of the original data set. 

I found that the youngest and oldest people tended to say 'yes' most often, so I created a feature for (age-40)^2. I also found that young students, and old retirees were most likely to say yes, while entrepreneurs, blue collar workers, and people working in the service industry were most likely to say no, so I created features to reflect this. People were least likely to say 'yes' on Monday or Friday, so I created a feature to reflect this as well. A few columns had missing values, so I had to drop some of the rows, leaving me with around 90% of the original data. I then exported this data as a 'cleaned_bank.csv' for further analysis. 

## Data Analysis
### See: 'Bank Project Analysis.ipynb' 

### Step 1: Train-test split, Class Imbalance, Scaling

I used a 70-30 train-test split, with the remaining 30 being split in two in the future for model validation and model testing. I then applied a min-max scaler, fit to my training data, on both my test and training data. Finally, I modified my training data to account for the class imbalance (only 11% positive), by randomly removing some of the negative results. 

### Step 2: Models

For each model, I grid searched to find the optimal hyper-parameters for the training set. After an initial search, I tightened my parameter ranges, to tune the model more precisely. Note that the parameters in my notebook do not show every parameter that I have tried. 

#### Logistic Regression

I tried to run both a lasso and ridge logistic regression. Both had similar results, with f1 scores around 0.71. Before running the ridge regression, I further scaled down the features which were highly correlated with other features, as discussed here: https://towardsdatascience.com/background-d5f101e00afc

#### KNN

I ran the K-nearest neighbours model, with features boosted by the square root of their ridge regression weight, and penalised by the correlation with other features, as discussed earlier. This boosting increased my f1 score on my training set by around 4%. I could not use the weight metric 'distance', as it would give me seemingly erroneously high f1 scores (0.975). This boosted KNN was among the best models in my training set, with a f1 score of 0.74. 

#### Trees

Decision trees gave me an f1 score of 0.72, random forests gave me  0.7, XG boost gave me 0.74, and Ada boost gave me 0.71. 

#### SVM

I tried support vector machines with kernels rbf, poly, and sigmoid, and got f1 scores of 0.72, 0.72, and 0.7 respectively.

### Conclusions from training

All models seemed to perform similarly well, giving me f1 scores in the range of 0.7-0.75. 

### Step 3: Validation

On my validation set, my model f1 scores dropped sharply, with most models giving me an f1 score around 0.45. This is to be expected, as the validation set has imbalanced classes, whereas the training set did not. A noteable exception was KNN, which gave me a validation f1 score of only 0.38. It is possible that using the parameters from ridge regression to boost certain features led to over fitting. 

I also attempted to use ensemble models to improve my score. I tried using 7 models, 5 models, and 3 models and taking their majority vote to predict y. 7 models did slightly better than 5 and 3, but did similarly well to my best single model in validation, ada boost. Finally, I corrected for the class imbalance in the validation set, and ran logistic regression and ada boost on my model predictions, to try and better predict my target. This gave me an f1 score of 0.71, but this is at the lower end of the single models, when they were tested on a balanced dataset. Before I corrected for the class imbalance, both models were predicting the majority class 100% of the time, giving me an f1 score of 0. 

### Step 4: Testing 

Since the 7 vote and ada boost seemed to do the best, and the 7 vote was likely more resilient to over fitting, I decided to use the 7 vote on my testing set. I also ran all the models individually, just to see the result, and once again, 7 vote and ada boost were very similar to each other, and also the best performing models. I had a testing f1 score of 0.45 using the votes from 7 models. I feel this is not bad given how arbitrary people buying a product from a phone call can be. 

Looking at the confusion matrix below, you can see that the model predicts 'True', i.e. will buy a product, 20% of the time, and only 7/20 actually end up buying the product, a 35% success rate. While that seems low, we have kept in around 2/3rds of the successes (7/11), while selecting only 20% of the customers. As the bank normally has an 11% success rate on their marketing calls, if they optimised using this model, their success rate would more than triple!

Note: P True and P False are predictions of the model, True/False are ground truths

![alt text](https://github.com/molron94/BankMarketing/blob/master/Confusion%20Matrix.png)




