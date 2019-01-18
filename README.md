# Apache-Spark-Pet-Owners-Prediction-on-YouTube-Comments
Apache Spark Pet Owners Prediction on YouTube Comments

## Prerequisites
Before we start, these packages are required for our analysis:

```Python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql.types import StructField, StringType, IntegerType, StructType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql.functions import col, size, udf
from pyspark.ml.feature import Word2Vec
from pyspark.sql import functions as f
from statsmodels.stats.proportion import proportions_ztest
from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

```

## Situation: 
The given data is from YouTube comments that contains creatorID, userID, and Comments.


## Task: 
The goal is trying to identify whether the user is the owner of dogs or cat through the comments. The biggest challenge here is that I had to create the label through the comments by myself so that it would fit into the classification problem.


## Action:
#### 1. Data Exploration and Cleaning
1. Use Spark DataFrame to load the data
2. Check for missing values and I found that missing values only occupy a small portion. Second, it is hard to impute the missing value of ID numbers or comments in this dataset so I would drop any NULL value in the future
3. Count the number of creators and users for a brief understanding of the data

#### 2. Data preprocessing building classifier
1. Create a label(Y). Find users who own dogs or cats. In my perspective, people who have pets are more likely to say something involving these three elements: title + the number of pets + the type of their pet. For example, (I/My/Our) + (a/2/3/one) + (dog/cat). Hence, I assigned comments with this combination as dog or cats owners. I dealt with missing values later because the labeling step does not affect by the missing values and the code will be clearer if I put into further steps.
2. Data preprocessing: Now, I removed the missing values tokenized the comments, and remove stop words so that I could fit the word lists into Word2Vec 
3. Word2Vec transformation: I transformed the word lists to vectors with setVectorSize of 256. The number 256 is because just by other professional's experience and number with the power of 2 is easy for the algorithm to compute.

#### 3. Get insights of Data
Still, we need to get some insights into the data so that we can make strategies from these insights with the assistant with the predictive model.
1. See if people who mention a dog or cat are more likely to have them
I did this analysis that if this statement holds, then people who talk about pets and yet to have pets are the ones to own pets as well in them, and we can target them as potential audience.
2. How many comments do pets owners leave in general
I wanted to know if there is a chance for me to segment users with different activeness
3. Top creators that have most viewers that are pet owners
I wanted to find out famous channels so that Youtube can set Ads according to the audience

#### 4. Identify Cat And Dog Owners through the Comments - Build the classifier 
1. Down-sampling: Because I was facing imbalance with rate 1 to 100 so I downsampled it to 50 percent: 50 percent
2. I implement LogisticRegression, GBTClassifier, and RandomForestClassifier without tuning the hyperparameters to see which model has the best performance
3. Use CrossValidator with Grid Search Concept for Hyper-Parameter Tuning
  
  
## Result
I had 88% of accuracy the 89% recall on the Logistic Regression. Here I emphasized more on the recall since the goal is to identify the pets owners as possible. After that, I could make sure that the three points I made on the insights could be made into practice.
