# Recommender-System

## Task 1 – small scale cosine similarity recommender system algorithm

Use the cosine similarity recommender system algorithm to train and predict ratings for a small (100K) set of items.

**Resources**

test_100K.csv
- testing dataset (20% from 100K)
- (Format: user_id (int), item_id (int), timestamp (int))

train_100K.csv
- training dataset (80% from 100K)
- (Format: user_id (int), item_id (int), rating (float), timestamp (int))

**Output format**

File Name: result.csv
Columns: user_id (int), item_id (int), rating_prediction (float), timestamp (int)
Note: output must be comma delimited without any whitespaces

## Task 2 – large scale matrix factorisation recommender system algorithm

Use the matrix factorization recommender system algorithm to train and predict ratings for a large (20M) set of items. You may need to use a database to handle the large data.
 
**Resources**

test_20M.csv
- testing dataset (20% from 20M)
- (Format: user_id (int), item_id (int), timestamp (int))

train_20M.csv
- training dataset (80% from 20M)
- (Format: user_id (int), item_id (int), rating (float), timestamp (int))

**Output format**

File Type: .csv
Columns: user_id (int), item_id (int), rating_prediction (float), timestamp (int)
