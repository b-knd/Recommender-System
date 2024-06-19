import math
import sys
import os
import csv
import numpy as np

# function to precompute similarity scores between items and store result in matrix
def precompute_similarity_scores(user_item_mat, user_avg_rating_mat):
    num_items = user_item_mat.shape[1]
    similarity_matrix = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(i+1, num_items):
            similarity_matrix[i, j] = cosine_sim(user_item_mat, i, j, user_avg_rating_mat)
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix

# function for cosine similarity calculation
def cosine_sim(user_item_mat, item_index1, item_index2, user_avg_rating_mat):
    # Get ratings of items
    ratings_item1 = user_item_mat[:, item_index1]
    ratings_item2 = user_item_mat[:, item_index2]
        
    # Indices of users who rated both items
    common_users_idx = (ratings_item1 != 0) & (ratings_item2 != 0)
    
    # Ratings of common users for both items
    ratings_item1_common = ratings_item1[common_users_idx]
    ratings_item2_common = ratings_item2[common_users_idx]
    
    # Average ratings of common users
    avg_ratings_common = user_avg_rating_mat[common_users_idx]
    
    # Check if there are common users
    if np.sum(common_users_idx) == 0:
        return 0.5  # Return NaN if there are no common users
    
    # Simplified  cosine similarity calculation
    diff_item1 = ratings_item1_common - avg_ratings_common
    diff_item2 = ratings_item2_common - avg_ratings_common

    dot_product = np.sum(diff_item1 * diff_item2)
    magnitude_item1 = np.linalg.norm(diff_item1)
    magnitude_item2 = np.linalg.norm(diff_item2)
    
    # Handle division by zero
    if magnitude_item1 == 0 or magnitude_item2 == 0:
        return np.mean(avg_ratings_common)  # Using average rating as similarity
    
    similarity = dot_product / (magnitude_item1 * magnitude_item2)
    
    return similarity

# function to round ratings for prediction depends of nearest value
def round_rating(predicted_rating):
    valid_ratings = [0.5 * i for i in range(11)]
    rounded_rating = min(valid_ratings, key=lambda x: abs(x-predicted_rating))
    return rounded_rating

# function to predict rating for an item by a user
def predict_rating(user_item_mat, user_index, item_index, similarity_matrix):
    # Get the similarity scores between the target item and all other items
    item_similarities = similarity_matrix[item_index, :]
    
    # Get all the ratings of the target user for all items
    user_ratings = user_item_mat[user_index, :]
    
    # Find items rated by the target user
    rated_items_indices = np.nonzero(user_ratings)[0]
    
    # Initialize variables for weighted sum and sum of weights
    weighted_sum = 0
    sum_of_weights = 0
    
    # Iterate over rated items
    for i in rated_items_indices:
        # Get similarity score between the target item and the rated item
        similarity_score = item_similarities[i]
        
        # If similarity score is non-negative
        if similarity_score >= 0.1:
            # Get the rating of the target user for the rated item
            rating = user_ratings[i]
            
            # Update weighted sum and sum of weights
            weighted_sum += similarity_score * rating
            sum_of_weights += similarity_score
    
    # If sum of weights is zero, return default rating
    if sum_of_weights == 0:
        return 0.5
    else:
        # Compute the predicted rating
        predicted_rating = weighted_sum / sum_of_weights
        return predicted_rating

if __name__ == "__main__":
    
    # LOADING DATASET
    print("[1/4] Loading dataset...")

    entries = []

    #read csv file to load entries
    with open('train_100K.csv', 'r') as file:
        for line in file:
            columns = line.strip().split(',')

            #extract information
            userid = columns[0]
            itemid = columns[1]
            rating = float(columns[2])
            timestamp = columns[3]

            entries.append([userid, itemid, rating, timestamp])

    # convert entries to numpy array
    entries_arr = np.array(entries)

    
    # COMPUTING USER-ITEM MATRIX
    print("[2/4] Extracting information from dataset, computing matrices and similarity scores...") 

    # Populate a user_item matrix to ease future calculation
    user_item_mat = {}

    user_id_list = np.unique(entries_arr[:, 0]) 
    item_id_list = np.unique(entries_arr[:, 1])

    # To keep track of indexing in the matrices
    user_idx = {user_id: i for i, user_id in enumerate(user_id_list)}
    item_idx = {item_id: i for i, item_id in enumerate(item_id_list)}

    num_users = len(user_id_list)
    num_items = len(item_id_list)
    user_item_mat = np.zeros((num_users, num_items))

    for entry in entries:
        user_id = entry[0]
        item_id = entry[1]
        rating = entry[2]

        user_index = user_idx[user_id]
        item_index = item_idx[item_id]

        user_item_mat[user_index, item_index] = rating


    # COMPUTING USERS' AVERAGE RATINGS
    # initialize a matrix to store average ratings of users
    user_avg_rating_mat = np.zeros((num_users,))

    # Compute the average rating for each user
    for i in range(num_users):
        user_ratings = user_item_mat[i, :]

        # Exclude zero ratings (unrated items)
        user_ratings = user_ratings[user_ratings != 0]

        # Compute average rating
        if len(user_ratings) > 0:
            user_avg_rating = np.mean(user_ratings)
        else:
            user_avg_rating = 0  # Handle case when user has not rated any item

        # Store the average rating in the matrix
        user_avg_rating_mat[i] = user_avg_rating
    

    
    # PREDICT AND OUTPUT
    print("[3/4] Prediction in progress, outputing to file...")

    test_entries = []
    count = 0
    similarity_matrix = precompute_similarity_scores(user_item_mat, user_avg_rating_mat)
    with open('test_100K.csv', 'r') as file:
        for line in file:
            columns = line.strip().split(',')

            #extract information
            userid = columns[0]
            itemid = columns[1]
            timestamp = columns[2]

            if itemid in item_idx:
                item_index = item_idx[itemid]
                user_index = user_idx[userid]
                predicted_rating = round_rating(predict_rating(user_item_mat, user_index, item_index, similarity_matrix))
                test_entries.append([userid, itemid, predicted_rating, timestamp])
            else:
                count += 1
                test_entries.append([userid, itemid, 0.5, timestamp])
    

    output_file = 'predicted_ratings_rounded_mean.csv'

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for entry in test_entries:
            writer.writerow(entry)
            
    print("[4/4] Output file written. Complete.")
