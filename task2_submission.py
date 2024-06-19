import torch
from torch import nn
from torch.optim import Adam
import csv
import numpy as np
import sqlite3
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading data to database (training and testing)

def create_and_load_data(database_name, train_file, test_file):
    try:
        logging.info("Connecting to database...")
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()
        
        # create tables to store train and test data
        cursor.execute('''CREATE TABLE IF NOT EXISTS train_data
                            (user_id INTEGER, item_id INTEGER, rating REAL, timestamp INTEGER)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS test_data
                            (user_id INTEGER, item_id INTEGER, timestamp INTEGER)''')
        
        # read and insert train data into database
        with open(train_file, 'r') as train_csv:
            reader = csv.reader(train_csv)
            for row in reader:
                cursor.execute("INSERT INTO train_data VALUES (?,?,?,?)", row)
                
        # read and insert test data into database
        with open(test_file, 'r') as test_csv:
            reader = csv.reader(test_csv)
            for row in reader:
                cursor.execute("INSERT INTO test_data (user_id, item_id, timestamp) VALUES (?,?,?)", (row[0], row[1], row[2]))
                
        conn.commit()
        conn.close()
        logging.info("Data loading successful.")
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")


train_file = 'train_20M.csv'
test_file = 'test_20M.csv'
database_name = 'recommender_data.db'

# only run once for storing
# create_and_load_data(database_name, train_file, test_file)

# Retrieve data from sql

def load_train_data(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM train_data")
    train_data = cursor.fetchall()
    conn.close()
    return train_data

def load_test_data(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test_data")
    test_data = cursor.fetchall()
    conn.close()
    return test_data

def verify_data(train_data, test_data):
    print("Training data sample:")
    print(train_data[:5])  # Print first 5 rows
    print("\nTesting data sample:")
    print(test_data[:5])   # Print first 5 rows


# Definition of Model

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        self.user_factors.weight.data.uniform_(0.25, 0.5)
        self.item_factors.weight.data.uniform_(0.25, 0.5)
        self.linear = nn.Linear(num_factors, 1)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_factors(user_ids)
        item_embedding = self.item_factors(item_ids)
        
        # Print the shapes of user_embedding and item_embedding
        print("Shape of user_embedding:", user_embedding.shape)
        print("Shape of item_embedding:", item_embedding.shape)

        prediction = (user_embedding * item_embedding).sum(dim=1)
        prediction = self.linear(prediction)
        return prediction

# Model initialisation and Training

num_factors = 12
learning_rate = 0.01
num_epochs = 200

# Load data from SQLite database
logging.info("\nLoading training data and test data from SQL database")
database_name = 'recommender_data.db'
train_file = 'train_20M.csv'
test_file = 'test_20M.csv'
# Load data from SQLite database
train_data = load_train_data(database_name)
test_data = load_test_data(database_name)
# Verify the loaded data
verify_data(train_data, test_data)

user_ids = [row[0] for row in train_data]
item_ids = [row[1] for row in train_data]
ratings = torch.tensor([row[2] for row in train_data], dtype=torch.float32)

unique_user_ids = list(set(user_ids))
unique_item_ids = list(set(item_ids))
print("Number of unique users: ", len(unique_user_ids))
print("Number of unique items: ", len(unique_item_ids))
user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}

user_ids_tensor = torch.tensor([user_to_idx[user_id] for user_id in user_ids])
item_ids_tensor = torch.tensor([item_to_idx[item_id] for item_id in item_ids])
ratings_tensor = ratings

num_users = len(unique_user_ids)
num_items = len(unique_item_ids)

# Initialize the model
logging.info("\nInitialising model")
model = MatrixFactorization(num_users, num_items, num_factors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if device.type == 'cuda':
    mode.user_factors.to(device)
    model.item_factors.to(device)

# Define loss function and optimizer
loss_function = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=learning_rate)

def train_model(model, num_epochs, user_ids_tensor, item_ids_tensor, ratings_tensor, loss_function, optimizer, num_factors, batch_size=64):

    num_samples = len(user_ids_tensor)
    print(f"Number of epochs: {num_epochs}")
    print(f"Number of factors: {num_factors}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"L2 regularization strength: {optimizer.param_groups[0]['weight_decay']}\n")

    for epoch in range(num_epochs):
        total_loss = 0.0  
        optimizer.zero_grad()

        user_embedding = model.user_factors(user_ids_tensor)
        item_embedding = model.item_factors(item_ids_tensor)

        predictions = torch.sum(user_embedding * item_embedding, dim=1)

        # Ensure they have the same size
        assert predictions.size() == ratings_tensor.size(), "Size mismatch between prediction and target tensors"

        # Calculate loss only if sizes match
        loss = loss_function(predictions, ratings_tensor)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Print the average loss for the epoch
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss}\n')

    logging.info("\nTraining completed.")

logging.info("\nStart model training")
train_model(model, num_epochs, user_ids_tensor, item_ids_tensor, ratings_tensor, loss_function, optimizer, num_factors, 64)

# Testing and output

def predict_rating(model, user_id, item_id, user_to_idx, item_to_idx):
    # Check if user_id or item_id is not in the index
    if user_id not in user_to_idx or item_id not in item_to_idx:
        # Default rating
        return 0.5
    
    # Convert user and item IDs to tensors
    user_id_tensor = torch.tensor(user_to_idx[user_id], dtype=torch.long)
    item_id_tensor = torch.tensor(item_to_idx[item_id], dtype=torch.long)

    # Predict rating using the trained model
    with torch.no_grad():
        user_embedding = model.user_factors(user_id_tensor)
        item_embedding = model.item_factors(item_id_tensor)
        prediction = (user_embedding * item_embedding).sum().item()
    
    # Round prediction to the nearest 0.5
    prediction = round(prediction * 2) / 2

    # Clip the prediction to ensure it stays within range
    prediction = max(0.5, min(prediction, 5))

    return prediction

def save_model(model, num_users, num_items, num_factors, model_file):
    torch.save({
        'num_users': num_users,
        'num_items': num_items,
        'num_factors': num_factors,
        'model_state_dict': model.state_dict()
    }, model_file)
    print(f"Trained model saved to '{model_file}'")

def predict_ratings(model, test_data, user_to_idx, item_to_idx):
    predictions = []
    for user_id, item_id, _ in test_data:  # Ignoring the timestamp
        rating = predict_rating(model, user_id, item_id, user_to_idx, item_to_idx)
        predictions.append(rating)
    return predictions

def save_predictions_to_csv(predictions, test_data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'item_id', 'predicted_rating', 'timestamp'])
        for (user_id, item_id, timestamp), rating in zip(test_data, predictions):
            writer.writerow([user_id, item_id, rating, timestamp])

model_file = "matrix_factorisation_model.pth"
save_model(model, num_users, num_items, num_factors, model_file)

loaded_model_data = torch.load(model_file)
loaded_model = MatrixFactorization(loaded_model_data['num_users'], loaded_model_data['num_items'], loaded_model_data['num_factors'])
loaded_model.load_state_dict(loaded_model_data['model_state_dict'])

# Predict ratings for test data
predictions = predict_ratings(model, test_data, user_to_idx, item_to_idx)

# Save predictions to CSV
logging.info("\nSaving output to file")
output_file = 'predicted_ratings.csv'
save_predictions_to_csv(predictions, test_data, output_file)