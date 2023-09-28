import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 10000
parent_status = np.random.randint(0, 4, num_samples)  # Status ranging from 0 to 3
child1_status = np.random.randint(0, 4, num_samples)
child2_status = np.random.randint(0, 4, num_samples)
child3_status = np.random.randint(0, 4, num_samples)
child4_status = np.random.randint(0, 4, num_samples)
child5_status = np.random.randint(0, 4, num_samples)

# Generate random creation dates for each sample within the past year
earliest_date = datetime.now() - timedelta(days=365)
creation_dates = [(earliest_date + timedelta(days=np.random.randint(0, 365))).date() for _ in range(num_samples)]

# Compute the number of days since the creation date for each sample
days_since = [(datetime.now().date() - date).days for date in creation_dates]

# Define the weights associated with each status
status_weights = {
    0: 0.15,  # Unknown
    1: 0.4,   # Error
    2: 0.25,  # Warning
    3: 0.1    # Good
}

# Convert the statuses to their respective weights
parent_status_weighted = np.vectorize(status_weights.get)(parent_status)
child1_status_weighted = np.vectorize(status_weights.get)(child1_status)
child2_status_weighted = np.vectorize(status_weights.get)(child2_status)
child3_status_weighted = np.vectorize(status_weights.get)(child3_status)
child4_status_weighted = np.vectorize(status_weights.get)(child4_status)
child5_status_weighted = np.vectorize(status_weights.get)(child5_status)

# Define the weights for the features
feature_weights = {
    'days_since': 0.2,
    'parent_status': 0.3,
    'child_status': 0.1
}

# Normalize days_since to a range of 0 to 1
MAX_DAYS = 365
normalized_days_since = np.array(days_since) / MAX_DAYS

# Compute the weighted features
weighted_days_since = normalized_days_since * feature_weights['days_since']
weighted_parent_status = parent_status_weighted * feature_weights['parent_status']
weighted_child1_status = child1_status_weighted * feature_weights['child_status']
weighted_child2_status = child2_status_weighted * feature_weights['child_status']
weighted_child3_status = child3_status_weighted * feature_weights['child_status']
weighted_child4_status = child4_status_weighted * feature_weights['child_status']
weighted_child5_status = child5_status_weighted * feature_weights['child_status']

# Calculate the combined score for each sample and normalize to a range of 0 to 1
combined_scores = weighted_parent_status + weighted_child1_status + weighted_child2_status + weighted_child3_status + weighted_child4_status + weighted_child5_status
attention_scores = (combined_scores - combined_scores.min()) / (combined_scores.max() - combined_scores.min())

# Combine the weighted features into a single dataset
data = np.column_stack((weighted_days_since, weighted_parent_status, weighted_child1_status, weighted_child2_status, weighted_child3_status, weighted_child4_status, weighted_child5_status))

# Split the dataset into training and testing sets
train_data, test_data = data[:8000], data[8000:]
train_labels, test_labels = attention_scores[:8000], attention_scores[8000:]

# Define and compile the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(train_data, train_labels, epochs=50, verbose=0)

# Save the trained model weights
checkpoint_path = "model_checkpoints/cp.ckpt"
model.save_weights(checkpoint_path)

# Load the model weights and make predictions on the test set
model.load_weights(checkpoint_path)
predictions = model.predict(test_data)

# Randomly select some samples and display their details
random_idx = np.random.randint(0, len(test_data))
predicted_score = predictions[random_idx][0]
actual_score = test_labels[random_idx]

# Define the number of samples to display
num_samples_to_display = 20

# Randomly select indices of samples to display
random_indices = np.random.choice(len(test_data), size=num_samples_to_display, replace=False)
predicted_scores = [predictions[i][0] for i in random_indices]

# Extract and display relevant data for the randomly selected samples
data_to_display = {
    'Creation Date': [creation_dates[i + 800] for i in random_indices],
    'Days Since Creation': [days_since[i + 800] for i in random_indices],
    'Parent Status': [parent_status[i + 800] for i in random_indices],
    'Child1 Status': [child1_status[i + 800] for i in random_indices],
    'Child2 Status': [child2_status[i + 800] for i in random_indices],
    'Child3 Status': [child3_status[i + 800] for i in random_indices],
    'Child4 Status': [child4_status[i + 800] for i in random_indices],
    'Child5 Status': [child5_status[i + 800] for i in random_indices],
    'Weighted Parent Status': [test_data[i][1] for i in random_indices],
    'Weighted Child1 Status': [test_data[i][2] for i in random_indices],
    'Weighted Child2 Status': [test_data[i][3] for i in random_indices],
    'Weighted Child3 Status': [test_data[i][4] for i in random_indices],
    'Weighted Child4 Status': [test_data[i][5] for i in random_indices],
    'Weighted Child5 Status': [test_data[i][6] for i in random_indices],
    'Weighted Days Since Creation': [test_data[i][0] for i in random_indices],
    'Priority Score (Attention Score)': [test_labels[i] for i in random_indices],
    'Predicted Priority Score': predicted_scores
}

# Convert the extracted data to a pandas DataFrame and sort by Priority Score
df = pd.DataFrame(data_to_display)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sorted_df = df.sort_values(by="Priority Score (Attention Score)", ascending=False)

ordered_columns = ['Creation Date', 'Days Since Creation', 'Parent Status', 'Child1 Status', 'Child2 Status',
                   'Child3 Status', 'Child4 Status', 'Child5 Status', 'Weighted Days Since Creation',
                   'Weighted Parent Status', 'Weighted Child1 Status', 'Weighted Child2 Status',
                   'Weighted Child3 Status', 'Weighted Child4 Status', 'Weighted Child5 Status',
                   'Priority Score (Attention Score)', 'Predicted Priority Score']
df = df[ordered_columns]
sorted_df = df.sort_values(by="Priority Score (Attention Score)", ascending=False)

# Calculate and display evaluation metrics
true_scores_variance = np.var(test_labels)
predicted_scores_variance = np.var(predictions)
mse, mae = model.evaluate(test_data, test_labels, verbose=0)
rmse = np.sqrt(mse)
r2 = r2_score(test_labels, predictions)
print(f"R-squared (R2): {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Variance of True Attention Scores: {true_scores_variance}")
print(f"Variance of Predicted Attention Scores: {predicted_scores_variance}")
print("\n")
print(sorted_df)

