# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor # You can change this to RandomForestRegressor or another model
import pickle
import os

print("Starting model training script...")

# --- Configuration ---
# IMPORTANT: Replace 'your_dataset.csv' with the actual name of your training data file.
# This file must be in the same directory as this script, or you must provide its full path.
DATASET_FILE = 'traffic volume.csv'

# Define the exact order of features that your app.py expects and that your model
# was originally designed to work with. This is crucial for consistency.
# This list must match the 'original_feature_names' in your app.py
EXPECTED_FEATURES_ORDER = [
    'holiday', 'temp', 'rain', 'snow', 'weather', 'year',
    'month', 'day', 'hours', 'minutes', 'seconds'
]

# Define which of the EXPECTED_FEATURES_ORDER are numerical and which are categorical
NUMERICAL_FEATURES = ['temp', 'rain', 'snow', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']
CATEGORICAL_FEATURES = ['holiday', 'weather']

# Ensure that all features in EXPECTED_FEATURES_ORDER are covered by numerical or categorical lists
if sorted(NUMERICAL_FEATURES + CATEGORICAL_FEATURES) != sorted(EXPECTED_FEATURES_ORDER):
    print("Configuration Error: NUMERICAL_FEATURES and CATEGORICAL_FEATURES do not cover all EXPECTED_FEATURES_ORDER.")
    print(f"Expected features: {EXPECTED_FEATURES_ORDER}")
    print(f"Combined defined features: {sorted(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)}")
    exit()

# --- 1. Load your dataset ---
try:
    data = pd.read_csv(DATASET_FILE)
    print(f"Dataset '{DATASET_FILE}' loaded successfully! Shape: {data.shape}")
    print("Dataset columns found:", data.columns.tolist())
except FileNotFoundError:
    print(f"Error: '{DATASET_FILE}' not found. Please place your training data in the same directory as this script.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- 2. Data Preprocessing (Feature Engineering) ---
# This part handles extracting date/time components and dropping original columns.
# It assumes your dataset has 'date' and 'Time' columns.
if 'date' in data.columns and 'Time' in data.columns:
    print("Extracting date and time features...")
    # Combine 'date' and 'Time' columns into a single string
    data['datetime_str'] = data['date'].astype(str) + ' ' + data['Time'].astype(str)

    # Convert the combined string to datetime objects, specifying the format
    # Based on previous errors, assuming DD-MM-YYYY HH:MM:SS format
    # Adjust 'format' if your date/time strings are different (e.g., "%Y-%m-%d %H:%M:%S")
    data['date_time'] = pd.to_datetime(data['datetime_str'], format="%d-%m-%Y %H:%M:%S", errors='coerce')

    # Check for any conversion errors (NaT - Not a Time)
    if data['date_time'].isnull().any():
        print("Warning: Some date/time values could not be parsed and were converted to NaT.")
        print("These rows will be dropped. Please check your 'date' and 'Time' column formats in your CSV.")
        data.dropna(subset=['date_time'], inplace=True)
        if data.empty:
            print("Error: All rows were dropped due to date/time parsing errors. Exiting.")
            exit()

    # Extract 'year', 'month', 'day', 'hours', 'minutes', 'seconds'
    data['year'] = data['date_time'].dt.year
    data['month'] = data['date_time'].dt.month
    data['day'] = data['date_time'].dt.day
    data['hours'] = data['date_time'].dt.hour
    data['minutes'] = data['date_time'].dt.minute
    data['seconds'] = data['date_time'].dt.second

    # Drop the original 'date', 'Time', and the intermediate 'datetime_str', 'date_time' columns
    data = data.drop(['date', 'Time', 'datetime_str', 'date_time'], axis=1)
    print("Date and time features extracted and original columns dropped.")
else:
    print("Warning: 'date' or 'Time' columns not found. Skipping date/time feature extraction.")
    print("Ensure your dataset already contains 'year', 'month', 'day', 'hours', 'minutes', 'seconds' if needed.")


# --- 3. Define Features (X) and Target (y) ---
# Assuming 'traffic_volume' is your target variable
TARGET_COLUMN = 'traffic_volume'
if TARGET_COLUMN not in data.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset. Please check your CSV file.")
    exit()

X = data.drop(TARGET_COLUMN, axis=1) # Features
y = data[TARGET_COLUMN] # Target

# --- 4. Verify and Reorder Features in X ---
# This step is critical to ensure X has the exact columns in the exact order
# that the preprocessor (and app.py) expects.
if not all(feature in X.columns for feature in EXPECTED_FEATURES_ORDER):
    missing_features = [f for f in EXPECTED_FEATURES_ORDER if f not in X.columns]
    extra_features = [f for f in X.columns if f not in EXPECTED_FEATURES_ORDER and f != TARGET_COLUMN]
    print(f"Error: Dataset columns after preprocessing do not match {EXPECTED_FEATURES_ORDER}.")
    if missing_features:
        print(f"Missing expected features: {missing_features}")
    if extra_features:
        print(f"Extra features found (will be ignored by preprocessor): {extra_features}")
    print(f"Features found in dataset after preprocessing: {list(X.columns)}")
    # Exit if critical features are missing, but allow if extra features are present (remainder='drop')
    if missing_features:
        exit()

X = X[EXPECTED_FEATURES_ORDER] # Reorder X to match the expected order
print(f"Features DataFrame (X) columns after reordering: {list(X.columns)}")
print(f"First 5 rows of X:\n{X.head()}")
print(f"Data types of X:\n{X.dtypes}")


# --- 5. Create Preprocessing and Model Pipeline ---
# This preprocessor will be part of the final pipeline.
# It handles numerical scaling and one-hot encoding for categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ],
    remainder='drop' # Drop any columns not explicitly handled (e.g., if you have other IDs)
)

# Create the full model pipeline (preprocessor + regressor)
# This is the object that will be saved as 'model.pkl'
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42)) # Or your chosen model (e.g., RandomForestRegressor())
])

# --- 6. Split data and Train the Model ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting model training...")
model_pipeline.fit(X_train, y_train)
print("Model training complete!")

# --- 7. Evaluate the Model (Optional) ---
score = model_pipeline.score(X_test, y_test)
print(f"Model R^2 Score on test set: {score:.4f}")

# --- 8. Save the Trained Pipeline ---
script_dir = os.path.dirname(__file__)
model_output_path = os.path.join(script_dir, "model.pkl")

try:
    with open(model_output_path, 'wb') as file:
        pickle.dump(model_pipeline, file)
    print(f"Full model pipeline saved successfully to: {model_output_path}")
except Exception as e:
    print(f"Error saving model pipeline: {e}")

print("Script finished.")