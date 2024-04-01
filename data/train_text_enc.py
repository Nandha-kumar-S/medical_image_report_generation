import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(dataset_path,model):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Drop any rows with missing values
    df.dropna(inplace=True)
    if model=='mlp':
    # Convert string representations of features to numeric arrays
        df['Frontal_features'] = df['Frontal_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
        df['Lateral_features'] = df['Lateral_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

        # Encode the target variable 'impressions'
        label_encoder = LabelEncoder()
        df['impressions_encoded'] = label_encoder.fit_transform(df['impression'])

        # Convert features to NumPy arrays
        X = np.array(df[['Frontal_features', 'Lateral_features']])
        X_frontal = np.array(df['Frontal_features'].tolist())
        X_lateral = np.array(df['Lateral_features'].tolist())
        X = np.concatenate((X_frontal, X_lateral), axis=1)
        
        # Convert target variable to NumPy array
        y = np.array(df['impressions_encoded'])

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert NumPy arrays to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)
        return X_train, X_test, y_train, y_test , label_encoder

    elif model=='xgb':
        df['Frontal_features'] = df['Frontal_features'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').astype(float))
        df['Lateral_features'] = df['Lateral_features'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').astype(float))

        # Preprocess textual features ('impressions' column) using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
        tfidf_features = tfidf_vectorizer.fit_transform(df['impression'])

        # Convert numerical features to numpy arrays
        X_numerical = df[['Frontal_features', 'Lateral_features']].values

        # Concatenate numerical features with TF-IDF features
        X_concatenated = np.hstack((X_numerical, tfidf_features.toarray()))

        y = df['impression'].values.flatten()  

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_concatenated, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test