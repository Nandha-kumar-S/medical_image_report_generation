import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class XGBModel:
    def __init__(self):
        self.model = None
        
    def train_model(self, X_train, y_train, X_valid, y_valid, params=None, num_round=100, model_path='xgb_model.model'):
        # Set default parameters if not provided
        if params is None:
            params = {
                'objective': 'multi:softmax',
                'num_class': len(set(y_train)),
                'eval_metric': 'merror'
            }

        y_train = np.array(y_train).flatten()
        encoder = OneHotEncoder()  # Set sparse=False for a dense matrix
        y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
        y_train_dense = y_train_encoded.toarray()
        y_valid_encoded = encoder.fit_transform(y_valid.reshape(-1, 1))
        
        dtrain = xgb.DMatrix(X_train, label=y_train_dense)
        dvalid = xgb.DMatrix(X_valid, label=y_valid_encoded)
        
        # Define evaluation sets
        evals = [(dtrain, 'train'), (dvalid, 'valid')]
        
        # Train the XGBoost model
        self.model = xgb.train(params, dtrain, num_round, evals=evals, verbose_eval=True)
        
        # Make predictions on the validation set
        y_pred = self.model.predict(dvalid)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_valid_encoded, y_pred)
        print(f'Validation Accuracy: {accuracy:.4f}')

        # Save the trained model
        self.model.save_model(model_path)

# Example usage:
# Instantiate XGBModel

