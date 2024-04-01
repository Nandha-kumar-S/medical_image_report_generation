import joblib

def load_mlp_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def generate_report(model, image_features):

    try:
        prediction = model.predict(image_features)
        return prediction
    except Exception as e:
        print(f"Error predicting with model: {str(e)}")
        return None
