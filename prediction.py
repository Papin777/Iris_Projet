import joblib

def predict(data):
    clf = joblib.load("rf_model.sav")  # Load the pre-trained model
    return clf.predict(data)  # Make a prediction using the loaded model
