import mlflow
import pandas as pd

def load_and_predict():
    MODEL_NAME = "titanic-classifier-prod"
    MODEL_STAGE = "Staging"

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error loading model: {e}")
        return

    df = pd.read_csv("processed_data/test.csv")

    if "Survived" in df.columns:
        X_sample = df.drop(columns=["Survived"]).iloc[:1]
        actual = df["Survived"].iloc[0]
    else:
        X_sample = df.iloc[:1]
        actual = None

    pred = model.predict(X_sample)

    print("-" * 30)
    print("Sample features (first row):")
    print(X_sample.to_string(index=False))
    if actual is not None:
        print(f"Actual Label: {actual}")
    print(f"Predicted Label: {int(pred[0])}")
    print("-" * 30)

if __name__ == "__main__":
    load_and_predict()
