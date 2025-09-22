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
        print(f"Ensure a model version exists in stage '{MODEL_STAGE}' in MLflow UI.")
        return

    df = pd.read_csv("processed_data/test.csv")
    sample = df.drop(columns=['Survived']).iloc[0:1]
    actual = df['Survived'].iloc[0]

    pred = model.predict(sample)

    print("------------------------------")
    print(f"Sample features:\n{sample.to_dict(orient='records')[0]}")
    print(f"Actual Label: {actual}")
    print(f"Predicted Label: {pred[0]}")
    print("------------------------------")

if __name__ == "__main__":
    load_and_predict()
