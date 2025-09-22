import pandas as pd
import mlflow

def validate_data():
    mlflow.set_experiment("Titanic - Data Validation")
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")

        train = pd.read_csv("train.csv")

        num_rows, num_cols = train.shape
        missing_values = train.isnull().sum().sum()
        num_classes = train['Survived'].nunique()

        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_param("num_classes", num_classes)

        status = "Success" if (num_classes == 2) else "Failed"
        mlflow.log_param("validation_status", status)

        print(f"Rows: {num_rows}, Cols: {num_cols}")
        print(f"Missing total: {missing_values}, Classes: {num_classes}")
        print(f"Validation status: {status}")

if __name__ == "__main__":
    validate_data()
