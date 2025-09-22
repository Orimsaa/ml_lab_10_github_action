import sys, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow, mlflow.sklearn
from mlflow.artifacts import download_artifacts

# ✅ บังคับใช้โลคัลถ้าไม่กำหนดจาก ENV
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("file:./mlruns")

def train_evaluate_register(preprocessing_run_id, n_estimators=300):
    ACCURACY_THRESHOLD = 0.80
    mlflow.set_experiment("Titanic - Model Training")

    with mlflow.start_run(run_name=f"rf_n{n_estimators}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        # ✅ โหลด artifacts จาก Run ID
        local_path = download_artifacts(run_id=preprocessing_run_id, artifact_path="processed_data")
        train_path = os.path.join(local_path, "train.csv")
        test_path  = os.path.join(local_path, "test.csv")

        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        X_train, y_train = train_df.drop('Survived', axis=1), train_df['Survived']
        X_test,  y_test  = test_df.drop('Survived', axis=1),  test_df['Survived']

        pipeline = Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample"
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            pipeline,
            "titanic_classifier_pipeline",
            input_example=X_train.head(1)
        )

        if acc >= ACCURACY_THRESHOLD:
            print("🎉 Accuracy meets threshold. Registering model...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/titanic_classifier_pipeline"
            registered = mlflow.register_model(model_uri, "titanic-classifier-prod")
            print(f"📦 Registered model: {registered.name}, version {registered.version}")
        else:
            print("ℹ️ Accuracy below threshold. Not registering.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [n_estimators]")
        sys.exit(1)

    run_id = sys.argv[1]
    n_est = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    train_evaluate_register(preprocessing_run_id=run_id, n_estimators=n_est)
