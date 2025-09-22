import os
import argparse
import tempfile
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def parse_args():
    p = argparse.ArgumentParser(
        description="Train/Evaluate/Register model using artifacts from a preprocessing run_id"
    )
    p.add_argument("run_id", help="MLflow run_id from preprocessing step")
    p.add_argument("--n-estimators", type=int, default=300,
                   help="Number of trees for RandomForest (default: 300)")
    p.add_argument("--acc-threshold", type=float, default=0.85,
                   help="Threshold used for tagging pass/fail (model is registered regardless)")
    p.add_argument("--model-name", type=str, default="titanic-classifier-prod",
                   help="Registered Model name (default: titanic-classifier-prod)")
    return p.parse_args()


def main():
    args = parse_args()
    src_run_id = args.run_id
    n_estimators = args.n_estimators
    acc_threshold = args.acc_threshold
    model_name = args.model_name

    mlflow.set_experiment("Titanic - Train/Evaluate/Register")

    with mlflow.start_run() as run:
        this_run_id = run.info.run_id
        mlflow.set_tag("ml.step", "train_evaluate_register")
        mlflow.log_param("source_run_id", src_run_id)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("acc_threshold", acc_threshold)
        mlflow.log_param("model_name", model_name)

        # 1) โหลด artifacts (processed_data) จาก run ก่อนหน้า
        client = MlflowClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = client.download_artifacts(src_run_id, "processed_data", tmpdir)
            train_path = os.path.join(local_dir, "train.csv")
            test_path = os.path.join(local_dir, "test.csv")
            if not (os.path.exists(train_path) and os.path.exists(test_path)):
                raise FileNotFoundError("train.csv/test.csv not found under processed_data/ from run_id=" + src_run_id)
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

        # 2) แยก X / y
        target_col = "Survived"
        if target_col not in train_df or target_col not in test_df:
            raise KeyError(f"Target column '{target_col}' not found in train/test")
        X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
        X_test,  y_test  = test_df.drop(columns=[target_col]),  test_df[target_col]

        # 3) เทรนโมเดล
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # 4) ประเมิน
        acc = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        passed = bool(acc >= acc_threshold)
        mlflow.set_tag("threshold_passed", str(passed))
        print(f"Accuracy = {acc:.4f} (threshold = {acc_threshold})  -> passed={passed}")

        # 5) ลงทะเบียนโมเดล "เสมอ" + ติดแท็กผล threshold
        # หมายเหตุ: ใช้ log_model พร้อม registered_model_name จะสร้าง/อัพเวอร์ชันอัตโนมัติ
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=model_name
        )
        mlflow.set_tag("registration_mode", "always")
        print(f"✅ Model '{model_name}' registered (run: {this_run_id}).")

        # (ทางเลือก) ใส่ input_example/ signature ในอนาคตเพื่อเลี่ยง warning:
        # from mlflow.models import infer_signature
        # sig = infer_signature(X_train, clf.predict(X_train))
        # mlflow.sklearn.log_model(clf, "model", registered_model_name=model_name, signature=sig, input_example=X_train.head(1))

        print(f"✅ Train/Eval finished. This run_id: {this_run_id}")


if __name__ == "__main__":
    main()
