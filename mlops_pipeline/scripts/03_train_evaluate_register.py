import os
import sys
import argparse
import tempfile
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/Evaluate/Register model using processed artifacts from a previous run_id"
    )
    parser.add_argument("run_id", help="MLflow run_id from preprocessing step")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees for RandomForest (default: 300)")
    parser.add_argument("--acc-threshold", type=float, default=0.85,
                        help="Accuracy threshold to allow model registration (default: 0.85)")
    parser.add_argument("--model-name", type=str, default="titanic-classifier-prod",
                        help="Registered Model name (default: titanic-classifier-prod)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id
    n_estimators = args.n_estimators
    acc_threshold = args.acc_threshold
    model_name = args.model_name

    mlflow.set_experiment("Titanic - Train/Evaluate/Register")

    with mlflow.start_run() as run:
        this_run_id = run.info.run_id
        mlflow.set_tag("ml.step", "train_evaluate_register")
        mlflow.log_param("source_run_id", run_id)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("acc_threshold", acc_threshold)
        mlflow.log_param("model_name", model_name)

        # -------- 1) ดาวน์โหลด artifacts จาก run ก่อนหน้า --------
        client = MlflowClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            # โฟลเดอร์ artifacts ที่เรา log ไว้ในขั้น preprocess คือ "processed_data"
            local_dir = client.download_artifacts(run_id, "processed_data", tmpdir)

            train_path = os.path.join(local_dir, "train.csv")
            test_path = os.path.join(local_dir, "test.csv")

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"train/test csv not found in artifacts of run_id={run_id}. "
                    f"Looked for: {train_path} and {test_path}"
                )

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

        # -------- 2) แยก X,y --------
        if "Survived" not in train_df.columns:
            raise KeyError("Target column 'Survived' not found in train.csv")
        if "Survived" not in test_df.columns:
            raise KeyError("Target column 'Survived' not found in test.csv")

        X_train = train_df.drop("Survived", axis=1)
        y_train = train_df["Survived"]
        X_test = test_df.drop("Survived", axis=1)
        y_test = test_df["Survived"]

        # -------- 3) เทรนโมเดล --------
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # -------- 4) ประเมินผล --------
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        print(f"Accuracy = {acc:.4f} (threshold = {acc_threshold})")

        # -------- 5) ล็อกโมเดล + ลงทะเบียนแบบมีเงื่อนไข threshold --------
        # เคล็ดลับ: log_model พร้อม registered_model_name จะ "สร้างเวอร์ชันใหม่" ใน Model Registry ให้อัตโนมัติ
        if acc >= acc_threshold:
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                registered_model_name=model_name
            )
            mlflow.set_tag("registration", "succeeded")
            print(f"✅ Registered model '{model_name}' (run: {this_run_id})")
        else:
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model"
            )
            mlflow.set_tag("registration", "skipped_below_threshold")
            msg = (f"Model accuracy {acc:.4f} < {acc_threshold}. "
                   f"Skip registration for '{model_name}'.")
            print(f"⚠️  {msg}")
            # ถ้าอยากให้ fail job เมื่อไม่ผ่าน threshold ให้ uncomment บรรทัดต่อไป:
            # raise SystemExit(1)

        print(f"✅ Train/Eval finished. This run_id: {this_run_id}")


if __name__ == "__main__":
    main()
