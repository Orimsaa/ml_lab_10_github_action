import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

# ✅ บังคับใช้โลคัลถ้าไม่กำหนดจาก ENV
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("file:./mlruns")

def preprocess_data(test_size=0.25, random_state=42):
    mlflow.set_experiment("Titanic - Data Preprocessing")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")

        df = pd.read_csv("train.csv")

        # ===== Feature engineering / cleaning =====
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
        df['Title'] = df['Title'].replace({
            'Mlle':'Miss','Ms':'Miss','Mme':'Mrs',
            'Lady':'Rare','Countess':'Rare','Dona':'Rare',
            'Dr':'Rare','Rev':'Rare','Col':'Rare','Major':'Rare','Sir':'Rare',
            'Jonkheer':'Rare','Capt':'Rare','Don':'Rare'
        })

        df = df.drop(columns=['Name','Ticket','Cabin','PassengerId'])
        df = pd.get_dummies(df, columns=['Sex','Embarked','Pclass','Title'], drop_first=True)

        X = df.drop('Survived', axis=1)
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        os.makedirs("processed_data", exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv("processed_data/train.csv", index=False)
        pd.concat([X_test,  y_test ], axis=1).to_csv("processed_data/test.csv",  index=False)

        # Log to MLflow
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))

        # ✅ ใช้ path relative (ไม่ให้ไป /C:)
        artifact_path = os.path.join("processed_data")
        mlflow.log_artifacts(local_dir=artifact_path, artifact_path="processed_data")

        print(f"Preprocessing Run ID: {run_id}")

        # ✅ ส่ง run_id ออกให้ GitHub Actions ใช้
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
                fh.write(f"run_id={run_id}\n")

if __name__ == "__main__":
    preprocess_data()
