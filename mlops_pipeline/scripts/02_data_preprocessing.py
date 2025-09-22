import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def preprocess_data(test_size=0.25, random_state=42):
    # ตั้งชื่อ experiment (ปรับได้ตามที่อาจารย์กำหนด)
    mlflow.set_experiment("Titanic - Data Preprocessing")

    # หาไฟล์ dataset แบบยืดหยุ่น: ใช้ ENV ก่อน, ไม่งั้นลอง data/train.csv แล้วค่อย train.csv
    dataset_path = os.environ.get("DATASET_PATH")
    if not dataset_path:
        dataset_path = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")

        # ----- Load -----
        df = pd.read_csv(dataset_path)

        # ----- Cleaning / Feature engineering -----
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Title from Name
        df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
        df['Title'] = df['Title'].replace({
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'Countess': 'Rare', 'Dona': 'Rare',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Sir': 'Rare',
            'Jonkheer': 'Rare', 'Capt': 'Rare', 'Don': 'Rare'
        })

        # Drop columns not used
        df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

        # One-hot encoding
        df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass', 'Title'], drop_first=True)

        # Split
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # ----- Save processed (ใช้ relative path เสมอ) -----
        processed_dir = "processed_data"
        os.makedirs(processed_dir, exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(processed_dir, "train.csv"), index=False)
        pd.concat([X_test,  y_test ], axis=1).to_csv(os.path.join(processed_dir, "test.csv"),  index=False)

        # ----- Log to MLflow -----
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))
        mlflow.log_artifacts(processed_dir, artifact_path="processed_data")

        # ส่ง run_id ออกไปให้ GitHub Actions step ถัดไป
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"run_id={run_id}\n")

        print(f"✅ Preprocessing Run ID: {run_id}")

if __name__ == "__main__":
    preprocess_data()
