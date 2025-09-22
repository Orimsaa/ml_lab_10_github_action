import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def preprocess_data(test_size=0.25, random_state=42):
    mlflow.set_experiment("Titanic - Data Preprocessing")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")

        df = pd.read_csv("train.csv")

        # ===== Feature engineering / cleaning =====
        # Age / Fare missing
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        # Family features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Title from Name
        df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
        df['Title'] = df['Title'].replace({
            'Mlle':'Miss','Ms':'Miss','Mme':'Mrs',
            'Lady':'Rare','Countess':'Rare','Dona':'Rare',
            'Dr':'Rare','Rev':'Rare','Col':'Rare','Major':'Rare','Sir':'Rare',
            'Jonkheer':'Rare','Capt':'Rare','Don':'Rare'
        })

        # Drop columns not used
        df = df.drop(columns=['Name','Ticket','Cabin','PassengerId'])

        # One-hot encoding
        df = pd.get_dummies(df, columns=['Sex','Embarked','Pclass','Title'], drop_first=True)

        # Split
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Save processed
        os.makedirs("processed_data", exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv("processed_data/train.csv", index=False)
        pd.concat([X_test,  y_test ], axis=1).to_csv("processed_data/test.csv",  index=False)

        # Log to MLflow
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))
        mlflow.log_artifacts("processed_data", artifact_path="processed_data")

        print(f"Preprocessing Run ID: {run_id}")

if __name__ == "__main__":
    preprocess_data()
