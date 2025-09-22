import mlflow
import pandas as pd

def load_and_predict():
    """
    Load the latest staged model from MLflow Model Registry and make a sample prediction.
    """
    MODEL_NAME = "titanic-classifier-prod"
    MODEL_STAGE = "Staging"   # ถ้าตั้งเป็น Production ก็เปลี่ยนตรงนี้ได้

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Make sure there is a '{MODEL_STAGE}' version in the MLflow UI.")
        return

    # เตรียม 1 แถวตัวอย่างจาก processed_data/test.csv (ที่สร้างจากสคริปต์ 02)
    test_path = "processed_data/test.csv"
    df = pd.read_csv(test_path)

    # แยกฟีเจอร์/เลเบล (ถ้าไฟล์ test มีคอลัมน์ Survived ติดมาด้วย)
    if "Survived" in df.columns:
        X_sample = df.drop(columns=["Survived"]).iloc[:1]
        actual = df["Survived"].iloc[0]
    else:
        X_sample = df.iloc[:1]
        actual = None

    # ทำนาย
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
