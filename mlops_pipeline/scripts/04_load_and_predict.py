import sys
import mlflow
from mlflow.tracking import MlflowClient

def main():
    if len(sys.argv) < 3:
        print("Usage: python 04_transition_model.py <model_name> <alias>")
        sys.exit(1)

    model_name = sys.argv[1]
    alias = sys.argv[2]

    client = MlflowClient()

    # หารุ่นล่าสุดของโมเดลนี้
    versions = client.get_latest_versions(model_name)
    if not versions:
        print(f"No versions found for model '{model_name}'.")
        sys.exit(1)

    latest = versions[-1]  # เอาเวอร์ชันล่าสุด
    version = latest.version

    # ตั้ง alias เช่น "Staging"
    client.set_registered_model_alias(model_name, alias, version)

    print(f"✅ Model '{model_name}' version {version} transitioned to alias '{alias}'.")

if __name__ == "__main__":
    main()
