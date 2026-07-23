# quick_tag.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:///home/mouhe/MLopsPipeline/backend/scr/mlops/mlruns")
client = MlflowClient()

# Assuming version 1 of 'siat_fraud_fusion_model' exists
client.set_registered_model_alias(
    name="siat_fraud_fusion_model",
    alias="champion",
    version="1"
)
print("✅ Explicitly assigned 'champion' alias to version 1.")