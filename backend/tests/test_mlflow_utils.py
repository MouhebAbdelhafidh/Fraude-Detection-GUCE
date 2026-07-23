import mlflow
from mlflow_utils import init_mlflow, start_pipeline_run, child_run, log_to_run

init_mlflow()
print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", mlflow.get_experiment_by_name("fraud_detection_pipeline"))

# 1. Root run
root_run_id = start_pipeline_run(dag_run_id="smoke_test_001")
print("root run_id:", root_run_id)

# 2. Nested child runs
with child_run("dummy_child_1", root_run_id) as run:
    mlflow.log_param("x", 1)
    mlflow.log_metric("y", 0.5)
    print("child_1 run_id:", run.info.run_id)

with child_run("dummy_child_2", root_run_id) as run:
    mlflow.log_param("z", "hello")
    mlflow.log_metric("score", 0.87)
    print("child_2 run_id:", run.info.run_id)

# 3. Reopen the root run and append summary metrics
log_to_run(
    root_run_id,
    metrics={"n_children_completed": 2},
    params={"pipeline_status": "smoke_test"},
)

print("done — check the UI at", mlflow.get_tracking_uri())