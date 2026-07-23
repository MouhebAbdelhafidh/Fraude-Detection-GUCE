"""
mlflow_utils.py

Central MLflow configuration + helpers for correlating runs across
Airflow tasks.

Each task in this DAG (PythonOperator) runs in its own subprocess, so a
single `with mlflow.start_run():` block can't span multiple tasks. The
pattern used here instead:

  1. The first task (task_prepare_data) creates one "pipeline root" run
     and immediately closes it -- it exists only as an anchor to group
     everything else under. Its run_id is pushed to XCom.
  2. Every later task/model pulls that run_id from XCom and opens its
     own run tagged with `mlflow.parentRunId=<root run_id>`. MLflow's UI
     groups any run carrying that tag under the parent run's "Nested
     Runs" view, regardless of which process created it -- this is the
     supported way to fake in-process nesting across subprocess/task
     boundaries.

Set MLFLOW_TRACKING_URI (and optionally MLFLOW_EXPERIMENT_NAME) as
environment variables before Airflow imports pipeline_tasks -- the DAG
file in this project sets sensible defaults via os.environ.setdefault()
so this works out of the box, but a real deployment should point
MLFLOW_TRACKING_URI at a proper tracking server (Postgres-backed or a
managed MLflow instance) rather than the local file store, since the
local file store doesn't handle concurrent writers from parallel Airflow
tasks well.
"""

import os
import mlflow

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "file:///home/mouhe/MLopsPipeline/mlruns"
)
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "fraud_detection_pipeline"
)

_initialized = False


def init_mlflow():
    """Idempotent -- safe to call at the top of every task/function."""
    global _initialized
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    _initialized = True


def start_pipeline_run(dag_run_id: str) -> str:
    """
    Call once, from the first task in the DAG. Creates and immediately
    closes a root run that exists purely to group the DAG run's child
    runs together in the MLflow UI. Returns the run_id to push to XCom.
    """
    init_mlflow()
    with mlflow.start_run(run_name=f"pipeline_{dag_run_id}") as run:
        mlflow.set_tag("dag_run_id", dag_run_id)
        mlflow.set_tag("stage", "pipeline_root")
        run_id = run.info.run_id
    return run_id


def child_run(name: str, parent_run_id: str = None):
    """
    Context manager for a single model/task-level run, nested under the
    pipeline root run if parent_run_id is provided.

    Usage:
        with child_run("tabular_ocsvm", parent_run_id) as run:
            mlflow.log_params(...)
            mlflow.log_metric(...)
    """
    init_mlflow()
    tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else {}
    return mlflow.start_run(run_name=name, tags=tags)


def log_to_run(run_id: str, metrics: dict = None, params: dict = None, artifact_paths=None):
    """
    Reopen an existing run (e.g. the pipeline root run) to append
    metrics/params/artifacts logged after it was first closed.
    """
    init_mlflow()
    with mlflow.start_run(run_id=run_id):
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if artifact_paths:
            for p in artifact_paths:
                mlflow.log_artifact(p)