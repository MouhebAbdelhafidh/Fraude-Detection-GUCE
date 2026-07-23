#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIAT Fraud Detection Complete MLOps Pipeline
Orchestrates:
  [Tabular Prep -> Tabular Model Training -> Tabular Ensemble Voting]   ||
  [GNN Graph Synthesis -> PyG Tensor Splitting -> GNN Training & Drift Audit]
  -> Final Fraud Fusion (combines the tabular weighted vote with the GNN's
       structural risk score into one final prediction)
"""

import os
import sys
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# Add the module directory to sys.path so Python can resolve both
# 'pipeline_tasks' AND sub-modules imported by tasks (like 'gnn_models')
# --------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # points to /backend/src/

for path in [CURRENT_DIR, PARENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# MLflow tracking config
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///home/mouhe/MLopsPipeline/backend/src/mlops/mlruns")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "fraud_detection_pipeline")

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# Now safe to import pipeline_tasks (which internally imports gnn_models)
from pipeline_tasks import (
    task_prepare_data,
    task_preprocess_graph_nodes,
    task_generate_graph_tensors,
    task_train_models,
    task_train_gnn_model,
    task_run_data_drift_audit,
    task_evaluate_ensemble,
    task_final_fraud_fusion,
    task_register_champion_model,
)

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlops_fraud_pipeline_v1',
    default_args=default_args,
    description='Tabular + GNN branches run in parallel; converge only at final fraud fusion',
    schedule='@weekly',
    catchup=False,
    tags=['fraud', 'pyspark', 'pytorch-geometric', 'xgboost', 'evidently', 'mlflow'],
) as dag:

    # --------------------------------------------------------------
    # Tabular branch
    # --------------------------------------------------------------
    prepare_data_task = PythonOperator(
        task_id='pyspark_data_preparation',
        python_callable=task_prepare_data,
    )

    train_models_task = PythonOperator(
        task_id='unsupervised_models_training',
        python_callable=task_train_models,
    )

    evaluate_ensemble_task = PythonOperator(
        task_id='ensemble_risk_evaluation',
        python_callable=task_evaluate_ensemble,
    )

    # --------------------------------------------------------------
    # Graph / GNN branch
    # --------------------------------------------------------------
    preprocess_graph_nodes_task = PythonOperator(
        task_id='preprocess_graph_nodes',
        python_callable=task_preprocess_graph_nodes,
    )

    generate_graph_tensors_task = PythonOperator(
        task_id='generate_graph_tensors',
        python_callable=task_generate_graph_tensors,
    )

    train_gnn_task = PythonOperator(
        task_id='gnn_model_training',
        python_callable=task_train_gnn_model,
    )

    # Data Drift Auditor -- plain PythonOperator, in-process, using the
    # current evidently API (see task_run_data_drift_audit's docstring in
    # pipeline_tasks.py for why the old evidently.report / metric_preset
    # import paths no longer work). No subprocess/venv isolation needed --
    # the earlier ModuleNotFoundError was an import-path issue, not an
    # environment-isolation one.
    evidently_drift_task = PythonOperator(
        task_id='evidently_data_drift_audit',
        python_callable=task_run_data_drift_audit,
    )

    # --------------------------------------------------------------
    # Convergence point: fusing the two branches is the explicit goal
    # of this task, so unlike the rest of the DAG, this one genuinely
    # needs to wait on both. The drift audit is observability only --
    # it's included in the wait list so a drift-report failure doesn't
    # silently vanish, but it does not feed into the fusion score itself.
    # --------------------------------------------------------------
    final_fusion_task = PythonOperator(
        task_id='final_fraud_fusion',
        python_callable=task_final_fraud_fusion,
    )
    

    # Tabular branch
    prepare_data_task >> train_models_task >> evaluate_ensemble_task

    # Graph branch: GNN training and the drift audit both run off of the
    # same generated tensors, in parallel with each other.
    preprocess_graph_nodes_task >> generate_graph_tensors_task
    generate_graph_tensors_task >> train_gnn_task
    generate_graph_tensors_task >> evidently_drift_task

    register_model_task = PythonOperator(
        task_id='register_champion_model',
        python_callable=task_register_champion_model,
    )
    
    # Fusion needs terminal outputs from all parallel tracking runs
    [evaluate_ensemble_task, train_gnn_task, evidently_drift_task] >> final_fusion_task
    final_fusion_task >> register_model_task