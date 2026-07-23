# tabular_data_prep.py
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

def prepare_tabular_data(spark: SparkSession, url: str, properties: dict, export_parquet: bool = False):
    """
    Complete production tabular data preparation pipeline for SIAT Fraud Detection.
    Consolidates JDBC ingestion, merging, structural deduplication, missing value handling,
    domain-specific feature engineering, and vector scaling.
    """
    print("📥 Step 1: Ingesting source tables via JDBC connection...")
    df_file_type = spark.read.jdbc(url=url, table="file_type", properties=properties)
    df_files = spark.read.jdbc(url=url, table="files", properties=properties)
    df_file_item = spark.read.jdbc(url=url, table="file_item", properties=properties)
    df_file_item_field_value = spark.read.jdbc(url=url, table="file_item_field_value", properties=properties)
    df_item_flow = spark.read.jdbc(url=url, table="item_flow", properties=properties)
    df_item_flow_data = spark.read.jdbc(url=url, table="item_flow_data", properties=properties)

    print("🔗 Step 2: Executing relational merge transformations...")
    # Prepare files & file type
    df_f_ready = df_files.withColumnRenamed("id", "file_id").withColumnRenamed("file_type_id", "original_file_type_id")
    df_ft_ready = df_file_type.withColumnRenamed("id", "file_type_id")
    df_merged_files = df_f_ready.join(df_ft_ready, df_f_ready.original_file_type_id == df_ft_ready.file_type_id, how="full")

    # Merge with file item
    df_fi_ready = df_file_item.withColumnRenamed("id", "file_item_id")
    df_master_enriched = df_merged_files.join(df_fi_ready, on="file_id", how="left")

    # Merge with file item field value
    df_final_merged = df_master_enriched.join(df_file_item_field_value, on="file_item_id", how="left") \
                                        .withColumnRenamed("value", "field_value")

    # Merge with item flow
    df_if_ready = df_item_flow.withColumnRenamed("id", "item_flow_id")
    df_ml_complete = df_final_merged.join(df_if_ready, on="file_item_id", how="left")

    # Merge with item flow data
    df_ifd_ready = df_item_flow_data.withColumnRenamed("id", "flow_data_id").withColumnRenamed("value", "flow_observation")
    df_siat_universe = df_ml_complete.join(df_ifd_ready, on="item_flow_id", how="left")

    print("🛠️ Step 3: Resolving duplicated schema column namespaces...")
    new_column_names = []
    seen = {}
    for name in df_siat_universe.columns:
        if name in seen:
            seen[name] += 1
            new_column_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            new_column_names.append(name)
    df = df_siat_universe.toDF(*new_column_names)

    print("🧹 Step 4: Cleaning schema anomalies and executing structural drop routines...")
    # Correct step tracking hierarchy
    df = df.drop("step_id").withColumnRenamed("step_id_2", "step_id")
    
    # Drop columns possessing 100% missing metadata values
    cols_to_drop = [
        "file_type_guce_annulation", "file_type_guce_paiement", "numero_demande_annulation", 
        "numero_demande_paiement", "reference_guce_annulation", "reference_guce_paiement", 
        "validity_date", "numero_dossier_base", "payment_file_type", "num_ebms_msg_annulation", 
        "num_ebms_msg_paiement", "subfamily_id", "history", "stage", "assigned_user_id_2"
    ]
    df = df.drop(*cols_to_drop)

    print("🧪 Step 5: Injecting multi-variable missing value imputation rules...")
    df = df.fillna({"flow_observation": ""})
    
    country_fills = {"country_of_origin": "CM", "country_of_destination": "DOMESTIC", "country_of_provenance": "DOMESTIC"}
    df = df.fillna(country_fills)

    # Flag pure server-side background procedures before default IP assignment
    df = df.withColumn("is_backend_process", F.when(F.col("ip_address").isNull(), 1).otherwise(0))
    
    system_fills = {"ip_address": "0.0.0.0", "message_id": "NO_SYSTEM_ID"}
    df = df.fillna(system_fills)

    # Complete operational indicators and handle decision timeline rollbacks
    df = df.withColumn("is_completed_transaction", F.when(F.col("signature_date").isNotNull(), 1).otherwise(0))
    df = df.withColumn("signature_date", F.coalesce(F.col("signature_date"), F.col("last_decision_date")))

    id_fills = {"signatory_user_id": -1, "assigned_user_id": -1, "bureau_id": -1, "nsh_id": -1, "flow_data_id": -1, "data_type_id": -1}
    df = df.fillna(id_fills)

    # Drop entries missing vital financial metadata and translate structural split indicators
    df = df.dropna(subset=["valeur_fob", "quantity", "file_type_id"])
    df = df.withColumn("is_amended_or_split", F.when(F.col("parent_file_id").isNotNull(), 1).otherwise(0)).drop("parent_file_id")
    df = df.fillna({"received": "DRAFT"})

    print("🚀 Step 6: Engineering analytical behavioral and financial risk features...")
    # Timeline Processing Metrics
    df = df.withColumn("processing_time_days", (F.col("signature_date").cast("timestamp").cast("double") - F.col("created_date").cast("timestamp").cast("double")) / 86400.0)
    df = df.withColumn("processing_time_days", F.when(F.col("processing_time_days").isNull(), 0.0).when(F.col("processing_time_days") < 0, 0.0).otherwise(F.col("processing_time_days")))

    # Market Price Normalization and Peer Deviation Checking
    df = df.withColumn("valeur_fob_clean", F.translate(F.col("valeur_fob").cast("string"), ",", ".").cast("double"))
    df = df.withColumn("quantity_clean", F.translate(F.col("quantity").cast("string"), ",", ".").cast("double"))
    df = df.withColumn("unit_price_fob", F.col("valeur_fob_clean") / (F.col("quantity_clean") + 0.001))
    
    nsh_window = Window.partitionBy("nsh_id")
    df = df.withColumn("median_nsh_unit_price", F.median("unit_price_fob").over(nsh_window))
    df = df.withColumn("price_deviation_ratio", F.col("unit_price_fob") / (F.col("median_nsh_unit_price") + 0.001))

    # Text Mining Flags via Case-Insensitive Regular Expressions
    lowercase_obs = F.lower(F.col("flow_observation"))
    df = df.withColumn("has_rejection_keywords", F.when(lowercase_obs.rlike("rejet|reject|annul|suspend"), 1).otherwise(0))
    df = df.withColumn("has_modification_keywords", F.when(lowercase_obs.rlike("modif|rectif|changement|update"), 1).otherwise(0))
    df = df.withColumn("has_authority_keywords", F.when(lowercase_obs.rlike("sgs|oncc|douane|customs|legal"), 1).otherwise(0))

    # Compliance & Cross-Silo Verification Signatures
    df = df.withColumn("is_not_cm_origin", F.when(F.col("country_of_origin") != "CM", 1.0).otherwise(0.0))
    df = df.withColumn("is_self_signed", F.when(F.col("assigned_user_id") == F.col("signatory_user_id"), 1.0).otherwise(0.0))
    df = df.withColumn("is_weekend_processing", F.when(F.dayofweek(F.col("created_date")).isin(1, 7), 1.0).otherwise(0.0))
    df = df.withColumn("is_night_processing", F.when((F.hour(F.col("created_date")) >= 22) | (F.hour(F.col("created_date")) <= 5), 1.0).otherwise(0.0))
    df = df.withColumn("is_coffee_transaction", F.when(F.col("code").contains("CAF") | F.lower(F.col("label_fr")).contains("café"), 1.0).otherwise(0.0))
    df = df.withColumn("draft_flag", F.col("draft").cast("double"))
    df = df.withColumn("fictive_flag", F.col("fictive").cast("double"))

    # Crucial Gateway State Transitions (step_id Encoding)
    df = df.withColumn("is_in_modification_phase", F.when((F.col("step_id") >= 131) & (F.col("step_id") <= 135), 1.0).otherwise(0.0))
    df = df.withColumn("is_at_signature_gate", F.when(F.col("step_id") == 112, 1.0).otherwise(0.0))
    df = df.withColumn("is_at_payment_gate", F.when(F.col("step_id") == 122, 1.0).otherwise(0.0))

    # Network Masking Footprints & Tax Haven Routings
    df = df.withColumn("is_anomaly_ip", F.when(F.col("ip_address").isin("127.0.0.1", "0.0.0.0", "localhost") | F.col("ip_address").isNull(), 1.0).otherwise(0.0))
    df = df.withColumn("is_high_risk_destination", F.when(F.col("country_of_destination") == "SG", 1.0).otherwise(0.0))

    print("📐 Step 7: Performing Vector Assembly and Feature Scaling Pipeline...")
    # Gather the 18 targeted features requested by the downstream model assembler
    feature_cols = [
        "processing_time_days", "is_weekend_processing", "is_night_processing", 
        "is_in_modification_phase", "is_at_signature_gate", "is_at_payment_gate", 
        "unit_price_fob", "price_deviation_ratio", "is_not_cm_origin", 
        "is_high_risk_destination", "is_coffee_transaction", "is_anomaly_ip", 
        "is_backend_process", "is_completed_transaction", "is_amended_or_split", 
        "draft_flag", "fictive_flag", "has_rejection_keywords"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)

    preprocessing_pipeline = Pipeline(stages=[assembler, scaler])
    pipeline_model = preprocessing_pipeline.fit(df)
    df_scaled = pipeline_model.transform(df)

    if export_parquet:
        os.makedirs("datamlops", exist_ok=True)
        print("💾 Saving final processed dataset matrix to 'datamlops/scaled_siat_fraud_dataset.parquet'...")
        df_scaled.write.mode("overwrite").parquet("datamlops/scaled_siat_fraud_dataset.parquet")

    print(f"✅ Extraction & Engineering Success! Extracted Matrix Dimensions: {len(df_scaled.columns)} Columns.")
    return df_scaled, pipeline_model


# =========================================================================================
# Local Integration Testing Block
# =========================================================================================
# if __name__ == "__main__":
#     # Create isolated standalone testing session
#     spark_session = SparkSession.builder \
#         .appName("PostgresMLOpsDataPrepPipeline") \
#         .config("spark.jars.packages", "org.postgresql:postgresql:42.7.2") \
#         .getOrCreate()

#     db_properties = {
#         "user": "postgres",
#         "password": "nobodycan@2H",
#         "driver": "org.postgresql.Driver"
#     }
#     db_url = "jdbc:postgresql://localhost:5432/guce_siat_cc"

#     # Run the integrated transformation logic
#     final_df, fitted_transformer = prepare_tabular_data(spark_session, db_url, db_properties, export_parquet=True)
    
#     # Output metrics directly comparable to the notebook
#     print(f"Final Count Verification: {final_df.count()} Rows verified in Spark processing context.")
#     final_df.select("numero_dossier", "unit_price_fob", "price_deviation_ratio", "scaled_features").show(5, truncate=False)