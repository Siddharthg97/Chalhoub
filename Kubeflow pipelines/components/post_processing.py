from kfp.v2.dsl import component, Artifact, Output
import logging
import yaml


@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu",
    packages_to_install=[
        "pandas==2.2.2"
    ]
)
def post_inference(
    config_path: str,  # Path to training_config.yaml in GCS
    inference_table: Input[Artifact],
    post_inference: Output[Artifact],
):

    # ========= IMPORTS REQUIRED WITHIN COMPONENT FUNCTION =========
    import pandas as pd
    from google.cloud import bigquery, storage
    import yaml
    import logging
    # =============================================================

    # ---------- Logging Setup ----------
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Feature Engineering Step...")

    # ---------- Load Config from GCS ----------
    if not config_path.startswith("gs://"):
        raise ValueError("Config path must start with gs://")

    bucket_name = config_path.split("/")[2]
    blob_name = "/".join(config_path.split("/")[3:])

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    config_content = blob.download_as_text()
    config = yaml.safe_load(config_content)

    logger.info("Configuration loaded successfully from GCS")

    # ---------- Read Raw Data from BigQuery ----------
    client = bigquery.Client(project=config["project"]["id"])
    query = f"SELECT * FROM `{config['inference_table_outputs']['inference_table']}`"
    logger.info(f"Executing query: {query}")
    df = client.query(query).to_dataframe()

    if df.empty:
        raise ValueError("No data found in source table")
    logger.info(f"Retrieved {len(df)} rows with shape {df.shape}")

    # -------------------- SAMPLE START --------------------

    test_data_predictions = f"SELECT * FROM `{config['post_processing_inputs']['test_data_predictions']}`"
    temporal_avg_unit_price = f"SELECT * FROM `{config['post_processing_inputs']['temporal_avg_unit_price']}`"
    feature_importance = f"SELECT * FROM `{config['post_processing_inputs']['feature_importance']}`"
    inference_table = f"SELECT * FROM `{config['inference_outputs']['inference_table']}`"

    # -------------------- SAMPLE END ----------------------

    # =====================================================================
    # SAVE DATAFRAMES TO BIGQUERY & SET OUTPUT ARTIFACTS
    # =====================================================================
    def save_to_bq_and_set_artifact(df_obj, table_name, artifact):
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(df_obj, table_name, job_config=job_config)
        job.result()
        logger.info(f"Saved table: {table_name} with {len(df_obj)} rows")
        artifact.uri = f"bigquery://{table_name}"
        artifact.metadata["table_name"] = table_name
        artifact.metadata["row_count"] = len(df_obj)
        artifact.metadata["description"] = "Feature engineered table output"

    table = f"{config['post_processing_outputs']['post_processing_inference_table']}"  # e.g., project.dataset

    save_to_bq_and_set_artifact(df1, f"{table}", post_inference)

    logger.info("Feature Engineering step completed successfully ")
