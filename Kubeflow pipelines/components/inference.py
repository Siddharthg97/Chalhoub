from kfp.v2.dsl import component, Artifact, Input, Model, Output, Metrics

@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu",
    packages_to_install=[
        "google-cloud-bigquery==3.31.0",
        "google-cloud-storage==2.10.0",
        "google-cloud-aiplatform==1.35.0",
        "PyYAML==6.0.1",
        "joblib==1.3.2",
        "scikit-learn==1.7.0",
        "lightgbm==4.6.0",
        "xgboost==3.0.2",
        "pandas==2.3.0",
        "numpy==1.26.4",
    ]
)
def inference(
    config_path: str,                               # Path to training_config.yaml in GCS
    artifact_master_table: Input[Artifact],        # Kept for compatibility; not used
    model_rf: Output[Model],
    model_xgb: Output[Model],
    model_lgbm: Output[Model],
    metrics_rf: Output[Metrics],
    metrics_xgb: Output[Metrics],
    metrics_lgbm: Output[Metrics],
    test_data_predictions: Output[Artifact],
):
    """
    Complete model training component for MLOps pipeline in Kubeflow.
    Trains multiple models (RandomForest, XGBoost, LightGBM), logs experiments,
    and registers models in Vertex AI Model Registry.
    """
    
    # ==================== IMPORTS AND SETUP ====================
    import os
    import pickle
    import warnings

    import numpy as np
    import pandas as pd
    from pandas.tseries.offsets import MonthEnd

    from Faces_temporal_utils import (
        get_lag_features,
        get_rate_of_sale_monthly,
        get_monthly_seasonality_index,
        get_moving_stats_features,
        add_monthly_derivative_features,
        add_monthly_flags
    )
    warnings.filterwarnings("ignore")
    
    # Setup logging and warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting model training component.")
    
    try:
        # ==================== CONFIGURATION LOADING ====================
        logger.info("Loading configuration from GCS...")
        if not config_path.startswith("gs://"):
            raise ValueError("Config path must be a GCS URI starting with gs://")
            
        bucket_name = config_path.split('/')[2]
        blob_name = '/'.join(config_path.split('/')[3:])
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        config_content = blob.download_as_text()
        config = yaml.safe_load(config_content)
        logger.info("Configuration loaded from GCS successfully.")
        
        # ==================== VERTEX AI CONFIGURATION ====================
        PROJECT_ID = config["project"]["id"]
        LOCATION = "europe-west4"
        STAGING_BUCKET = f"gs://trd-sf-mlops/faces/monthly/training/{config['target']['value']}/staging_bucket"
        BASE_EXPERIMENT_PREFIX = f"sales-forecasting-faces-monthly-{config['target']['value']}"
        
        # ==================== UTILITY FUNCTIONS ====================
        def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
            """Parse GCS URI into bucket and path components."""
            assert gcs_uri.startswith("gs://")
            path = gcs_uri[5:]
            parts = path.split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            return bucket, prefix
        
        def upload_to_gcs(local_path: str, gcs_uri: str):
            """Upload local file to GCS."""
            bucket_name, blob_name = parse_gcs_uri(gcs_uri)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} -> {gcs_uri}")
        
        def init_experiment_for_model(model_name: str):
            """Initialize Vertex AI context for a specific model's experiment."""
            try:
                experiment = aiplatform.Experiment(f"{BASE_EXPERIMENT_PREFIX}-{model_name}-{config['target']['value']}")
            except Exception:
                experiment = aiplatform.Experiment.create(f"{BASE_EXPERIMENT_PREFIX}-{model_name}-{config['target']['value']}")
            
            aiplatform.init(
                project=PROJECT_ID,
                location=LOCATION,
                experiment=f"{BASE_EXPERIMENT_PREFIX}-{model_name}-{config['target']['value']}",
                staging_bucket=STAGING_BUCKET,
            )
        
        def start_experiment_run(model_name: str, key: str | None = None) -> str:
            """Start (and set active) a run inside the current experiment."""
            ts = time.strftime("%Y%m%d-%H%M%S")
            run_name = f"{model_name}{('-' + key) if key else ''}-{ts}"
            aiplatform.start_run(run=run_name)
            return run_name
        
        # ==================== DATA LOADING ====================
        client = bigquery.Client(project=config["project"]["id"])
        
        def load_dataframe(table_key: str):
            """
            Loads data from a BigQuery table using a key from the global config dictionary
            and returns the resulting DataFrame.
            """
            table_path = f"{config['master_table_outputs'][table_key]}_{config['target']['value']}"
            query = f"SELECT * FROM `{table_path}`"
            logger.info(f"Executing query: {query}")
            
            df = client.query(query).to_dataframe(create_bqstorage_client=False)
            
            if df.empty:
                raise ValueError("No data found in source table")
            
            logger.info(f"Retrieved {len(df)} rows with shape {df.shape}")
            return df
        
        # ==================== TRAINING PARAMETERS ====================
        # ---- Inputs ----
        fc_horizon = 0  # Only input needed
        fc_horizon_inf = 15 # Only input needed
        horizon = 3
        brand = "FACES"
        country = "KSA"
        gcs_path = "gs://trd-sf-ntb"
        target = f"{config['target']['value']}"
        experiment_name = "mlops"
        granularity = "monthly"
        brand = brand.lower().replace(" ", "_")
        
        base_vars = {"key": "key", "target": "target", "date": "date"}
        
        # ---- Dates ----
        today = pd.Timestamp.today().normalize()
        ref_month_end = today if today.is_month_end else today - MonthEnd(1)
        
        # Fixed train start date
        train_start_date = pd.Timestamp("2023-06-01")
        
        # Dynamic train end date based on ref_month_end and fc_horizon
        train_end_date = (ref_month_end - pd.DateOffset(months=fc_horizon)).replace(day=1) + MonthEnd(0)
        
        # Test period
        test_start_date = (train_end_date + pd.DateOffset(months=1)).replace(day=1)
        test_end_date = (test_start_date + pd.DateOffset(months= fc_horizon - 1)).replace(day=1) + MonthEnd(0)
        
        # Target label month (typically the last test month)
        target_month = test_end_date.strftime("%Y-%m")
        
        # Inference end date (for forecast window)
        inference_end_date = (train_end_date + pd.DateOffset(months=fc_horizon + fc_horizon_inf)).replace(day=1) + MonthEnd(0)
        
        # Format for output
        train_start_date = train_start_date.strftime("%Y-%m-%d")
        train_end_date = train_end_date.strftime("%Y-%m-%d")
        test_start_date = test_start_date.strftime("%Y-%m-%d")
        test_end_date = test_end_date.strftime("%Y-%m-%d")
        inference_end_date = inference_end_date.strftime("%Y-%m-%d")
        
        # ---- Output ----
        print(f"Today: {today.strftime('%Y-%m-%d')} (Is Month End: {today.is_month_end})")
        print(f"Train Start Date: {train_start_date}")
        print(f"Train End Date: {train_end_date}")
        print(f"Test Start Date: {test_start_date}")
        print(f"Test End Date: {test_end_date}")
        print(f"Target Month: {target_month}")
        print(f"Inference End Date: {inference_end_date}")
        
        test_start_date, test_end_date
        
        # ==================== DATA LOADING AND PREPARATION ====================
        file_name = "feature_store.parquet"
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_name}"
        logger.info(f"Feature store path: {full_path}")
        
        # Load master table
        df_ml = load_dataframe("master_table")
        logger.info(f"Loaded master table with shape: {df_ml.shape}")
        
        # ==================== EDA FUNCTION ====================
        def eda_features(df_input: pd.DataFrame, feature_name: str):
            """
            Perform EDA for a given feature group in the DataFrame.
            
            Parameters:
            - df_input: pd.DataFrame – The input dataset.
            - feature_name: str – Must be one of ["temporal", "promotion", "marketing", "store"].
            """
            logger.info(f"------EDA on {feature_name} features------------------")
            
            try:
                logger.info(
                    f"min & max dates: {df_input.date.min().date()}, {df_input.date.max().date()}"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch date range: {e}")
            
            logger.info(f"Shape of dataset: {df_input.shape}")
            logger.info(f"{feature_name} features: {df_input.columns.values}")
            logger.info(f"Missing values:\n{df_input.isnull().sum()}")
        
        # Perform EDA
        eda_features(df_ml, "complete")
        
        # Clean data
        df_ml.drop(columns=["monthly_max_qty", "monthly_min_qty"], inplace=True, errors='ignore')
        
        def feature_combine_pd(dataframe: pd.DataFrame, dfu_columns: list[str], feature_list: dict[str, any],
           key: str, ds: str, target: str, target_mean: float) -> pd.DataFrame:
            '''
                Function combines row prediction with older dataframe and re-creates temporal features using new target
            '''
            temp_df = dataframe.copy()

            temp_df = get_lag_features(temp_df, [12], key=key, date_col=ds, target=target, target_mean = target_mean)
            temp_df = get_moving_stats_features(temp_df, months_back=[6])
            temp_df = get_monthly_seasonality_index(temp_df, 'date', 'target')

            # temp_df = create_monthly_seasonal_features(temp_df,'date', 'target','key')

            # Add year column if not already there
            if 'year' not in temp_df.columns:
                temp_df['year'] = pd.to_datetime(temp_df[ds]).dt.year

            # Generate & merge rate_of_sale
            rate_of_sale_df = get_rate_of_sale_monthly(temp_df, [key])
            temp_df = temp_df.merge(rate_of_sale_df, on=[key, 'year'], how='left')

            monthly_df = add_monthly_derivative_features(temp_df)
            monthly_df = add_monthly_flags(monthly_df)

            # print("1", monthly_df.columns)

            monthly_df = monthly_df.rename(columns={'year_month': 'date'})
            # print("2", monthly_df.columns)

            # List of columns to replace from monthly_df
            columns_to_replace = ['deriv_1_pct', 'deriv_2_pct', 'deriv_3_pct', 'deriv_6_pct', 'deriv_1_flag']  # customize this

            # Drop those from temp_df first
            temp_df.drop(columns=columns_to_replace, inplace=True)

            # Merge with monthly_df — include deriv_2_trend_flag without dropping it
            cols_to_merge = ['key', 'date'] + columns_to_replace + ['deriv_2_trend_flag']

            # Now merge without suffixes (since there’s no conflict)
            temp_df = temp_df.merge(monthly_df[cols_to_merge], on=['key', 'date'], how='left')

            # temp_df = temp_df.merge(monthly_df, on=['key', 'date'], how='left')
            # print("3", temp_df.columns)

            # Load the mapping DataFrame from the pickle file
            mapping_df = pd.read_pickle(file_path)
            # mapping_df = pd.read_pickle('f_deriv_flag.pkl')

            # Create a mapping dictionary from the DataFrame
            mapping_dict = dict(zip(mapping_df['deriv_2_trend_flag'], mapping_df['deriv_2_trend_flag_encode']))

            # print("Mapping file loaded with keys:", list(mapping_dict.keys()))

            # Apply the mapping to temp_df to create the encoded column
            temp_df['deriv_2_trend_flag_encode'] = temp_df['deriv_2_trend_flag'].map(mapping_dict)

            temp_df.drop(columns=['deriv_2_trend_flag'], inplace=True)
            return temp_df

    
    def forecast_loop_store(key, df_in, model_pred, expected_features, feature_list,model_name, target_mean):
        '''
            Function creates row wise predictions and then append the row to the training data,
            and uses these new values to create target based temporal features
        '''
        future_rows_to_predict = df_in[df_in["Pred_Flag"]==1].copy()
        future_rows_to_predict.sort_values("date", inplace=True)
        main_df = df_in[df_in["Pred_Flag"]==0].copy()
        main_df.sort_values("date", inplace=True)

        for i in range(len(future_rows_to_predict)):
            data_sv = []
            row_to_forecast = future_rows_to_predict.iloc[[i], :].copy()
            date = row_to_forecast["date"].values[0]
            # print(date)
            pred_flag = row_to_forecast["Pred_Flag"].values[0]
            temp_df = pd.concat([main_df, row_to_forecast], ignore_index=True)
            # print(temp_df.columns)
            temp_df.drop(columns=feature_list, errors='ignore', inplace=True)

            # Generate features (All temporal features)
            feature_df = feature_combine_pd(temp_df, ['key'], feature_list, 'key', 'date', 'target', target_mean)

            model_input = feature_df.reindex(columns=expected_features, fill_value=0)

            row_pred = model_input.iloc[[-1]]  # Last row is the forecast row
            column_name = expected_features.copy()
            column_name.extend(["target", 'key', 'date', 'Pred_Flag'])

            # Predict
            pred = model_pred.predict(row_pred)
            record = list(np.concatenate([row_pred.values.flatten(), pred.flatten()]).reshape(len(row_pred.values[0])+1))
            record.append(key)
            record.append(date)
            record.append(pred_flag)
            data_sv.append(record)

            if 'actual' not in future_rows_to_predict.columns:
                future_rows_to_predict['actual'] = future_rows_to_predict['target']  # store original target before predictions

            future_rows_to_predict.iloc[i, future_rows_to_predict.columns.get_loc('target')] = pred[0]

            # Append predicted row to df for next iteration
            row_to_forecast['target'] = pred[0]
            df_res = pd.DataFrame(data_sv, columns=column_name)
            main_df = pd.concat([main_df[column_name], df_res], ignore_index=True)

        # df_res = pd.DataFrame(data_sv,columns=column_name)
        # df_res2 = pd.concat([main_df[column_name],df_res],ignore_index=True)
        local_forecast_path = os.path.join(model_save_path, f"{model_name}_forecast_.csv")
        print(f"{local_forecast_path}")
        main_df.to_csv(local_forecast_path)

         # Prepare forecast_df with predictions
        forecast_df = future_rows_to_predict[['key', 'date', 'target']].copy()
        forecast_df.rename(columns={'target': 'forecast'}, inplace=True)

        # Add 'actual' column from the input df_in if available
        actual_df = df_in[df_in['Pred_Flag'] == 1][['key', 'date', 'target']].copy()

        # Merge forecasts with actuals
        forecast_df = forecast_df.merge(actual_df, on=['key', 'date'], how='left')

        return forecast_df

    
    def run_inference(df_input, model_list, model_save_path, feature_list, target_mean):
        '''
            Main function to run inference for the forecast horizon
        '''
        final = pd.DataFrame()

        for model_name in model_list:
            with open(os.path.join(model_save_path, f"{model_name}_model.pkl"), 'rb') as f:
                model, expected_features = pickle.load(f)

            all_keys = df_input["key"].unique().tolist()
            forecasts = pd.DataFrame()

            for k in all_keys:
                df_key = df_input[df_input["key"] == k].copy()
                df_forecast = forecast_loop_store(
                    k, df_key, model, expected_features, feature_list, model_name, target_mean
                )
                forecasts = pd.concat([forecasts, df_forecast], axis=0)

            forecasts.rename(columns={'forecast': f'forecast_{model_name}'}, inplace=True)
            if final.empty:
                final = forecasts
            else:
                final = pd.merge(final, forecasts, on=['key', 'date'], how='inner')

        return final

    
    feature_list = ['Lag12_y', 'MA6_y', 'STD6_y','Seasonality_Index']
    
    # %%time
    completed_fc = run_inference(
    df_input=df_inference,
    model_list=["xgboost", "lightgbm", "random_forest"],
    model_save_path=model_save_path,
    feature_list=feature_list,
    target_mean=target_mean)
    
    # List all columns that start with 'target' but are not exactly 'target'
    cols_to_drop = [col for col in completed_fc.columns if col.startswith('target') and col != 'target']

    # Drop those columns
    completed_fc_clean = completed_fc.drop(columns=cols_to_drop)
    
    # Construct full dynamic path
    full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{output_file_name}"
    print(full_path)

    completed_fc_clean.to_parquet(full_path)

    
