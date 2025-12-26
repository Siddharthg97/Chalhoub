"""
Master Table Generation Component for Vertex AI Pipelines.

This component:
1. Loads feature-engineered datasets (6 BigQuery tables) from previous pipeline stage.
2. Combines/Processes them into a consolidated master dataset.
3. Applies preprocessing steps (placeholder section for Data Science team).
4. Saves processed dataset to BigQuery.
5. Creates an initial 'best_model' table with row_id & actual values for model evaluation consistency.

Production Features:
- Robust logging and structured error handling.
- Explicit GCP project set from config to avoid accidental job execution in unintended projects.
- Artifact metadata enrichment for traceability.

NOTE: All dataset-specific logic will be provided by the Data Science team in clearly marked placeholder.
"""

from kfp.v2.dsl import component, Artifact, Input, Output

# =====================================================================
# Add python libraries with exact version in packages_to_install
# =====================================================================

@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu",
    packages_to_install=[
        "google-cloud-bigquery==3.31.0",
        "google-cloud-storage==2.10.0",
        "PyYAML==6.0.1",
        "pandas==2.3.0",
        "numpy==1.26.4",
        "holidays==0.73",
        "category_encoders==2.8.1",
        "seaborn==0.13.2",
        "matplotlib==3.10.0",
        "typing_extensions==4.13.0",
        "typing-inspection==0.4.0",
        "scikit-learn==1.7.0",
        "lightgbm==4.6.0",
        "xgboost==3.0.2",
        "pandas==2.3.0",
        "numpy==1.26.4",
        "scipy==1.15.3",
        "cloudpickle==3.1.1",
    ]
)

def master_table_inference(
    config_path: str,  # Path to training_config.yaml in GCS
    artifact_pre_process: Input[Artifact],
    artifact_temporal_features: Input[Artifact],
    artifact_store_features: Input[Artifact],
    artifact_online_marketing_features: Input[Artifact],
    artifact_promotional_features: Input[Artifact],
    artifact_master_table: Output[Artifact],        # Final processed dataset for training
    artifact_feature_target_encodings: Output[Artifact]  # Initial best_model table for evaluation
):
    """
    Combines multiple feature-engineered BigQuery tables into a master dataset
    and prepares processed outputs for training and evaluation.

    Args:
        config_path: Path to YAML config in GCS.
        table_1..table_6: Input BigQuery tables from the feature engineering step.
        processed_dataset: Output processed training dataset.
        best_model_artifact_init: Output initial best_model table.
    """
    
    # =====================================================================
    # Logging Setup
    # =====================================================================
    
    import logging
    
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Master Table creation step...")
    
    try:
        
        # =====================================================================
        # Import libraries here as required by the code
        # =====================================================================
        
        import pandas as pd
        import numpy as np
        import category_encoders as ce
        
        import category_encoders as ce
        import seaborn as sns
        
        from pandas.tseries.offsets import MonthEnd
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        from typing import List
        import re
        import holidays
        # import pickle
        
        import matplotlib.pyplot as plt
        
        import warnings
        
        warnings.filterwarnings("ignore")
        
        # =====================================================================
        # Download Faces_temporal_utils.py from GCS into /tmp
        # =====================================================================

        import os
        from google.cloud import storage

        helper_path = "gs://trd-sf-mlops/faces/monthly/training/utils/Faces_temporal_utils.py"

        if not helper_path.startswith("gs://"):
            raise ValueError("Helper path must start with gs://")

        bucket_name = helper_path.split('/')[2]
        blob_name = '/'.join(helper_path.split('/')[3:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        local_helper_file = "/tmp/Faces_temporal_utils.py"
        blob.download_to_filename(local_helper_file)

        import sys
        sys.path.insert(0, "/tmp")  # make /tmp importable
        
        from Faces_temporal_utils import (
            add_monthly_KSA_holiday_count,
            create_peak_calendar,
            merge_peak_calendar_info,
        )  

    
        # =====================================================================
        # Load config.yaml from GCS
        # =====================================================================
        
        if not config_path.startswith("gs://"):
            raise ValueError("Config path must start with gs://")
        bucket_name = config_path.split('/')[2]
        blob_name = '/'.join(config_path.split('/')[3:])
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        config_content = blob.download_as_text()
        config = yaml.safe_load(config_content)
        logger.info("Configuration loaded from GCS successfully.")
    
        # =====================================================================
        # Read Data from BigQuery
        # =====================================================================
        
        from google.cloud import bigquery
        
        client = bigquery.Client(project=config["project"]["id"])
    
        def load_dataframe(table_key: str):
            """
            Loads data from a BigQuery table using a key from the global `config` dictionary
            and returns the resulting DataFrame.
            """
            table_path = f"{config['feature_engineering_outputs'][table_key]}_{config['target']['value']}"
            query = f"SELECT * FROM `{table_path}`"
            logger.info(f"Executing query: {query}")
        
            df = client.query(query).to_dataframe(create_bqstorage_client=False)
        
            if df.empty:
                raise ValueError("No data found in source table")
        
            logger.info(f"Retrieved {len(df)} rows with shape {df.shape}")
            
            return df


            
        # ---------- Load All Feature-Engineered Tables ----------
    #     def load_bq_table(artifact):
    #         table_uri = artifact.uri.replace("bigquery://", "")
    #         logger.info(f"Loading table from BigQuery: {table_uri}")
    #         return client.query(f"SELECT * FROM `{table_uri}`").to_dataframe()
    
    #     df1 = load_bq_table(table_1)
    #     df2 = load_bq_table(table_2)
    #     df3 = load_bq_table(table_3)
    #     df4 = load_bq_table(table_4)
    #     df5 = load_bq_table(table_5)
    #     df6 = load_bq_table(table_6)
            
        # load_dataframe("pre_process", "df1")
        # load_dataframe("temporal_features", "df1")
        # load_dataframe("store_features", "df1")
        # load_dataframe("online_marketing_features", "df1")
        # load_dataframe("promotional_features", "df1")
    
        # ---- Inputs ----
        fc_horizon = 0  # Only input needed
        fc_horizon_inf = 15 # Only input needed
        brand = "FACES"
        country = "KSA"
        gcs_path = "gs://trd-sf-ntb"
        target = f"{config['target']['column']}"
        experiment_name = "mlops"
        granularity = "monthly"
        brand = brand.lower().replace(" ", "_")
        
        file_input1 = "temporal_features.parquet"
        file_input2 = "promotional_features.parquet"
        file_input3 = "online_marketing_features.parquet"
        file_input4 = "store_features.parquet"
        file_output = "feature_store.parquet"
        
        file_trend_flag = "deriv_flag.pkl"
        
        # Fixed train start date
        train_start_date = pd.Timestamp("2023-06-01")
        
        today = pd.Timestamp.today().normalize()
        ref_month_end = today if today.is_month_end else today - MonthEnd(1)
        
        # Train end date is up to the latest available month (i.e., ref_month_end)
        train_end_date = ref_month_end
        
        # Inference period: starts the month after train ends, and spans fc_horizon_inf months
        inference_start_date = (train_end_date + pd.DateOffset(months=1)).replace(day=1)
        inference_end_date = (train_end_date + pd.DateOffset(months=fc_horizon_inf)).replace(day=1) + MonthEnd(0)
        
        # ---- Output Formatting ----
        train_start_date = train_start_date.strftime("%Y-%m-%d")
        train_end_date = train_end_date.strftime("%Y-%m-%d")
        inference_start_date = inference_start_date.strftime("%Y-%m-%d")
        inference_end_date = inference_end_date.strftime("%Y-%m-%d")
        
        # # Target label month (typically the last test month)
        target_month = ref_month_end.strftime("%Y-%m")
        
        # ---- Output ----
        print(f"Today: {today.strftime('%Y-%m-%d')} (Is Month End: {today.is_month_end})")
        print(f"Train Start Date: {train_start_date}")
        print(f"Train End Date: {train_end_date}")
        print(f"Inference Start Date: {inference_start_date}")
        print(f"Inference End Date: {inference_end_date}")
        print(f"Target month: {target_month}")
        
        inference_start_date, inference_end_date
        
        def eda_features(df_input: pd.DataFrame, feature_name: str):
            """
            feature_name can only be following : temporal, promotion, marketing, store
            """
            print(f"------EDA on {feature_name} features------------------")
            try:
                print(
                    f"min & max dates {df_input.date.min().date()},{df_input.date.max().date()}",
                    end="\n\n",
                )
            except:
                pass
            print(f"shape of dataset : {df_input.shape}", end="\n\n")
            print(f"{feature_name} features: {df_input.columns.values}", end="\n\n")
            print(f"missing values :\n{df_input.isnull().sum()}")
            
        # Construct full dynamic path
        full_path1 = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_input1}"
        print(full_path1)
        # Construct full dynamic path
        
        # Save the DataFrame
        # df = pd.read_parquet(full_path1)
        df = load_dataframe("temporal_features")
        
        df = df[df["date"] <= train_end_date]
        # train_end_date=str(df.date.max().date())
        try:
            df.drop(
                columns=[
                    "ppu",
                    "business_type",
                    "is_month_start",
                    "is_month_end",
                    "days_in_month",
                ],
                inplace=True,
            )
        except:
            pass
        eda_features(df, "temporal")
        
        df.tail(2)
        
        # Construct full dynamic path
        full_path2 = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_input2}"
        print(full_path2)
        # Save the DataFrame
        # promo_df = pd.read_parquet(full_path2)
        promo_df = load_dataframe("promotional_features")
        
        promo_df.rename(columns={"locationId": "key"}, inplace=True)
        promo_df["key"] = promo_df["key"].astype(str)
        eda_features(promo_df, "Promotion")
        
        promo_df.head(2)
        
        # Construct full dynamic path
        full_path4 = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_input4}"
        print(full_path4)
        
        # Save the DataFrame
        # stores_df = pd.read_parquet(full_path4)
        
        stores_df = load_dataframe("store_features")
        
        stores_df = stores_df.rename(columns={"store": "key"})
        stores_df.key = stores_df.key.astype(str)
        str_cnt = len(stores_df.key.unique())
        print(f"Total store counts: {str_cnt}")
        str_vals = stores_df.key.unique()
        print(f"List of unique store id: {str_vals}")
        eda_features(stores_df, "Store")
        
        stores_df.head(2)
        
        # Construct full dynamic path
        full_path3 = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_input3}"
        print(full_path3)
        # mmm_df = pd.read_parquet(full_path3)
        
        mmm_df = load_dataframe("online_marketing_features")
        
        columns_to_keep = [
            "date",
            "total_cost_usd_facebook",
            "total_cost_usd_google",
            "total_cost_usd_instagram",
            "total_cost_usd_tiktok",
        ]
        mmm_df = mmm_df[columns_to_keep]
        int_mmm_df = mmm_df[mmm_df["date"] <= train_end_date]
        print(eda_features(mmm_df, "Marketing"))
        
        mmm_df.head(2)
        
        ext_mmm_df = mmm_df[mmm_df["date"] > train_end_date]
        ext_mmm_df.isnull().sum()
        eda_features(ext_mmm_df, "Extended markting")
        
        ext_mmm_df = pd.concat([ext_mmm_df] * str_cnt, ignore_index=True)
        dates = ext_mmm_df.date.unique().tolist()
        ext_mmm_df["key"] = ""
        for date in dates:
            ext_mmm_df.loc[ext_mmm_df["date"] == date, "key"] = list(str_vals)
        
        eda_features(ext_mmm_df, "Extended marketing")
        
        ext_mmm_df.head(2)
        
        df.columns
        
        print(f"temporal features {df.date.min()},{df.date.max()}")
        
        print(
            f"trimmed marketing features {int_mmm_df.date.min()},{int_mmm_df.date.max()}"
        )
        print(f"promotion features {promo_df.date.min()},{promo_df.date.max()}")
        # print(f"store features {stores_df.date.min()},{stores_df.date.max()}")
        start_date = max(
            [
                df.date.min().date(),
                int_mmm_df.date.min().date(),
                promo_df.date.min().date(),
            ]
        )
        print(start_date)
        
        # df = df.merge(promo_df, on = ['key','date'], how = 'left').fillna(0)
        df = df[df["date"].dt.date >= start_date]
        
        temp_mm = df.merge(int_mmm_df, on="date", how="left").fillna(0)
        print(temp_mm.columns)
        
        temp_ext_mm = pd.concat([temp_mm, ext_mmm_df], axis=0, join="outer")
        print(temp_ext_mm.columns)
        
        df_temp_mm_str = (
            temp_ext_mm.merge(stores_df, on="key", how="left")
            .sort_values(["key", "date"])
            .reset_index(drop=True)
        )
        print(df_temp_mm_str.columns)
        df = df_temp_mm_str.merge(promo_df, on=["key", "date"], how="left")
        print(df.columns)
        df.isnull().sum()
        
        df.reset_index(inplace=True, drop=True)
        df.sort_values(["key", "date"], inplace=True)
        
        df.date.max()
        
        df.columns
        
        # imputing promotion features with 0 value, feature avg_promo_duration missing in FACES
        df[[
                "distinct_discount_levels",
                "avg_discount",
                "max_discount",
                "min_discount",
                "avg_promo_duration",
                "promo_days_in_month",
                "percentage_products_on_promo",
            ]] = df[[
                "distinct_discount_levels",
                "avg_discount",
                "max_discount",
                "min_discount",
                "avg_promo_duration",
                "promo_days_in_month",
                "percentage_products_on_promo",
            ]].fillna(0)
        
        df_fut_hz = df[
            (df["date"] > train_end_date) & (df["date"] <= inference_end_date)
        ]
        print(df_fut_hz.date.min().date(), df_fut_hz.date.max().date())
        
        df_fut_hz = add_monthly_KSA_holiday_count(df_fut_hz[["date"]], "date")
        
        df_fut_hz.columns
        
        df_fut_hz = merge_peak_calendar_info(
            df_fut_hz, start_date=train_start_date, end_date=inference_end_date
        )
        df_fut_hz.festive_peak_flag.value_counts()
        
        df_fut_hz.columns
        
        df.tail(2)
        
        print(len(df[df["date"] > train_end_date]))
        print(len(df_fut_hz))
        
        df = df[df["date"] <= inference_end_date]
        
        print(len(df[df["date"] > train_end_date]))
        print(len(df_fut_hz))
        
        print(f"shape and max date:{df.shape},{df.date.max().date()}", end="\n\n")
        print(f"missing values:\n{df.isnull().sum()}", end="\n\n")
        
        qualt_features = [
            "channel",
            "festive_peak_flag",
            "month",
            "quarter",
            "year",
            "store_format",
            "total_square_ft_cat",
            "selling_square_ft_cat",
            "key",
        ]
        print(f"qualitative features:\n{qualt_features}")
        
        for col in qualt_features:
            print(df[col].value_counts())
            
        def target_encode_percentage_contribution(
            df_input: pd.DataFrame, cat_features: list, target_col: str
        ) -> pd.DataFrame:
            # Total target sum across entire dataset
            total_target = int(df_input[target_col].sum())
        
            for col in cat_features:
                # Sum of target per category
                var = col + "_target_sum"
                df_n = df_input.dropna(subset=target_col)
                contrib_df = df_n.groupby(col).agg(var=("target", "sum")).reset_index()
                contrib_df = contrib_df.rename(columns={"var": var})
        
                # Compute percentage contribution
                contrib_df[f"{col}_target_pct"] = contrib_df[var].apply(
                    lambda x: x / total_target
                )
                contrib_df.drop(columns=var, errors="ignore", inplace=True)
        
                # Join back to original dataframe
                df_input = pd.merge(df_input, contrib_df, on=col, how="left")
        
            return df_input
        
        
        # peak_festive_flag, channel
        df_encoding = df.copy()
        
        df[df["target"].isnull()]
        
        df_ml = df.copy()
        
        df_train = df_ml[pd.to_datetime(df_ml.date)<=train_end_date]
        df_test = df_ml[(pd.to_datetime(df_ml.date)>train_end_date) &((pd.to_datetime(df_ml.date)<=inference_end_date))]

        df_train["Pred_Flag"] = 0
        df_test["Pred_Flag"] = 1

        # Combine the two subsets
        df_inference = pd.concat([df_train, df_test], ignore_index=True)
        
        # Traget encoding for Key
        for col in ["deriv_2_trend_flag", "store_format", "key"]:
            encoder = ce.TargetEncoder(cols=[col])
            df_w_may_apr = df_ml[pd.to_datetime(df_ml["date"]) <= train_end_date][
                [col, "target"]
            ]
            # print(df_w_may_apr.head())
            # Fit and transform
            df_w_may_apr[f"{col}_encoded"] = encoder.fit_transform(
                df_w_may_apr[col], df_w_may_apr["target"]
            )
            df_storecat = df_w_may_apr[[f"{col}_encoded", col]].drop_duplicates()
        
            key_dict = {}
            for index, row in df_storecat.iterrows():
                key_dict[row[col]] = row[f"{col}_encoded"]
                df_ml[f"{col}_encode"] = df_ml[col].apply(
                    lambda x: key_dict[x] if x in key_dict.keys() else None
                )
        
        # Encoding features - quarter, channel,city
        df_ml["quarter_sin"] = df_ml.quarter.apply(
            lambda x: np.sin(2 * np.pi * (x - 1) / 4)
        )
        df_ml["quarter_cos"] = df_ml.quarter.apply(
            lambda x: np.cos(2 * np.pi * (x - 1) / 4)
        )
        
        df_ml = df_ml[df_ml["is_ecom_fullfillment_loc"].notna()]
        df_ml["is_ecom_fullfillment_loc"] = df_ml["is_ecom_fullfillment_loc"].astype(
            int
        )
        
        # Convert channel to binary: Retail = 0, Ecom = 1
        df_ml["channel"] = (
            df_ml["channel"].str.lower().map({"retail": 0, "ecommerce": 1})
        )
        
        df_ml["city"] = df_ml["city"].str.upper()
        
        # Define primary cities
        primary_cities = ["RIYADH", "JEDDAH", "MAKKAH", "MADINAH", "DAMMAM", "KHOBAR"]
        
        # Map city to binary flag: 1 for primary, 0 for secondary
        df_ml["city"] = df_ml["city"].apply(lambda x: 1 if x in primary_cities else 0)
        
        flag = [dt for dt in df_ml[df_ml["festive_peak_flag"] == 1].date.unique()]
        one_month_ago = [date + relativedelta(months=-1) for date in flag]
        one_month_ahead = [date + relativedelta(months=1) for date in flag]
        ttl_months = one_month_ahead + one_month_ago
        
        df_ml["pre_post_rm_flg"] = 0
        df_ml.loc[df_ml["date"].isin(ttl_months), "pre_post_rm_flg"] = 1
        
        df_ml.isnull().sum()
        
        df_ml[
            ["deriv_2_trend_flag", "deriv_2_trend_flag_encode"]
        ].dropna().drop_duplicates()
        
        # Assuming df is the DataFrame
        trend_dict = dict(
            df_ml[["deriv_2_trend_flag", "deriv_2_trend_flag_encode"]]
            .dropna()
            .drop_duplicates()
            .values
        )
        
        print(trend_dict)
        
        trend = pd.DataFrame(
            list(trend_dict.items()),
            columns=["deriv_2_trend_flag", "deriv_2_trend_flag_encode"],
        )
        
        # trend.to_pickle(file_trend_flag)
        
        # or to GCS path
        # full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{file_trend_flag}"
        # trend.to_pickle(full_path)
        
        df_ml.drop(
            columns=["month", "quarter", "year", "deriv_2_trend_flag", "year_month"],
            inplace=True,
            errors="ignore",
        )
        df_ml.sort_values(["key", "date"], inplace=True)
        print(f"features & target:{df_ml.columns.values}")
        
        df_ml.deriv_2_trend_flag_encode.unique()
        
        df.info()
        
        # corr = df_ml.corr()
        
        # 2. Mask the upper triangle to avoid duplicate information
        # mask = np.triu(np.ones_like(corr, dtype=bool))
        
#         # 3. Plot the heatmap
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(
#             corr,
#             mask=mask,
#             cmap=sns.diverging_palette(220, 20, as_cmap=True),
#             vmin=-1,
#             vmax=1,
#             annot=True,
#             fmt=".2f",
#             linewidths=0.5,
#             cbar_kws={"shrink": 0.8},
#         )
#         plt.title("Correlation Heatmap of df_ml Columns")
#         plt.tight_layout()
#         plt.show()
        
#         df_ml.columns
        
#         cols = [
#             "Seasonality_Index",
#             # "rate_of_sale",
#             "total_cost_usd_facebook",
#             "total_cost_usd_google",
#             "total_cost_usd_instagram",
#             "total_cost_usd_tiktok",
#             "target_seasonal_monthly",
#             "target_seasonal_quarterly",
#         ]
        
#         corr_df = df_ml[cols].corr()
#         mask = np.triu(np.ones_like(corr_df, dtype=bool), k=0)
#         corr_masked = corr_df.mask(mask, other=0)
        
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(
#             corr_masked,
#             annot=True,
#             fmt=".2f",
#             cmap="coolwarm",
#             vmin=-1,
#             vmax=1,
#             linewidths=0.5,
#             square=True,
#         )
#         plt.title("Lower-Triangle Correlation Heatmap (Zeros Above)")
#         plt.tight_layout()
#         plt.show()
        
#         # store related features
#         cols = [
#             "total_square_ft",
#             "selling_square_ft",
#             "total_square_ft_cat",
#             "selling_square_ft_cat",
#         ]
        
#         corr_df = df_ml[cols].corr()
#         mask = np.triu(np.ones_like(corr_df, dtype=bool), k=0)
#         corr_masked = corr_df.mask(mask, other=0)
        
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(
#             corr_masked,
#             annot=True,
#             fmt=".2f",
#             cmap="coolwarm",
#             vmin=-1,
#             vmax=1,
#             linewidths=0.5,
#             square=True,
#         )
#         plt.title("Lower-Triangle Correlation Heatmap (Zeros Above)")
#         plt.tight_layout()
#         plt.show()
        
#         cols = [
#             # "rate_of_sale",
#             "Lag12_y",
#             "MA6_y",
#             "STD6_y",
#             "EMA6_y",
#             "target_seasonal_monthly",
#             "target_seasonal_quarterly",
#             "key_encode",
#         ]
#         corr_df = df_ml[cols].corr()
#         mask = np.triu(np.ones_like(corr_df, dtype=bool), k=0)
#         corr_masked = corr_df.mask(mask, other=0)
        
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(
#             corr_masked,
#             annot=True,
#             fmt=".2f",
#             cmap="coolwarm",
#             vmin=-1,
#             vmax=1,
#             linewidths=0.5,
#             square=True,
#         )
#         plt.title("Lower-Triangle Correlation Heatmap (Zeros Above)")
#         plt.tight_layout()
#         plt.show()
        
#         df_ml.drop(
#             [
#                 "store_format",
#                 # "rate_of_sale",
#                 "target_seasonal_monthly",
#                 "target_seasonal_quarterly",
#                 "EMA6_y",
#                 "total_square_ft",
#                 "total_square_ft_cat",
#             ],
#             axis=1,
#             inplace=True,
#         )
        
        df_ml.head(20)
        
        df_ml[df_ml["key"] == "3024"].tail(20)
        
        # Construct full dynamic path
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_output}"
        print(full_path)
        
        full_path
        
        # df_ml.to_parquet(full_path, index=False)
        
        df_ml.date.max()
        
        df_ml.key.unique()
        
        df_ml.tail(2)
    
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
    
        save_to_bq_and_set_artifact(df_ml, f"{config['master_table_outputs']['master_table']}_{config['target']['value']}", artifact_master_table)
        save_to_bq_and_set_artifact(trend, f"{config['master_table_outputs']['feature_target_encodings']}_{config['target']['value']}", artifact_feature_target_encodings)
    
        logger.info("Master Table Creation step completed successfully ")
    
    except Exception as e:
        logger.exception("Unhandled error occurred during Master Table Creation Component execution.")
        raise
