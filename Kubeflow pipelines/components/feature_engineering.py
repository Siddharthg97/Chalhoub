"""
Feature Engineering Component for Vertex AI Pipelines.

This component is responsible for:
1. Loading configuration from GCS.
2. Reading raw data from BigQuery.
3. Providing clearly marked placeholders for Data Science feature engineering logic.
4. Writing multiple processed tables to BigQuery for downstream components.

Production Notes:
- Well-structured logging for observability.
- Explicit GCP project specification to avoid accidental dataset creation in unintended projects.
- Handles empty dataset scenarios gracefully.
"""

from kfp.v2.dsl import component, Artifact, Output

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
    ]
)

def feature_engineering_inference(
    config_path: str,  # Path to inference_config.yaml in GCS
    artifact_pre_process: Output[Artifact],
    artifact_temporal_features: Output[Artifact],
    artifact_store_features: Output[Artifact],
    artifact_online_marketing_features: Output[Artifact],
    artifact_promotional_features: Output[Artifact],
):
    """
    Executes feature engineering pipeline stage.

    Args:
        config_path: Path to YAML configuration in GCS (e.g., gs://bucket/configs/inference_config.yaml)
        table_1..table_6: Output BigQuery tables produced by feature engineering logic.
    """
    
    import logging
    import yaml
    
    # =====================================================================
    # Logging Setup
    # =====================================================================

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Feature Engineering Step...")
    
    try:
        logger.info("Import libraries here as required by the code")
        
        
        # =====================================================================
        # Import libraries here as required by the code
        # =====================================================================
    
        import datetime
        import re
        import warnings
        from datetime import timedelta
        from typing import List
        
        import holidays
        import numpy as np
        import pandas as pd
        from google.cloud import bigquery, storage
        from pandas.tseries.offsets import MonthEnd
        logger.info("Importing yaml")
        import yaml
    
        warnings.filterwarnings("ignore")
        
        logger.info("Load inference_config.yaml from GCS")
    
        # =====================================================================
        # Load inference_config.yaml from GCS
        # =====================================================================
    
        if not config_path.startswith("gs://"):
            raise ValueError("Config path must start with gs://")
    
        bucket_name = config_path.split("/")[2]
        
        logger.info(f"############## Received config_path as {config_path}")
        
        logger.info(f"Found Bucket as {bucket_name}")
        
        blob_name = "/".join(config_path.split("/")[3:])
        logger.info(f"Found Bucket as {blob_name}")
    
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        config_content = blob.download_as_text()
        logger.info("Using yaml")
        config = yaml.safe_load(config_content)
    
        logger.info("Configuration loaded successfully from GCS")
        
        # =====================================================================
        # Read Data from BigQuery
        client = bigquery.Client(project=config["project"]["id"])
        # client = bigquery.Client()
        
        logger.info("Configured BigQuery client")
        # =====================================================================
        
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
            artifact.metadata["description"] = "Feature Engineering component output"
        
        logger.info("Read Data from BigQuery")
    
        # ---- Inputs ----
        # fc_horizon = 0  # Only input needed
        fc_horizon_inf = 15 # Only input needed
        target = f"{config['target']['column']}"
        
        brand = "FACES"
        country = "KSA"
        gcs_path = "gs://trd-sf-ntb"
        experiment_name = "mlops"
        granularity = "monthly"
        brand = brand.lower().replace(" ", "_")
        
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
        
        
        '''
            1. PreProcess
        '''
        file_name = "pre_process.parquet"
        
        # Construct full dynamic path
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_name}"
        print(full_path)
        
        # Transaction data
        query_df = f"""
            SELECT *
            FROM `{config['data_sources']['sales_table']}` sales
            JOIN `{config['data_sources']['dim_product_table']}` products
            ON sales.bk_productid = products.item
            WHERE sales.brand = '{brand.upper()}'
              AND sales.bu_country = '{country}'
              AND sales.bk_businessdate >= '{train_start_date}' 
              AND sales.bk_businessdate <= '{train_end_date}'
        """
        logger.info(f"In config the sales table is : {config['data_sources']['sales_table']}")
        
        query_job_trans = client.query(query_df)
        df_full = query_job_trans.to_dataframe(create_bqstorage_client=False)
        
        df_full.rename(columns={"bk_storeid": "key", "bk_businessdate": "date"}, inplace=True)
        
        # Convert 'date' column to datetime
        df_full["date"] = pd.to_datetime(df_full["date"])
        
        # Convert 'date' to the first day of the month
        df_full["month"] = df_full["date"].values.astype("datetime64[M]")
        
        # Ensure numeric columns are the right type
        df_full["amountusd_beforetax"] = df_full["amountusd_beforetax"].astype(int)
        df_full["mea_quantity"] = df_full["mea_quantity"].astype(int)
        
        # Perform all aggregations in one groupby operation
        df = (
            df_full.groupby(["key", "business_type", "month"])
            .agg(
                num_transactions=("tran_seq_no", "nunique"),
                sales_amount=("amountusd_beforetax", "sum"),
                units_quantity=("mea_quantity", "sum"),
            )
            .reset_index()
        )
        
        # Calculate price per unit
        df["ppu"] = df["sales_amount"] / df["units_quantity"]
        
        # Ensure 'key' is an integer
        df["key"] = df["key"].astype(float).astype(int).astype(str)
        
        # Rename 'month' column to 'date' to match expected function parameter
        df = df.rename(columns={"month": "date"})
        
        # Convert 'date' to string before passing to the function
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")  # Or any format if you want to simulate ambiguity
        
        def detect_date_format(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
            """
            Detect and apply the correct date format to a specified column in a pandas DataFrame.
        
            This function attempts to parse the dates in the specified column using a list of common date formats.
            If a format is found that can successfully parse all dates without resulting in null values, the column
            is converted to this date format.
        
            Parameters:
            df (pd.DataFrame): Input DataFrame containing the date column.
            date_col (str): The name of the column containing date strings to be parsed.
        
            Returns:
            pd.DataFrame: DataFrame with the specified date column converted to datetime. If no format is successful,
                          the original DataFrame is returned unchanged.
            """
            formats = [
                "%m-%d-%Y",
                "%-m/%-d/%Y",
                "%-d/%-m/%Y",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%Y/%m/%d %H:%M:%S",
                "%m-%d-%Y %H:%M:%S",
                "%m-%d-%Y %I:%M %p",
                "%d-%b-%Y",
                "%Y%m%d",
            ]
        
            for fmt in formats:
                try:
                    parsed_dates = pd.to_datetime(df[date_col], format=fmt, errors="raise")
                    df_copy = df.copy()
                    df_copy[date_col] = parsed_dates
                    return df_copy
                except Exception:
                    continue
        
            # If none of the formats match, return original DataFrame
            return df
        
        
        df = detect_date_format(df, "date")
        df.head(2)
        
        def remove_inactive_stores(df, store_col="store_id", date_col="date", target_month="2025-07"):
            """
            Removes inactive stores from the DataFrame based on store activity in the target month.
        
            Parameters
            ----------
            df : pd.DataFrame
                Input transaction data with at least [store_col, date_col].
            store_col : str, default 'store_id'
                Name of the column containing store IDs.
            date_col : str, default 'date'
                Name of the column containing transaction dates.
            target_month : str, default '2025-05'
                Target month in 'YYYY-MM' format to define active stores.
        
            Returns
            -------
            pd.DataFrame
                Filtered DataFrame containing only rows from active stores.
            """
            df = df.copy()
        
            # Ensure date column is datetime
            df[date_col] = pd.to_datetime(df[date_col])
        
            # Extract month
            df["month"] = df[date_col].dt.to_period("M").astype(str)
        
            # Get active stores in target month
            active_stores = set(df[df["month"] == target_month][store_col].unique())
        
            # Filter to only active stores
            filtered_df = df[df[store_col].isin(active_stores)].drop(columns="month").reset_index(drop=True)
        
            return filtered_df
        
        filtered_df = remove_inactive_stores(
            df=df, store_col="key", date_col="date", target_month=target_month
        )
        
        filtered_df["key"].nunique()
        
        filtered_df.key.unique()
        
        filtered_df.head(2)
        
        filtered_df.tail(2)
        
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_name}"
        print(full_path)
        
        # Save the DataFrame
        # filtered_df.to_parquet(full_path, index=False)
        
        table_pre_process = f"{config['feature_engineering_outputs']['pre_process']}"
    
        save_to_bq_and_set_artifact(filtered_df, f"{table_pre_process}_{config['target']['value']}", artifact_pre_process)
        
        
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
            
        eda_features(filtered_df, "pre_process")
        
        
        '''
            2. Temporal Features
        '''
        file_input = "pre_process.parquet"
        file_output = "temporal_features.parquet"
        
        # Construct full dynamic path
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{file_input}"
        print(full_path)
        
        # Save the DataFrame
        # df = pd.read_parquet(full_path)
        df = filtered_df
        df = df[["key", "date", "business_type", "sales_amount", "units_quantity", "num_transactions","ppu"]]
        
        logger.info("############## List of Columns in df currently: %s", df.columns.tolist())
        logger.info("############## Looking for target column: '%s'", config['target']['column'])
        
        df.rename(columns={f"{config['target']['column']}": "target"}, inplace=True)
        
        
        
        def add_monthly_KSA_holiday_count(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
            """
            Adds a column with the count of KSA holidays per month.
        
            Parameters:
            df (pd.DataFrame): DataFrame with a monthly date column (e.g., '2023-03-01').
            date_col (str): Name of the date column.
        
            Returns:
            pd.DataFrame: DataFrame with an additional 'holiday_count' column.
            """
            # Ensure date column is in datetime format
            df[date_col] = pd.to_datetime(df[date_col])
        
            # Extract year and month
            df["year"] = df[date_col].dt.year
            df["month_num"] = df[date_col].dt.month
        
            # Build KSA holidays for all years in your data
            years = df["year"].unique()
            KSA_holidays = holidays.country_holidays("SA", years=years)
        
            # Count holidays per (year, month)
            # Convert holiday_dates to pandas datetime
            holiday_dates = pd.to_datetime(pd.Series(list(KSA_holidays.keys())))
        
            holiday_df = pd.DataFrame(
                {
                    "year": holiday_dates.dt.year,
                    "month_num": holiday_dates.dt.month,
                    "is_holiday": 1,
                }
            )
        
            holiday_counts = (
                holiday_df.groupby(["year", "month_num"]).size().reset_index(name="holiday_count")
            )
        
            # Merge with original df
            df = df.merge(holiday_counts, on=["year", "month_num"], how="left")
        
            # Fill NaN (months with no holidays) with 0
            if "holiday_count" in df.columns:
                df["holiday_count"] = df["holiday_count"].fillna(0).astype(int)
            else:
                df["holiday_count"] = 0
        
            # Drop helper columns if not needed
            df = df.drop(columns=["year", "month_num"])
            return df
        
        
        df = add_monthly_KSA_holiday_count(df, date_col="date")
        df.head(2)
        
        
        def get_date_based_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
            """
                Generate various date-based features from a date column in a Pandas DataFrame.

                Parameters:
                -----------
                df : pd.DataFrame
                    Input Pandas DataFrame containing the date column from which features will be derived.
                date_column : str
                    Name of the date column in the DataFrame.
                date_formats : list
                    List of possible date formats to try for parsing.

                Returns:
                --------
                pd.DataFrame
                    A new DataFrame with the following additional columns:
                    - 'week': Week number of the year (1-52).
                    - 'year': The year of the date.
                    - 'month': The month of the date.
                    - 'adjusted_year': The year adjusted based on week number (for cross-year week handling).
                    - 'year_month': A string in the format 'YYYYMM' representing the year and month.
                    - 'year_week': A string in the format 'YYYYWW' representing the adjusted year and week number.

                Raises:
                -------
                Exception
                    If none of the date formats match, an exception is raised indicating invalid date formats.
            """
        
            df["date"] = pd.to_datetime(df["date"])
        
            df["month"] = df["date"].dt.month
            df["quarter"] = df["date"].dt.quarter
            df["year"] = df["date"].dt.year
            df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
            df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
            df["days_in_month"] = df["date"].dt.days_in_month
        
            # Optional: Fourier features for seasonality
            df["fourier_year_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["fourier_year_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
            return df
        
        
        df = get_date_based_features(df, "date")
        df.head(2)
        
        # Ensure the column is in datetime format
        df["date"] = pd.to_datetime(df["date"])
        
        # Filter the DataFrame up to and including the train_end_date
        df_train = df[df["date"] <= pd.to_datetime(train_end_date)].copy()
        
        target_mean = df_train["target"].mean()
        print(target_mean)  # 2696.7343260188086
        
        
        def get_lag_features(df: pd.DataFrame, lag_check: List[int], key: str = "key",
                    date_col: str = "date", target: str = "target", target_mean: float = target_mean,) -> pd.DataFrame:
            """
            Generate lag features for the given Pandas DataFrame.

            Parameters:
            df (pd.DataFrame): The input DataFrame.
            lag_check (list of int): A list of integers specifying the lag periods.
            key (str): The column name used as the grouping key.
            date_col (str): The column name used to order within each group.
            target (str): The name of the target column to lag.

            Returns:
            pd.DataFrame: DataFrame with new lag feature columns.
            """
            df = df.copy()
            df.sort_values(by=[key, date_col], inplace=True)

            # mean_target = target_mean

            for lag_num in lag_check:
                df[f"Lag{lag_num}_y"] = df.groupby(key)[target].shift(lag_num).fillna(target_mean)

            return df


        df = get_lag_features(df, [12], "key", "date", "target", target_mean)
        df.head(2)
        
        
        def get_moving_stats_features(df: pd.DataFrame, months_back: List[int], key: str = "key",
                    date_col: str = "date", target: str = "target",) -> pd.DataFrame:
            """
                Generate past-only moving average (MA), rolling std (STD), and exponential moving average (EMA)
                features for monthly time-series data.

                Parameters:
                - df (pd.DataFrame): Input DataFrame containing at least the key, date, and target columns.
                - months_back (List[int]): List of window sizes (in months) for calculating moving statistics.
                - key (str): Column name identifying the entity (e.g., product or store).
                - date_col (str): Column name for the date (must be monthly datetime or convertible).
                - target (str): Column name for the value to compute statistics on.

                Returns:
                - pd.DataFrame: Original DataFrame with added moving statistical features.
            """

            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df.sort_values(by=[key, date_col], inplace=True)

            for window in months_back:
                ma_col = f"MA{window}_y"
                std_col = f"STD{window}_y"
                ema_col = f"EMA{window}_y"

                # Use shifted target to avoid peeking into the current period
                shifted = df.groupby(key)[target].shift(1)

                # Moving Average
                df[ma_col] = (
                    shifted.groupby(df[key])
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

                # Rolling Standard Deviation
                df[std_col] = (
                    shifted.groupby(df[key])
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )

                # Exponential Moving Average
                df[ema_col] = shifted.groupby(df[key]).transform(
                    lambda x: x.ewm(span=window, adjust=False).mean()
                )

            return df


        df = get_moving_stats_features(df, months_back=[6])
        df.head(2)
        
        for prefix in ["EMA", "MA", "STD"]:
            cols = [col for col in df.columns if col.startswith(prefix) and col.endswith("_y")]
            df[cols] = df[cols].fillna(0)
        
        
        def get_monthly_seasonality_index(df: pd.DataFrame, date_col: str = "date", target: str = "target") -> pd.DataFrame:
            """
                Computes a monthly seasonality index using only past data (up to previous month).

                Parameters:
                - df (pd.DataFrame): Input DataFrame with at least [date_col, target].
                - date_col (str): Name of the datetime column.
                - target (str): Name of the target column (e.g., sales).

                Returns:
                - pd.DataFrame: Original DataFrame with an added 'Seasonality_Index' column.
            """
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df.sort_values(by=date_col, inplace=True)

            # Extract month and year
            df["month"] = df[date_col].dt.month
            df["year"] = df[date_col].dt.year

            # Shift target so current month doesn't influence its own index
            df["target_prev"] = df.groupby("month")[target].shift(1)

            # Compute monthly average of past values
            monthly_avg = df.groupby("month")["target_prev"].mean().rename("monthly_avg")

            # Normalize to get a seasonality index
            seasonality_index = (monthly_avg / monthly_avg.mean()).round(2)
            seasonality_index.name = "Seasonality_Index"

            # Map back to original df
            df = df.merge(seasonality_index, on="month", how="left")

            # Fill missing (e.g. first year) with neutral seasonality
            df["Seasonality_Index"] = df["Seasonality_Index"].fillna(1.0)

            return df.drop(columns=["target_prev"])


        df = get_monthly_seasonality_index(df, date_col="date", target="target")
        df.head(2)
        
        
        def create_peak_calendar(start_date=train_start_date, end_date=train_end_date):
        
            calendar = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date)})
            calendar["month"] = calendar["date"].dt.month
            calendar["year"] = calendar["date"].dt.year
            calendar["year_month"] = calendar["date"].dt.to_period("M")
        
            # Peak months
            peak_months = [1, 7, 12]
            calendar["peak_flag"] = calendar["month"].isin(peak_months)
        
            # Ramadan and Eid logic
            ramadan_ranges = [
                ("2023-03-23", "2023-04-20"),
                ("2024-03-10", "2024-04-08"),
                ("2025-02-28", "2025-03-29"),
                ("2026-02-17", "2026-03-19"),
            ]
            eid_fitr = pd.to_datetime(
                ["2023-04-21", "2024-04-10", "2025-03-30", "2026-03-19", "2026-03-20"]
            )
            eid_adha = pd.to_datetime(
                ["2023-06-28", "2024-06-16", "2025-06-06", "2026-05-26", "2026-05-27"]
            )
        
            # Add festive_peak_flag (only Ramadan + Eid)
            calendar["festive_peak_flag"] = False
            for start, end in ramadan_ranges:
                calendar.loc[
                    (calendar["date"] >= start) & (calendar["date"] <= end),
                    "festive_peak_flag",
                ] = True
            calendar.loc[
                calendar["date"].isin(eid_fitr) | calendar["date"].isin(eid_adha),
                "festive_peak_flag",
            ] = True
        
            # Also add Ramadan + Eid to peak_flag
            for start, end in ramadan_ranges:
                calendar.loc[
                    (calendar["date"] >= start) & (calendar["date"] <= end),
                    "peak_flag",
                ] = True
            calendar.loc[
                calendar["date"].isin(eid_fitr) | calendar["date"].isin(eid_adha),
                "peak_flag",
            ] = True
        
            # National Day
            calendar.loc[
                (calendar["month"] == 9)
                & (calendar["date"].dt.day == 23)
                & (calendar["year"].isin([2023, 2024, 2025])),
                "peak_flag",
            ] = True
        
            # Black Friday / Cyber Monday
            calendar.loc[
                (calendar["month"] == 11)
                & (calendar["date"].dt.day.between(20, 30))
                & (calendar["year"].isin([2023, 2024, 2025])),
                "peak_flag",
            ] = True
        
            # Mid-term promotions
            calendar.loc[
                ((calendar["month"] == 2) & (calendar["date"].dt.day.between(10, 16)))
                | ((calendar["month"] == 3) & (calendar["date"].dt.day.between(15, 25))),
                "peak_flag",
            ] = True
        
            # Convert flags to int
            calendar[["peak_flag", "festive_peak_flag"]] = calendar[
                ["peak_flag", "festive_peak_flag"]
            ].astype(int)
        
            return calendar
        
        
        def merge_peak_calendar_info(df, start_date=train_start_date, end_date=train_end_date):
            import pandas as pd
        
            # Generate the calendar with flags
            calendar = create_peak_calendar(start_date=start_date, end_date=end_date)
        
            # Compute monthly peak ratio
            monthly_peak_ratio = (
                calendar.groupby("year_month")["peak_flag"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "peak_days", "count": "days_in_month"})
                .assign(KSA_shopping_peak_ratio=lambda d: d["peak_days"] / d["days_in_month"])
                .reset_index()
            )
        
            # Prepare df for merge
            df = df.copy()
            df["year_month"] = df["date"].dt.to_period("M")
        
            # Drop existing columns if any
            df.drop(columns=["KSA_shopping_peak_ratio"], errors="ignore", inplace=True)
        
            # Merge shopping peak ratio
            df = df.merge(
                monthly_peak_ratio[["year_month", "KSA_shopping_peak_ratio"]],
                on="year_month",
                how="left",
            )
        
            # Merge festive_peak_flag (only for Ramadan and Eid)
            df = df.merge(calendar[["date", "festive_peak_flag"]], on="date", how="left")
        
            # Clean up
            df.drop(columns=["year_month"], inplace=True)
        
            return df
        
        
        df = merge_peak_calendar_info(df)
        df.head(2)
        
        
        def create_weather_season_features(df, date_column):
            """Add a single KSA-specific season feature as a numeric value"""
        
            def assign_season_code(month):
                if month in [12, 1, 2]:
                    return 1  # Winter (Cool season)
                elif month in [3, 4, 5]:
                    return 2  # Spring (Pleasant season)
                elif month in [6, 7, 8, 9]:
                    return 3  # Summer (Hot season)
                elif month in [10, 11]:
                    return 4  # Autumn (Pleasant season)
        
            df["KSA_seasons"] = df[date_column].dt.month.apply(assign_season_code)
            return df
        
        
        df = create_weather_season_features(df, "date")
        df.head(2)
        
        
        def create_monthly_seasonal_features(df, date_column="date", target_col="target", group_key="key"):
            """
            Create seasonal decomposition features for monthly-level forecasting, optionally grouped by a key.
        
            Parameters:
            -----------
            df : pd.DataFrame
                Input DataFrame with monthly time series.
            target_col : str or pd.Series
                Name of the target column to analyze (or a reference to it).
            date_column : str or pd.Series
                Name of the date column.
            group_key : str or list of str, optional
                Column(s) to group by when calculating seasonal patterns. If None, computes over entire dataset.
        
            Returns:
            --------
            pd.DataFrame
                DataFrame with added monthly seasonal features.
            """
            import pandas as pd
        
            # Handle Series inputs
            if isinstance(target_col, pd.Series):
                target_col = target_col.name
            if isinstance(date_column, pd.Series):
                date_column = date_column.name
        
            df_month = df.copy()
        
            # Ensure date is datetime
            if date_column not in df_month.columns:
                raise KeyError(f"Date column '{date_column}' not found.")
            if not pd.api.types.is_datetime64_any_dtype(df_month[date_column]):
                df_month[date_column] = pd.to_datetime(df_month[date_column])
        
            # Extract month & quarter from date
            df_month["month"] = df_month[date_column].dt.month
            df_month["quarter"] = df_month[date_column].dt.quarter
        
            # Define groupings
            if group_key is not None:
                if isinstance(group_key, str):
                    group_key = [group_key]
                monthly_groups = ["month"] + group_key
                quarterly_groups = ["quarter"] + group_key
            else:
                monthly_groups = ["month"]
                quarterly_groups = ["quarter"]
        
            # Shift target so current month doesn't influence its own index
            df_month["monthly_target_prev"] = df_month.groupby(monthly_groups)[target_col].shift(1)
            df_month["quarterly_target_prev"] = df_month.groupby(quarterly_groups)[target_col].shift(1)
        
            # Add seasonal features
            df_month[f"{target_col}_seasonal_monthly"] = df_month.groupby(monthly_groups)[
                "monthly_target_prev"
            ].transform("mean")
            df_month[f"{target_col}_seasonal_quarterly"] = df_month.groupby(quarterly_groups)[
                "quarterly_target_prev"
            ].transform("mean")
        
            df_month.drop(["monthly_target_prev", "quarterly_target_prev"], axis=1, inplace=True)
            return df_month
        
        
        df = create_monthly_seasonal_features(df, "date", "target", "key")
        df.head(2)
        
        
        def add_monthly_derivative_features(df: pd.DataFrame) -> pd.DataFrame:
            """
                Adds monthly-level derivative features:
                - deriv_1_pct: % change in monthly total target vs previous month (based on previous-to-previous)
                - deriv_2_pct: % change in deriv_1_pct vs previous month
                - deriv_3_pct: % change in monthly total target 3 months ago vs previous month
                - deriv_6_pct: % change in monthly total target 6 months ago vs previous month
                - monthly_max_qty, monthly_min_qty: across months per key

                Parameters:
                - df: DataFrame with ['key', 'date', 'target'] (daily data)

                Returns:
                - Monthly aggregated DataFrame with added features
            """
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["year_month"] = df["date"].dt.to_period("M").astype(str)

            # Aggregate daily data to monthly total per key
            monthly = df.groupby(["key", "year_month"])["target"].sum().reset_index()
            monthly.rename(columns={"target": "monthly_target"}, inplace=True)

            # Sort for group-wise operations
            monthly = monthly.sort_values(["key", "year_month"])

            # Lagged targets
            monthly["prev_month_target"] = monthly.groupby("key")["monthly_target"].shift(1)
            monthly["prev_to_prev_month_target"] = monthly.groupby("key")["monthly_target"].shift(2)
            monthly["target_3_months_ago"] = monthly.groupby("key")["monthly_target"].shift(4)
            monthly["target_6_months_ago"] = monthly.groupby("key")["monthly_target"].shift(7)

            # Derivatives
            monthly["deriv_1_pct"] = (
                (monthly["prev_month_target"] - monthly["prev_to_prev_month_target"])
                / monthly["prev_to_prev_month_target"]
                * 100
            ).round(2)

            monthly["prev_deriv_1_pct"] = monthly.groupby("key")["deriv_1_pct"].shift(1)
            monthly["deriv_2_pct"] = (
                (monthly["deriv_1_pct"] - monthly["prev_deriv_1_pct"]) / monthly["prev_deriv_1_pct"] * 100
            ).round(2)

            monthly["deriv_3_pct"] = (
                (monthly["prev_month_target"] - monthly["target_3_months_ago"])
                / monthly["target_3_months_ago"]
                * 100
            ).round(2)

            monthly["deriv_6_pct"] = (
                (monthly["prev_month_target"] - monthly["target_6_months_ago"])
                / monthly["target_6_months_ago"]
                * 100
            ).round(2)

            # Clean up inf/nan
            for col in ["deriv_1_pct", "deriv_2_pct", "deriv_3_pct", "deriv_6_pct"]:
                monthly[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                monthly[col] = monthly[col].fillna(0)

            # Monthly max/min
            monthly_stats = (
                monthly.groupby("key")["prev_month_target"]
                .agg(monthly_max_qty="max", monthly_min_qty="min")
                .reset_index()
            )

            monthly = monthly.merge(monthly_stats, on="key", how="left")

            # Final cleanup
            monthly.drop(
                columns=[
                    "prev_month_target",
                    "prev_to_prev_month_target",
                    "prev_deriv_1_pct",
                    "target_3_months_ago",
                    "target_6_months_ago",
                ],
                inplace=True,
            )

            # Convert year_month back to datetime
            monthly["year_month"] = pd.to_datetime(monthly["year_month"])

            # Drop original target if not needed
            monthly.drop(columns=["monthly_target"], inplace=True)
            return monthly
        
        
        monthly_df = add_monthly_derivative_features(df)
        monthly_df.head(2)
        
        
        def add_monthly_flags(df: pd.DataFrame) -> pd.DataFrame:
            """
                Adds monthly flags based on derivative features:
                - deriv_1_flag: 1 if abs(deriv_2_pct) > 50 else 0
                - deriv_2_trend_flag:
                    'increasing' if deriv_2_pct > 0,
                    'decreasing' if deriv_2_pct < 0,
                    'fluctuating' otherwise

                Parameters:
                - df: Monthly-level DataFrame with deriv_2_pct

                Returns:
                - DataFrame with added flag columns
            """
            df = df.copy()

            df["deriv_1_flag"] = np.where(df["deriv_2_pct"].abs() > 50, 1, 0)

            df["deriv_2_trend_flag"] = np.select(
                [df["deriv_2_pct"] > 0, df["deriv_2_pct"] < 0],
                ["increasing", "decreasing"],
                default="fluctuating",
            )

            return df

        monthly_df = add_monthly_flags(monthly_df)
        monthly_df.head(2)
        print(monthly_df.deriv_2_trend_flag.unique())
        
        monthly_df = monthly_df.rename(columns={"year_month": "date"})
        
        # Ensure 'date' columns are datetime and represent month start
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
        monthly_df["date"] = pd.to_datetime(monthly_df["date"]).dt.to_period("M").dt.to_timestamp()

        # Merge on ['key', 'date']
        df = df.merge(monthly_df, on=["key", "date"], how="left", suffixes=("", "_monthly"))
        
        
        #### Ramadan features
        
        def add_ramadan_features(df, date_column):
            """Add Ramadan-related features (approximate dates)"""
        
            # Ramadan approximate dates (you should use hijri-converter for exact dates)
            ramadan_periods = {
                2023: (pd.Timestamp("2023-03-22"), pd.Timestamp("2023-04-21")),
                2024: (pd.Timestamp("2024-03-10"), pd.Timestamp("2024-04-09")),
                2025: (pd.Timestamp("2025-02-28"), pd.Timestamp("2025-03-30")),
                2026: (pd.Timestamp("2026-02-17"), pd.Timestamp("2026-03-19")),
            }
        
            df["is_ramadan"] = 0
        
            for year, (start_date, end_date) in ramadan_periods.items():
                year_mask = df[date_column].dt.year == year
                ramadan_mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
                df.loc[year_mask & ramadan_mask, "is_ramadan"] = 1
        
            # Pre and post Ramadan periods
            df["pre_ramadan"] = 0
            df["post_ramadan"] = 0
        
            for year, (start_date, end_date) in ramadan_periods.items():
                year_mask = df[date_column].dt.year == year
        
                # 2 weeks before Ramadan
                pre_ramadan_mask = (df[date_column] >= (start_date - timedelta(days=14))) & (
                    df[date_column] < start_date
                )
                df.loc[year_mask & pre_ramadan_mask, "pre_ramadan"] = 1
        
                # 1 week after Ramadan
                post_ramadan_mask = (df[date_column] > end_date) & (
                    df[date_column] <= (end_date + timedelta(days=7))
                )
                df.loc[year_mask & post_ramadan_mask, "post_ramadan"] = 1
        
            return df
        
        
        # -------------------------------------------X--------------------------------------------------
        
        
        def add_eid_features(df, date_column):
            """Add Eid-related features"""
        
            # Eid Al-Fitr approximate dates (end of Ramadan)
            eid_fitr_dates = {
                2023: [
                    pd.Timestamp("2023-04-21"),
                    pd.Timestamp("2023-04-22"),
                    pd.Timestamp("2023-04-23"),
                ],
                2024: [
                    pd.Timestamp("2024-04-09"),
                    pd.Timestamp("2024-04-10"),
                    pd.Timestamp("2024-04-11"),
                ],
                2025: [
                    pd.Timestamp("2025-03-30"),
                    pd.Timestamp("2025-03-31"),
                    pd.Timestamp("2025-04-01"),
                ],
                2026: [
                    pd.Timestamp("2026-03-19"),
                    pd.Timestamp("2026-03-20"),
                    pd.Timestamp("2026-03-21"),
                ],
            }
        
            # Eid Al-Adha approximate dates
            eid_adha_dates = {
                2023: [
                    pd.Timestamp("2023-06-28"),
                    pd.Timestamp("2023-06-29"),
                    pd.Timestamp("2023-06-30"),
                ],
                2024: [
                    pd.Timestamp("2024-06-16"),
                    pd.Timestamp("2024-06-17"),
                    pd.Timestamp("2024-06-18"),
                ],
                2025: [
                    pd.Timestamp("2025-06-06"),
                    pd.Timestamp("2025-06-07"),
                    pd.Timestamp("2025-06-08"),
                ],
                2026: [
                    pd.Timestamp("2026-05-26"),
                    pd.Timestamp("2026-05-27"),
                    pd.Timestamp("2026-05-28"),
                ],
            }
        
            df["is_eid_fitr"] = 0
            df["is_eid_adha"] = 0
            df["pre_eid_shopping"] = 0
        
            for year in eid_fitr_dates.keys():
                year_mask = df[date_column].dt.year == year
        
                # Eid Al-Fitr
                for eid_date in eid_fitr_dates[year]:
                    df.loc[year_mask & (df[date_column] == eid_date), "is_eid_fitr"] = 1
        
                    # Pre-Eid shopping period (1 week before)
                    pre_eid_mask = (df[date_column] >= (eid_date - timedelta(days=7))) & (
                        df[date_column] < eid_date
                    )
                    df.loc[year_mask & pre_eid_mask, "pre_eid_shopping"] = 1
        
                # Eid Al-Adha
                for eid_date in eid_adha_dates[year]:
                    df.loc[year_mask & (df[date_column] == eid_date), "is_eid_adha"] = 1
        
            return df
        
        
        # -------------------------------------------X--------------------------------------------------
        
        
        def add_shopping_events(df, date_column):
            """Add major shopping events and sales periods for KSA"""
        
            # Saudi National Day Sales (around Sept 23, assume Sept month-long sale)
            df["saudi_national_day_sales"] = (df[date_column].dt.month == 9).astype(int)
        
            # Ramadan shopping spikes (already marked separately, but can flag the full Ramadan month)
            # Optional - if you want an event flag for Ramadan shopping
            df["ramadan_shopping"] = df["ramadan_period"] if "ramadan_period" in df.columns else 0
        
            # Eid shopping periods (pre-eid shopping flags already exist, but you can add full month flags)
            # We'll skip this as you have 'pre_eid_shopping' separately
        
            # End-of-season sales months (assumed March and August here; adjust as needed)
            df["end_of_season_sales"] = df[date_column].dt.month.isin([3, 8]).astype(int)
        
            # Saudi Summer Sales (June, July)
            df["saudi_summer_sales"] = df[date_column].dt.month.isin([6, 7]).astype(int)
        
            # Other potential KSA-specific sales/events can be added here
        
            return df
        
        
        # -------------------------------------------X--------------------------------------------------
        
        
        def add_luxury_seasons_ksa(df, date_column):
            """Add luxury retail specific seasonal features including KSA-specific events"""
        
            # Global Fashion Weeks (Feb/March and September)
            df["fashion_week_season"] = df[date_column].dt.month.isin([2, 3, 9]).astype(int)
        
            # KSA Fashion Weeks (approximate months, adjust if you have exact dates)
            # Riyadh Fashion Week (March)
            df["riyadh_fashion_week"] = (df[date_column].dt.month == 3).astype(int)
            # Jeddah Fashion Week (October)
            df["jeddah_fashion_week"] = (df[date_column].dt.month == 10).astype(int)
        
            # Luxury sales spike during Ramadan and Eid months (usually March/April or Feb/March depending on year)
            # Assume Ramadan months flagged previously, here just add a luxury spike flag for those months
            df["luxury_ramadan_eid_season"] = df[date_column].dt.month.isin([2, 3, 4]).astype(int)
        
            # Saudi National Day month (September) - luxury promos & shopping spike
            df["saudi_national_day_season"] = (df[date_column].dt.month == 9).astype(int)
        
            # Combine all to a single luxury_season flag (optional)
            df["luxury_season"] = df[
                [
                    "fashion_week_season",
                    "riyadh_fashion_week",
                    "jeddah_fashion_week",
                    "luxury_ramadan_eid_season",
                    "saudi_national_day_season",
                ]
            ].max(axis=1)
        
            # You can choose to keep individual columns or just 'luxury_season'
            return df
        
        
        def create_holiday_features(df, date_column):
            """
            Create holiday and KSA-specific retail calendar features.
        
            Adds daily binary flags for:
            - Ramadan (pre, during, post)
            - Eid Al-Fitr / Eid Al-Adha
            - National/public holidays
            - High sales intent periods (Ramadan only: months 3 & 4)
            - Shopping events
            - Luxury weeks (luxury retail)
        
            Returns a DataFrame with all holiday & seasonal flags.
            """
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
        
            # --------------------------
            # KSA National Public Holidays
            df["month_day"] = df[date_column].dt.strftime("%m-%d")
            ksa_national_holidays = [
                "2022-02-22",  # Founding Day (since 2022)
                "2022-09-23",  # National Day
                "2023-02-22",
                "2023-09-23",
                "2024-02-22",
                "2024-09-23",
                "2025-02-22",
                "2025-09-23",
                "2026-02-22",
                "2026-09-23",
            ]
        
            df["national_holiday_flag"] = (
                df[date_column].isin(pd.to_datetime(ksa_national_holidays)).astype(int)
            )
            df.drop(columns="month_day", inplace=True)
        
            # --------------------------
            # Ramadan, Pre-, Post-
            df = add_ramadan_features(df, date_column)
            df["ramadan_period"] = 0
            df.loc[df["pre_ramadan"] == 1, "ramadan_period"] = 1
            df.loc[df["post_ramadan"] == 1, "ramadan_period"] = 1
            df.loc[df["is_ramadan"] == 1, "ramadan_period"] = 1
            df.drop(columns=["pre_ramadan", "post_ramadan", "is_ramadan"], inplace=True)
        
            # --------------------------
            # Eid Holidays
            df = add_eid_features(df, date_column)
        
            # Public Holidays Flag
            df["public_holidays"] = 0
            df.loc[(df["is_eid_fitr"] == 1) | (df["is_eid_adha"] == 1), "public_holidays"] = 1
            df.loc[(df["national_holiday_flag"] == 1), "public_holidays"] = 1
        
            # --------------------------
            # Shopping Events
            df = add_shopping_events(df, date_column)
        
            # --------------------------
            # High Sales Intent Periods for Ramadan ONLY (months 3 & 4)
            df["is_high_sales_intent_flag"] = 0
            ramadan_months = [3, 4]  # March and April
        
            mask_ramadan_high_sales = (df["ramadan_period"] == 1) & (
                df[date_column].dt.month.isin(ramadan_months)
            )
            df.loc[mask_ramadan_high_sales, "is_high_sales_intent_flag"] = 1
        
            # --------------------------
            # Encode Shopping Event Priority
            df["shopping_event_code"] = 0
            event_priority = {
                "end_of_season_sales": 1,
                "saudi_summer_sales": 1,
                "saudi_national_day_sales": 1,
                "pre_eid_shopping": 1,
            }
            for event, code in event_priority.items():
                if event in df.columns:
                    df.loc[df[event] == 1, "shopping_event_code"] = code
            # Drop these columns after encoding
            df.drop(columns=list(event_priority.keys()), inplace=True)
        
            # --------------------------
            # Fashion Weeks and Luxury Seasons
            df = add_luxury_seasons_ksa(df, date_column)
        
            # Cleanup
            df.drop(
                columns=["is_eid_fitr", "is_eid_adha", "national_holiday_flag"],
                inplace=True,
            )
        
            print("Holiday & KSA retail features created")
            return df
        
        def generate_ksa_holiday_monthly_summary(start_date, end_date):
            """
                Generate simplified monthly aggregated calendar features for KSA retail focus.

                Returns:
                    pd.DataFrame: Aggregated features with minimal key indicators.
            """
            # Create daily calendar
            calendar = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date)})
            calendar["year_month"] = calendar["date"].dt.to_period("M")
        
            # Create retail + holiday calendar flags
            df = create_holiday_features(calendar, "date")
        
            # Combine fashion and luxury into one flag
            df["fashion_luxury_flag"] = df[["fashion_week_season", "luxury_season"]].max(axis=1)
        
            # Aggregate
            df_grp = (
                df.groupby("year_month")
                .agg(
                    ramadan_days=("ramadan_period", "sum"),
                    public_holiday_days=("public_holidays", "sum"),
                    fashion_luxury_days=("fashion_luxury_flag", "sum"),
                )
                .reset_index()
            )
        
            # Add count of days per month and a date column
            df_grp["count_of_days"] = df_grp["year_month"].apply(lambda x: x.days_in_month)
            df_grp["date"] = df_grp["year_month"].dt.to_timestamp()
        
            # Add high_sales_days as same as ramadan_days
            # df_grp['high_sales_days'] = df_grp['ramadan_days']
        
            # Reorder columns
            df_grp = df_grp[
                [
                    "date",
                    "count_of_days",
                    "ramadan_days",
                    "public_holiday_days",
                    "fashion_luxury_days",
                ]
            ]
        
            return df_grp
        
        monthly_ksa_features = generate_ksa_holiday_monthly_summary(train_start_date, inference_end_date)
        print(monthly_ksa_features.head())
        
        
        def merge_holiday_features(business_df, holiday_features_df, date_col="date"):
            """
            Merge business data with monthly KSA holiday & retail calendar features.
        
            Args:
                business_df (pd.DataFrame): Data with at least a 'date' column.
                holiday_features_df (pd.DataFrame): Monthly aggregated calendar features.
                date_col (str): Column name for the date in business_df.
        
            Returns:
                pd.DataFrame: Merged dataframe with holiday features.
            """
            # Ensure datetime
            business_df = business_df.copy()
            business_df[date_col] = pd.to_datetime(business_df[date_col])
            holiday_features_df = holiday_features_df.copy()
            holiday_features_df["date"] = pd.to_datetime(holiday_features_df["date"])
        
            # Merge on date
            merged = business_df.merge(holiday_features_df, on="date", how="left")
        
            # Fill missing holiday feature columns
            feature_cols = [
                "ramadan_days",
                "public_holiday_days",
                "fashion_luxury_days",
                "count_of_days",
            ]
            for col in feature_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0)
        
            return merged
        
        # Assuming df contains your business KPIs and has a 'date' column
        merged_df = merge_holiday_features(df, monthly_ksa_features, date_col="date")
        merged_df.head(2)
        
        target_value = {config['target']['value']}
        
        if target_value == "sales":
            merged_df.drop(columns=["count_of_days", "units_quantity", "num_transactions"], inplace=True)
            logger.info(f"Target Value is : {target_value}. columns dropped units_quantity and num_transactions")
            logger.info(f"Remaining columns: {merged_df.columns.tolist()}")
        elif target_value == "units":
            merged_df.drop(columns=["count_of_days", "sales_amount", "num_transactions"], inplace=True)
            logger.info(f"Target Value is : {target_value}. columns dropped sales_amount and num_transactions")
            logger.info(f"Remaining columns: {merged_df.columns.tolist()}")
        elif target_value == "transactions":
            merged_df.drop(columns=["count_of_days", "units_quantity", "sales_amount"], inplace=True)
            logger.info(f"Target Value is : {target_value}. columns dropped units_quantity and sales_amount")
            logger.info(f"Remaining columns: {merged_df.columns.tolist()}")
        else:
            logger.warning(f"Unknown target value: '{target_value}'. No columns dropped.")
            logger.info(f"Remaining columns: {merged_df.columns.tolist()}")

        
        # Construct full dynamic path
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_output}"
        print(full_path)
        
        # Save the DataFrame
        # merged_df.to_parquet(full_path, index=False)
        
        table_temporal_features = f"{config['feature_engineering_outputs']['temporal_features']}"
        save_to_bq_and_set_artifact(merged_df, f"{table_temporal_features}_{config['target']['value']}", artifact_temporal_features)
        
        print(merged_df.columns)
        
        '''
            3. Store Features
        '''
        file_output = "store_features.parquet"
        
        # Define list of store IDs
        faces_store_ids = [
            "3024",
            "3025",
            "3026",
            "3029",
            "3031",
            "3032",
            "3033",
            "3034",
            "3037",
            "3040",
            "3043",
            "3044",
            "3046",
            "3048",
            "3049",
            "3050",
            "3051",
            "3052",
            "3053",
            "3054",
            "3059",
            "3060",
            "3066",
            "3067",
            "3068",
            "3069",
            "3072",
            "3073",
            "3074",
            "3076",
            "3077",
            "3092",
            "3122",
            "3123",
            "3135",
            "3136",
            "3139",
            "3142",
            "3144",
            "3145",
            "3146",
            "3152",
            "3157",
            "3158",
            "3159",
            "3177",
        ]
        
        # Convert list to comma-separated string for SQL IN clause
        faces_store_ids_str = ", ".join(str(sid) for sid in faces_store_ids)
        
        # client = bigquery.Client()
        
        # Define the SQL query
        query_stores = f"""
        SELECT *
        FROM `{config['data_sources']['store_table']}`
        WHERE store IN ({faces_store_ids_str}) 
        """
        
        # Run query and convert to DataFrame
        query_job_stores = client.query(query_stores)
        stores_df = query_job_stores.to_dataframe(create_bqstorage_client=False)
        
        stores_df.channel.unique()
        
        stores_df.is_ecom_fullfillment_loc.unique()
        
        stores_df.store_format.unique()
        
        recommended_features = [
            "store",
            "city",
            "is_ecom_fullfillment_loc",
            "channel",
            "total_square_ft",
            "selling_square_ft",
            "store_format",
        ]
        
        stores_df = stores_df[recommended_features].fillna(0)
        print(f"selected features:\n{recommended_features}")
        quant_features = ["total_square_ft", "selling_square_ft"]
        qualit_features = recommended_features.copy()
        [qualit_features.remove(i) for i in quant_features if i in qualit_features]
        for col in quant_features:
            stores_df[col] = stores_df[col].astype(float)
        
        print(f"quantitave features:\n{quant_features}")
        
        print(f"qualitative features:\n{qualit_features}")
        
        stores_df["total_square_ft_cat"] = pd.qcut(stores_df["total_square_ft"], q=4, labels=[1, 2, 3, 4])
        stores_df["selling_square_ft_cat"] = pd.qcut(
            stores_df["selling_square_ft"], q=4, labels=[1, 2, 3, 4]
        )
        
        stores_df.city.unique()
        
        stores_df
        
        # Construct full dynamic path
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_output}"
        print(full_path)
        
        # stores_df.to_parquet(full_path, index=False)
        
        table_store_features = f"{config['feature_engineering_outputs']['store_features']}"

        save_to_bq_and_set_artifact(stores_df, f"{table_store_features}_{config['target']['value']}", artifact_store_features)
        
        '''
            4. Online Marketing Features
        '''
        marketing_feature_start_dt = "2023-03-01"
        file_output = "online_marketing_features.parquet"
        
        # client = bigquery.Client()
        
        def fetch_bigquery_data(
            project_id: str = "chb-svc-tredence-d001",
            dataset_table: str = "shared_analytics_prod.mkting_brands_union",
            brand: str = "FACES",
            start_date: str = marketing_feature_start_dt,
            end_date: str = None,
            limit: int = None,) -> pd.DataFrame:
            """
                Fetches marketing data from BigQuery for a specific brand and date range.

                Args:
                    project_id (str): GCP project ID. Default: 'chb-svc-tredence-d001'
                    dataset_table (str): Dataset and table name. Default: 'shared_analytics_prod.mkting_brands_union'
                    brand (str): Brand to filter. Default: 'FACES'
                    start_date (str): Start date in YYYY-MM-DD format. Default: '2023-03-01'
                    end_date (str): End date in YYYY-MM-DD format. If None, uses current date.
                    limit (int): Maximum number of rows to return. If None, returns all rows.

                Returns:
                    pd.DataFrame: DataFrame containing the query results
            """
            client = bigquery.Client(project=project_id)
        
            query = f"""
                SELECT *
                FROM `{config['data_sources']['online_marketing_table']}`
                WHERE brand = '{brand}'
                AND date >= '{start_date}'
            """
        
            if end_date:
                query += f" AND date <= '{end_date}'"
        
            if limit:
                query += f" LIMIT {limit}"
        
            query_job = client.query(query)
            return query_job.to_dataframe(create_bqstorage_client=False)
        
        
        mmm = fetch_bigquery_data(brand="FACES", start_date=marketing_feature_start_dt)
        
        df = mmm.copy()
        
        df.head(2)
        
        df["date"] = pd.to_datetime(df["date"])
        
        df.date.min(), df.date.max()
        
        def extract_region_from_campaign(
            df: pd.DataFrame,
            campaign_col: str = "campaign_name",
            regions: list = ["KSA"],
            output_col: str = "region",
        ) -> pd.DataFrame:
            """
            Extracts the first matching region code from a campaign name column.
        
            Args:
                df (pd.DataFrame): Input DataFrame.
                campaign_col (str): Column containing campaign names. Default: 'campaign_name'.
                regions (list): List of region codes to match. Default: ['KSA', 'KSA'].
                output_col (str): Name of the output column. Default: 'region'.
        
            Returns:
                pd.DataFrame: DataFrame with the extracted region column.
            """
            # Create regex pattern (case-insensitive)
            pattern = r"(" + "|".join(regions) + r")"
        
            # Extract matches and take the first one (converted to uppercase)
            df[output_col] = (
                df[campaign_col]
                .str.findall(pattern, flags=re.IGNORECASE)
                .apply(lambda x: x[0].upper() if x else None)
            )
        
            return df
        
        
        df = extract_region_from_campaign(df)
        
        df.head(2)
        
        df.business_type.value_counts()
        
        def calculate_monthly_metrics(
            df: pd.DataFrame,
            brand_region_map: dict = {"FACES": "KSA"},
            metrics: list = [
                "total_cost_usd",
                "impressions",
                "clicks",
                "ctr",
                "cpm",
                "cpc",
            ],
        ) -> pd.DataFrame:
            """
            Calculate monthly marketing metrics for specific brand-region combinations,
            including pivoted total_cost_usd by source.
        
            Args:
                df: Input DataFrame containing marketing data
                brand_region_map: Dictionary of {brand: region} pairs to analyze
                metrics: List of metrics to include in aggregation
        
            Returns:
                DataFrame with monthly metrics including total_cost_usd pivoted by source.
            """
            # Filter for specified brand-region pairs
            filter_condition = df.apply(
                lambda row: any(
                    row["brand"] == brand and row["region"] == region
                    for brand, region in brand_region_map.items()
                ),
                axis=1,
            )
            filtered_df = df[filter_condition].copy()
        
            # Ensure date is datetime and create month column
            filtered_df["date"] = pd.to_datetime(filtered_df["date"])
            filtered_df["month"] = filtered_df["date"].dt.to_period("M")
        
            # Base monthly aggregation
            monthly_features = (
                filtered_df.groupby(["month", "brand", "region"])
                .agg(
                    {
                        "total_cost_usd": "sum",
                        "impressions": "sum",
                        "clicks": "sum",
                        "ctr": "mean",
                        "cpm": "mean",
                        "cpc": "mean",
                    }
                )
                .reset_index()
            )
            
            # Rename lowercase metrics to avoid BigQuery schema conflicts
            rename_map = {"ctr": "ctr_l", "cpm": "cpm_l", "cpc": "cpc_l"}
            monthly_features = monthly_features.rename(columns=rename_map)

        
            # Pivot total_cost_usd by source
            cost_by_source = (
                filtered_df.pivot_table(
                    index=["month", "brand", "region"],
                    columns="source",
                    values="total_cost_usd",
                    aggfunc="sum",
                    fill_value=0,
                )
                .add_prefix("total_cost_usd_")
                .reset_index()
            )
        
            # Merge source-specific cost into monthly features
            monthly_features = pd.merge(
                monthly_features,
                cost_by_source,
                on=["month", "brand", "region"],
                how="left",
            )
        
            # Derived metrics
            monthly_features["CPM"] = (
                monthly_features["total_cost_usd"] / monthly_features["impressions"]
            ) * 1000
            monthly_features["CPC"] = monthly_features["total_cost_usd"] / monthly_features["clicks"]
            monthly_features["CTR"] = (monthly_features["clicks"] / monthly_features["impressions"]) * 100
        
            # Column ordering
            column_order = [
                "month",
                "brand",
                "region",
                "total_cost_usd",
                "impressions",
                "clicks",
                "CTR",
                "CPM",
                "CPC",
                "ctr_l",
                "cpm_l",
                "cpc_l",
            ] + [col for col in monthly_features.columns if col.startswith("total_cost_usd_")]
            column_order = [col for col in column_order if col in monthly_features.columns]
        
            return monthly_features[column_order]
        
        
        monthly_features_source = calculate_monthly_metrics(df)
        monthly_features_source.head(2)
        
        monthly_features_source.month.max()
        
        filtered_df = monthly_features_source[monthly_features_source["month"] <= target_month]
        
        filtered_df.month.max()
        
        def extend_monthly_spend_by_sply(df: pd.DataFrame, target_month) -> pd.DataFrame:
            """
            Extend monthly marketing data by copying same months from the previous year
            for each brand-region combination, until October 2026.
        
            Parameters:
            -----------
            df : pd.DataFrame
                Input DataFrame with columns ['month', 'brand', 'region', ...]
                where 'month' is a pandas Period ('M' frequency).
        
            Returns:
            --------
            pd.DataFrame
                DataFrame with original + extended synthetic data.
            """
            df = df.copy()
        
            if not pd.api.types.is_period_dtype(df["month"]):
                df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")
        
            df["is_actual"] = True
            extended_rows = []
        
            target_month = pd.Period(target_month, freq="M")
        
            for (brand, region), group in df.groupby(["brand", "region"]):
                current_df = group.copy()
                current_max_month = current_df["month"].max()
        
                while current_max_month < target_month:
                    months_to_extend = pd.period_range(
                        start=current_max_month + 1,
                        end=min(current_max_month + 12, target_month),
                        freq="M",
                    )
        
                    source_months = months_to_extend - 12
                    source_data = current_df[current_df["month"].isin(source_months)].copy()
        
                    if source_data.empty:
                        break  # No more historical data to copy
        
                    source_data["month"] = source_data["month"] + 12
                    source_data["is_actual"] = False
        
                    extended_rows.append(source_data)
                    current_df = pd.concat([current_df, source_data], ignore_index=True)
                    current_max_month = source_data["month"].max()
        
            df_extended = pd.concat([df] + extended_rows, ignore_index=True) if extended_rows else df
        
            return df_extended.sort_values(["brand", "region", "month"]).reset_index(drop=True)
        
        df_extended = extend_monthly_spend_by_sply(filtered_df, inference_end_date)
        df_extended.tail(13)
        
        df_extended[df_extended["month"].isin(["2026-09", "2025-09"])]
        
        df_extended["date"] = df_extended["month"].dt.to_timestamp()
        df_extended = df_extended.drop(columns="month")
        
        df_extended.date.max()
        
        df_extended.head(2)
        
        # Construct full dynamic path
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_output}"
        print(full_path)
        
        # df_extended.to_parquet(full_path, index=False)

        table_online_marketing_features = f"{config['feature_engineering_outputs']['online_marketing_features']}"

        save_to_bq_and_set_artifact(df_extended, f"{table_online_marketing_features}_{config['target']['value']}", artifact_online_marketing_features)
        
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
            
        eda_features(df_extended, "marketing")
        
        
        '''
            5. Promotional Features
        '''
        file_output = "promotional_features.parquet"
        EffFromDate = "2023-03-01"
        EffUpToDate = "2023-03-30"
        
        # inactive_stores = [3106, 3143, 3173, 4022]
        
        # Define inputs
        faces_storeid = (
            "3024",
            "3025",
            "3026",
            "3029",
            "3031",
            "3032",
            "3033",
            "3034",
            "3037",
            "3040",
            "3043",
            "3044",
            "3046",
            "3048",
            "3049",
            "3050",
            "3051",
            "3052",
            "3053",
            "3054",
            "3059",
            "3060",
            "3066",
            "3067",
            "3068",
            "3069",
            "3072",
            "3073",
            "3074",
            "3076",
            "3077",
            "3092",
            "3122",
            "3123",
            "3135",
            "3136",
            "3139",
            "3142",
            "3144",
            "3145",
            "3146",
            "3152",
            "3157",
            "3158",
            "3159",
            "3177",
        )
        
        # Format store list for SQL IN clause
        faces_storeid_str = ", ".join(str(sid) for sid in faces_storeid)
        
        # BigQuery client
        # client = bigquery.Client()
        
        # Define query with f-string
        query_promo = f"""
        SELECT * FROM `{config['data_sources']['promo_table']}`
        WHERE locationId IN ({faces_storeid_str})
          AND DATE(eligibilityRuleEffFromDate) >= DATE('{EffFromDate}')
          AND DATE(eligibilityRuleEffUpToDate) >= DATE('{EffUpToDate}')
        """
        
        # Run query
        query_job_pro_dis = client.query(query_promo)
        promo_la = query_job_pro_dis.to_dataframe(create_bqstorage_client=False)
        
        promo_la.head(2)
        
        promo_la.strategy.unique()
        
        promo = promo_la[promo_la["strategy"] == "RELATIVE"]
        
        promo.head(2)
        
        # Ensure date columns are in datetime format
        promo["eligibilityRuleEffFromDate"] = pd.to_datetime(promo["eligibilityRuleEffFromDate"])
        promo["eligibilityRuleEffUpToDate"] = pd.to_datetime(promo["eligibilityRuleEffUpToDate"])
        
        # Group by locationId and get min/max dates
        promo.groupby("locationId").agg(
            min_eff_from_date=("eligibilityRuleEffFromDate", "min"),
            max_eff_up_to_date=("eligibilityRuleEffUpToDate", "max"),
        ).reset_index()
        
        promo.shape
        
        promo.eligibilityRuleEffFromDate.min()
        
        promo.eligibilityRuleEffUpToDate.max()
        
        # %%time
        def parse_promo_dates(df: pd.DataFrame) -> pd.DataFrame:
            df["eligibilityRuleEffFromDate"] = pd.to_datetime(df["eligibilityRuleEffFromDate"])
            df["eligibilityRuleEffUpToDate"] = pd.to_datetime(df["eligibilityRuleEffUpToDate"])
            df["promo_length"] = (
                df["eligibilityRuleEffUpToDate"] - df["eligibilityRuleEffFromDate"]
            ).dt.days + 1
            df["discount_efficiency"] = df["amountOfDiscount"].astype(float) / df["promo_length"]
            return df
        
        
        def expand_monthly_records(df: pd.DataFrame) -> pd.DataFrame:
            def expand_months(row):
                return pd.date_range(
                    row["eligibilityRuleEffFromDate"],
                    row["eligibilityRuleEffUpToDate"],
                    freq="MS",
                )
        
            df["months"] = df.apply(expand_months, axis=1)
            exploded = df.explode("months").rename(columns={"months": "date"})
            exploded["date"] = exploded["date"].dt.to_period("M")
            return exploded
        
        
        def expand_promo_days(df: pd.DataFrame) -> pd.DataFrame:
            def expand_days(row):
                return pd.date_range(
                    row["eligibilityRuleEffFromDate"],
                    row["eligibilityRuleEffUpToDate"],
                )
        
            df["promo_days"] = df.apply(expand_days, axis=1)
            exploded = df.explode("promo_days")
            exploded["year_month"] = exploded["promo_days"].dt.to_period("M")
            return exploded
        
        
        def compute_promo_days_per_month(promo_days_df: pd.DataFrame) -> pd.DataFrame:
            return (
                promo_days_df.drop_duplicates(["locationId", "promo_days"])
                .groupby(["locationId", "year_month"])
                .agg(promo_days_in_month=("promo_days", "count"))
                .reset_index()
                .rename(columns={"year_month": "date"})
            )
        
        
        def aggregate_monthly_features(promo_monthly: pd.DataFrame) -> pd.DataFrame:
            return (
                promo_monthly.groupby(["locationId", "date"])
                .agg(
                    unique_items_on_promo=("itemId", "nunique"),
                    distinct_discount_levels=("amountOfDiscount", "nunique"),
                    avg_discount=("amountOfDiscount", "mean"),
                    max_discount=("amountOfDiscount", "max"),
                    min_discount=("amountOfDiscount", "min"),
                    avg_promo_duration=("promo_length", "mean"),
                )
                .reset_index()
            )
        
        
        def calculate_percentage_products_on_promo(
            monthly_df: pd.DataFrame, promo_la: pd.DataFrame
        ) -> pd.DataFrame:
            total_items_available = (
                promo_la.groupby("locationId")["itemId"]
                .nunique()
                .reset_index()
                .rename(columns={"itemId": "total_items_available"})
            )
            merged = pd.merge(monthly_df, total_items_available, on="locationId", how="left")
            merged["percentage_products_on_promo"] = (
                merged["unique_items_on_promo"] / merged["total_items_available"]
            ) * 100
            merged["percentage_products_on_promo"] = merged["percentage_products_on_promo"].round(2)
            return merged.drop(columns=["unique_items_on_promo", "total_items_available"])
        
        
        def monthly_aggregated_promo_features(promo_la: pd.DataFrame) -> pd.DataFrame:
            promo_la = parse_promo_dates(promo_la)
            promo_monthly = expand_monthly_records(promo_la)
            promo_days_expanded = expand_promo_days(promo_la)
            promo_day_counts = compute_promo_days_per_month(promo_days_expanded)
            monthly_agg = aggregate_monthly_features(promo_monthly)
        
            # Merge promo days
            monthly_agg = pd.merge(monthly_agg, promo_day_counts, on=["locationId", "date"], how="left")
            monthly_agg["promo_days_in_month"] = monthly_agg["promo_days_in_month"].fillna(0).astype(int)
        
            # Calculate % of products on promo
            monthly_agg = calculate_percentage_products_on_promo(monthly_agg, promo_la)
        
            # Round selected numeric columns
            for col in [
                "avg_discount",
                "max_discount",
                "min_discount",
                "avg_promo_duration",
            ]:
                try:
                    monthly_agg[col] = monthly_agg[col].fillna(0).astype(float).round(2)
                except:
                    print(monthly_agg[col].apply(lambda x: type(x)).unique())
        
            return monthly_agg
        
        
        monthly_promo_df = monthly_aggregated_promo_features(promo)
        monthly_promo_df.head(2)
        
        monthly_promo_df.tail()
        
        filtered_df = monthly_promo_df[monthly_promo_df["date"] <= pd.Period(target_month)]
        
        filtered_df.date.min()
        
        def extend_promo_until_oct_2026(monthly_promo_df: pd.DataFrame, target_end_date) -> pd.DataFrame:
            """
            Extend monthly promo data until October 2026 using same-month values from the previous year.
            Dates are returned as datetime (1st of each month).
            """
            df = monthly_promo_df.copy()
        
            # Ensure 'date' column is datetime at start of month
            if pd.api.types.is_period_dtype(df["date"]):
                df["date"] = df["date"].dt.to_timestamp(how="start")
            elif pd.api.types.is_datetime64_any_dtype(df["date"]):
                df["date"] = df["date"].values.astype("datetime64[M]")
        
            target_end_date = pd.to_datetime(target_end_date)
            repeated_dfs = []
        
            for loc in df["locationId"].unique():
                loc_df = df[df["locationId"] == loc].copy()
                last_date = loc_df["date"].max()
        
                while last_date < target_end_date:
                    future_start = last_date + pd.DateOffset(months=1)
                    future_end = min(last_date + pd.DateOffset(months=15), target_end_date)
        
                    sply_start = future_start - pd.DateOffset(years=1)
                    sply_end = future_end - pd.DateOffset(years=1)
        
                    sply_data = loc_df[(loc_df["date"] >= sply_start) & (loc_df["date"] <= sply_end)].copy()
        
                    if sply_data.empty:
                        break
        
                    sply_data["date"] = sply_data["date"] + pd.DateOffset(months=12)
                    repeated_dfs.append(sply_data)
                    loc_df = pd.concat([loc_df, sply_data], ignore_index=True)
                    last_date = loc_df["date"].max()
            print(last_date)
            df_extended = pd.concat([df] + repeated_dfs, ignore_index=True)
            df_extended = df_extended.drop_duplicates(subset=["locationId", "date"])
            df_extended = df_extended.sort_values(["locationId", "date"]).reset_index(drop=True)
        
            return df_extended
        
        monthly_df_extended = extend_promo_until_oct_2026(filtered_df, inference_end_date)
        monthly_df_extended.tail(2)
        
        monthly_df_extended["date"].max()
        
        # Ensure 'date' is datetime
        monthly_df_extended["date"] = pd.to_datetime(monthly_df_extended["date"])
        
        # Extract year-month as Period (e.g., 2024-05)
        monthly_df_extended["year_month"] = monthly_df_extended["date"].dt.to_period("M")
        
        # Group and aggregate unique months per location
        available_months = (
            monthly_df_extended.groupby("locationId")["year_month"]
            .unique()
            .reset_index()
            .rename(columns={"year_month": "available_months"})
        )
        
        # Optionally sort months
        available_months["available_months"] = available_months["available_months"].apply(
            lambda x: sorted(x)
        )
        
        # Display
        available_months.head()
        # available_months.to_csv("faces_available_months.csv")
        
        monthly_df_extended.drop("year_month", axis=1, inplace=True)
        
        monthly_df_extended.columns
        
        monthly_df_extended.date.max()
        
        # Construct full dynamic path
        full_path = f"{gcs_path}/{experiment_name}/{granularity}/{target}/{brand}/{target_month}/train_{train_end_date}/{file_output}"
        print(full_path)
        
        # Save the DataFrame
        # monthly_df_extended.to_parquet(full_path, index=False)
        
        table_promotional_features = f"{config['feature_engineering_outputs']['promotional_features']}"
    
        save_to_bq_and_set_artifact(monthly_df_extended, f"{table_promotional_features}_{config['target']['value']}", artifact_promotional_features)
    
        logger.info("Feature Engineering step completed successfully ")
        
    
    except Exception as e:
        logger.exception("Unhandled error occurred during Feature Engineering Component execution.")
        raise
