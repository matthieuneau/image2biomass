import numpy as np
import pandas as pd


def preprocess_biomass_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pivots biomass dataset from long to wide format.
    Ensures each image has exactly one row with 5 target columns.
    """
    # 1. Extract the base image ID from sample_id (removes the suffix like __Dry_Clover_g)
    # This ensures we have a clean ID to group by
    df["image_id"] = df["image_path"].apply(lambda x: x.split("/")[-1].split(".")[0])

    # 2. Define the columns that are constant for a single image (metadata)
    index_cols = [
        "image_id",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]

    # 3. Pivot the table
    # target_name becomes the new columns, target provides the values
    df_wide = df.pivot_table(
        index=index_cols, columns="target_name", values="target"
    ).reset_index()

    # 4. Clean up column names (remove the 'target_name' axis label)
    df_wide.columns.name = None

    target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

    X_train = df_wide[index_cols].copy()  # Get a new object
    y_train = df_wide[["image_id"] + target_cols]

    # 5. Encode Sampling_Date as cyclical features
    X_train["Sampling_Date"] = pd.to_datetime(X_train["Sampling_Date"]).dt.dayofyear
    X_train["date_sin"] = np.sin(2 * np.pi * X_train["Sampling_Date"] / 365)
    X_train["date_cos"] = np.cos(2 * np.pi * X_train["Sampling_Date"] / 365)
    X_train.drop(columns=["Sampling_Date"], inplace=True)

    # 6. One-hot encode State
    X_train = pd.get_dummies(X_train, columns=["State"])

    return X_train, y_train


if __name__ == "__main__":
    X_train, y_train = preprocess_biomass_data(pd.read_csv("./data/train.csv"))
    X_train.to_csv("./data/X_train.csv", index=False)
    y_train.to_csv("./data/y_train.csv", index=False)
