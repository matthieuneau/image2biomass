import pandas as pd

data = pd.read_csv("data/train.csv")


def extract_labels(data: pd.DataFrame) -> None:
    # Extract 'id' and 'biomass' columns
    labels = data[["sample_id", "target"]]
    labels.to_csv("data/labels.csv", index=False)


if __name__ == "__main__":
    extract_labels(data)
