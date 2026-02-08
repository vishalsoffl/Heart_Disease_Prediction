import pandas as pd

def load_data(path="data/raw/heart.csv"):
    df = pd.read_csv(path)
    return df