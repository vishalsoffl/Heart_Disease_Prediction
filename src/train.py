from sklearn.linear_model import LogisticRegression
from src.data_loader import load_data
from src.preprocess import preprocess_data

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test
