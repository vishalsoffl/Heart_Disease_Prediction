from src.train import train_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate():
    model, X_test, y_test = train_model()
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))

if __name__ == "__main__":
    evaluate()
