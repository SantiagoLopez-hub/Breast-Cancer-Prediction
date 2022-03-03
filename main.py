from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


def classifier():
    # Loading dataset
    data = load_breast_cancer()

    # Set training 80% and testing 20% of data
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2
    )

    # Create a classification model
    model = RandomForestClassifier()
