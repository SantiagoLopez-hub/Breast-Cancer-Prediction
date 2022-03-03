from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy


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

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    model_predictions = model.predict(X_test)

    # Set sizes
    plt.figure(figsize=(10, 5))

    # First plot (Training data)
    plt.subplot(121)
    plt.title('Training data')
    plt.plot(X_train)

    # Second plot (Testing data)
    plt.subplot(122)
    plt.title('Testing data')
    plt.plot(X_test)

    # Display accuracy
    accuracy = numpy.mean((model_predictions == y_test) * 100).round(2).item()
    plt.suptitle('Breast Cancer Predictions (Classification)\nAccuracy: ' +
                 repr(accuracy) +
                 '%')
    plt.show()


if __name__ == '__main__':
    classifier()
