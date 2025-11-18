from sklearn.metrics import balanced_accuracy_score, classification_report

from preprocessing import data_diabetes, preprocessing
from src.knn import KNNModel


def main():
    samples = preprocessing(data=data_diabetes, test_size=0.3, validation_size=0.1)
    model = KNNModel(k=3)
    model.train(x=samples.X_train, y=samples.y_train)
    report = classification_report(
        y_pred=model.predict(samples.X_test),
        y_true=samples.y_test,
    )
    acc = balanced_accuracy_score(y_pred=model.predict(samples.X_test), y_true=samples.y_test, adjusted=True)
    print(acc)
    print(report)


if __name__ == "__main__":
    main()
