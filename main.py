from preprocessing import data_diabetes, data_spam, preprocessing
from src.knn import KNNModel
from src.RForest import RForest


def knn():
    samples = preprocessing(data=data_spam, test_size=0.3, validation_size=0.1)
    model = KNNModel()
    model.train(x=samples.X_train, y=samples.y_train)
    model.benchmark(x=samples.X_validation, y=samples.y_validation)


def random_forest():
    samples = preprocessing(data=data_diabetes, test_size=0.3, validation_size=0.1)
    model = RForest()
    model.train(x=samples.X_train, y=samples.y_train)
    model.benchmark(x=samples.X_validation, y=samples.y_validation)


if __name__ == "__main__":
    pass
