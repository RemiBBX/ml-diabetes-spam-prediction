from src.nn_interface import MLPModel
from src.preprocessing import preprocessing, data_spam, data_diabetes
from sklearn.metrics import confusion_matrix, classification_report

def nn() :
    samples = preprocessing(data=data_diabetes, test_size=0.15, validation_size=0.15)
    model = MLPModel()
    model.train(samples)
    model.benchmark(samples.X_test, samples.y_test)

if __name__ == "__main__":
    pass

