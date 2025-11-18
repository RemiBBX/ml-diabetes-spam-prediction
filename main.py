from src.nn_interface import MLPModel
from src.preprocessing import preprocessing, data_spam, data_diabetes
from sklearn.metrics import confusion_matrix, classification_report

def main() :
    samples = preprocessing(data=data_diabetes, test_size=0.15, validation_size=0.15)
    model = MLPModel()
    model.train(samples)
    y_pred = samples.y_test,model.predict()
    model.benchmark(y_pred, samples.y_test)

if __name__ == "__main__":
    main()

