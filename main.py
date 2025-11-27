from src.interpretability import compare_models_shap, explain_random_forest, explain_mlp
from src.kernel_methods import LinearSVC_
from src.knn import KNNModel
from src.nn_interface import MLPModel
from src.preprocessing import data_diabetes, data_spam, preprocessing, load_data
from src.RForest import RForest


def nn():
    samples = preprocessing(data=data_diabetes, test_size=0.15, validation_size=0.15)
    model = MLPModel()
    model.train(samples)
    model.benchmark(samples.X_test, samples.y_test)


def knn():
    samples = preprocessing(data=data_spam, test_size=0.3, validation_size=0.1)
    model = KNNModel()
    model.train(x=samples.X_train, y=samples.y_train)
    model.benchmark(x=samples.X_validation, y=samples.y_validation)


def random_forest():
    samples = preprocessing(data=data_spam, test_size=0.3, validation_size=0.1)
    model = RForest()
    model.train(x=samples.X_train, y=samples.y_train)
    model.benchmark(x=samples.X_validation, y=samples.y_validation)


def linear_svc():
    samples = preprocessing(data=data_diabetes, test_size=0.3, validation_size=0.1)
    model = LinearSVC_()
    model.train(x=samples.X_train, y=samples.y_train)
    model.benchmark(x=samples.X_validation, y=samples.y_validation)


def test_shap_analysis():
    """Test des analyses SHAP sur diabetes"""

    # 1. Preprocessing
    samples = preprocessing(data=data_diabetes, test_size=0.15, validation_size=0.15)

    # 2. Entraîner Random Forest
    print("\n=== Entraînement Random Forest ===")
    #rf_model = RForest()
    #rf_model.train(x=samples.X_train, y=samples.y_train)

    # 3. Entraîner MLP
    print("\n=== Entraînement MLP ===")
    mlp_model = MLPModel(input_size=21, epochs=15)
    mlp_model.train(samples)

    # 4. Récupérer les noms de features
    _, _, df, feature_names = load_data(data_diabetes)
    feature_names.remove("Class")  # Enlever la target


    # 5. SHAP Random Forest
    print("\n" + "=" * 50)
    #rf_shap, rf_explainer = explain_random_forest(
        #model=rf_model.model, X_train=samples.X_train, X_test=samples.X_test, feature_names=feature_names
    #)

    # 6. SHAP MLP
    print("\n" + "=" * 50)
    mlp_shap, mlp_explainer = explain_mlp(
        model=mlp_model.model,
        X_train=samples.X_train,
        X_test=samples.X_test,
        feature_names=feature_names,
        n_background=100,
        n_test_samples=200,
    )

    # 7. Comparaison
    print("\n" + "=" * 50)
    # comparison = compare_models_shap(
    #     rf_shap_values=rf_shap,
    #     mlp_shap_values=mlp_shap,
    #     X_test_rf=samples.X_test,
    #     X_test_mlp=samples.X_test[:200],  # Limité pour MLP
    #     feature_names=feature_names,
    # )
    #
    # return comparison


def test_shap_integrated():
    """Test SHAP avec la méthode explain() intégrée aux modèles"""

    print("\n" + "=" * 60)
    print("ANALYSE SHAP AVEC INTERFACE INTÉGRÉE")
    print("=" * 60)

    # 1. Preprocessing
    print("\n📊 Préparation des données...")
    samples = preprocessing(data=data_diabetes, test_size=0.15, validation_size=0.15)

    # Récupérer les noms de features
    _, _, df, feature_names = load_data(data_diabetes)
    feature_names.remove('Class')
    print(f"Features: {len(feature_names)}")

    # 2. Random Forest
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    print("\n📚 Entraînement...")
    rf_model = RForest()
    rf_model.train(x=samples.X_train, y=samples.y_train)

    print("\n📊 Benchmark...")
    rf_model.benchmark(x=samples.X_test, y=samples.y_test)

    print("\n🔍 Explication SHAP...")
    rf_shap, rf_explainer = rf_model.explain(
        X_train=samples.X_train,
        X_test=samples.X_test,
        feature_names=feature_names,
        max_samples=1000
    )

    # 3. MLP
    print("\n" + "=" * 60)
    print("MLP PYTORCH")
    print("=" * 60)

    print("\n📚 Entraînement...")
    mlp_model = MLPModel(input_size=21, epochs=15)
    mlp_model.train(samples)

    print("\n📊 Benchmark...")
    mlp_model.benchmark(x=samples.X_test, y=samples.y_test)

    print("\n🔍 Explication SHAP...")
    mlp_shap, mlp_explainer = mlp_model.explain(
        X_train=samples.X_train,
        X_test=samples.X_test,
        feature_names=feature_names,
        max_samples=200,
        n_background=100
    )


if __name__ == "__main__":
    # Ancienne version
    # test_shap_analysis()

    # Nouvelle version avec interface intégrée
    test_shap_integrated()
