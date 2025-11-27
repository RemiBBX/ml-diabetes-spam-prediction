import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.model_interface import LearningModelInterface


class RForest(LearningModelInterface):
    def __init__(self):
        self.model = None
        self.best_params_ = None

        self.param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_leaf": [
                1,
                2,
            ],
        }

    def train(self, x, y):
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=0), param_grid=self.param_grid, cv=5, n_jobs=1, verbose=0
        )

        grid.fit(x, np.ravel(y))

        self.best_params_ = grid.best_params_
        self.model = grid.best_estimator_
        print("Best hyperparameters found with Grid Search:", self.best_params_)

    def predict(self, x):
        return self.model.predict(x)

    def explain(self, X_train: np.ndarray, X_test: np.ndarray, feature_names: list, **kwargs):
        """
        Analyse SHAP pour Random Forest

        Args:
            X_train: Données d'entraînement
            X_test: Données de test
            feature_names: Noms des features
            **kwargs:
                - max_samples: Limite de samples à analyser (défaut: 1000)
                - show_plots: Afficher les graphiques (défaut: True)
        """
        import shap
        import matplotlib.pyplot as plt

        max_samples = kwargs.get('max_samples', 1000)
        show_plots = kwargs.get('show_plots', True)

        print(f"=== Analyse SHAP - Random Forest ===")

        # Limiter le nombre de samples
        if len(X_test) > max_samples:
            print(f"Limitation à {max_samples} samples (sur {len(X_test)})...")
            indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_test = X_test[indices]

        # Créer l'explainer
        print("Création du TreeExplainer...")
        explainer = shap.TreeExplainer(self.model)

        # Calculer SHAP values
        print(f"Calcul des SHAP values sur {len(X_test)} samples...")
        shap_values = explainer.shap_values(X_test)

        # Gérer classification binaire
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            print("Classification binaire → classe positive utilisée")

        if show_plots:
            # Summary plot - importance
            print("Génération des graphiques...")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=21, show=False)
            plt.title("Random Forest - Feature Importance (SHAP)")
            plt.tight_layout()
            plt.show()

            # Summary plot détaillé
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                              plot_type="dot", show=False)
            plt.title("Random Forest - Impact des features")
            plt.tight_layout()
            plt.show()

        print("✅ Analyse SHAP Random Forest terminée")
        return shap_values, explainer

    def save_model(self, path: str):
        """Sauvegarde le modèle entraîné (grid.best_estimator_)"""
        if self.model is not None:
            joblib.dump(self.model, path)
            print(f"✅ Modèle Random Forest sauvegardé dans {path}")
        else:
            print("❌ Erreur: Le modèle n'a pas été entraîné. Veuillez appeler train() d'abord.")

    def load_model(self, path: str):
        """Charge un modèle entraîné depuis un fichier"""
        try:
            self.model = joblib.load(path)
            # Récupérer les paramètres pour référence si nécessaire
            self.best_params_ = self.model.get_params()
            print(f"✅ Modèle Random Forest chargé depuis {path}")
            return True
        except FileNotFoundError:
            print(f"❌ Erreur: Fichier non trouvé à {path}")
            return False
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
