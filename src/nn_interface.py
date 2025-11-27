from typing import Literal

import numpy as np
import torch.nn
from torch.utils.data import DataLoader, TensorDataset

from src.model_interface import LearningModelInterface
from src.neural_network import MLP_nn, train_model


class MLPModel(LearningModelInterface):
    def __init__(self, input_size: Literal[57, 21], epochs) -> None:
        ## Parameters
        self.input_size = input_size
        self.layer_size = 3
        self.hidden_size = 128

        self.train_size = 0.7
        self.validation_size = 0.15
        self.test_size = 0.15

        self.learning_rate = 0.001
        self.batch_size = 512
        self.epochs = epochs

        self.model = MLP_nn(
            input_size=self.input_size, hidden_size=self.hidden_size, output_size=1, layer_size=self.layer_size
        )

    def train(self, samples) -> None:
        self.X_train = torch.tensor(samples.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(samples.y_train, dtype=torch.float32)
        self.X_val = torch.tensor(samples.X_validation, dtype=torch.float32)
        self.y_val = torch.tensor(samples.y_validation, dtype=torch.float32)
        self.X_test = torch.tensor(samples.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(samples.y_test, dtype=torch.float32)

        self.train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.validation_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.batch_size,
            shuffle=False,
        )
        pos = torch.count_nonzero(self.y_train)
        pos_weight = torch.tensor([len(self.y_train) / pos], dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=self.epochs,
            print_every_epochs=1,
        )

    def predict(self, x) -> np.ndarray:
        X_test = torch.tensor(x, dtype=torch.float32)
        probs = torch.sigmoid(self.model(X_test)).detach().cpu().numpy().flatten()

        t = 0.5
        pred = (probs >= t).astype(int)
        return pred

    def explain(self, X_train: np.ndarray, X_test: np.ndarray, feature_names: list, idx_explain: int, **kwargs):
        """
        Analyse SHAP pour MLP PyTorch

        Args:
            X_train: Données d'entraînement
            X_test: Données de test
            feature_names: Noms des features
            **kwargs:
                - max_samples: Limite de samples à analyser (défaut: 200)
                - n_background: Taille du background dataset (défaut: 100)
                - show_plots: Afficher les graphiques (défaut: True)
                - sample_index: Index de l'échantillon à expliquer localement (défaut: 0) <--- AJOUTÉ
        """
        import matplotlib.pyplot as plt
        import shap

        max_samples = kwargs.get("max_samples", 200)
        n_background = kwargs.get("n_background", 100)
        show_plots = kwargs.get("show_plots", True)
        # 💡 NOUVEAU: Récupérer l'index de l'échantillon pour l'analyse locale
        sample_index = kwargs.get("sample_index", 0)

        print("=== Analyse SHAP - MLP PyTorch ===")

        # Mode évaluation
        self.model.eval()
        print("Modèle en mode évaluation")

        # Wrapper pour les probabilités
        def model_predict_proba(X):
            """Convertit numpy → tensor → proba → numpy"""
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs = torch.sigmoid(logits)
            return probs.numpy().flatten()

        # Background dataset
        print(f"Sélection de {n_background} samples pour le background...")
        indices = np.random.choice(len(X_train), n_background, replace=False)
        background = X_train[indices]

        # Limiter test samples
        if len(X_test) > max_samples:
            print(f"Limitation à {max_samples} samples (sur {len(X_test)})...")
            test_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_test = X_test[test_indices]

            # ⚠️ MISE À JOUR: Si X_test est limité, l'index doit être ajusté ou vérifié
            # On s'assure que l'index n'est pas hors limite pour le nouveau X_test
            if sample_index >= len(X_test):
                sample_index = 0
                print("Index de l'échantillon ajusté à 0 après la limitation des échantillons de test.")

        # KernelExplainer
        print("Création du KernelExplainer...")
        explainer = shap.KernelExplainer(model_predict_proba, background)

        # Calculer SHAP values
        print(f"Calcul des SHAP values sur {len(X_test)} samples...")
        print("(Cela peut prendre 2-3 minutes...)")
        shap_values = explainer.shap_values(X_test)

        # Gérer classification binaire
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            print("Classification binaire détectée")

        print(f"SHAP values calculées ! Shape: {shap_values.shape}")

        if show_plots:
            # Summary plot - importance
            print("Génération des graphiques...")

            # # Summary plot Global (Importance)
            # plt.figure(figsize=(10, 6))
            # shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            # plt.title("MLP PyTorch - Feature Importance (SHAP)")
            # plt.tight_layout()
            # plt.show()

            # Summary plot Détaillé (Impact des features)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="dot", show=False)
            plt.title("MLP PyTorch - Impact des features")
            plt.tight_layout()
            plt.show()

            # WaterFall Plot (Explication Locale)
            print(f"\nGénération du WaterFall Plot pour l'échantillon {sample_index}...")

            # Créer l'objet Explanation pour le WaterFall Plot
            explanation = shap.Explanation(
                values=shap_values[sample_index],
                base_values=explainer.expected_value[idx_explain]
                if isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim > 0
                else explainer.expected_value,
                data=X_test[sample_index],
                feature_names=feature_names,
            )

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                explanation, max_display=10, show=False
            )  # Affiche les 10 features les plus influentes localement
            plt.title(f"MLP PyTorch - Explication Locale pour l'échantillon {sample_index}")
            plt.tight_layout()
            plt.show()

        print("✅ Analyse SHAP MLP terminée")
        return shap_values, explainer
