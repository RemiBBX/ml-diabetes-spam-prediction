import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import pearsonr


# Fonction 1 : SHAP pour Random Forest
def explain_random_forest(model, X_train, X_test, feature_names, max_samples=1000):
    """
    Analyse SHAP pour un modèle Random Forest
    """
    print(f"=== Analyse SHAP pour Random Forest ===")

    # Limiter le nombre de samples si nécessaire
    if len(X_test) > max_samples:
        print(f"Limitation à {max_samples} samples (sur {len(X_test)}) pour la vitesse...")
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_test = X_test[indices]

    # 1. Créer le TreeExplainer
    print(f"Création du TreeExplainer pour Random Forest...")
    explainer = shap.TreeExplainer(model)
    print(f"Expected value (baseline): {explainer.expected_value}")

    # 2. Calculer les SHAP values sur le test set
    print(f"Calcul des SHAP values sur {len(X_test)} samples...")
    shap_values = explainer.shap_values(X_test)

    # Pour la classification binaire, shap_values peut être une liste
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Classe positive
        print("Classification binaire détectée → utilisation classe positive")

    # 3. Summary plot (importance des features)
    print("Génération du summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("Random Forest - Feature Importance (SHAP)")
    plt.tight_layout()
    plt.show()

    # 4. Summary plot détaillé (avec distribution des valeurs)
    print("Génération du summary plot détaillé...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="dot", show=False)
    plt.title("Random Forest - Impact des features (valeurs hautes en rouge)")
    plt.tight_layout()
    plt.show()

    print("Analyse SHAP Random Forest terminée !")
    return shap_values, explainer


# Fonction 2 : SHAP pour MLP PyTorch
def explain_mlp(model, X_train, X_test, feature_names, n_background=100, n_test_samples=200):
    """Analyse SHAP pour un MLP PyTorch"""
    print(f"=== Analyse SHAP pour MLP PyTorch ===")

    # Mode évaluation (désactive dropout, batch norm, etc.)
    model.eval()
    print("Modèle mis en mode évaluation")

    # Wrapper pour retourner des probabilités (pas des logits)
    def model_predict_proba(X):
        """Wrapper qui transforme numpy → tensor → proba → numpy"""
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits)
        return probs.numpy().flatten()  # Flatten pour avoir shape (n,) au lieu de (n,1)

    print("Wrapper de prédiction créé")

    # Background dataset (NUMPY, pas tensor !)
    print(f"Sélection de {n_background} samples pour le background dataset...")
    indices = np.random.choice(len(X_train), n_background, replace=False)
    background = X_train[indices]
    print(f"Background dataset shape: {background.shape}")

    # Créer le KernelExplainer
    print("Création du KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict_proba, background)
    print("KernelExplainer créé avec succès")

    # Limiter le nombre de samples pour la vitesse
    if len(X_test) > n_test_samples:
        print(f"Limitation à {n_test_samples} samples (sur {len(X_test)}) pour la vitesse...")
        test_indices = np.random.choice(len(X_test), n_test_samples, replace=False)
        X_test_sample = X_test[test_indices]
    else:
        X_test_sample = X_test

    # Calculer les SHAP values (NUMPY, pas tensor !)
    print(f"Calcul des SHAP values sur {len(X_test_sample)} samples...")
    print("(Cela peut prendre 2-3 minutes...)")
    shap_values = explainer.shap_values(X_test_sample)  # ← Direct en numpy

    # KernelExplainer peut retourner une liste pour la classification binaire
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # KernelExplainer retourne souvent [0] pour binaire
        print("Classification binaire détectée")

    print(f"SHAP values calculées ! Shape: {shap_values.shape}")

    # Summary plot (importance des features)
    print("Génération du summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.title("MLP PyTorch - Feature Importance (SHAP)")
    plt.tight_layout()
    plt.show()

    # Summary plot détaillé
    print("Génération du summary plot détaillé...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      plot_type="dot", show=False)
    plt.title("MLP PyTorch - Impact des features (valeurs hautes en rouge)")
    plt.tight_layout()
    plt.show()

    print("Analyse SHAP MLP terminée !")
    return shap_values, explainer


# Fonction 3 : Comparaison des deux
def compare_models_shap(rf_shap_values, mlp_shap_values, X_test_rf, X_test_mlp, feature_names):
    print("=== Comparaison Random Forest vs MLP ===")

    # Calculer l'importance moyenne de chaque feature
    rf_importance = np.abs(rf_shap_values).mean(axis=0)
    mlp_importance = np.abs(mlp_shap_values).mean(axis=0)

    print(f"RF importance shape: {rf_importance.shape}")
    print(f"MLP importance shape: {mlp_importance.shape}")

    # Créer un DataFrame pour faciliter la comparaison
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Random Forest': rf_importance,
        'MLP': mlp_importance
    })

    # Trier par importance Random Forest
    comparison_df = comparison_df.sort_values('Random Forest', ascending=False)

    print("\nTop 5 features - Random Forest:")
    print(comparison_df.head())

    # Graphique de comparaison côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Random Forest
    axes[0].barh(comparison_df['Feature'], comparison_df['Random Forest'], color='forestgreen')
    axes[0].set_xlabel('Importance SHAP moyenne')
    axes[0].set_title('Random Forest - Feature Importance')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # MLP
    axes[1].barh(comparison_df['Feature'], comparison_df['MLP'], color='steelblue')
    axes[1].set_xlabel('Importance SHAP moyenne')
    axes[1].set_title('MLP PyTorch - Feature Importance')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Graphique de corrélation
    plt.figure(figsize=(8, 6))
    plt.scatter(comparison_df['Random Forest'], comparison_df['MLP'], alpha=0.6, s=100)

    for idx, row in comparison_df.iterrows():
        plt.annotate(row['Feature'],
                     (row['Random Forest'], row['MLP']),
                     fontsize=8, alpha=0.7)

    max_val = max(comparison_df['Random Forest'].max(), comparison_df['MLP'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Accord parfait')

    plt.xlabel('Importance SHAP - Random Forest')
    plt.ylabel('Importance SHAP - MLP')
    plt.title('Corrélation des importances entre RF et MLP')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Calculer la corrélation de Pearson
    from scipy.stats import pearsonr

    corr, p_value = pearsonr(comparison_df['Random Forest'], comparison_df['MLP'])
    print(f"\nCorrélation de Pearson entre RF et MLP: {corr:.3f} (p-value: {p_value:.4f})")

    if corr > 0.7:
        print("→ Les deux modèles se basent sur des features similaires")
    elif corr > 0.4:
        print("→ Les modèles ont des stratégies partiellement différentes")
    else:
        print("→ Les modèles utilisent des stratégies très différentes")

    return comparison_df


# Fonction 4 : Analyse détaillée (waterfalls, dependence plots)
def detailed_analysis(shap_values, X_test, feature_names, model_name):
    """Analyse approfondie avec différents plots"""
    pass
