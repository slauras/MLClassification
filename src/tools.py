import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    classification_report, confusion_matrix, roc_curve, auc
)

def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle de classification binaire et affiche la courbe ROC avec les thresholds annotés.

    Parameters:
    - model : Modèle entraîné (ex: LogisticRegression)
    - X_test : Features de test
    - y_test : Labels de test
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe 1

    # Métriques de régression (indicatives pour évaluation globale)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"🔹 R² Score: {r2:.4f}")
    print(f"🔹 MAE: {mae:.4f}")
    print(f"🔹 MSE: {mse:.4f}\n")

    # Rapport de classification
    print("📊 Rapport de Classification:\n", classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    plt.xlabel("Prédictions")
    plt.ylabel("Vraie Classe")
    plt.title("Matrice de Confusion")
    plt.show()

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)  # Diagonale aléatoire

    # Ajouter quelques thresholds sur la courbe
    for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):  # Afficher environ 10 thresholds
        plt.annotate(f"{thresholds[i]:.2f}", (fpr[i], tpr[i]), fontsize=8, color="red")

    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.title("Courbe ROC avec Thresholds")
    plt.legend()
    plt.grid()
    plt.show()
