import functools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, roc_auc_score
)
import xgboost as xgb

def evaluate_model(data_proc, model_name):
    model = data_proc.models[model_name]
    y_true = data_proc.y_valid
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(data_proc.X_valid)
        y_proba = model.predict_proba(data_proc.X_valid)[:, 1]
    else:
        # xgboost
        y_proba = model.predict(xgb.DMatrix(data_proc.X_valid))
        y_pred = (y_proba >= 0.5).astype(int)
    plot_metrics(data_proc.status, y_true, y_pred, y_proba)

def evaluate_multi_model(data_proc_list, model_name):
    y_true = np.concatenate(
        list(map(
            lambda data_proc: data_proc.y_valid, data_proc_list
        ))
    )
    if model_name != "xgboost":
        y_pred = np.concatenate(
            list(map(
                lambda data_proc: data_proc.models[model_name].predict(data_proc.X_valid), data_proc_list
            ))
        )
        y_proba = np.concatenate(
            list(map(
                lambda data_proc: data_proc.models[model_name].predict_proba(data_proc.X_valid)[:, 1], data_proc_list
            ))
        )
    else:
        y_proba = np.concatenate(
            list(map(
                lambda data_proc: data_proc.models[model_name].predict(xgb.DMatrix(data_proc.X_valid)), data_proc_list
            ))
        )
        y_pred = (y_proba >= 0.5).astype(int)
    plot_metrics("All data_procs", y_true, y_pred, y_proba)

def plot_metrics(experience_name, y_true, y_pred, y_proba):
    
    print(f"### Rapport de classification [{experience_name}] : \n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    
    precision = precision[:-1] #if precision.shape[0] > y_pred.shape[0] else precision
    recall = recall[:-1]# if recall.shape[0] > y_pred.shape[0] else recall

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0],
                xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    axs[0].set_xlabel("Prédictions")
    axs[0].set_ylabel("Vraie Classe")
    axs[0].set_title("Matrice de Confusion")

    axs[1].plot(fpr, tpr, color="blue", lw=2, label=f"Roc curve (AUC = {roc_auc:.2f})")
    axs[1].plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):
        axs[1].annotate(f"{thresholds[i]:.2f}", (fpr[i], tpr[i]), fontsize=8, color="red")
    axs[1].set_xlabel("Taux de Faux Positifs (FPR)")
    axs[1].set_ylabel("Taux de Vrais Positifs (TPR)")
    axs[1].set_title("Courbe ROC avec Thresholds")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(pr_thresholds, recall, color='orange', label='Recall')
    axs[2].plot(pr_thresholds, precision, color='blue', label='Precision')
    axs[2].axvline(x=0.5, color='grey', linestyle='--', label='Seuil à 0.5')
    axs[2].set_xlabel("Seuil de confiance")
    axs[2].set_ylabel("Recall ou Precision")
    axs[2].set_title("Courbe Rappel/Confiance")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def summarize_categoricals(df: pd.DataFrame, show_levels=False):
    data = [[
        df[c].unique(), len(df[c].unique()), df[c].isnull().sum(),
        np.round((df[c].isnull().sum() / df.shape[0]) * 100, 1), df.dtypes[c]
    ] for c in df.columns]
    df_temp = pd.DataFrame(data, index=df.columns,
                        columns=['Levels', 'No. of Levels', 'No. of Missing Values', '% empty', 'type'])
    return df_temp.iloc[:, 0 if show_levels else 1:]
