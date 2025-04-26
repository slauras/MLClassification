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


def plot_loss(data_proc_lst, model_name):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))  # 1 ligne, 3 colonnes
    
    for i, data_proc in enumerate(data_proc_lst):
        evals_result = data_proc.evals_results[model_name]
        
        try:
            # Pour XGBoost
            epochs = len(evals_result['train']['logloss'])
            df_loss = pd.DataFrame({
                'Epoch': list(range(epochs)),
                'Train': evals_result['train']['logloss'],
                'Validation': evals_result['validation']['logloss']
            })
        except KeyError:
            try:
                # Pour CatBoost
                epochs = len(evals_result['learn']['Logloss'])
                df_loss = pd.DataFrame({
                    'Epoch': list(range(epochs)),
                    'Train': evals_result['learn']['Logloss'],
                    'Validation': evals_result['validation']['Logloss']
                })
            except KeyError:
                raise ValueError(f"Unsupported evals_result format for model {model_name}")

        # Format long pour seaborn
        df_loss_melted = df_loss.melt(id_vars='Epoch', value_vars=['Train', 'Validation'], var_name='set', value_name='logloss')

        # Tracer dans le subplot correspondant
        sns.lineplot(data=df_loss_melted, x='Epoch', y='logloss', hue='set', ax=ax[i])
        ax[i].set_title(f'Loss pendant le training - [{data_proc.status}]', fontsize=14)
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Logloss')
        ax[i].grid(True)

    plt.tight_layout()
    plt.show()
    

def plot_importance_xgb(data_proc):

    _, ax = plt.subplots(1, 2, figsize=(12, 6)) 

    booster = data_proc.models["xgboost"]
    importance_dict = booster.get_score(importance_type='gain')  # importance par 'gain'
    feature_names = data_proc.feature_names

    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in range(len(feature_names))],
        'Importance': [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
    })

    importance_df['innit_feature'] = importance_df['Feature'].apply(lambda x: x.split('_')[0])

    # 2. Grouper par "variable_initiale" et calculer les stats
    df_grouped = importance_df.groupby('innit_feature')['Importance'].agg(['mean', 'median', 'max']).reset_index()


    importance_mean_df = df_grouped.sort_values(by="mean", ascending=False).head(30)
    sns.barplot(x='mean', y='innit_feature', data=importance_mean_df, hue='innit_feature', palette='viridis', ax=ax[0])
    ax[0].set_title('Importance des Features moyennes (XGBoost)', fontsize=16)
    ax[0].set_xlabel('Gain moyen', fontsize=14)
    ax[0].set_ylabel('Feature', fontsize=14)

    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(30)
    sns.barplot(x='Importance', y='Feature', data=importance_df, hue='innit_feature', palette='husl', ax=ax[1])
    ax[1].set_title('Importance des modalitées (OHE) (XGBoost)', fontsize=16)
    ax[1].set_xlabel('Gain', fontsize=14)
    ax[1].set_ylabel('Modalité', fontsize=14)

    # importance_median_df = df_grouped.sort_values(by="median", ascending=False).head(30)
    # sns.barplot(x='median', y='innit_feature', data=importance_median_df, hue='innit_feature', palette='viridis', ax=ax[0])
    # ax[0].set_title('Importance des Features max (XGBoost)', fontsize=16)
    # ax[0].set_xlabel('Gain moyen', fontsize=14)
    # ax[0].set_ylabel('Feature', fontsize=14)

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


