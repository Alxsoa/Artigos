# =====================================================================
# Bibliotecas Necessárias
# =====================================================================
import optuna
from optuna.visualization import plot_optimization_history
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd

# =====================================================================
# Criação do dataset sintético
# =====================================================================
X, y = make_classification (
        n_samples=10000, 
        n_features=20, 
        n_informative=15,
        n_redundant=3, 
        n_classes=3, 
        flip_y=0.1, 
        random_state=42
    )

# =====================================================================
# Definindo a Função objetivo
# =====================================================================
def objective(trial):
    params = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'num_class': 3,
        'n_estimators': 200   
    }
    
    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0)
    
    # Validação cruzada sem early stopping
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    return np.mean(scores)

# =====================================================================
# Executando a otimização e armazenando os resultados completos
# =====================================================================
def run_optimization(n_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Armazenar resultados completos
    results = {
        'best_accuracy': study.best_value,
        'best_params': study.best_params,
        'all_trials': study.trials_dataframe(),
        'optimization_history': plot_optimization_history(study)
    }
    return results

# =====================================================================
# Configurações da otimização
# =====================================================================
trials_counts = [10, 30 , 50, 100, 150]
final_results = {}

# =====================================================================
# Configurações para suprimir warnings
# =====================================================================
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
xgb.set_config(verbosity=0)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# Executando as Otimizações
# =====================================================================
print ("# =====================================================================")
print (f"# 🚀 Início das Tentativas de Otimização")
print ("# =====================================================================")
for n_trials in trials_counts:
    print(f"Executando a Otimização com {n_trials} Tentativas")
    results = run_optimization(n_trials)
    final_results[n_trials] = results
    
    # Exibir resultados parciais
    print ("# =====================================================================")
    print (f"# 🏅 Resultados com {n_trials} Tentativas ")
    print ("# =====================================================================")
    print (f"Melhor Acurácia ......: {results['best_accuracy']:.4f}")
    print ( "Melhores Parâmetros ..: ")
    for key, value in results['best_params'].items():
        print(f"{key} = {value}")
    print ("# =====================================================================\n")

    # Salvar resultados em arquivos
    results['all_trials'].to_csv(f'Resultado_Historico_{n_trials}_Tentativas.csv', index=False)
    results['optimization_history'].write_image(f'Resultado_Historico_{n_trials}_Tentativas.png')
    
    print ("# =====================================================================")
    print (f"# 💾 Armazenando Reultados ")
    print ("# =====================================================================")
    print (f"✅ Resultados do Optuna para {n_trials} Tentativas")
    print (f"✅ Histórico das Otimizações para {n_trials} Tentativas")
    print ("# =====================================================================\n")

# =====================================================================
# Análise final comparativa
# =====================================================================
best_overall = max(final_results.items(), key=lambda x: x[1]['best_accuracy'])
print ("# =====================================================================")
print (f"# 🏁 Melhor Resultado ")
print ("# =====================================================================")
print (f"Melhor performance ..........: {best_overall[1]['best_accuracy']:.4f} (com {best_overall[0]} trials)")
print ( "Melhores Parâmetros Globais..:")
for key, value in best_overall[1]['best_params'].items():
    print(f"{key} = {value}")
    print ("# =====================================================================\n")

# =====================================================================
# Gráfico comparativo
# =====================================================================
plt.figure(figsize=(10, 6))
plt.plot(
    list(final_results.keys()),
    [res['best_accuracy'] for res in final_results.values()],
    marker='o',
    linestyle='--'
)
plt.title  ('Evolução da Acurácia')
plt.xlabel ('Número de Tentativas')
plt.ylabel ('Melhor Acurácia')
plt.grid(True)
plt.savefig('melhoria_com_trials.png')
plt.show()

# =====================================================================
# FIM DO PROGRAMA
# =====================================================================

