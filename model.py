import os
import copy
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import lightgbm as lgb
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from supplementary import *


root = os.getcwd()
time_stamp = sys.argv[1]
path = os.path.join(root, f'results/{time_stamp}')
data = pd.read_pickle(os.path.join(path, 'general_data_for_training.pkl'))

# Let's redefine the target variables taking into account the EDA results
general_targets = ['BG30', 'BG60', 'BG90', 'BG120', 'BGMax',
                   'AUC60', 'AUC120', 'iAUC60', 'iAUC120']
selected_targets = ['BGMax', 'iAUC120', 'AUC60']

clinical_features = {'meal_type_n', 'gi', 'gl', 'carbo', 'prot', 'fat', 'kcal',
                     'mds', 'kr', 'pv', 'gl_b3h', 'gl_b6h', 'gl_b12h',
                     'carbo_b3h', 'carbo_b6h', 'carbo_b12h', 'prot_b3h',
                     'prot_b6h', 'prot_b12h', 'fat_b3h', 'fat_b6h', 'fat_b12h',
                     'pv_b3h', 'pv_b6h', 'pv_b12h', 'kcal_b3h', 'kcal_b6h',
                     'kcal_b12h', 'prec_meal_gi', 'prec_meal_gl',
                     'prec_meal_carbo', 'prec_meal_prot', 'prec_meal_fat',
                     'prec_meal_pv', 'daytime', 'part_of_day', 'срок_берем1',
                     'ИЦН', 'СПКЯ', 'Группа', 'Срок_анализ_V1', 'CGMS_срок',
                     'Глюкоза_нт_общая', 'прибавка_m1', 'отеки1', 'АД_сист1',
                     'АД_диаст1', 'N_беременностей', 'N_родов',
                     'N_невынашивание', 'N_абортов', 'menses', 'АГ',
                     'СД_у_родственников', 'КОК', 'курение_до_беременности',
                     'курение_во_время', 'возраст', 'рост', 'вес_до_берем',
                     'ИМТ', 'фрукты1', 'молочное_необезжир1', 'сосиски1',
                     'бобовые2', 'овощи2', 'сосиски2', 'подъем2',
                     'rs10830963_MTNR1B_N', 'глю_нт_V1', 'кетоны_V1',
                     'HbA1C_V1', 'инсулин_V1', 'лептин_V1', 'ФР_V1',
                     'БОМК_визит1', 'Хол_V1', 'ТГ_V1', 'ЛПВП_V1', 'ЛПОНП_V1',
                     'ЛПНП_V1', 'АБ_бер_ть', 'diet_before_V1',
                     'Diet_duration_V1'}

micr_features = {'OTU_110', 'OTU_123', 'OTU_124', 'OTU_13', 'OTU_132',
                 'OTU_15', 'OTU_152', 'OTU_153', 'OTU_187', 'OTU_190',
                 'OTU_193', 'OTU_195', 'OTU_196', 'OTU_197', 'OTU_20',
                 'OTU_210', 'OTU_230', 'OTU_239', 'OTU_241', 'OTU_25',
                 'OTU_256', 'OTU_257', 'OTU_305', 'OTU_312', 'OTU_33',
                 'OTU_337', 'OTU_338', 'OTU_34', 'OTU_354', 'OTU_359',
                 'OTU_396', 'OTU_399', 'OTU_402', 'OTU_420', 'OTU_439',
                 'OTU_44', 'OTU_454', 'OTU_459', 'OTU_496', 'OTU_51',
                 'OTU_529', 'OTU_587', 'OTU_609', 'OTU_677', 'OTU_68',
                 'OTU_712', 'OTU_76', 'OTU_77', 'OTU_79', 'OTU_84', 'OTU_98'}

cgms_features = {'iAUCb240', 'iAUCb120', 'iAUCb60', 'BGRiseb240', 'BGRiseb120',
                 'BGRiseb60', 'BGb240', 'BGb120', 'BGb60', 'BGb50', 'BGb40',
                 'BGb30', 'BGb25', 'BGb20', 'BGb15', 'BGb10', 'BGb5', 'BG0'}

feature_group_1 = copy.deepcopy(clinical_features)
feature_group_2 = feature_group_1 | micr_features
feature_group_3 = feature_group_1 | cgms_features
feature_group_4 = feature_group_1 | micr_features | cgms_features

# Setup control over randomnes.
# You can use the same seed we used during training,
# but it is generally recommended to run multiple times
# and use some statistics to aggregate the results.
seed = 8781

# Tracker
history = dict()

# Split the dataset by patient number
def spliter(df: pd.DataFrame, y: str, rng: int = None):
    train_n, test_n = train_test_split(df.index.unique(), test_size=.3, shuffle=True, random_state=rng)
    return train_n.tolist(), test_n.tolist()


# Drop the outliers
# Skip this step if you want to use a dataset of 2,706 meal records.
# Generally speaking, these records are not outliers,
# they are clinical cases that occur quite often in practice,
# but our sample is not sufficiently diverse and large, so they are defined as outliers.
def drop_outliers(df: pd.DataFrame, y: str):
    q1 = df[y].quantile(0.25)
    q3 = df[y].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[y] > lower_bound) & (df[y] < upper_bound)]
# --------------------------------------------------------------------------------------


# Data transformation for training
def transform(df: pd.DataFrame, y: str, targets: list[str], rng, describe=False):
    num_train, num_test = spliter(df, y, rng)
    feature_names = [col for col in df.columns if col not in targets]
    train_data = df.loc[df.index.isin(num_train)]
    test_data = df.loc[df.index.isin(num_test)]
    train_data_clean = drop_outliers(train_data, y)
    if describe:
        describe_data(train_data_clean, path, f'clean_train_{y}')
        train_data_clean.drop(columns=['meal_id', 'ts_meal_diary'], axis=1, inplace=True, errors='ignore')
    test_data_clean = drop_outliers(test_data, y)
    if describe:
        describe_data(test_data_clean, path, f'clean_test_{y}')
        test_data_clean.drop(columns=['meal_id', 'ts_meal_diary'], axis=1, inplace=True, errors='ignore')
    return (
        (train_data_clean.loc[:, train_data_clean.columns.isin(feature_names)].to_numpy(), train_data_clean[y].to_numpy()),
        (test_data_clean.loc[:, test_data_clean.columns.isin(feature_names)].to_numpy(), test_data_clean[y].to_numpy()),
        num_train, num_test, feature_names
    )


def basline_info(y, train_batch, test_batch, num_train, num_test):
    print(f'{y}')
    print(f'Median target value of train: {np.median(train_batch)}')
    print(f'Median target value of test: {np.median(test_batch)}')
    print(f'Train size: {len(num_train)}')
    print(f'Test size: {len(num_test)}')
    print(f'Total number of patients: {len(num_train) + len(num_test)}')
    print(f'Total number of rows: {len(train_batch) + len(test_batch)}')
    intersection = set(num_train) & set(num_test)
    print(f'Intersection: {intersection if intersection else "none"}')
    print(f'Train shape: {train_batch.shape}')
    print(f'Test shape: {test_batch.shape}')
    print('------------------\n')


history['baseline_model'] = {}

print('\n\nBASELINE MODEL INFO\n')
for target in selected_targets:
    (train, test, n_train, n_test, features) = \
        transform(data, target, general_targets, seed, describe=True)
    basline_info(target, train[1], test[1], n_train, n_test)
    model = simple_model(train[0][:, 3].reshape(-1, 1), train[1])
    y_pred = model.predict(test[0][:, 3].reshape(-1, 1))
    history['baseline_model'][target] = \
        {'Data': {'train': (train[0][:, 3], train[1]),
                  'test': (test[0][:, 3], test[1])},
         'Features': ['carbo'],
         'Patients': {'train': n_train,
                      'test': n_test},
         'Model': model,
         'Metrics': {'mae': mean_absolute_error(test[1], y_pred),
                     'mse': mean_squared_error(test[1], y_pred),
                     'rmse': root_mean_squared_error(test[1], y_pred),
                     'pearson': f'''{pearsonr(test[1], y_pred)[0]}, p-value {pearsonr(test[1], y_pred)[1]}''',
                     'r2': r2_score(test[1], y_pred)}
         }

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, train_batch, rng):
    param_grid = {
        'objective': 'regression',
        'device_type': 'cuda' if torch.cuda.is_available() else 'cpu',
        'metric': 'rmse',
        'verbosity': -1,
        'force_col_wise': 'true',
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 10.0, log=True),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 200),
        'feature_fraction ': trial.suggest_float('feature_fraction', 0.3, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    cv_results = \
        lgb.cv(param_grid, train_batch, num_boost_round=trial.suggest_int('num_boost_round', 100, 5000),
               nfold=3, stratified=False, shuffle=True, seed=rng)
    cv_mean = cv_results['valid rmse-mean'][-1]
    return cv_mean


def run_the_process(train_batch, name, rng):
    process = \
        optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=rng),
                            pruner=optuna.pruners.SuccessiveHalvingPruner(),
                            study_name=name)
    process.optimize(lambda trial: objective(trial, train_batch, rng),
                     n_trials=500,
                     show_progress_bar=True)
    return process


# Get string representation of the variable
def var_name(var, namespace):
    return [name for name in namespace if namespace[name] is var][0]


print('ADVANCED MODEL TRAINING')
for subset in [feature_group_1, feature_group_2, feature_group_3,
               feature_group_4]:
    feature_group = var_name(subset, locals())
    history[feature_group] = {}
    print(feature_group.upper())
    for target in selected_targets:
        print(target.upper())
        (train, test, n_train, n_test, features) = \
            transform(data.loc[:, data.columns.isin([*subset,
                                                     *general_targets,
                                                     target])],
                      target, general_targets, seed)
        study = run_the_process(lgb.Dataset(train[0], label=train[1]),
                                target, seed)
        params = study.best_trial.params
        n_estimators = params.pop('num_boost_round')
        model = lgb.train(params, lgb.Dataset(train[0], label=train[1]),
                          num_boost_round=n_estimators)
        y_pred = model.predict(test[0])

        history[feature_group][target] = \
            {'Data': {'train': (train[0], train[1]),
                      'test': (test[0], test[1])},
             'Features': features,
             'Patients': {'train': n_train,
                          'test': n_test},
             'Model': {'file': model,
                       'parameters': study.best_trial.params},
             'Metrics': {'mae': mean_absolute_error(test[1], y_pred),
                         'mse': mean_squared_error(test[1], y_pred),
                         'rmse': root_mean_squared_error(test[1], y_pred),
                         'pearson': f'''{pearsonr(test[1], y_pred)[0]}, p-value {pearsonr(test[1], y_pred)[1]}''',
                         'r2': r2_score(test[1], y_pred),
                         'adj_r2': r2_score_adj(test[1], y_pred,
                                                test[0].shape[0],
                                                test[0].shape[1])}}

plt.style.use('default')

for experiment in history:
    experiment_path = os.path.join(path, experiment)
    os.makedirs(experiment_path, exist_ok=True)
    for target in history[experiment]:
        target_path = os.path.join(experiment_path, target)
        os.makedirs(target_path, exist_ok=True)
        with open(os.path.join(target_path, 'results.txt'),
                  'w', encoding='UTF-8') as file:
            model_metrics = history[experiment][target]['Metrics']
            file.write(f'{model_metrics}')
            patients_in_experiment = history[experiment][target]['Patients']
            file.write(f'{patients_in_experiment}')
            model_features = history[experiment][target]['Features']
            file.write(f'{model_features}')
            if experiment != 'baseline_model':
                model_params = \
                    history[experiment][target]['Model']['parameters']
                file.write(f'{model_params}')
        X_train, y_train = history[experiment][target]['Data']['train']
        X_test, y_test = history[experiment][target]['Data']['test']
        if experiment != 'baseline_model':
            estimator = history[experiment][target]['Model']['file']
            estimator.save_model(os.path.join(target_path, 'model.json'))
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)
            plot_scatter(true=y_test, predicted=explainer.expected_value + np.sum(shap_values, axis=1),
                         focal_feature_group=experiment,
                         focal_target=target, path_to_save=target_path)
            plot_shap(shap_values=shap_values, batch=X_test, feature_names=model_features, path_to_save=target_path)
            plot_grouped_shap(shap_values=shap_values, feature_names=model_features, path_to_save=target_path)
        else:
            estimator = history[experiment][target]['Model']
            with open(os.path.join(target_path, 'model.pkl'), 'wb') as f:
                pickle.dump(estimator, f)
            plot_scatter(true=y_test, predicted=estimator.predict(X_test.reshape(-1, 1)), focal_feature_group=experiment,
                         focal_target=target, path_to_save=target_path)
            plot_scatter(true=y_test, predicted=X_test, focal_feature_group=experiment,
                         focal_target=target, path_to_save=target_path, only_carbo=True)
        np.savetxt(os.path.join(target_path, 'X_train.csv'), X_train,
                   delimiter=',', header=','.join(model_features), comments='',
                   fmt='%.4f', encoding='UTF-8')
        np.savetxt(os.path.join(target_path, 'y_train.csv'), y_train,
                   delimiter=',', header=target, comments='',
                   fmt='%.4f', encoding='UTF-8')
        np.savetxt(os.path.join(target_path, 'X_test.csv'), X_test,
                   delimiter=',', header=','.join(model_features), comments='',
                   fmt='%.4f', encoding='UTF-8')
        np.savetxt(os.path.join(target_path, 'y_test.csv'), y_test,
                   delimiter=',', header=target, comments='',
                   fmt='%.4f', encoding='UTF-8')
