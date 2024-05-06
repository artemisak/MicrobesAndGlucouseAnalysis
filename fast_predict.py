import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from supplementary import *


os.makedirs('fast_predict_results', exist_ok=True)

aics = {}

for target in ['BGMax', 'iAUC120']:

    aics[target] = {} 
    best_model_timestamp = '15-04-2024T19-11-41' if target == 'BGMax' else '15-04-2024T14-04-41'

    for feature_group in ['baseline_model', *[f'feature_group_{i}' for i in range(1, 5)]]:
        
        path_to_get = f"results/{best_model_timestamp}/{feature_group}/{target}"
        
        path_to_put = f"fast_predict_results/{best_model_timestamp}/{feature_group}/{target}"
        os.makedirs(path_to_put, exist_ok=True)
        
        X_train = np.genfromtxt(fname=os.path.join(path_to_get, f'X_test.csv'),
                            delimiter=",", skip_header=1, encoding='UTF-8')
        y_train = np.genfromtxt(fname=os.path.join(path_to_get, f'y_test.csv'),
                            delimiter=",", skip_header=1, encoding='UTF-8')
        X_test = np.genfromtxt(fname=os.path.join(path_to_get, f'X_test.csv'),
                            delimiter=",", skip_header=1, encoding='UTF-8')
        y_test = np.genfromtxt(fname=os.path.join(path_to_get, f'y_test.csv'),
                            delimiter=",", skip_header=1, encoding='UTF-8')
        
        with open(os.path.join(path_to_get, f'X_test.csv'), 'r', encoding='UTF-8') as file:
            column_names = file.readline().strip().lstrip('# ').split(',')

        features_mapper = pd.read_csv('supplementary.csv')
        features_mapper.dropna(subset=['Current names'], axis=0, inplace=True)
        features_mapper = dict(zip(features_mapper['Current names'], features_mapper['New ones']))

        if feature_group == 'baseline_model':
            model = simple_model(X_train.reshape(-1, 1), y_train)
            y_pred = model.predict(X_test.reshape(-1, 1))
            plot_scatter(true=y_test, predicted=y_pred, focal_feature_group=feature_group, focal_target=target,
                        path_to_save=os.path.join(os.getcwd(), path_to_put))
            plot_scatter(true=y_test, predicted=X_test, focal_feature_group=feature_group, focal_target=target,
                        path_to_save=os.path.join(os.getcwd(), path_to_put), only_carbo=True)
        else:
            model = lgb.Booster(model_file=os.path.join(path_to_get, f'model.json'))
            y_pred = model.predict(X_test)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            plot_shap(shap_values=shap_values, batch=X_test, feature_names=[features_mapper.get(feature, feature)
                                                                            for feature in column_names],
                    path_to_save=os.path.join(os.getcwd(), path_to_put))

            plot_grouped_shap(shap_values=shap_values, feature_names=column_names,
                            path_to_save=os.path.join(os.getcwd(), path_to_put))
            
            plot_scatter(true=y_test, predicted=y_pred, focal_feature_group=feature_group, focal_target=target,
                        path_to_save=os.path.join(os.getcwd(), path_to_put))
            
        aics[target][feature_group] = calculate_aic(y_test, y_pred, 2 if feature_group == 'baseline_model' else model.num_trees())
        
    print(f'AIC for {target} with different features: ', aics[target])
