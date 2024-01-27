import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import bootstrap

try:    
    os.remove('params.txt')
except FileNotFoundError:
    pass

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
general_targets = ['BG60', 'BG120', 'BGMax', 'AUC60', 'AUC120', 'iAUC60', 'iAUC120']

for j in general_targets:
    for i in [0, 1]:
        data = pd.read_pickle(f'data_for_training_{i}.pkl')
        data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data.columns.values]

        target_feature = j

        train_valid, test_patients = train_test_split(np.unique(data['N']), test_size=0.2, shuffle=True)

        train_patients, validation_patients = train_test_split(train_valid, test_size=0.2, shuffle=True)

        X_train = data.loc[data['N'].isin(train_patients),
                           ~data.columns.isin(['N', *general_targets])]
        y_train = data.loc[data['N'].isin(train_patients), target_feature]

        X_valid = data.loc[data['N'].isin(validation_patients),
                           ~data.columns.isin(['N', *general_targets])]
        y_valid = data.loc[data['N'].isin(validation_patients), target_feature]

        X_test = data.loc[data['N'].isin(test_patients),
                          ~data.columns.isin(['N', *general_targets])]
        y_test = data.loc[data['N'].isin(test_patients), target_feature]

        cv_params = {'learning_rate': [x/100 for x in range(10, 45, 25)],
                     'max_depth': [x for x in range(4, 10, 2)],
                     'n_estimators': [x for x in range(500, 1200, 200)],
                     'reg_lambda': [x for x in range(0, 3, 1)],
                     'reg_alpha': [x for x in range(0, 3, 1)]}
        
        regressor = XGBRegressor(tree_method='hist', eval_metric=mean_absolute_error, early_stopping_rounds=10)
        cv = RandomizedSearchCV(estimator=regressor, param_distributions=cv_params, n_iter=10, cv=3)
        model = cv.fit(X_train, y_train, eval_set = [(X_valid, y_valid)])

        y_pred = model.predict(X_test)

        explainer = shap.TreeExplainer(model.best_estimator_)
        values = explainer.shap_values(pd.concat([X_train, X_test]))
        fig = plt.figure()
        shap.summary_plot(values, pd.concat([X_train, X_test]), show=False)
        plt.savefig(f'SHAP_{i}_{j}')
        plt.close(fig)

        r = round(pearsonr(y_test, y_pred)[0], 4)
        r_boot = bootstrap((y_test, y_pred), pearsonr, paired=True)
        r_low = round(r_boot.confidence_interval.low[0], 4)
        r_high = round(r_boot.confidence_interval.high[0], 4)

        r2 = round(r2_score(y_test, y_pred), 4)
        r2_boot = bootstrap((y_test, y_pred), r2_score, paired=True)
        r2_low = round(r2_boot.confidence_interval.low, 4)
        r2_high = round(r2_boot.confidence_interval.high, 4)

        mae = round(mean_absolute_error(y_test, y_pred), 4)
        mae_boot = bootstrap((y_test, y_pred), mean_absolute_error, paired=True)
        mae_low = round(mae_boot.confidence_interval.low, 2)
        mae_high = round(mae_boot.confidence_interval.high, 4)

        mse = round(mean_squared_error(y_test, y_pred), 4)
        mse_boot = bootstrap((y_test, y_pred), mean_squared_error, paired=True)
        mse_low = round(mse_boot.confidence_interval.low, 4)
        mse_high = round(mse_boot.confidence_interval.high, 4)

        with open('params.txt', 'a+') as file:
            file.write(f'\n\t{j}')
            if i == 0:
                file.write('\n\t\tWith microbiom (0)')
            else:
                file.write('\n\t\tWithout microbiom (1)')
            file.write(f'\n\t\t\t{model.best_estimator_.get_params()}')
            txt = f'\n\t\t\tR: {r_low} < {r} < {r_high}\
              \n\t\t\tR2: {r2_low} < {r2} < {r2_high}\
              \n\t\t\tMAE: {mae_low} < {mae} < {mae_high}\
              \n\t\t\tRMSE: {mse_low} < {mse} < {mse_high}'
            file.write(txt)
        
        features = pd.Series(X_train.columns.tolist(), name='Features')
        features_shap = pd.Series(model.best_estimator_.feature_importances_, name='SHAP')
        features_shap = pd.concat([features, features_shap], axis=1)
        sorted_features_shap = features_shap.sort_values(by='SHAP', ascending=False).reset_index(drop=True)
        selected_features = sorted_features_shap.iloc[:35, 0].tolist()

        X_train_selected = X_train.loc[:, selected_features]
        X_valid_selected = X_valid.loc[:, selected_features]
        X_test_selected = X_test.loc[:, selected_features]

        regressor = XGBRegressor(tree_method='exact', eval_metric=mean_absolute_error, early_stopping_rounds=20)
        cv = RandomizedSearchCV(estimator=regressor, param_distributions=cv_params, n_iter=20, cv=5)
        model = cv.fit(X_train_selected, y_train, eval_set = [(X_valid_selected, y_valid)])

        y_pred = model.predict(X_test_selected)

        explainer = shap.TreeExplainer(model.best_estimator_)
        values = explainer.shap_values(pd.concat([X_train_selected, X_test_selected]))
        fig = plt.figure()
        shap.summary_plot(values, pd.concat([X_train_selected, X_test_selected]), show=False)
        plt.savefig(f'SHAP_{i}_{j}_selected')
        plt.close(fig)

        r = round(pearsonr(y_test, y_pred)[0], 4)
        r_boot = bootstrap((y_test, y_pred), pearsonr, paired=True)
        r_low = round(r_boot.confidence_interval.low[0], 4)
        r_high = round(r_boot.confidence_interval.high[0], 4)

        r2 = round(r2_score(y_test, y_pred), 4)
        r2_boot = bootstrap((y_test, y_pred), r2_score, paired=True)
        r2_low = round(r2_boot.confidence_interval.low, 4)
        r2_high = round(r2_boot.confidence_interval.high, 4)

        mae = round(mean_absolute_error(y_test, y_pred), 4)
        mae_boot = bootstrap((y_test, y_pred), mean_absolute_error, paired=True)
        mae_low = round(mae_boot.confidence_interval.low, 2)
        mae_high = round(mae_boot.confidence_interval.high, 4)

        mse = round(mean_squared_error(y_test, y_pred), 4)
        mse_boot = bootstrap((y_test, y_pred), mean_squared_error, paired=True)
        mse_low = round(mse_boot.confidence_interval.low, 4)
        mse_high = round(mse_boot.confidence_interval.high, 4)

        with open('params.txt', 'a+') as file:
            file.write(f'\n\t{j}')
            if i == 0:
                file.write('\n\t\tWith microbiom (0, selected)')
            else:
                file.write('\n\t\tWithout microbiom (1, selected)')
            file.write(f'\n\t\t\t{model.best_estimator_.get_params()}')
            txt = f'\n\t\t\tR: {r_low} < {r} < {r_high}\
              \n\t\t\tR2: {r2_low} < {r2} < {r2_high}\
              \n\t\t\tMAE: {mae_low} < {mae} < {mae_high}\
              \n\t\t\tRMSE: {mse_low} < {mse} < {mse_high}'
            file.write(txt)
