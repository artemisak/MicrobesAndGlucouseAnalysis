import os
import math
import statistics
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


groups = {'meal_composition': ['gi', 'gl', 'prot', 'carbo', 'fat',
                               'kcal', 'mds', 'kr', 'pv'],
          'meal_context': ['meal_type_n', 'daytime', 'part_of_day', 'gl_b3h',
                           'gl_b6h', 'gl_b12h', 'carbo_b3h', 'carbo_b6h',
                           'carbo_b12h', 'prot_b3h', 'prot_b6h', 'prot_b12h',
                           'fat_b3h', 'fat_b6h', 'fat_b12h', 'pv_b3h', 'pv_b6h',
                           'pv_b12h', 'kcal_b3h', 'kcal_b6h', 'kcal_b12h',
                           'prec_meal_gi', 'prec_meal_gl', 'prec_meal_carbo',
                           'prec_meal_prot', 'prec_meal_fat', 'prec_meal_pv'],
          'cgm_data': ['iAUCb240', 'iAUCb120', 'iAUCb60', 'BGRiseb240',
                       'BGRiseb120', 'BGRiseb60', 'BGb240', 'BGb120', 'BGb60',
                       'BGb50', 'BGb40', 'BGb30', 'BGb25', 'BGb20', 'BGb15',
                       'BGb10', 'BGb5', 'BG0'],
          'gynecological_data': ['срок_берем1', 'ИЦН', 'СПКЯ', 'прибавка_m1',
                                 'отеки1', 'N_беременностей', 'N_родов',
                                 'N_невынашивание', 'N_абортов',
                                 'menses', 'КОК'],
          'other': ['АГ', 'СД_у_родственников', 'курение_до_беременности',
                    'курение_во_время', 'Группа', 'Срок_анализ_V1', 'АД_сист1',
                    'АД_диаст1', 'CGMS_срок', 'срок_кал1', 'возраст',
                    'АБ_бер_ть', 'diet_before_V1', 'Diet_duration_V1'],
          'anthropometric': ['рост', 'вес_до_берем', 'ИМТ'],
          'lifestyle_survey': ['фрукты1', 'молочное_необезжир1', 'сосиски1',
                               'бобовые2', 'овощи2', 'сосиски2', 'подъем2'],
          'genetics': ['rs10830963_MTNR1B_N'],
          'laboratory_glycemic_markers': ['Глюкоза_нт_общая', 'глю_нт_V1',
                                          'HbA1C_V1', 'ФР_V1'],
          'serum_lipid_markers': ['Хол_V1', 'ТГ_V1', 'ЛПВП_V1',
                                  'ЛПОНП_V1', 'ЛПНП_V1'],
          'other_laboratory_data': ['кетоны_V1', 'инсулин_V1',
                                    'лептин_V1', 'БОМК_визит1'],
          'microbiome': ['OTU_110', 'OTU_123', 'OTU_124', 'OTU_13', 'OTU_132',
                         'OTU_15', 'OTU_152', 'OTU_153', 'OTU_187', 'OTU_190',
                         'OTU_193', 'OTU_195', 'OTU_196', 'OTU_197', 'OTU_20',
                         'OTU_210', 'OTU_230', 'OTU_239', 'OTU_241', 'OTU_25',
                         'OTU_256', 'OTU_257', 'OTU_305', 'OTU_312', 'OTU_33',
                         'OTU_337', 'OTU_338', 'OTU_34', 'OTU_354', 'OTU_359',
                         'OTU_396', 'OTU_399', 'OTU_402', 'OTU_420', 'OTU_439',
                         'OTU_44', 'OTU_454', 'OTU_459', 'OTU_496', 'OTU_51',
                         'OTU_529', 'OTU_587', 'OTU_609', 'OTU_677', 'OTU_68',
                         'OTU_712', 'OTU_76', 'OTU_77', 'OTU_79', 'OTU_84',
                         'OTU_98']}


def plot_scatter(true, predicted, focal_feature_group, focal_target, path_to_save, only_carbo=False):
    plt.figure()
    if only_carbo:
        y_true_jittered = true + np.random.normal(0, 0.05, true.size)
        y_pred_jittered = predicted + np.random.normal(0, 0.05, true.size)
        p = np.poly1d(np.polyfit(y_true_jittered, y_pred_jittered, 1))
        x = np.linspace(min(y_true_jittered), max(y_true_jittered), 50)
        plt.plot(x, p(x), color='tab:orange')
        plt.scatter(y_true_jittered, y_pred_jittered)
        scale = 'mmol/l*h' if focal_target == 'iAUC120' else 'mmol/l'
        plt.xlabel(f'Measured target, {scale}')
        plt.ylabel('Carbohydrates, g')
        plt.title(f'{focal_feature_group} | {"GLUmax" if focal_target == "BGMax" else focal_target}')
        r = np.round(pearsonr(true, predicted)[0], 2)
        p_value = pearsonr(true, predicted)[1]
        plt.legend([f'Pearson R: {r}, p-value: {p_value}'], loc='upper left')
        plt.grid()
        plt.savefig(os.path.join(path_to_save, f'scatter_plot_only_carbo.png'))
    else:
        y_true_jittered = true + np.random.normal(0, 0.05, true.size)
        y_pred_jittered = predicted + np.random.normal(0, 0.05, predicted.size)
        p = np.poly1d(np.polyfit(y_true_jittered, y_pred_jittered, 1))
        x = np.linspace(min(y_true_jittered), max(y_true_jittered), 50)
        plt.plot(x, p(x), color='tab:orange')
        plt.scatter(y_true_jittered, y_pred_jittered, color='tab:blue')
        scale = 'mmol/l*h' if focal_target == 'iAUC120' else 'mmol/l'
        plt.xlabel(f'Measured target, {scale}')
        plt.ylabel(f'Predicted, {scale}')
        plt.title(f'{focal_feature_group} | {"GLUmax" if focal_target == "BGMax" else focal_target}\n' +
                  f'R2 = {np.round(r2_score(true, predicted), 2)}')
        r = np.round(pearsonr(true, predicted)[0], 2)
        p_value = pearsonr(true, predicted)[1]
        plt.legend([f'Pearson R: {r}, p-value: {p_value}'], loc='upper left')
        plt.grid()
        plt.savefig(os.path.join(path_to_save, f'scatter_plot.png'))
    plt.close()


def plot_shap(shap_values, batch, feature_names, path_to_save):
    plt.figure()
    shap.summary_plot(shap_values, batch, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, 'shap_summary.png'))
    plt.close()


def plot_grouped_shap(shap_values, feature_names, path_to_save):
    shap_matrix = pd.DataFrame(shap_values, columns=feature_names)
    grouped_shap = pd.DataFrame([])
    for group, cols in groups.items():
        if shap_matrix.columns.isin(cols).any():
            grouped_shap[group] = \
                shap_matrix.loc[:, shap_matrix.columns.isin(cols)].abs().sum(axis=1)
    plt.figure()
    shap.summary_plot(grouped_shap.to_numpy(), grouped_shap.columns.tolist(),
                      plot_type='bar', show=False)
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0, rect=(0.0, 0.0, 0.8, 1.0))
    plt.savefig(os.path.join(path_to_save, 'shap_grouped.png'))
    plt.close()


def r2_score_adj(true, predicted, observations_num, predictors_num):
    return 1 - (1 - r2_score(true, predicted)) * (observations_num - 1) / (observations_num - predictors_num - 1)


def calculate_aic(y_true, y_pred, k):
    return len(y_true) * np.log(np.sum((y_true - y_pred)**2) / len(y_true)) + 2 * k


def describe_data(df, path_to_save, mark):
    with open(os.path.join(path_to_save, f'{mark}_dataset_description.txt'),
          'w', encoding='UTF-8') as file:
        file.write(f'Total number of unique patients: {len(df.index.unique())}\n')
        file.write(f'Meals logged per participant (GDM): {calculate_confidence_interval(df[df["Группа"] == 1].pivot_table(index="N", values="meal_id", aggfunc="count")["meal_id"])}\n')
        file.write(f'Meals logged per participant (Healhy): {calculate_confidence_interval(df[df["Группа"] == 2].pivot_table(index="N", values="meal_id", aggfunc="count")["meal_id"])}\n')
        file.write(f'T-test between them: {ttest_ind(df[df["Группа"] == 1].pivot_table(index="N", values="meal_id", aggfunc="count")["meal_id"], df[df["Группа"] == 2].pivot_table(index="N", values="meal_id", aggfunc="count")["meal_id"])[1]}\n')
        file.write(f'Days logged per participant (GDM): {calculate_confidence_interval(df[df["Группа"] == 1].pivot_table(index="N", values="ts_meal_diary", aggfunc="nunique")["ts_meal_diary"])}\n')
        file.write(f'Days logged per participant (Healhy): {calculate_confidence_interval(df[df["Группа"] == 2].pivot_table(index="N", values="ts_meal_diary", aggfunc="nunique")["ts_meal_diary"])}\n')
        file.write(f'T-test between them: {ttest_ind(df[df["Группа"] == 1].pivot_table(index="N", values="ts_meal_diary", aggfunc="nunique")["ts_meal_diary"], df[df["Группа"] == 2].pivot_table(index="N", values="ts_meal_diary", aggfunc="nunique")["ts_meal_diary"])[1]}\n')
        file.write(f'Total number of rows: {df.shape[0]}\n')
        file.write(f'Total number of features: {df.shape[1]}')

def calculate_confidence_interval(data, confidence_level=0.95):

    sample_mean = statistics.mean(data)
    sample_std_dev = statistics.pstdev(data)

    z_score = statistics.NormalDist().inv_cdf((1 + confidence_level) / 2)

    standard_error_of_mean = sample_std_dev / math.sqrt(len(data))

    margin_of_error = z_score * standard_error_of_mean

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    confidence_interval = (lower_bound, upper_bound)

    return sample_mean, sample_std_dev, confidence_interval

def simple_model(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model
