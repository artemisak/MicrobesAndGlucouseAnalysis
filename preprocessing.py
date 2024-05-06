import os
import re
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from supplementary import *

root = os.getcwd()

original_micr_data = pd.read_csv(os.path.join(root, 'microbiom_data.csv'),
                                 dtype='string', encoding='UTF-8')

# As the warning suggests, the provided table stores mixed data types
type_constrained_micr_data = \
    pd.concat([original_micr_data.iloc[0:413, 0].astype('string'),
               original_micr_data.iloc[0:413, 1:].astype('float')], axis=1)

numerical_micr_data = type_constrained_micr_data.copy()
sample = numerical_micr_data.pop('Sample')
visit = numerical_micr_data.pop('Visit')

# The sample name contains the patient number,
# which will be required to merge the tables
N = sample.apply(lambda row: int(row[0:3]))
N.name = 'N'


# Reading genetic sequences can be very costly in terms of memory consumption
def read_line(source):
    while True:
        line = source.readline()
        if not line:
            break
        yield line


def read_files(files):
    LefSe_otu = []
    for file_name in files:
        try:
            with open(file_name, mode='r', encoding='UTF-8') as source:
                for line in read_line(source):
                    bacteria = re.findall(r'\bOTU_\d+\b', line)
                    if bacteria:
                        LefSe_otu.append(*bacteria)
        except FileNotFoundError:
            print(f"File '{file_name}' not found.")
        except IOError as e:
            print(f"Error reading file '{file_name}': {e}")
    return LefSe_otu


# From the list of representatives of the gut microbiota,
# it was proposed to select for the study the bacteria that showed
# the greatest differences between patient groups as a result of LefSe analysis
LefSe_files = [os.path.join(root, 'LefSe_above_median_BgMax_20_01.csv'),
               os.path.join(root, 'LefSe_above_median_iauc120_20_01.csv')]
selected_otu = np.unique(read_files(LefSe_files))
otu = numerical_micr_data.loc[:, selected_otu]

micr_data = pd.concat([N, visit, otu], axis=1)
micr_data.set_index(['N', 'Visit'], inplace=True)

# The column reflects the period in which the analytical sample was processed
# The filter is designed to cut off a small portion of the data
# that has been corrupted in transit
micr_data = micr_data.loc[(micr_data.index.get_level_values('Visit') == 99) |
                          (micr_data.index.get_level_values('Visit') == 146)]
micr_data = micr_data.reset_index(level='Visit', drop=True)

# Deciphering OTU sequences
microbial_sequences = original_micr_data.iloc[413, 2:]
otu_micr_pairs = dict(zip(microbial_sequences.index,
                          microbial_sequences.to_numpy()))
selected_otu_micr_pairs = \
    {key: otu_micr_pairs[key] for key in micr_data.columns}

# Set of clinical parameters proposed by the principal investigator for analysis
clinical_param = ['quality_cgm1', 'CGMS_срок', 'диета_срок',
                  'диета_кал', 'срок_кал1', 'Глюкоза_нт_общая',
                  'прибавка_m1', 'отеки1', 'АД_сист1', 'АД_диаст1',
                  'N_беременностей', 'N_родов', 'N_невынашивание',
                  'N_абортов', 'menses', 'АГ', 'НТГ', 'СД_у_родственников',
                  'КОК', 'курение_до_беременности', 'курение_во_время',
                  'возраст', 'рост', 'вес_до_берем', 'ИМТ', 'фрукты1',
                  'молочное_необезжир1', 'сосиски1', 'бобовые2', 'овощи2',
                  'сосиски2', 'подъем2', 'rs10830963_MTNR1B_N',
                  'глю_нт_V1', 'кетоны_V1', 'HbA1C_V1',
                  'инсулин_V1', 'лептин_V1', 'ФР_V1', 'БОМК_визит1',
                  'Хол_V1', 'ТГ_V1', 'ЛПВП_V1', 'ЛПОНП_V1', 'ЛПНП_V1',
                  'КА_V1', 'АБ_бер_ть']

original_clinical_data = pd.read_csv(os.path.join(root, 'clinical_data.csv'),
                                     index_col='N',
                                     usecols=['N', *clinical_param],
                                     dtype='float', encoding='UTF-8')
original_clinical_data.index = original_clinical_data.index.astype('int')

additional_clinical_data = \
    pd.read_csv(os.path.join(root, 'additional_clinical_data.csv'),
                index_col='N', usecols=['N', 'CGM_g_age1', 'GM_g_age2',
                                        'diet_before_V1', 'Diet_duration_V1'],
                dtype='float', encoding='UTF-8')
additional_clinical_data.index = additional_clinical_data.index.astype('int')

original_clinical_data = original_clinical_data.join(additional_clinical_data,
                                                     on='N')

# The filter is designed to cut off some patients who carelessly filled out data
clinical_data = original_clinical_data.copy()
clinical_data = \
    clinical_data.loc[((clinical_data['quality_cgm1'] == 0) |
                       (clinical_data['quality_cgm1'] == 5)), :]
clinical_data.drop(columns=['quality_cgm1'], axis=1, inplace=True)

# It was suggested that diaries of patients who had taken antibiotics
# less than 4 weeks prior to the study should not be analyzed
clinical_data.drop(labels=[712, 724], axis=0, inplace=True)

# Let's recalculate some biochemical markers as there may have been errors
# or omissions in them
clinical_data['ЛПОНП_V1'] = clinical_data['ТГ_V1'] / 2.18
clinical_data['ЛПНП_V1'] = (clinical_data['Хол_V1'] -
                            clinical_data['ЛПОНП_V1'] -
                            clinical_data['ЛПВП_V1'])

# Set of garbage monitoring parameters (time lags, device info etc.)
garbage_params = ['ts_meal_start', 'ts_meal_end',
                  'ts_meal_diary', 'meal_shift', 'meal_length',
                  'preg_week', 'meal_type', 'bg_after', 'bg_after_t',
                  'pa_before', 'pa_before_t', 'pa_after', 'prec_meal_shift',
                  'meal_items', 'meal_cats', 'mass_d', 'CGM_range',
                  'CGM_mean', 'CGM_below', 'CGM_above', 'BGb60_to_mean',
                  'BGTrend240', 'BGTrend120', 'BGTrend60', 'BGLag',
                  'BGRise', 'BG65_70', 'BGabove70', 'BG_traffic_light',
                  'uncooked_dishes', 'quality_meal', 'diacompanion',
                  'prediction_function', 'bg_predicted', 'predictionTimeShift',
                  'predictionFunction', 'bg_before30', 'bgBeforeApp',
                  'BGMaxMore', 'count_meals_b12h', 'quality_diary',
                  'quality_cgm1', 'quality_cgm2', 'bg_after_hour',
                  'preg_week_calc', 'Дата_анализ_V1']

# Set of redundant monitoring parameters provided by the principal investigator
redundant_params = ['сладкие_напитки1', 'овощи1', 'хлеб_любой2', 'бобовые1',
                    'спорт2', 'рыба1', 'мясо2', 'хлеб_цельнозерновой_2',
                    'молочное_необезжир2', 'кофе1', 'молочное_обезжир1',
                    'ca', 'хлеб_цельное1', 'мясо1', 'выпечка2', 'сухофрукты1',
                    'ходьба2', 'шоколад2', 'b1', 'пирожные2', 'kar',
                    'спорт1', 'ходьба1', 'пирожные1', 'b2', 'fe', 'выпечка1',
                    'mg', 'сухофрукы2', 'water', 'ne', 'zola', 're', 'фрукты2',
                    'ok', 'хлеб_любой1', 'рыба2', 'овощи2_сырые', 'k', 'соусы2',
                    'подъем1', 'сладкие_напитки2', 'алкоголь1',
                    'молочное_обезжир2', 'соусы1', 'овощи1_сырые', 'c',
                    'алкоголь2', 'a', 'кофе2', 'срок_анализы_V1', 'шоколад1',
                    'p', 'na']

# Continuous glucose monitoring parameters
# (clearly very important, but interfere with this analysis)
cgms_a = ['CGM_range', 'CGM_mean', 'CGM_below', 'CGM_above',
          'BGTrend240', 'BGTrend120', 'BGTrend60', 'BGLag',
          'BGRise', 'BG65_70', 'BGabove70', 'BG_traffic_light']
cgms_b = ['BGb60_to_mean', 'iAUCb240', 'iAUCb120', 'iAUCb60',
          'BGRiseb240', 'BGRiseb120', 'BGRiseb60', 'BGb240',
          'BGb120', 'BGb60', 'BGb50', 'BGb40', 'BGb30', 'BGb25',
          'BGb20', 'BGb15', 'BGb10', 'BGb5']

original_monitoring_data = \
    pd.read_csv(os.path.join(root, 'monitoring_data.csv'),
                index_col='N', dtype='string', encoding='UTF-8')

for column in original_monitoring_data.columns:
    original_monitoring_data[column] = \
        original_monitoring_data[column].astype('float', errors='ignore')

original_monitoring_data.index = \
    pd.to_numeric(original_monitoring_data.index, errors='raise')

# Remove all unnecessary items
pattern = f'Unnamed|meal_items|meal_mass|meal_time|without'
redundant_params.extend(original_monitoring_data.filter(regex=pattern,
                                                        axis=1).columns)
monitoring_data = original_monitoring_data.copy()
monitoring_data = \
    monitoring_data.loc[:, ~monitoring_data.columns.isin([*garbage_params,
                                                          *redundant_params,
                                                          *cgms_a])]

# Exclude patients who were taking insulin
insulin_features = ['i_before', 'i_before_t', 'i_type']
monitoring_data = monitoring_data[
    (monitoring_data['project'] == 3) &
    (monitoring_data['i_before'].isna())
    ]
monitoring_data.drop(labels=['project', *insulin_features],
                     axis=1, inplace=True)

# During the analysis of the literature, the most
# significant parameters for forecasting were identified
targets = ['BG30', 'BG60', 'BG90', 'BG120', 'BGMax',
           'AUC60', 'AUC120', 'iAUC60', 'iAUC120']
factors = ['BG0', 'gi', 'gl', 'carbo', 'prot', 'fat']
monitoring_data.dropna(subset=[*factors, *targets], inplace=True)

data = clinical_data.merge(micr_data, on='N', how='inner')

data = monitoring_data.merge(data, on='N', how='inner',
                             suffixes=('_monitoring', '_clinical'))

redundant_params = data.filter(regex='_monitoring').columns
data.drop(labels=redundant_params, axis=1, inplace=True)
data.rename(columns=lambda x: re.sub(r'_clinical$', '', x), inplace=True)

data['CGMS_срок'] = np.where(data['n_cgm'] == 1,
                             data['CGM_g_age1'],
                             data['GM_g_age2'])
data.drop(labels=['CGM_g_age1', 'GM_g_age2', 'n_cgm'],
          axis=1, inplace=True)

data.sort_index(inplace=True)

# Let's examine the data characterizing the patients
# included in the preliminary dataset
N = data.index.unique()
noutof_clinical_data = clinical_data[clinical_data.index.isin(N)]
noutof_clinical_data.describe()

fig = px.imshow(noutof_clinical_data.corr())
fig.show()

data.drop(labels='НТГ', axis=1, inplace=True)

fig = px.box(pd.concat([noutof_clinical_data['Глюкоза_нт_общая'],
                        data['BGMax'].groupby('N').median(),
                        data['AUC120'].groupby('N').median(),
                        noutof_clinical_data['СД_у_родственников']],
                       axis=1), color='СД_у_родственников',
             notched=True, points='all')
fig.show()

fig = px.box(pd.concat([data['BG0'].groupby('N').median(),
                        noutof_clinical_data['СД_у_родственников']],
                       axis=1),
             color='СД_у_родственников', notched=True, points='all')
fig.show()

numerator = (data['BGMax'].groupby('N').median() -
             data['BG120'].groupby('N').median())
denominator = data['BG120'].groupby('N').median()
bg_drop_rate = numerator / denominator
fig = px.box(pd.concat([bg_drop_rate, noutof_clinical_data['СД_у_родственников']],
                       axis=1),
             color='СД_у_родственников', notched=True, points='all')
fig.show()

scaler = MinMaxScaler()
df = pd.concat([data['gl'], data['carbo'],
                data['gi'], data['meal_type_n']], axis=1)
df[['gl', 'carbo', 'gi']] = scaler.fit_transform(df[['gl', 'carbo', 'gi']])
fig = px.scatter_ternary(df, a='gl', b='carbo', c='gi', color='meal_type_n')
fig.show()

df = pd.concat([data['carbo'], data['prot'],
                data['fat'], data['meal_type_n']], axis=1)
df[['carbo', 'prot', 'fat']] = scaler.fit_transform(df[['carbo',
                                                        'prot',
                                                        'fat']])
fig = px.scatter_ternary(df, a='carbo', b='prot', c='fat', color='meal_type_n')
fig.show()

nbins = 4
fig = make_subplots(rows=1, cols=nbins - 1,
                    specs=[[{'type': 'ternary'} for n in range(nbins - 1)]],
                    subplot_titles=[f'nbins = {n + 2}' for n in range(nbins - 1)],
                    horizontal_spacing=.2)

for n in range(2, nbins + 1):
    binned_daytime = pd.cut(data['daytime'], bins=n)
    df = pd.concat([data['carbo'], data['prot'], data['fat'], binned_daytime],
                   axis=1)
    df[['carbo', 'prot', 'fat']] = scaler.fit_transform(df[['carbo',
                                                            'prot',
                                                            'fat']])
    daytimes = df['daytime'].unique()
    encoder = LabelEncoder()
    colors = encoder.fit_transform(daytimes)
    for i, daytime in enumerate(daytimes):
        mask = df['daytime'] == daytime
        fig.add_trace(go.Scatterternary(a=df.loc[mask, 'carbo'],
                                        b=df.loc[mask, 'prot'],
                                        c=df.loc[mask, 'fat'],
                                        mode='markers',
                                        name=str(daytime),
                                        marker={'color': colors[i]}),
                      row=1, col=n - 1)
    fig.update_ternaries(aaxis_title='Carbo', baxis_title='Prot',
                         caxis_title='Fat', row=1, col=n - 1)
title = 'Changes in the nutritional composition of food throughout the day'
fig.update_layout(title=title, legend_title='Time intervals')
fig.show()

nans_percentage = data.isna().mean()
any_na = nans_percentage[nans_percentage > 0]

prec_meal_params = any_na.filter(regex='prec_').index
data[prec_meal_params] = data[prec_meal_params].fillna(value=0)

matches = re.findall(r"iAUCb\d+|BGRiseb\d+|BGb\d+",
                     ', '.join(any_na.index.to_list()))
data.dropna(subset=matches, inplace=True)

data.drop('bgBefore_glu', axis=1, inplace=True)

row_per_patient = data.groupby('N').mean(numeric_only=True)
nans_percentage = row_per_patient.isna().mean()
any_na = nans_percentage[nans_percentage > 0]

data.drop(labels=nans_percentage[nans_percentage > 0.15].index, axis=1, inplace=True)

data.drop(labels=[668, 704], inplace=True)

data = np.round(data, 4)

time_stamp = sys.argv[1]
path = os.path.join(root, f'results/{time_stamp}')

data.to_csv(os.path.join(path, 'general_data_with_meals_id.csv'), encoding='UTF-8')
data.drop(columns=['meal_id'], axis=1, inplace=True)

describe_data(data, path, 'general')

data.to_pickle(os.path.join(path, 'general_data_for_training.pkl'))
