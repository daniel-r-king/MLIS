import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix

def normalize(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def invert_response(n):
    if n == 1:
        n = 5
    elif n == 2:
        n = 4
    elif n == 3:
        n = 3
    elif n == 4:
        n = 1
    elif n == 5:
        n = 1
    else:
        n = 3
    return n

extroversion = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
neuroticism = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10']
agreeableness = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
conscientiousness = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
openness = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10']
all_responses = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10']

full_df = pd.read_csv('data.tsv', sep = '\t', index_col = None, low_memory = False)
full_df.columns = ['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10']
full_df['E2'] = full_df['E2'].apply(invert_response)
full_df['E4'] = full_df['E4'].apply(invert_response)
full_df['E6'] = full_df['E6'].apply(invert_response)
full_df['E8'] = full_df['E8'].apply(invert_response)
full_df['E10'] = full_df['E10'].apply(invert_response)
full_df['N2'] = full_df['N2'].apply(invert_response)
full_df['N4'] = full_df['N4'].apply(invert_response)
full_df['A1'] = full_df['A1'].apply(invert_response)
full_df['A3'] = full_df['A3'].apply(invert_response)
full_df['A5'] = full_df['A5'].apply(invert_response)
full_df['A7'] = full_df['A7'].apply(invert_response)
full_df['C2'] = full_df['C2'].apply(invert_response)
full_df['C4'] = full_df['C4'].apply(invert_response)
full_df['C6'] = full_df['C6'].apply(invert_response)
full_df['C8'] = full_df['C8'].apply(invert_response)
full_df['O2'] = full_df['O2'].apply(invert_response)
full_df['O4'] = full_df['O4'].apply(invert_response)
full_df['O6'] = full_df['O6'].apply(invert_response)

iso_codes = ['A1', 'AF', 'AX', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BH', 'BS', 'BD', 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW', 'BV', 'BR', 'IO', 'BN', 'BG', 'BF', 'BI', 'KH', 'CM', 'CA', 'CV', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX', 'CC', 'CO', 'KM', 'CG', 'CD', 'CK', 'CR', 'CI', 'HR', 'CU', 'CW', 'CY', 'CZ', 'DK', 'DJ', 'DM', 'DO', 'EC', 'EG', 'SV', 'GQ', 'ER', 'EE', 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR', 'GF', 'PF', 'TF', 'GA', 'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL', 'GD', 'GP', 'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 'HM', 'VA', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IM', 'IL', 'IT', 'JM', 'JP', 'JE', 'JO', 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW', 'KG', 'LA', 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 'LT', 'LU', 'MO', 'MK', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR', 'MU', 'YT', 'MX', 'FM', 'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'NC', 'NZ', 'NI', 'NE', 'NG', 'NU', 'NF', 'MP', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA', 'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 'PT', 'PR', 'QA', 'RE', 'RO', 'RU', 'RW', 'BL', 'SH', 'KN', 'LC', 'MF', 'PM', 'VC', 'WS', 'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL', 'SG', 'SX', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS', 'SS', 'ES', 'LK', 'SD', 'SR', 'SJ', 'SZ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TK', 'TO', 'TT', 'TN', 'TR', 'TM', 'TC', 'TV', 'UG', 'UA', 'AE', 'GB', 'US', 'UM', 'UY', 'UZ', 'VU', 'VE', 'VN', 'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW', '(nu']
country_numbers = []
for i in range(0, 251):
    country_numbers.append(i)
country_dict = dict(zip(iso_codes, country_numbers))
full_df['country'] = full_df['country'].map(country_dict)
full_df.replace([np.inf, -np.inf], np.nan)
full_df.dropna(inplace=True)

for entry in full_df['age']:
    if entry > 0 and entry < 27:
        entry = 1
    if entry > 26 and entry < 35:
        entry = 2
    if entry > 34 and entry < 50:
        entry = 3
    if entry > 49 and entry < 65:
        entry = 5
    if entry > 64:
        entry = 6

ext_df = full_df[extroversion]
neuro_df = full_df[neuroticism]
agree_df = full_df[agreeableness]
consc_df = full_df[conscientiousness]
open_df = full_df[openness]

all_responses_df = full_df[all_responses]

print(full_df.head())

X_train, X_test, y_train, y_test = train_test_split(all_responses_df, full_df['gender'].values, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Gender Prediction Accuracy: ', gnb.score(X_test, y_test))
print(cm)

X_train, X_test, y_train, y_test = train_test_split(all_responses_df, full_df['country'].values, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Country Prediction Accuracy: ', gnb.score(X_test, y_test))
print(cm)

X_train, X_test, y_train, y_test = train_test_split(all_responses_df, full_df['age'].values, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Age Prediction Accuracy: ', gnb.score(X_test, y_test))
print(cm)

X_train, X_test, y_train, y_test = train_test_split(all_responses_df, full_df['race'].values, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Race Prediction Accuracy: ', gnb.score(X_test, y_test))
print(cm)

X_train, X_test, y_train, y_test = train_test_split(all_responses_df, full_df['hand'].values, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Handedness Prediction Accuracy: ', gnb.score(X_test, y_test))
print(cm)



