import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import sklearn.datasets as dt
import pandas as pd

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

full_df = pd.read_csv('data.tsv', sep = '\t', index_col = None, low_memory = False).head(n = 500)
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

ext_df = full_df[extroversion]
neuro_df = full_df[neuroticism]
agree_df = full_df[agreeableness]
consc_df = full_df[conscientiousness]
open_df = full_df[openness]

all_responses_df = full_df[all_responses]

pca = PCA(n_components = 5)
all_responses_pc = pca.fit_transform(all_responses_df)
all_responses_pca_df = pd.DataFrame(data = all_responses_pc)

print(all_responses_pca_df.to_string())
print ("PCA total explained variance = ", round(pca.explained_variance_ratio_.sum(), 2))