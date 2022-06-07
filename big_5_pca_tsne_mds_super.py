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

full_df = pd.read_csv('data.tsv', sep = '\t', index_col = None, low_memory = False).head(n = 100)
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

pca = PCA(n_components = 1)
tsne = TSNE(n_components = 1, verbose = 0, perplexity = 40, n_iter = 300)
mds = MDS(n_components = 1)

ext_pc = pca.fit_transform(ext_df)
ext_pc_df = pd.DataFrame(data = ext_pc, columns = ['extroversion'])
print("Extroversion PCA total explained variance = ", round(pca.explained_variance_ratio_.sum(), 2))

ext_tsne = tsne.fit_transform(ext_df)
ext_tsne_df = pd.DataFrame(data = ext_tsne, columns = ['extroversion'])
ext_mds = mds.fit_transform(ext_df)
ext_mds_df = pd.DataFrame(data = ext_mds, columns = ['extroversion'])

neuro_pc = pca.fit_transform(neuro_df)
neuro_pc_df = pd.DataFrame(data = neuro_pc, columns = ['neuroticism'])
print("Neuroticism PCA total explained variance = ", round(pca.explained_variance_ratio_.sum(), 2))

neuro_tsne = tsne.fit_transform(neuro_df)
neuro_tsne_df = pd.DataFrame(data = neuro_tsne, columns = ['neuroticism'])
neuro_mds = mds.fit_transform(neuro_df)
neuro_mds_df = pd.DataFrame(data = neuro_mds, columns = ['neuroticism'])

agree_pc = pca.fit_transform(agree_df)
agree_pc_df = pd.DataFrame(data = agree_pc, columns = ['agreeableness'])
print("Agreeableness PCA total explained variance = ", round(pca.explained_variance_ratio_.sum(), 2))

agree_tsne = tsne.fit_transform(agree_df)
agree_tsne_df = pd.DataFrame(data = agree_tsne, columns = ['agreeableness'])
agree_mds = mds.fit_transform(agree_df)
agree_mds_df = pd.DataFrame(data = agree_mds, columns = ['agreeableness'])

consc_pc = pca.fit_transform(consc_df)
consc_pc_df = pd.DataFrame(data = consc_pc, columns = ['conscientiousness'])
print("Conscientiousness PCA total explained variance = ", round(pca.explained_variance_ratio_.sum(), 2))

consc_tsne = tsne.fit_transform(consc_df)
consc_tsne_df = pd.DataFrame(data = consc_tsne, columns = ['conscientiousness'])
consc_mds = mds.fit_transform(consc_df)
consc_mds_df = pd.DataFrame(data = consc_mds, columns = ['conscientiousness'])

open_pc = pca.fit_transform(open_df)
open_pc_df = pd.DataFrame(data = open_pc, columns = ['openness'])
print("Openness PCA total explained variance = ", round(pca.explained_variance_ratio_.sum(), 2))

open_tsne = tsne.fit_transform(open_df)
open_tsne_df = pd.DataFrame(data = open_tsne, columns = ['openness'])
open_mds = mds.fit_transform(open_df)
open_mds_df = pd.DataFrame(data = open_mds, columns = ['openness'])

trait_pca_dfs = [ext_pc_df, neuro_pc_df, agree_pc_df, consc_pc_df, open_pc_df]
traits_pca_df = pd.concat(trait_pca_dfs, axis = 1)

trait_tsne_dfs = [ext_tsne_df, neuro_tsne_df, agree_tsne_df, consc_tsne_df, open_tsne_df]
traits_tsne_df = pd.concat(trait_tsne_dfs, axis = 1)

trait_mds_dfs = [ext_mds_df, neuro_mds_df, agree_mds_df, consc_mds_df, open_mds_df]
traits_mds_df = pd.concat(trait_mds_dfs, axis = 1)

print('***T-SNE***')
print(traits_tsne_df.to_string())
print('***MDS***')
print(traits_mds_df.to_string())
print('***PCA***')
print(traits_pca_df.to_string())