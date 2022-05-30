import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import sklearn.datasets as dt
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# separate the variables from the subject to which those variables apply
x = df.loc[:, features].values
y = df.loc[:,['target']].values
# standardize variables
x = StandardScaler().fit_transform(x)

# reduce 4-dimensional iris data set to 3 principal components
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', 'pc3'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

print('***3D PCA DATA FRAME***')
print(finalDf.to_string())

# reduce 3-dimensional principal component space to 2-dimensions via t-sne
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_transform = tsne.fit_transform(principalComponents)

print('\n***3D PCA --> 2D T-SNE***')
print(tsne_transform)

# reduce 3-dimensional principal component space to 2-dimensions via multidimensional scaling
mds = MDS(random_state=0)
mds_transform = mds.fit_transform(principalComponents)

print('\n***3D PCA --> 2D MDS***')
print(mds_transform)

# check mds stress 
stress = mds.stress_

# terrible fit?...
print(stress)




