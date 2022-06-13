import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

random_numbers = np.random.randint(0, 10000, size = (900, 5))

classes = np.array(['Class 1'] * 300 + ['Class 2'] * 300 + ['Class 3'] * 300)

training_data, test_data, training_classes, test_classes = train_test_split(random_numbers, classes, test_size=0.2, random_state=0)

sc = StandardScaler()
training_data = sc.fit_transform(training_data)
test_data = sc.fit_transform(test_data)

pca_training_model = PCA(n_components = 2)
training_data_pcs = pca_training_model.fit_transform(training_data)

lda_training_model = LinearDiscriminantAnalysis(n_components = 2)
training_data = lda_training_model.fit_transform(training_data, training_classes)
test_data = lda_training_model.transform(test_data)

lda_df = pd.DataFrame(data = training_data, index = training_classes)
pca_df = pd.DataFrame(data = training_data_pcs, index = training_classes)

print(lda_df.head(n = 50))
print(pca_df.head(n = 50))

classifier = RandomForestClassifier(max_depth = 2, random_state = 0)

classifier.fit(training_data, training_classes)
class_prediction = classifier.predict(test_data)

cm = confusion_matrix(test_classes, class_prediction)

print('LDA model accuracy = ' + str(accuracy_score(test_classes, class_prediction)))

classifier.fit(training_data_pcs, training_classes)
class_prediction = classifier.predict(test_data)

cm = confusion_matrix(test_classes, class_prediction)

print('PCA model accuracy = ' + str(accuracy_score(test_classes, class_prediction)))
