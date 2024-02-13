import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


SEED = 1234
random.seed(SEED)

datas = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura_Care-Exercise/exames.csv')
print(datas)
print(datas.isnull().sum())

examsValues = datas.drop(['id', 'diagnostico'], axis=1)
diagnostico = datas.diagnostico
examsValuesV1 = examsValues.drop(['exame_33'], axis=1)
print(examsValues)
print(diagnostico)

trainX, testX, trainY, testY = train_test_split(examsValuesV1, diagnostico, test_size=0.3)

forestClassifier = RandomForestClassifier(n_estimators = 100)
forestClassifier.fit(trainX, trainY) 
print(forestClassifier.score(testX, testY) * 100)

dummyClassifier = DummyClassifier(strategy = "most_frequent")
dummyClassifier.fit(trainX, trainY)
print(dummyClassifier.score(testX, testY) * 100)

standardizer = StandardScaler()
standardizer.fit(examsValuesV1)
examsValuesV2 = standardizer.transform(examsValuesV1)
examsValuesV2 = pd.DataFrame(data = examsValuesV2, columns = examsValuesV1.keys()) 
print(examsValuesV2)

def violinGraphic(values, start, end):
    datasPlot = pd.concat([diagnostico, values.iloc[:, start:end]], axis=1)
    datasPlot = pd.melt(datasPlot, id_vars="diagnostico", var_name="exams", value_name='valores')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x = 'exams', y = 'valores', hue = 'diagnostico', data = datasPlot, split= True)
    plt.xticks(rotation = 90)
    plt.show()

examsValuesV3 = examsValuesV2.drop(['exame_4', 'exame_29'], axis=1)
print(".")
print(examsValuesV3)

def classify(values):
    SEED = 1234
    random.seed(SEED)
    trainX, testX, trainY, testY = train_test_split(values, diagnostico, test_size=0.3)
    forestClassifier = RandomForestClassifier(n_estimators = 100)
    forestClassifier.fit(trainX, trainY) 
    print(forestClassifier.score(testX, testY) * 100)

classify(examsValuesV3)

correlationMatrix = examsValuesV3.corr()
plt.figure(figsize=(17, 15))
sns.heatmap(correlationMatrix, annot= True, fmt= ".1f") 
#plt.show()

correlationMatrixv1 = correlationMatrix[correlationMatrix > 0.99]
print(correlationMatrixv1)

correlationMatrixv2 = correlationMatrixv1.sum()
print(correlationMatrixv2)

correlatedVariables = correlationMatrixv2[correlationMatrixv2 > 1]
print(correlatedVariables)

examsValuesV4 = examsValuesV3.drop(columns=correlatedVariables.keys())
classify(examsValuesV4)

examsValuesV5 = examsValuesV3.drop(columns=['exame_3', 'exame_24'])
classify(examsValuesV5)

selectKBest = SelectKBest(chi2, k = 5)

examsValuesV6 = examsValuesV1.drop(columns=['exame_3', 'exame_24', 'exame_4', 'exame_29'])
print(examsValuesV6)
trainX, testX, trainY, testY = train_test_split(examsValuesV6, diagnostico, test_size=0.3)
selectKBest.fit(trainX, trainY)
trainKBest = selectKBest.transform(trainX)
testKBest = selectKBest.transform(testX)

classifier = RandomForestClassifier(n_estimators=100, random_state=1234)
classifier.fit(trainKBest, trainY)
print(classifier.score(testKBest, testY) * 100)

confusionMatrix = confusion_matrix(testY, classifier.predict(testKBest))
print(confusionMatrix)
plt.figure(figsize=(17, 15))
sns.set()
sns.heatmap(confusionMatrix, annot= True, fmt= "d").set(xlabel = "Predict", ylabel = "Real")
#plt.show()

classifier = RandomForestClassifier(n_estimators=100, random_state=1234)
classifier.fit(trainX, trainY)
rfeSelector = RFE(estimator = classifier, n_features_to_select = 5, step = 1)
rfeSelector.fit(trainX, trainY)
trainRfe = rfeSelector.transform(trainX)
testRfe = rfeSelector.transform(testX)
classifier.fit(trainRfe, trainY)
print(classifier.score(testRfe, testY) * 100)


confusionMatrix = confusion_matrix(testY, classifier.predict(testRfe))
print(confusionMatrix)
plt.figure(figsize=(17, 15))
sns.set()
sns.heatmap(confusionMatrix, annot= True, fmt= "d").set(xlabel = "Predict", ylabel = "Real")
#plt.show()

classifier = RandomForestClassifier(n_estimators=100, random_state=1234)
classifier.fit(trainX, trainY)
rfecvSelector = RFECV(estimator = classifier, cv = 5, step = 1, scoring = "accuracy")
rfecvSelector.fit(trainX, trainY)
trainRfecv = rfecvSelector.transform(trainX)
testRfecv = rfecvSelector.transform(testX)
classifier.fit(trainRfecv, trainY)
print(classifier.score(testRfecv, testY) * 100)
print(rfecvSelector.n_features_)
print(trainX.columns[rfecvSelector.support_])

confusionMatrix = confusion_matrix(testY, classifier.predict(testRfecv))
print(confusionMatrix)
plt.figure(figsize=(17, 15))
sns.set()
sns.heatmap(confusionMatrix, annot= True, fmt= "d").set(xlabel = "Predict", ylabel = "Real")
#plt.show()

plt.figure(figsize= (14, 8))
plt.xlabel("Nº exams")
plt.ylabel("accuracy")
plt.plot(range(1, len(rfecvSelector.cv_results_['mean_test_score']) + 1), rfecvSelector.cv_results_['mean_test_score'])
#plt.show()

pca = PCA(n_components=2)
examsValuesV7 = pca.fit_transform(examsValuesV5) 
plt.figure(figsize= (14, 8))
sns.scatterplot(x = examsValuesV7[:,0], y = examsValuesV7[:,1], hue = diagnostico)
#plt.show()

tsne = TSNE(n_components=2)
examsValuesV8 = tsne.fit_transform(examsValuesV5)
plt.figure(figsize= (14, 8))
sns.scatterplot(x = examsValuesV8[:,0], y = examsValuesV8[:,1], hue = diagnostico)
#plt.show()

#Juntamente com as técnicas de técnicas analíticas de seleção de features(Detecção de valores não preenchidos, valores constantes e correlacionados), 
#ficou concluído que para este projeto, a melhor técnica de seleção de features automática foi a RFECV, onde por si só escolhe quais e quantas features
#adquirem o melhor resultado 
