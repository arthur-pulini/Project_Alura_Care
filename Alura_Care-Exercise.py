import pandas as pd
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SEED = 123143
random.seed(SEED)

datas = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura_Care-Exercise/exames.csv')
print(datas)
print(datas.isnull().sum())#descobrindo quais colunas passuem valores nulos e quantos são

#Preparando quais colunas serão utilizadas em x e y
examsValues = datas.drop(['id', 'diagnostico'], axis=1)
diagnosis = datas.diagnostico
examsValuesV1 = examsValues.drop(['exame_33'], axis=1)
print(examsValues)
print(diagnosis)

#separando entre treino e teste, X e Y
trainX, testX, trainY, testY = train_test_split(examsValuesV1, diagnosis, test_size=0.3)

#instanciando o RandomForestClassifier, usando o numero padrão de árvores
forestClassifier = RandomForestClassifier(n_estimators = 100)
forestClassifier.fit(trainX, trainY) #Ajustando o classificador
print(forestClassifier.score(testX, testY) * 100)