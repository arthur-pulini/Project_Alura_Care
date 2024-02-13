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

SEED = 1234
random.seed(SEED)

datas = pd.read_csv('/home/arthur-pulini/Documentos/Programação/Machine learning Alura/Alura_Care-Exercise/exames.csv')
print(datas)
print(datas.isnull().sum())#descobrindo quais colunas passuem valores nulos e quantos são

#Preparando quais colunas serão utilizadas em x e y
examsValues = datas.drop(['id', 'diagnostico'], axis=1)
diagnostico = datas.diagnostico
examsValuesV1 = examsValues.drop(['exame_33'], axis=1)
print(examsValues)
print(diagnostico)

#separando entre treino e teste, X e Y
trainX, testX, trainY, testY = train_test_split(examsValuesV1, diagnostico, test_size=0.3)

#instanciando o RandomForestClassifier, usando o numero padrão de árvores
forestClassifier = RandomForestClassifier(n_estimators = 100)
forestClassifier.fit(trainX, trainY) #Ajustando o classificador
print(forestClassifier.score(testX, testY) * 100)

dummyClassifier = DummyClassifier(strategy = "most_frequent")
dummyClassifier.fit(trainX, trainY)
print(dummyClassifier.score(testX, testY) * 100)
#a partir dos resultados, o forestClassifier será usado como baseline para escolher as melhores features

#Normalizando os valores a fim de melhorar a vizualização do gráfico
standardizer = StandardScaler()
standardizer.fit(examsValuesV1)
examsValuesV2 = standardizer.transform(examsValuesV1)
examsValuesV2 = pd.DataFrame(data = examsValuesV2, columns = examsValuesV1.keys()) #transformando o examsValuesV2 em um Dataframe, o nome das clunas serão iguais os examsValuesV1
print(examsValuesV2)

def violinGraphic(values, start, end):
    #fazendo a concatenação e informando que diagnosis é a primeira coluna
    datasPlot = pd.concat([diagnostico, values.iloc[:, start:end]], axis=1)
    datasPlot = pd.melt(datasPlot, id_vars="diagnostico", var_name="exams", value_name='valores')
    #configurando o gráfico de tipo violino
    plt.figure(figsize=(10, 10))
    sns.violinplot(x = 'exams', y = 'valores', hue = 'diagnostico', data = datasPlot, split= True)
    plt.xticks(rotation = 90)
    plt.show()

#Retirando as colunas que possuem valores constantes, estes vaçores não fornecem informações
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

#gerando a correlação entre os exames 
correlationMatrix = examsValuesV3.corr()
plt.figure(figsize=(17, 15))
sns.heatmap(correlationMatrix, annot= True, fmt= ".1f") #com o heat map a visualização fica mais prética
#plt.show()

#pegando os dados da matrix acima de 0.99
correlationMatrixv1 = correlationMatrix[correlationMatrix > 0.99]
print(correlationMatrixv1)

#vendo qual coluna possui valores acima de 1, ou seja, colunas que são altamente correlacionadas
correlationMatrixv2 = correlationMatrixv1.sum()
print(correlationMatrixv2)

#separando apenas as variaveis altamente correlacionadas
correlatedVariables = correlationMatrixv2[correlationMatrixv2 > 1]
print(correlatedVariables)

#retirando as colunas altamente correlacionadas
examsValuesV4 = examsValuesV3.drop(columns=correlatedVariables.keys())
classify(examsValuesV4)

#adicionando apenas duas das colunas altamente correlacionadas, pois, a acuracia de examsValuesV4 deu menor que a baseline
examsValuesV5 = examsValuesV3.drop(columns=['exame_3', 'exame_24'])
classify(examsValuesV5)

#selecionando as 5 melhores features com a função chi2
selectKBest = SelectKBest(chi2, k = 5)

examsValuesV6 = examsValuesV1.drop(columns=['exame_3', 'exame_24', 'exame_4', 'exame_29'])
print(examsValuesV6)
trainX, testX, trainY, testY = train_test_split(examsValuesV6, diagnostico, test_size=0.3)
selectKBest.fit(trainX, trainY)
trainKBest = selectKBest.transform(trainX)
testKBest = selectKBest.transform(testX)

#medindo a acurácia a partir das 5 features selecionadas pelo SelectKBest
classifier = RandomForestClassifier(n_estimators=100, random_state=1234)
classifier.fit(trainKBest, trainY)
print(classifier.score(testKBest, testY) * 100)

#aplicando a matrix de confusão, a fim de saber se com as features escolhidas o modelo se sai bem
confusionMatrix = confusion_matrix(testY, classifier.predict(testKBest))
print(confusionMatrix)
plt.figure(figsize=(17, 15))
sns.set()
sns.heatmap(confusionMatrix, annot= True, fmt= "d").set(xlabel = "Predict", ylabel = "Real")
#plt.show()
