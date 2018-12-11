# -*- coding: utf-8 -*-
#Transformações de variáveis categóricas(STRING) para valores numéricos.
#Alguns algoritmos não conseguem lidar com variáveis categoricas e por isso fazemos a transformação. Por exemplo, SVM ou regressão linear, somente trabalham com valores numéricos.

import pandas as pd

base = pd.read_csv('../pre-processamento/bases/census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

#CONVERSÃO DE VALORES CATEGÓRICOS PARA NUMÉRICOS
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_previsores = LabelEncoder()
#labels = label_previsores.fit_transform(previsores[:, 1])
previsores[:, 1] = label_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = label_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = label_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = label_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = label_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = label_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = label_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = label_previsores.fit_transform(previsores[:, 13])
##

#Em previsores.workclass temos agora os valores: 1,2,3...7. Pode acontencer que o algoritmo considere o número 7 mais importante que os outros na hora de gerar o cálculo. Para contornar tal situação podemos aplicar o dummy.
#O ideal é aplicar quando as variáveis não forem ordinais.
onehot = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
previsores = onehot.fit_transform(previsores).toarray()

#transformação da classe
label_classe = LabelEncoder()
classe = label_classe.fit_transform(classe) 

#ESCALONAMENTO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores[:, 102:108] = scaler.fit_transform(previsores[:, 102:108])

#VALIDAÇÃO CRUZADA
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(previsores, classe, test_size=0.15, random_state=0)
