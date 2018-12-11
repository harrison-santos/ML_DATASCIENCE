# -*- coding: utf-8 -*-
#1-carregamento de dados. 2-Tratamento de valores inconsistentes. 3-Tratamento de valores faltantes. 4- Escalonamento de atributos
import pandas as pd
pd.set_option("display.max_columns", 8)
base = pd.read_csv('../pre-processamento/bases/credit-data.csv')
base.describe()


#1) Valores inconsistentes: valores negativos em idade. 
#Possíveis Passos: Apagar os registros com problemas, preencher manualmente, preencher valores com a média/mediana/most_frequent.
base.loc[base['age'] < 0]
base.mean()#faz a média com valores negativos
age_mean = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = age_mean 
base.loc[base['age'] < 0]#valor corrigido

#2) Valores faltantes
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values#todas as linhas, e colunas do index 1 até 3
classe = base.iloc[:, 4].values#todas as linhas, e a coluna no index 4


#tratamento de valores null com sklearn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

#Escalonamento de Atributos. Podemos escalonar através de padronização,ou também normalização. Com isso os algoritmos não farão cálculos matemáticos com números muito grandes.
#Padronização: x = x - média(x)/desvio padrão(x). Normalização: x = x - mínimo(x)/máximo(x)-mínimo(x)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#X = previsores e Y = classe preditora.
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(previsores, classe, test_size=0.25, random_state=0)#25% da base para teste.





