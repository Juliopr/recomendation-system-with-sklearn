# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 20:59:32 2017

@author: julio
"""

import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder




#Você pode achar a descrição do conjuto de dados na seção Datasets.
#No arquivo README.md esta contido um relatorio das variaveis envolvidas no problema
dados = pd.read_csv("arquivos_usu_1_com_acess_id.csv")


lb = LabelEncoder()
dados.ddat_dia_semana = lb.fit_transform(dados.ddat_dia_semana)


#Para aprendizado supervisionado
#x_data se refere as features e y_data se refere ao target
x_data = dados.drop("acess_id",axis=1)
y_data = dados['acess_id']

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=1)

#O random forest é o modelo de entrada.
#Ele é usado apenas para associar probabilidades dado uma classe a cada instância.
forest = RandomForestClassifier(n_estimators=161, max_depth=42,
                                 criterion='entropy',
                                 min_samples_split=2,
                                 min_samples_leaf=1)
forest.fit(x_train,y_train)

#Aqui são criadas novas features onde as linhas são as instâncias e
#as colunas são as probabilidades delas pertecerem a uma determinada classe.
prob_distribuition = forest.predict_proba(x_data)

x_train,x_test,y_train,y_test = train_test_split(prob_distribuition,y_data,
                                              test_size=0.15)
#O Multilayer perceptron usa essas distribuições de probabilidades para
#treino. similar a softmax regression.
out_mlp = MLPClassifier(activation="logistic",verbose=True,
                          hidden_layer_sizes=(len(dados.acess_id.value_counts()) +1,),
                          max_iter=5000)

out_mlp.fit(x_train,y_train)


#Essa função retorna uma 2d array onde cada linha
#contem 5 recomendações de arquivos para o usuário naquele dia.
def recomendations():
    classes = out_mlp.classes_

    pesos = out_mlp.predict_proba(x_test)

    
    recomendations = [] 
    
    for x in pesos:
        indicado = []
        for _ in range(5):
            indicado.append(classes[x.argmax()])
            x[x.argmax()] = 0
        recomendations.append(indicado)
    
    errado = 0
    certo = 0
    
    y = y_test.values
    
    for x in range(len(recomendations)):
        if y[x] in recomendations[x]:
            certo += 1
        else:
            errado += 1
    return recomendations,("certo:{}".format(certo),"errado{}:".format(errado))
    

#Variavel para teste
#recomendation,taxas = recomendations()












