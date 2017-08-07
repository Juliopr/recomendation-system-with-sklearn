# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 20:59:32 2017

@author: julio
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


#Todas as classes contidas nesse conjunto contem mais que 100 ocorrencias
#Testando a hipotese das probabilidades
dados = pd.read_csv("arquivos_usu_1_com_acess_id.csv")
lb = LabelEncoder()
dados.ddat_dia_semana = lb.fit_transform(dados.ddat_dia_semana)

#analise = dados[["ddat_dia","darq_id"]]


#Para aprendizado supervisionado
x_data = dados.drop("acess_id",axis=1)

#Para aprendizado não supervisionado
#x_data = dados
y_data = dados['acess_id']
#
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=1)
#
#
#mlp = MLPClassifier(hidden_layer_sizes=8,verbose=True,activation="logistic")
#mlp.fit(x_data,y_data)
#
#
#
##
#forest = RandomForestClassifier(n_estimators=161, max_depth=42,
#                                 criterion='entropy',
#                                 min_samples_split=2,
#                                 min_samples_leaf=1)
#forest.fit(x_train,y_train)
#####
#prob_distribuition = forest.predict_proba(x_data)
#x_train,x_test,y_train,y_test = train_test_split(prob_distribuition,y_data,
#                                              test_size=0.15)
##
##
#out_mlp = MLPClassifier(activation="logistic",verbose=True,
#                          hidden_layer_sizes=(len(dados.acess_id.value_counts()) +1,),
#                          max_iter=5000)
##
#out_mlp.fit(x_train,y_train)
#
#
#predictions = out_mlp.predict(x_test)





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
    

#
#recomendation,taxas = recomendations()

prop = {}

for x in recomendation:
    for y in x:
        if y not in prop.keys():
            prop[y] = 1
        else:
            prop[y] += 1














#Funções temporárias
#--------------------------------------------------------------


info = {
'tipo':'forest',
'loops':(1,100),
'x_train':x_train,
'x_test':x_test,
'y_train':y_train,
'y_test':y_test,
}



def make_acuracy_plot(info,verbose=False):
    acur = []
    cohen = []
    
    if info['tipo'] == "tree":
        for x in range(info['loops'][0],info['loops'][1]):
            Model = DecisionTreeClassifier(n_estimators=161, max_depth=82,criterion='giny',
                                           min_sample_split=2,
                                           min_sample_life=(x/200.)
                                           )
            Model.fit(info['x_train'],info['y_train'])
            predictions = Model.predict(info['x_test'])
            acur.append(metrics.accuracy_score(info['y_test'],predictions))
   
    if info['tipo'] == "forest":
        for x in range(info['loops'][0],info['loops'][1]):
            Model = RandomForestClassifier(n_estimators=10,min_samples_split=(x/200.))
                                           
            Model.fit(info['x_train'],info['y_train'])
            predictions = Model.predict(info['x_test'])
            acur.append(metrics.accuracy_score(info['y_test'],predictions))
            cohen.append(metrics.cohen_kappa_score(info['y_test'],predictions))
            if verbose==True:
               print("{} iteração".format(x))
    
    fig = plt.figure(224)
    fig.add_subplot(221)
    plt.plot(range(info['loops'][0],info['loops'][1]),acur)
    plt.title("acuracy_score")
    fig.add_subplot(222)
    plt.plot(range(info['loops'][0],info['loops'][1]),cohen)
    plt.title("Cohen_kappa_score")









