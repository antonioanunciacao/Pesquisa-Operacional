#Nome: Antonio Carlos da Anunciação
#Disciplina: Pesquisa Operacional, Eng.Sistemas UFMG
#Exercicio Computacional

#Problema:
#         Minimizacao do custo de Manutencao

import sys, os
import cplex
import docplex.mp
from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt

clusterDB = np.genfromtxt('clusterDB.csv',delimiter=',')
EquipDB = np.genfromtxt('EquipDB.csv',delimiter=',')
MPDB = np.genfromtxt('MPDB.csv',delimiter=',')

aux = np.zeros((len(EquipDB), 2))
for i in range(0, len(EquipDB)):
    aux[i,] = clusterDB[int((EquipDB[i,2]-1)),1:]
data = np.hstack((EquipDB, aux))
data = np.delete(data, 2, 1)

def Weibull(t, ni, beta):
    F = 1 - np.exp(-(t/ni)**beta)
    return F

def probabilidadeFalha(function, delta_t, x, k):
    t, ni, beta =x[1], x[3], x[4]
    p = np.zeros((len(k)))
    for i in range(0, len(k)):
        p[i] = (function(t+k[i]*delta_t, ni, beta)-function(t, ni, beta))/(1-function(t, ni, beta))
    return p
	
k, delta_t, c_m = MPDB[:,1], 5, MPDB[:,-1]

p_falha_i = np.zeros((500,3))
for i in range(0, len(EquipDB)):
    p_falha_i[i,] = probabilidadeFalha(Weibull, delta_t, data[i,], k)
	
cpf = np.zeros((500,3))
for i in range(0,500):
    cpf[i,] = p_falha_i[i,]*data[i,2]
	
def objFunction_1(x):
    custo_esperado = 0
    if type(x)==dict:
        for i in range(0,500):
            custo_esperado = custo_esperado + cpf[i,0]*x[(i,0)] + cpf[i,1]*x[(i,1)] + cpf[i,2]*x[(i,2)]
            #print("Custo acumulado esperado:", custo_esperado)
        return custo_esperado
    else:
        x_array = np.array(x)
        X_matrix = x_array.reshape((500,3))
        for i in range(0,500):
            custo_esperado = custo_esperado + cpf[i,0]*X_matrix[i,0] + cpf[i,1]*X_matrix[i,1] + cpf[i,2]*X_matrix[i,2]
            #print("Custo acumulado esperado:", custo_esperado)
        return custo_esperado


def objFunction_2(x):
    custo_manutencao = 0
    if type(x)==dict:
        for i in range(0,500):
            custo_manutencao = custo_manutencao + c_m[0]*x[(i,0)] + c_m[1]*x[(i,1)] + c_m[2]*x[(i,2)]
            #print("Custo acumulado de manuntencao:", custo_manutencao)
        return custo_manutencao
    else:
        x_array = np.array(x)
        X_matrix = x_array.reshape((500,3))
        for i in range(0,500):
            custo_manutencao = custo_manutencao + c_m[0]*X_matrix[i,0] + c_m[1]*X_matrix[i,1] + c_m[2]*X_matrix[i,2]
            #print("Custo acumulado de manutencao:", custo_manutencao)
        return custo_manutencao


		
lista_solucoes = []
lista_Pareto = []

for custo_max in range(0, 1001):

    modelo = Model(name='Rotina de Manutenção')
    x = modelo.binary_var_matrix(500, 3)

    modelo.add_constraint(objFunction_2(x) <= custo_max)
    for i in range(0,500):
        modelo.add_constraint(x[(i,0)]+x[(i,1)]+x[(i,2)] == 1)

    modelo.minimize(objFunction_1(x))
    solution = modelo.solve()
    X_solution = solution.get_all_values()

    X_array = np.array(X_solution)
    X_matrix = X_array.reshape((500,3))

    pareto = np.array([objFunction_1(X_solution), objFunction_2(X_solution)])

    print("Iteracao:", custo_max, "de 1000")

    lista_solucoes.append(X_matrix)
    lista_Pareto.append(pareto)

resultado = np.zeros((len(lista_solucoes), len(EquipDB)))
for i in range(0, len(lista_solucoes)):
    for j in range(0,len(lista_solucoes[i])):
        resultado[i,j] = 1 + np.argmax(lista_solucoes[i][j,:], axis=0)
        
np.savetxt("resultados.csv", resultado, delimiter=",")
print("Fim da execução")