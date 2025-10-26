"""
Criação de programa para calcular as Seções de Cônicas
"""
#Bibliotecas utilizadas
from Interface import extracao
from linear_algebra_conics import classificacao_conica
from graphic import graph
from sympy import Matrix

#Iniciando variáveis
A = B = C = D = E = F = None
a = b = f = tipo = None
Q = Matrix([[1,0],
            [0,1]]) #Definindo uma Matriz Q numérica

try:
    A, B, C, D, E, F = extracao() #Interface
    tipo, Q, λ1, λ2, a, b, f = classificacao_conica(A, B, C, D, E, F) #Classificação da cônica
except ValueError as VE:
    print(f"Erro: {VE}")

#'Plotagem' do gráfico
coef_eqg = [A, B, C, D, E, F]#Coeficientes da equação geral
clasf_c = [λ1, λ2, a, b, f] #Coeficientes da equação geral reduzida
graph(coef_eqg, clasf_c, Q, tipo)