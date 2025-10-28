# ------------------------------------------------------------
#|Criação de software para ministrar todas os outros arquivos  |
# ------------------------------------------------------------
"""
Módulo: main

Este módulo é responsável por juntar todos os outros módulos.

Objetivo:
- Criar programa que irá inicializar os outros módulos de forma harmônica.

Com isto será possível extrair dados, classificar e plotar a cônica.

"""
#Bibliotecas utilizadas
from Interface import extracao
from linear_algebra_conics import classificacao_conica
from graphic import graph
from sympy import Matrix


while (True):
    #Iniciando variáveis
    A = B = C = D = E = F = None
    boolean_value = False
    λ1 = λ2 = a = b = f = tipo = None
    Q = Matrix([[1,0],
                [0,1]]) #Definindo uma Matriz Q numérica

    try:
        A, B, C, D, E, F, boolean_value = extracao() #Interface
        tipo, Q, λ1, λ2, a, b, f = classificacao_conica(A, B, C, D, E, F) #Classificação da cônica
    except ValueError as VE:
        print(f"Erro: {VE}")

    if(boolean_value == True):
        break

    #'Plotagem' do gráfico
    coef_eqg = [A, B, C, D, E, F] #Coeficientes da equação geral
    clasf_c = [λ1, λ2, a, b, f] #Variáveis necessárias para plotar a quação reduzida
    graph(coef_eqg, clasf_c, Q, tipo)