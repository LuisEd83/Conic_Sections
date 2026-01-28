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
from numpy import array


while (True):
    #Iniciando variáveis
    A = B = C = D = E = F = None
    λ1 = λ2 = a = b = f = tipo = None
    boolean_value = autova_r = None
    Q = array([[1, 0],
                [0, 1]], float) #Definindo uma Matriz Q numérica

    try:
        A, B, C, D, E, F, boolean_value = extracao() #Interface
        tipo, Q, λ1, λ2, a, b, f, autova_r = classificacao_conica(A, B, C, D, E, F) #Classificação da cônica
    except ValueError as VE:
        print(f"Erro: {VE}")

    if(boolean_value == True):
        break

    #'Plotagem' do gráfico
    coef_eqg = [A, B, C, D, E, F] #Coeficientes da equação geral
    clasf_c = [λ1, λ2, a, b, f] #Variáveis necessárias para plotar a quação reduzida
    graph(coef_eqg, clasf_c, Q, tipo, autova_r)

    with open("scr/results.txt", "w", encoding="utf-8") as arq:
        arq.write(f"Equacao digitada pelo usuario: ({A:5.4f})x² + ({B:5.4f})xy + ({C:5.4f})y² + ({D:5.4f})x + ({E:5.4f})y + ({F:5.4f}) = 0" + '\n')
        arq.write(f"Autovalores calculados: λ1 = {λ1:5.4f}, λ2 = {λ2:5.4f}" + '\n')
        arq.write(f"Autovetores calculados: V1 = ({Q[0][0]:5.4f}, {Q[1][0]:5.4f}), V2 = ({Q[0][1]:5.4f}, {Q[1][1]:5.4f})" + '\n')
        if(λ1*λ2 != 0):
            arq.write(f"Equacao da Forma Padrao: ({λ1:5.4f})x² + ({λ2:5.4f})y² + ({f:5.4f}) = 0")
        elif(autova_r):
            arq.write(f"Equacao da Forma Padrao: ({a:5.4f})x + ({b:5.4f})y² + ({f:5.4f}) = 0")
        else:
            arq.write(f"Equacao da Forma Padrao: ({a:5.4f})x² + ({b:5.4f})y + ({f:5.4f}) = 0")