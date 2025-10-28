# -----------------------------------------
#|Criação de software para plotar gráficos |
# -----------------------------------------

"""
Módulo: Gráfico

Este módulo será responsável pela parte visual do resultado encontrado pela parte matemática
- i.e, da parte de classificação da cônica. Em suma, este módulo implementa um algoritmo ca-
paz de 'plotar' o gráfico da seção cônica encontrada.

Objetivo:
- Mostrar, de forma visual (por meio de uma janela), as consequências das mudanças de variá-
veis, da rotação dos eixos, etc. utilizando as bibliotecas Matplotlib e Numpy.

Com isso feito, será possível deixar mais dinâmico a forma como é calculada/encontrada a se-
ção cônica.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import Matrix, symbols, solve

def centro_conica(A, B, C, D, E):
    x,y = symbols("x y") 
    
    eqx = sp.Eq(2*A*x + B*y + D, 0) 
    eqy = sp.Eq(B*x + 2*C*y + E, 0) #Resolvendo este sistema, teremos o centro da seção cônica 
    
    sol = solve([eqx, eqy], [x, y]) 

    if(sol):
        if(len(sol) == 1):
            try:
                var_x = var_y = float(sol[x]) 
            except:
                var_x = var_y = float(sol[y])
        else:
            var_x = float(sol[x])
            var_y = float(sol[y])
        return [var_x, var_y] 
    else: 
        return [0,0]

def graph(coef_eqg: list, clasf_c : list, Q : Matrix, tipo : str):
    G = coef_eqg[:] #Cria uma cópia de uma lista (para facilitar na escrita)
    R = clasf_c[:] #[λ1, λ2, a, b, f]

    #Determinando os eixos do primeiro gráfico:
    r = 18

    x1 = np.linspace(-r, r, int(np.sqrt(r))*200) #intervalo x ∈ [-(r + 1), (r + 1)]. Este intervalo é dividido em 1000 partes
    y1 = np.linspace(-r, r, int(np.sqrt(r))*200) #Análogo ao x
    X1, Y1 = np.meshgrid(x1, y1) #Cria uma grade com dois eixos, i.e, um plano cartesiano
    
    #Determinando a equação geral:
    Z1 = G[0]*(X1**2) + G[1]*X1*Y1 + G[2]*(Y1**2) + G[3]*X1 + G[4]*Y1 + G[5]

    #Criando o gráfico:
    plt.figure(figsize = (12, 4.5)) #Determina o tamanho da figura onde estará o gráfico

    #Determinando o centro da cônica original
    centro_og = centro_conica(G[0], G[1], G[2], G[3], G[4])
    
    #Configurando o primeiro gráfico
    plt.subplot(1, 2, 1)
    plt.contour(X1, Y1, Z1, levels=[0], colors='blue') #Determina a curva de nível (Z = 0)
    
    #'Plotando' os vetores canônicos (para melhor visualização):
    plt.quiver(centro_og[0], centro_og[1], 1, 0, scale_units = 'xy', scale = 1, color = 'k')
    plt.quiver(centro_og[0], centro_og[1], 0, 1, scale_units = 'xy', scale = 1, color = 'k')

    #'Plotando' o centro da cônica geral
    plt.scatter(centro_og[0], centro_og[1], color='r', s=50, marker='.', label='Centro')
    plt.legend() 

    #Criando os eixos:
    plt.axhline(0, color = 'k', linewidth = 0.5) #Determina o eixo horizontal
    plt.axvline(0, color = 'k', linewidth = 0.5) #Determina o eixo vertical
    plt.gca().set_aspect('equal', adjustable = 'box') #Configura a forma como os valores são exibidos visualmente
    plt.title(f"Cônica original \n ({G[0]:5.2f})x² + ({G[1]:5.2f})xy + ({G[2]:5.2f})y² + ({G[3]:5.2f})x + ({G[4]:5.2f})y + ({G[5]:5.2f}) = 0",
              fontdict={
                    'weight': 'bold',      
                    'size': 10           
                })
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    #Determinando os eixos do segundo gráfico:
    x2 = np.linspace(-r, r, int(np.sqrt(r))*200) #intervalo x ∈ [-(r + 1), (r + 1)]. Este intervalo é dividido em 1000 partes
    y2 = np.linspace(-r, r, int(np.sqrt(r))*200) #Análogo ao x
    X2, Y2 = np.meshgrid(x2, y2) #Cria uma grade com dois eixos, i.e, um plano cartesiano

    #Determinando a equação reduzida:
    Z2 = None
    if(R[0]*R[1] !=0):
        Z2 = R[2]*(X2**2) + R[3]*(Y2**2) + R[4] #a*x² + b*y² + f
    elif((R[0] == 0) and (R[1] != 0)):
        Z2 = R[2]*X2 + R[3]*(Y2**2) + R[4] #a*x + b*y² + f
    elif((R[0] != 0) and (R[1] == 0)):
        Z2 = R[2]*(X2**2) + R[3]*(Y2) + R[4] #a*x² + b*y + f

    #Configurando o segundo gráfico
    plt.subplot(1, 2, 2)
    plt.contour(X2, Y2, Z2, levels=[0], colors='blue') #Determina a curva de nível (Z = 0)

    #'Plotando' os vetores associados aos autovalores (para melhor visualização):
    Q = np.array(Q.tolist(), dtype=float) #Convertendo Q para numpy 
    plt.quiver(0, 0, Q[0][0], Q[1][0], scale_units = 'xy', scale = 1, color = 'r')
    plt.quiver(0, 0, Q[0][1], Q[1][1], scale_units = 'xy', scale = 1, color = 'r')

    #'Plotando' o centro da cônica reduzida
    plt.scatter(0, 0, color='k', s=50, marker='.', label='Centro')
    plt.legend()

    #Criando os eixos:
    plt.axhline(0, color = 'k', linewidth = 0.5) #Determina o eixo horizontal
    plt.axvline(0, color = 'k', linewidth = 0.5) #Determina o eixo vertical
    plt.gca().set_aspect('equal', adjustable = 'box') #Configura a forma como os valores são exibidos visualmente
    if(R[0]*R[1] !=0):
        plt.title(f"{tipo} \n ({R[2]:5.2f})x² + ({R[3]:5.2f})y² + ({R[4]:5.2f}) = 0",
              fontdict={
                    'weight': 'bold',      
                    'size': 10           
                }) #O título será, também a classificação da Cônica
    elif((R[0] == 0) and (R[1] != 0)):
        plt.title(f"{tipo} \n ({R[2]:5.2f})x + ({R[3]:5.2f})y² + ({R[4]:5.2f}) = 0",
              fontdict={
                    'weight': 'bold',      
                    'size': 10           
                })
    elif((R[0] != 0) and (R[1] == 0)):
        plt.title(f"{tipo} \n ({R[2]:5.2f})x² + ({R[3]:5.2f})y + ({R[4]:5.2f}) = 0",
              fontdict={ 
                    'weight': 'bold',      
                    'size': 10           
                })
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()