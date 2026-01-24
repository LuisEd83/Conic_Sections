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
import graphic_functions as gf

from sympy import Matrix
from matplotlib.animation import FuncAnimation

def graph(coef_eqg: list, clasf_c : list, Q : Matrix, tipo : str, autova_r : bool):
    G0 = coef_eqg[:] #[A, B, C, D, E, F]
    R = clasf_c[:] #[λ1, λ2, a, b, f]

    #Ponto de referência da cônica. 
    ponto_ref = gf.ponto_representativo(G0[0], G0[1], G0[2], G0[3], G0[4], G0[5], tipo)

    #Definindo o tamanho do gráfico
    r = gf.raio_plot_conica(G0[0], G0[1], G0[2], G0[3], G0[4], ponto_ref)
    
    #Definindo o linspace do tempo (variável temporal).
    t = np.linspace(0, 1, 200) #t ∈ [0, 1] com 200 repartições.

    #Definindo e tratando o ângulo:
    Q = np.array(Q.tolist(), float)
    theta = np.arctan2(Q[1][0], Q[0][0])
    if(tipo == "Vazio"):
        theta = 0.0

    print(f"Valor do ângulo de rotação (em graus): {np.degrees(theta)}")

    #Calculando a posição dos vetores em relação ao tempo t.
    def vectors_rot(Q : Matrix, t):
        #Ângulo de rotação:
        alpha = theta

        #Interpolando o ângulo
        alpha_t = t * alpha #alpha_t ∈ [0, alpha]

        #Vetor 1 (eixo x rotacionado)
        xf1_t = np.cos(alpha_t)
        yf1_t = np.sin(alpha_t)

        #Vetor 2 (eixo y rotacionado)
        xf2_t = -np.sin(alpha_t)
        yf2_t = np.cos(alpha_t)

        #Origem
        xi = 0
        yi = 0

        return [xi, yi, xf1_t, yf1_t, xf2_t, yf2_t]

    #Criando uma figura e seus eixos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    #----------------PRIMEIRA PARTE----------------#

    ax1.set_xlim(-r, r)
    ax1.set_ylim(-r, r)
    ax1.set_aspect('equal', adjustable='box')

    #Parametrização da cônica:
    U, V = gf.parametrizar_conica(tipo, R[0], R[1], R[2], R[3], R[4])
    P = np.stack([U, V])

    #Calculando possível vértice no sistema reduzido (caso a cônica seja uma parábola)
    V_red = np.zeros(2) #Inicializa na origem (0, 0)
    if(tipo == "parabola"):
        if(abs(R[0]) < 1e-12):
            V_red = np.array([-R[4]/R[2], 0.0])
        elif(abs(R[1]) < 1e-12):
            V_red = np.array([0.0, -R[4]/R[3]])

    #Plotando cônica estática as coordenadas xy com correção de ângulo:
    alpha = theta
    Q_co = Q
    
    if((G0[1] == 0) and (tipo == "Parabola") and (autova_r)):
        alpha -= np.pi/2

        #Matriz de rotação corrigida para animação: 
        Q_co = np.array([[np.cos(alpha), -np.sin(alpha)],
                        [np.sin(alpha), np.cos(alpha)]], float)

    XY = Q_co @ (P - V_red.reshape(2, 1)) + ponto_ref.reshape(2, 1) #Rotaciona e translada a cônica parametrizada
    Xp, Yp = XY
    ax1.plot(Xp, Yp, 'b', linewidth = 1.5, zorder = 2)

    #Elementos do gráfico (textos):
    ax1.set_title(f"Animação - Mudança de eixo.\nEquação Geral:\n ({G0[0]:5.2f})x² + ({G0[1]:5.2f})xy + ({G0[2]:5.2f})y² + ({G0[3]:5.2f})x + ({G0[4]:5.2f})y + ({G0[5]:5.2f}) = 0",
              fontdict={
                    'weight': 'bold',      
                    'size': 10           
                })

    #----- Animação -> eixos -----#
    #Inicialização de variáveis
    eixo_autovetor1, = ax1.plot([], [], 'k', linewidth=1)
    eixo_autovetor2, = ax1.plot([], [], 'k', linewidth=1)

    ponto = None
    def update_axis(frame):
        nonlocal ponto
        tp = 0 #Variável para o ponto de referência

        #Limpeza do frame anterior
        try:
            if(ponto is not None):
                ponto.remove()
        except:
            pass

        #Variáveis temporais de frames:
        total_frames = len(t)
        N_rot = total_frames / 2        #Frames para rotação
        N_trans = total_frames - N_rot  #frames para translação

        X = np.linspace(-5*r, 5*r, 200) #Variável auxiliar para criar os eixos.
        
        if((frame < N_rot)):                                         #-----Rotação de eixos -----#
            tt = frame / (N_rot - 1)

            ang = tt * theta
            Q_current = np.array([
                [np.cos(ang), -np.sin(ang)],
                [np.sin(ang),  np.cos(ang)]
            ])

            v1_t = np.array([Q_current[0][0], Q_current[1][0]], float)
            x1 = v1_t[0] * X
            y1 = v1_t[1] * X
            
            v2_t = np.array([Q_current[0][1], Q_current[1][1]], float)
            x2 = v2_t[0] * X
            y2 = v2_t[1] * X

        elif((frame >= N_rot)): #-----Translação dos eixos-----#
            tt = (frame - N_rot)/(N_trans - 1)

            #Com rotação concluída:
            ang = 1 * theta

            Q_current = np.array([
                [np.cos(ang), -np.sin(ang)],
                [np.sin(ang),  np.cos(ang)]
            ])

            v1_t = np.array([Q_current[0][0], Q_current[1][0]], float)
            x1 = v1_t[0] * X + tt * ponto_ref[0] #Translação para a coordenada x do ponto de referência
            y1 = v1_t[1] * X + tt * ponto_ref[1] #Translação para a coordenada y do ponto de referência

            v2_t = np.array([Q_current[0][1], Q_current[1][1]], float)
            x2 = v2_t[0] * X + tt * ponto_ref[0] #Translação para a coordenada x do ponto de referência
            y2 = v2_t[1] * X + tt * ponto_ref[1] #Translação para a coordenada y do ponto de referência

        #Definindo eixos:
        eixo_autovetor1.set_data(x1, y1)
        eixo_autovetor2.set_data(x2, y2)

        return [eixo_autovetor1, eixo_autovetor2]
    
    #----- Animação -> grid -----#

    #Definindo a quantidade de linhas do grid:
    N = 6*(int)(r) #Número de elementos da lista
    
    #Criando listas no formato [None, None, ..., None]:
    grid_list_x = [None for _ in range(N)]
    grid_list_y = [None for _ in range(N)]

    for i in range(N):
        grid_list_x[i], = ax1.plot([], [], 'k', linewidth=0.2, zorder = 1)
        grid_list_y[i], = ax1.plot([], [], 'k', linewidth=0.2, zorder = 1)
    
    def update_grid(frame):

        #Inicialização de variáveis relaciondas a APENAS rotação:
        total_frame = len(t)
        N_rot = total_frame/2

        #Inicialização de variável para criação das linhas do grid:
        X = np.linspace(-5*r, 5*r, 50)

        if((frame < N_rot)): #-----Rotação do grid -----#
            tt = frame / (N_rot - 1)
            
            ang = tt * theta
            Q_current = np.array([
                [np.cos(ang), -np.sin(ang)],
                [np.sin(ang),  np.cos(ang)]
            ])

            #Vetores canônicos:
            v1 = np.array([1, 0], float)
            v2 = np.array([0, 1], float)

            for i in range(N):
                #Para as retas horizontais
                x1 = v1[0] * X + (-N/2 + i) * v2[0]
                y1 = v1[1] * X + (-N/2 + i) * v2[1]
                
                P1 = np.stack([x1, y1])
                XY1r = Q_current @ P1
                x1r, y1r = XY1r

                #Para as retas verticais
                x2 = v2[0] * X + (-N/2 + i) * v1[0]
                y2 = v2[1] * X + (-N/2 + i) * v1[1]
                
                P2 = np.stack([x2, y2])
                XY2r = Q_current @ P2
                x2r, y2r = XY2r

                grid_list_x[i].set_data(x1r, y1r)
                grid_list_y[i].set_data(x2r, y2r)

        grid_list = [grid_list_x, grid_list_y]
        return grid_list
    

    #----- Animação -> vetores -----#
    vetores_quiver = None #Inicialização de valor
    def update_vectors(frame):
        nonlocal vetores_quiver
        
        #Remove os vetores anteriores (caso existam)
        if(vetores_quiver is not None):
            for v in vetores_quiver:
                v.remove()

        total_frames = len(t)
        N_rot = total_frames/2          #Frames para rotação
        N_trans = total_frames - N_rot  #frames para translação

        if((frame < N_rot)): #-----Rotação de vetores-----#
            tt = frame / (N_rot - 1)

            coord_vetores = vectors_rot(Q, tt) #Extração de vetores

            #'Plotando' os vetores associados aos autovalores (para melhor visualização):
            v1 = ax1.quiver(coord_vetores[0], coord_vetores[1], coord_vetores[2], coord_vetores[3], scale_units = 'xy', scale = 1, color = 'g', zorder = 3,)
            v2 = ax1.quiver(coord_vetores[0], coord_vetores[1], coord_vetores[4], coord_vetores[5], scale_units = 'xy', scale = 1, color = 'r', zorder = 3)

        elif((frame >= N_rot)): #-----Translação dos vetores-----#
            tt = (frame - N_rot)/(N_trans - 1)

            coord_vetores_est = vectors_rot(Q, 1) #Extração de coordenadas com os vetores já rotacionados 

            #'Plotando' a translação dos vetores estacionários
            x_t = tt * ponto_ref[0] #Coordenada x(t) para translação
            y_t = tt * ponto_ref[1] #Coordenada y(t) para translação

            v1 = ax1.quiver(x_t, y_t, coord_vetores_est[4], coord_vetores_est[5], scale_units = 'xy', scale = 1, color = 'r', zorder = 3)
            v2 = ax1.quiver(x_t, y_t, coord_vetores_est[2], coord_vetores_est[3], scale_units = 'xy', scale = 1, color = 'g', zorder = 3,)

        vetores_quiver = [v1, v2]

        return vetores_quiver
    

    #---Variáveis que realizam a animação---#
    Animation_axis = FuncAnimation( 
        fig,
        update_axis,
        frames = len(t),                                 #---Animação dos eixos---#
        interval = 25,
        repeat = False
    )

    Animation_grid = FuncAnimation(
        fig,
        update_grid,
        frames = len(t),                                 #---Animação do grid---#
        interval = 25,
        repeat = False
    )

    Animation_vectors = FuncAnimation(
        fig,
        update_vectors,
        frames = len(t),                                 #---Animação dos vetores---#
        interval = 25,
        repeat = False
    )


    #'Plotando' os vetores canônicos:
    ax1.quiver(0, 0, 1, 0, scale_units = 'xy', scale = 1, color = 'g', zorder = 3)
    ax1.quiver(0, 0, 0, 1, scale_units = 'xy', scale = 1, color = 'r', zorder = 3)

    #'Plotando' a origem do sistema xy
    ax1.scatter(0, 0, color='k', s=50, marker='.', label = "Origem do plano xy",zorder = 4)
    ax1.legend() 

    #Mostrando a 'plotagem'
    ax1.set_xlabel("x -> u")
    ax1.set_ylabel("y -> v")


    #----------------|SEGUNDA PARTE|----------------

    ax2.set_xlim(-r, r)
    ax2.set_ylim(-r, r)
    ax2.set_aspect('equal', adjustable='box')

    #Criando os eixos:
    ax2.axhline(0, color = 'k', linewidth = 0.9) #Determina o eixo horizontal
    ax2.axvline(0, color = 'k', linewidth = 0.9) #Determina o eixo vertical

    #'Plotando' o centro do sistema de coordenadas uv (base de autovetores)
    ax2.scatter(0, 0, color='r', s=50, marker='.', label='Origem do sistema uv', zorder = 4)
    ax2.legend()

    #'Plotando' os vetores canônicos (para melhor visualização):
    ax2.quiver(0, 0, 1, 0, scale_units = 'xy', scale = 1, color = 'g', zorder = 3)
    ax2.quiver(0, 0, 0, 1, scale_units = 'xy', scale = 1, color = 'r', zorder = 3)
    
    #Criando os eixos:
    ax2.axhline(0, color = 'k', linewidth = 0.5) #Determina o eixo horizontal
    ax2.axvline(0, color = 'k', linewidth = 0.5) #Determina o eixo vertical
    if(R[0]*R[1] !=0):
        ax2.set_title(f"Forma Padrão \n {tipo} \n ({R[2]:5.2f})u² + ({R[3]:5.2f})v² + (0.00)u + (0.00)v + ({R[4]:5.2f}) = 0",
              fontdict={
                    'weight': 'bold',      
                    'size': 10           
                }) #O título será, também a classificação da Cônica
    else:
        if(autova_r):
            ax2.set_title(f"Forma Padrão \n {tipo} \n (0.00)u² + ({R[3]:5.2f})v² + ({R[2]:5.2f})u + (0.00)v + ({R[4]:5.2f}) = 0",
                fontdict={
                        'weight': 'bold',      
                        'size': 10           
                    })
        else:
            ax2.set_title(f"Forma Padrão \n {tipo} \n ({R[2]:5.2f})u² + (0.00)v² + (0.00)u + ({R[3]:5.2f})v + ({R[4]:5.2f}) = 0",
                fontdict={ 
                        'weight': 'bold',      
                        'size': 10           
                    })
            
    ax2.set_xlabel("u")
    ax2.set_ylabel("v")
    ax2.grid(True)

    #"Plotando" a cônica reduzida:
    #U e V já estão carregados em "U, V = parametrizar_conica(tipo_norm, R[0], R[1], R[2], R[3], R[4])"
    if(autova_r): #Troca de eixos
        U, V = V, U

    ax2.plot(U, V, 'b', linewidth=1.5, zorder=2)
    ax2.legend()
    
    plt.show()