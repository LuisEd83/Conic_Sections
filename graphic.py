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

from unicodedata import normalize, category
from sympy import Matrix
from matplotlib.animation import FuncAnimation

def normalize_tipo(s: str) -> str:
    """
    Modifica uma string deixando todos os caracteres minúsculos e sem acento
    """
    s = s.lower()
    s = normalize('NFD', s)
    s = ''.join(ch for ch in s if category(ch) != 'Mn') #remove acentos
    return s

def graph(coef_eqg: list, clasf_c : list, Q : Matrix, tipo : str):
    G0 = coef_eqg[:] #Cria uma cópia de uma lista (para facilitar na escrita)
    R = clasf_c[:] #[λ1, λ2, a, b, f]
    tipo_norm = normalize_tipo(tipo) #transforma as letras em minúculo

    #Determinando o centro da cônica original
    centro_og = gf.ponto_representativo(G0[0], G0[1], G0[2], G0[3], G0[4], G0[5], tipo_norm, 1e-12)

    #Automatizando o r
    r = gf.raio_plot_conica(G0[0], G0[1], G0[2], G0[3], G0[4], centro_og)

    t = np.linspace(0, 1, 200) #Variável temporal. Intervalo t ∈ [0, 1].

    #Calculando as coordenadas dos vetores em função de t
    def vectors_rot(Q, t): #Essa parte irá rotacionar o vetor
        # Calcula o ângulo de rotação da matriz Q
        angle = np.arctan2(Q[1, 0], Q[0, 0])
        
        # Ângulo interpolado
        angle_t = (1 - t)*angle

        #Vetor 1 (eixo x rotacionado)
        xf1_t = np.cos(angle_t)
        yf1_t = np.sin(angle_t)

        #Vetor 2 (eixo y rotacionado)
        xf2_t = -np.sin(angle_t)
        yf2_t = np.cos(angle_t)

        #Centro: linearmente do centro_og para a origem
        xi_t = 0 #centro_og[0]
        yi_t = 0 #centro_og[1]

        return [xi_t, yi_t, xf1_t, yf1_t, xf2_t, yf2_t]

    #----------------PRIMEIRA PARTE----------------#

    #Criando uma figura e seus eixos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.set_xlim(-r, r)
    ax1.set_ylim(-r, r)
    ax1.set_aspect('equal', adjustable='box')

    #Autovetores ortonormais (convertidos para numpy)
    Q_np = np.array(Q.tolist(), float)

    #Normalização das colunas
    for j in range(2):
        Q_np[:, j] /= np.linalg.norm(Q_np[:, j])

    #Parametrização da cônica no sistema (u, v)
    U, V = gf.parametrizar_conica(
        tipo_norm, R[0], R[1], R[2], R[3], R[4]
    )
    P = np.vstack([U, V])

    #Inicialização de objetos gráficos
    grafico_conica = None
    centro = None        
    Q_current = None     #Matriz de rotação atual
    V_pa = None          #Vértice da parábola
    V_red = None         #Vértive da parábola reduzida
    C_current = None     #Ponto de referência para o plot


    #Ângulo de rotação associado aos autovetores
    theta = np.arctan2(Q_np[1, 0], Q_np[0, 0])

    #Função de atualização dos frames da animação
    def update_conica(frame):
        #----------------| Controle de escopo |----------------#
        nonlocal grafico_conica
        nonlocal centro
        nonlocal Q_current
        nonlocal V_pa
        nonlocal V_red
        nonlocal C_current

        #----------------| Limpeza do frame anterior |----------#
        try:
            if(grafico_conica is not None):
                grafico_conica.remove()
            if(centro is not None):
                centro.remove()
        except:
            pass

        #Parâmetros temporais da animação
        total_frames = len(t)
        N_rot = total_frames / 2          #Primeira metade: rotação
        N_trans = total_frames - N_rot    #Segunda metade: translação

        C_current = np.array([0.0, 0.0], float)
        V_red = np.array([0.0, 0.0], float)

        #Vértice da parábola reduzida
        if(tipo_norm == "parabola"):
            if(abs(R[0]) < 1e-12):
                V_red = np.array([-R[4]/R[2], 0.0])
            elif(abs(R[1]) < 1e-12):
                V_red = np.array([0.0, -R[4]/R[3]])
        else:
            V_red = np.zeros(2) #Uma vez que este vértice é da PARÁBOLA, então, para outras cônicas, ela não existe

        #Fase 1: Rotação progressiva
        if(frame < N_rot) and (theta != 0):
            tt = frame / (N_rot - 1)
            ang = (1 - tt) * theta   #Ângulo interpolado

            #---------- Sem translação nesta fase ----------#
            if(tipo_norm == "parabola"):
                #Rotação da vértice
                V_pa = np.array(gf.rot_centro(centro_og, theta, tt), float)
                C_current = V_pa.copy()

            else:
                #Outras cônicas: rotação em torno do centro
                C_current = np.array(gf.rot_centro(centro_og, theta, tt), float) 

        #Fase 2: Translação progressiva
        else:
            if theta == 0:
                tt = frame / (N_trans - 1)
                tt /= 2
            else:
                tt = (frame - N_rot) / (N_trans - 1)

            ang = 0.0  #Rotação já concluída

            if(tipo_norm == "parabola"):
                #Vértice rotacionado (posição inicial)
                V_ini = np.array(gf.rot_centro(centro_og, theta, 1), float)

                #Interpolação correta até o vértice reduzido
                C_current = (1 - tt) * V_ini + tt * V_red

                V_pa = C_current.copy()

            else:
                C_current = (1 - tt) * np.array(gf.rot_centro(centro_og, theta, 1), float)

        #Plot do ponto de referência
        centro = ax1.scatter(C_current[0], C_current[1], color='r', s=50, marker='.', label='Centro', zorder=4)

        ax1.legend()

        #Rotação e plot da cônica
        Q_current = np.array([
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang),  np.cos(ang)]
        ])

        XY = Q_current @ (P - V_red.reshape(2,1)) + C_current.reshape(2, 1)
        Xp, Yp = XY

        grafico_conica, = ax1.plot(Xp, Yp, 'k', linewidth=1.5)

        return [grafico_conica]


    #Criando a animação
    animation_conica = FuncAnimation(
        fig,
        update_conica,
        frames = len(t),
        interval = 25,
        repeat = True
    )

    vetores_quiver = None #Inicialização de valor
    def update_vetor(frame):
        #----------------|Plotagem dos vetores|----------------#
        
        nonlocal vetores_quiver

        #Remove os vetores anteriores (se existirem)
        if vetores_quiver is not None:
            for v in vetores_quiver:
                v.remove()

        total_frames = len(t)
        N_rot = total_frames/2
        if((frame < N_rot) and (theta != 0)):
            tt = frame/(N_rot - 1)

            coord_vetores = vectors_rot(Q, tt) #vetores rotacionando.
            #'Plotando' os vetores associados aos autovalores (para melhor visualização):
            v1 = ax1.quiver(coord_vetores[0], coord_vetores[1], coord_vetores[4], coord_vetores[5], scale_units = 'xy', scale = 1, color = 'r', zorder = 3)
            v1_i = ax1.quiver(coord_vetores[0], coord_vetores[1], -coord_vetores[4], -coord_vetores[5], scale_units = 'xy', scale = 1, color = 'purple', zorder = 3)
            v2 = ax1.quiver(coord_vetores[0], coord_vetores[1], coord_vetores[2], coord_vetores[3], scale_units = 'xy', scale = 1, color = 'g', zorder = 3,)
            v2_i = ax1.quiver(coord_vetores[0], coord_vetores[1], -coord_vetores[2], -coord_vetores[3], scale_units = 'xy', scale = 1, color = 'b', zorder = 3)
        else:
            coord_vetores = vectors_rot(Q, 1)

            #Vetores estáticos na origem (0,0)
            v1 = ax1.quiver(coord_vetores[0], coord_vetores[1], coord_vetores[4], coord_vetores[5], scale_units = 'xy', scale = 1, color = 'r', zorder = 3)
            v1_i = ax1.quiver(coord_vetores[0], coord_vetores[1], -coord_vetores[4], -coord_vetores[5], scale_units = 'xy', scale = 1, color = 'purple', zorder = 3)
            v2 = ax1.quiver(coord_vetores[0], coord_vetores[1], coord_vetores[2], coord_vetores[3], scale_units = 'xy', scale = 1, color = 'g', zorder = 3,)
            v2_i = ax1.quiver(coord_vetores[0], coord_vetores[1], -coord_vetores[2], -coord_vetores[3], scale_units = 'xy', scale = 1, color = 'b', zorder = 3)

        vetores_quiver = [v1, v1_i, v2, v2_i]

        return vetores_quiver

    animation_vetores = FuncAnimation(
        fig,
        update_vetor,
        frames = len(t),
        interval = 25,
        repeat = True
    )

    #Criando os eixos:
    ax1.axhline(0, color = 'k', linewidth = 0.9) #Determina o eixo horizontal
    ax1.axvline(0, color = 'k', linewidth = 0.9) #Determina o eixo vertical
    
    #Criando plotagem estática da cônica original
    V_red = np.zeros(2)
    if(tipo_norm == "parabola"):
        if(abs(R[0]) < 1e-12):
            V_red = np.array([-R[4]/R[2], 0.0])
        elif(abs(R[1]) < 1e-12):
            V_red = np.array([0.0, -R[4]/R[3]])
    
    XY = Q_np @ (P - V_red.reshape(2,1)) + centro_og.reshape(2, 1)
    Xp, Yp = XY
    ax1.plot(Xp, Yp, 'red', linewidth=0.4, zorder=2)

    #Título - 
    ax1.set_title(f"Animação\nCônica original -> Equação reduzida.\n ({G0[0]:5.2f})x² + ({G0[1]:5.2f})xy + ({G0[2]:5.2f})y² + ({G0[3]:5.2f})x + ({G0[4]:5.2f})y + ({G0[5]:5.2f}) = 0",
              fontdict={
                    'weight': 'bold',      
                    'size': 10           
                })

    #Mostrando a 'plotagem'
    ax1.set_xlabel("x -> u")
    ax1.set_ylabel("y -> v")
    ax1.grid(True)
    
    #----------------|SEGUNDA PARTE|----------------

    ax2.set_xlim(-r, r)
    ax2.set_ylim(-r, r)
    ax2.set_aspect('equal', adjustable='box')

    #Criando os eixos:
    ax2.axhline(0, color = 'k', linewidth = 0.9) #Determina o eixo horizontal
    ax2.axvline(0, color = 'k', linewidth = 0.9) #Determina o eixo vertical

    #'Plotando' o centro da cônica reduzida
    if(tipo_norm == "parabola"):
        ax2.scatter(V_red[0], V_red[1], color='r', s=50, marker='.', label='Vértice', zorder = 4)
    else:
        ax2.scatter(0, 0, color='r', s=50, marker='.', label='Centro', zorder = 4)
    ax2.legend()

    #'Plotando' os vetores canônicos (para melhor visualização):
    ax2.quiver(0, 0, 1, 0, scale_units = 'xy', scale = 1, color = 'k', zorder = 3)
    ax2.quiver(0, 0, 0, 1, scale_units = 'xy', scale = 1, color = 'k', zorder = 3)
    ax2.quiver(0, 0, -1, 0, scale_units = 'xy', scale = 1, color = 'k', zorder = 3)
    ax2.quiver(0, 0, 0, -1, scale_units = 'xy', scale = 1, color = 'k', zorder = 3)
    
    #Criando os eixos:
    ax2.axhline(0, color = 'k', linewidth = 0.5) #Determina o eixo horizontal
    ax2.axvline(0, color = 'k', linewidth = 0.5) #Determina o eixo vertical
    if(R[0]*R[1] !=0):
        ax2.set_title(f"{tipo} \n ({R[2]:5.2f})u² + ({R[3]:5.2f})v² + ({R[4]:5.2f}) = 0",
              fontdict={
                    'weight': 'bold',      
                    'size': 10           
                }) #O título será, também a classificação da Cônica
    elif((R[0] == 0) and (R[1] != 0)):
        ax2.set_title(f"{tipo} \n ({R[2]:5.2f})u + ({R[3]:5.2f})v² + ({R[4]:5.2f}) = 0",
              fontdict={
                    'weight': 'bold',      
                    'size': 10           
                })
    elif((R[0] != 0) and (R[1] == 0)):
        ax2.set_title(f"{tipo} \n ({R[2]:5.2f})u² + ({R[3]:5.2f})v + ({R[4]:5.2f}) = 0",
              fontdict={ 
                    'weight': 'bold',      
                    'size': 10           
                })
    ax2.set_xlabel("u")
    ax2.set_ylabel("v")
    ax2.grid(True)

    #"Plotando" a cônica reduzida:
    #U e V já estão carregados em "U, V = parametrizar_conica(tipo_norm, R[0], R[1], R[2], R[3], R[4])"
    ax2.plot(U, V, 'b', linewidth=1.5, label='Cônica reduzida', zorder=2)
    ax2.legend()
    
    plt.show()