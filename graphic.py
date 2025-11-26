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

def centro_conica(A, B, C, D, E):
    """
    Calcula o centro de uma cônica dada por Ax² + Bxy + Cy² + Dx + Ey + F = 0
    """
    #Matriz do sistema para encontrar o centro
    M = np.array([[2*A, B], [B, 2*C]])
    b = np.array([-D, -E])
    
    #Verifica se a matriz é singular
    if(abs(np.linalg.det(M)) > 1e-10):
        # Sistema não singular - solução única
        try:
            sol = np.linalg.solve(M, b)
            return [sol[0], sol[1]]
        except:
            return [0, 0]
    else:
        #Sistema singular - cônica degenerada (par de retas ou reta única)
        #Encontra a solução de mínimos quadrados (ponto mais próximo da origem que satisfaz o sistema)
        try:
            sol, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
            
            #Verifica se o sistema é consistente
            if((rank < 2) or (residuals.size > 0 and residuals[0] > 1e-8)):
                #Sistema inconsistente - não há centro único
                #Para reta única, encontramos um ponto na reta mais próximo da origem
                #A reta é dada por uma das equações: 2Ax + By + D = 0 ou Bx + 2Cy + E = 0
                if ((abs(B) > 1e-10) or (abs(2*A) > 1e-10)):
                    #Usa a primeira equação: 2Ax + By + D = 0
                    a, b_coef, c = 2*A, B, D
                else:
                    #Usa a segunda equação: Bx + 2Cy + E = 0  
                    a, b_coef, c = B, 2*C, E
                
                #Encontra o ponto na reta ax + by + c = 0 mais próximo da origem
                denom = a**2 + b_coef**2
                if(denom > 1e-10):
                    x0 = (-a * c) / denom
                    y0 = (-b_coef * c) / denom
                    return [x0, y0]
                else:
                    return [0, 0]
            else:
                return [sol[0], sol[1]]
        except:
            return [0, 0]

def parametrizar_conica(tipo, λ1, λ2, A, B, f, n_pts=400):
    #Elipse / Circunferência
    if(tipo in ["elipse", "circunferencia"]):
        if ((A <= 0) or (B <= 0) or (f >= 0)):
            return np.array([]), np.array([])
        rx = np.sqrt(-f / A)
        ry = np.sqrt(-f / B)
        t = np.linspace(0, 2*np.pi, n_pts)
        u = rx * np.cos(t)
        v = ry * np.sin(t)
        return u, v
    
    #Hipérbole
    if(tipo == "hiperbole"):
        #hiperbole exige A*B < 0
        if (A*B >= 0):
            return np.array([]), np.array([])
        
        #Corrigindo possíveis erros de plotagem:
        if(f > 0):
            A = -A
            B = -B

        #Dividir n_pts em dois para os dois ramos
        n_half = n_pts // 2
        t = np.linspace(-5, 5, n_half)

        #Caso 1: A > 0 , B < 0  (eixo real = u)
        if ((A > 0) and (B < 0)):
            a = np.sqrt(abs(-f / A))
            b = np.sqrt(abs(-f / (-B)))
            u1 =  a * np.cosh(t)
            v1 =  b * np.sinh(t)
            u2 = -a * np.cosh(t)
            v2 = -b * np.sinh(t)

        #Caso 2: A < 0 , B > 0  (eixo real = v)
        elif ((A < 0 )and (B > 0)):
            a = np.sqrt(abs(-f / B))
            b = np.sqrt(abs(-f / (-A)))
            u1 =  b * np.sinh(t)
            v1 =  a * np.cosh(t)
            u2 = -b * np.sinh(t)
            v2 = -a * np.cosh(t)

        else:
            return np.array([]), np.array([])

        U = np.concatenate([u1, u2])
        V = np.concatenate([v1, v2])
        return U, V

    #Parábola
    if(tipo == "parabola"):
        t = np.linspace(-10, 10, n_pts)

        #λ1 = 0  -> A*u + B*v² + f = 0
        if(abs(λ1) < 1e-12):
            v = t
            u = -(B*v*v + f)/A
            return u, v

        #λ2 = 0 -> A*u² + B*v + f = 0
        if(abs(λ2) < 1e-12):
            u = t
            v = -(A*u*u + f)/B
            return u, v

        return np.array([]), np.array([])

    #Par de retas concorrentes
    if(tipo == "par de retas concorrentes"):
        t = np.linspace(-10, 10, n_pts)

        if ((A > 0) and (B < 0)):
            k = np.sqrt(A/(-B))
            u1, v1 = t,  k*t
            u2, v2 = t, -k*t
        elif ((A < 0) and (B > 0)):
            k = np.sqrt((-A)/B)
            u1, v1 = t,  k*t
            u2, v2 = t, -k*t
        else:
            return np.array([]), np.array([])

        #Inserir NaN para separar as retas
        separator = np.array([np.nan])
        U = np.concatenate([u1, separator, u2])
        V = np.concatenate([v1, separator, v2])
        return U, V
    
    #Par de retas paralelas
    if(tipo == "par de retas paralelas"):
        t = np.linspace(-10, 10, n_pts)
        
        #Caso 1: A·u² + f = 0 (retas verticais)
        if((abs(B) < 1e-12) and (A * f < 0)):
            u_val = np.sqrt(-f / A)
            u1 = np.full_like(t, u_val)   #Primeira reta: u = sqrt(-f/A)
            v1 = t
            u2 = np.full_like(t, -u_val)  #Segunda reta: u = -sqrt(-f/A)
            v2 = t
            
            #Inserir NaN para separar as retas
            separator = np.array([np.nan])
            U = np.concatenate([u1, separator, u2])
            V = np.concatenate([v1, separator, v2])
            return U, V
        
        #Caso 2: B·v² + f = 0 (retas horizontais)
        elif((abs(A) < 1e-12) and (B * f < 0)):
            v_val = np.sqrt(-f / B)
            u1 = t
            v1 = np.full_like(t, v_val)   #Primeira reta: v = sqrt(-f/B)
            u2 = t
            v2 = np.full_like(t, -v_val)  #Segunda reta: v = -sqrt(-f/B)
            
            #Inserir NaN para separar as retas
            separator = np.array([np.nan])
            U = np.concatenate([u1, separator, u2])
            V = np.concatenate([v1, separator, v2])
            return U, V
        
        else:
            return np.array([]), np.array([])

    #Reta única
    if(tipo == "reta unica"):
        t = np.linspace(-10, 10, n_pts)

        #Caso 1: Equação linear Au + Bv + f = 0
        if((abs(A) > 1e-12) and (abs(B) > 1e-12)):
            u = t
            v = -(A*u + f)/B
        #Caso 2: A = 0, equação Bv + f = 0 -> reta horizontal
        elif((abs(A) < 1e-12) and (abs(B) > 1e-12)):
            u = t
            v = np.full_like(t, -f/B)
        #Caso 3: B = 0, equação Au + f = 0 -> reta vertical  
        elif((abs(B) < 1e-12) and (abs(A) > 1e-12)):
            u = np.full_like(t, -f/A)
            v = t
        #Caso 4: Equação quadrática degenerada (como 25u² = 0)
        elif((abs(A) > 1e-12) and (abs(B) < 1e-12) and (abs(f) < 1e-12)):
            u = np.full_like(t, 0.0)
            v = t
        elif((abs(B) > 1e-12) and (abs(A) < 1e-12) and (abs(f) < 1e-12)):
            u = t
            v = np.full_like(t, 0.0)
        else:
            return np.array([]), np.array([])

        return u, v
    
    #Ponto
    if(tipo == "ponto"):
        return np.array([0]), np.array([0])

    #Vazio
    if(tipo == "vazio"):
        return np.array([]), np.array([])

    return np.array([]), np.array([])


def graph(coef_eqg: list, clasf_c : list, Q : Matrix, tipo : str):
    G0 = coef_eqg[:] #Cria uma cópia de uma lista (para facilitar na escrita)
    R = clasf_c[:] #[λ1, λ2, a, b, f]

    #Determinando o centro da cônica original
    centro_og = centro_conica(G0[0], G0[1], G0[2], G0[3], G0[4])

    #Automatizando o r
    r = np.sqrt(centro_og[0]**2 + centro_og[1]**2)
    r *= 3
    while((r <= 1.5) or (r<(np.sqrt(abs(R[4]))))):
        if(r == 0):
            r += 0.5
        else:
            r *= 2

    t = np.linspace(0, 1, 50) #Variável temporal. Intervalo t ∈ [0, 1].

    #Calculando as coordenadas dos vetores em função de t
    def vectors(Q, t):
        # Calcula o ângulo de rotação da matriz Q
        angle = np.arctan2(Q[1, 0], Q[0, 0])
        
        # Ângulo interpolado
        angle_t = (1 - t)*angle

        # Vetor 1 (eixo x rotacionado)
        xf1_t = np.cos(angle_t)
        yf1_t = np.sin(angle_t)

        # Vetor 2 (eixo y rotacionado)
        xf2_t = -np.sin(angle_t)
        yf2_t = np.cos(angle_t)

        # Centro: interpolando linearmente do centro_og para a origem
        xi_t = centro_og[0] * (1 - t)
        yi_t = centro_og[1] * (1 - t)

        return [xi_t, yi_t, xf1_t, yf1_t, xf2_t, yf2_t]

    #----------------PRIMEIRA PARTE----------------#

    #Criando uma figura e seus eixos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.set_xlim(-r, r)
    ax1.set_ylim(-r, r)
    ax1.set_aspect('equal', adjustable='box')

    #Autovetores ortonormais
    Q_np = np.array(Q.tolist(), float)
    # normaliza colunas
    for j in range(2):
        Q_np[:, j] /= np.linalg.norm(Q_np[:, j])

    #Parametrização em (u,v)
    tipo_norm = normalize_tipo(tipo) #transforma as letras em minúculo
    U, V = parametrizar_conica(tipo_norm, R[0], R[1], R[2], R[3], R[4])
    P = np.vstack([U, V])

    #Inicialização do gráfico
    grafico_conica = None 
    
    #Extrai o ângulo de rotação da matriz Q
    theta = np.arctan2(Q_np[1,0], Q_np[0,0])

    #Criando funçao de atualização de frame:
    def update_conica(frame):
        nonlocal grafico_conica
        
        try:
            if grafico_conica is not None:
                grafico_conica.remove()
        except:
            pass

        tt = frame/(len(t)-1)

        #Rotação correta (não deforma)
        ang = (1 - tt)*theta
        Q_current = np.array([
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang),  np.cos(ang)]
        ])

        #Translação suave
        C_current = (1 - tt)*np.array(centro_og, float)

        #Aplica movimento
        XY = Q_current @ P + C_current.reshape(2,1)

        Xp, Yp = XY
        grafico_conica, = ax1.plot(Xp, Yp, 'b', linewidth=1.5)

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

        #Coordenadas dos vetores
        tt = frame/(len(t)-1)
        coord_vetores = vectors(Q, tt)
        
        #'Plotando' os vetores associados aos autovalores (para melhor visualização):
        v1 = ax1.quiver(coord_vetores[0], coord_vetores[1], coord_vetores[4], coord_vetores[5], scale_units = 'xy', scale = 1, color = 'r', zorder = 3)
        v1_i = ax1.quiver(coord_vetores[0], coord_vetores[1], -coord_vetores[4], -coord_vetores[5], scale_units = 'xy', scale = 1, color = 'r', zorder = 3)
        v2 = ax1.quiver(coord_vetores[0], coord_vetores[1], coord_vetores[2], coord_vetores[3], scale_units = 'xy', scale = 1, color = 'r', zorder = 3,)
        v2_i = ax1.quiver(coord_vetores[0], coord_vetores[1], -coord_vetores[2], -coord_vetores[3], scale_units = 'xy', scale = 1, color = 'r', zorder = 3)

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