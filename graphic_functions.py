# ----------------------------------------------------------
#|Criação de funções para auxiliar na plotagem de gráficos |
# ----------------------------------------------------------


"""
Módulo: Depósito de funções para plotagem de gráfico

Objetivo:
- Desenvolver funções em Python para auxiliar na plotagem das cônicas.

Com este objetivo concluído, será possível plotar as cônicas com maior precisão
"""

import numpy as np

def parametrizar_conica(tipo, λ1, λ2, A, B, f, n_pts=200):
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

def vertice_parabola(F, H, g, tol=1e-12):

    #Decomposição espectral
    vals, vecs = np.linalg.eigh(H)

    #identifica autovalor não nulo
    idx = np.argmax(np.abs(vals))
    lam = vals[idx]

    if(abs(lam) < tol):
        raise ValueError("Parábola degenerada ou erro numérico")

    #Matriz de rotação
    P = vecs

    #Coeficientes no sistema rotacionado
    g_rot = P.T @ g
    alpha = g_rot[idx]
    beta  = g_rot[1 - idx]

    #Vértice no sistema rotacionado
    Xv = -alpha / (2 * lam)
    Yv = -(F - alpha**2 / (4 * lam)) / beta

    X = np.zeros(2)
    X[idx] = Xv
    X[1 - idx] = Yv

    #Volta para coordenadas originais
    x_v = P @ X
    return x_v

def ponto_representativo(A, B, C, D, E, F, tipo, tol = 1e-12):
    """
    Retorna um ponto representativo da cônica:
    - centro (elipse, hipérbole, circunferência)
    - vértice (parábola)
    - ponto mais próximo da origem (reta)
    - centro geométrico (pares de retas, ponto)
    """
    H = np.array([[A, B/2],
                  [B/2, C]], dtype=float)

    g = np.array([D, E], dtype=float)

    #Pseudoinversa (funciona mesmo se H for singular)
    H_pinv = np.linalg.pinv(H, rcond = tol)

    x0, y0 = [0,0]
    #Cálculo do vértice da Parábola
    if(tipo == "parabola"): 
        x0, y0 = vertice_parabola(F, H, g)
    else:
        x0, y0 = -0.5 * H_pinv @ g #Cálculo de coordenadas
    return np.array([x0, y0], float)

def raio_plot_conica(A, B, C, D, E, centro):
    #Base mínima
    r = 1.1

    #Escala dos termos lineares
    r += 0.09 * (abs(D) + abs(E))

    #Escala dos termos quadráticos
    r += 1.0 / max(1e-6, (abs(A) + abs(B) + abs(C))*2)

    #Se houver centro conhecido, use ele
    if(centro is not None):
        r += 0.5 * (np.hypot(centro[0], centro[1]))
        r *= 1.01
    #Garantia mínima
    return max(r, 3.5)