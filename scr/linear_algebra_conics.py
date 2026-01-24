# -------------------------------------------------
#|Criação de software de classificação de cônicas  |
# -------------------------------------------------

"""
Módulo: Classificação de Cônicas

Este módulo implementa um algoritmo capaz de classificar cônicas a partir da equação ge-
ral Ax² + Bxy + Cy² + Dx + Ey + F = 0

O algoritmo utilizará conceitos vistos em Álgebra Linear, como:
- Transformação linear;
- Autovalores e autovetores;
- Diagonalização de matrizes;
- Forma quadrática e suas propriedades.

Objetivos:
- Determinar o tipo de cônica: parábola, hipérbole, elipse, circunferência, par de retas
concorrentes, reta única e ponto. Observação: a partir dos cálculos, pode ocorrer do tipo
da cônica ser "vazio";
- Determinar a forma reduzida;
- Retornar o tipo de cônica e a forma reduzida - neste caso, os coeficientes da equação
reduzida.

Com os objetivos concluídos, será possível plotar, em um plano cartesiano, o gráfico da cô-
nica calculada.
"""

#Biblioteca utilizada
import numpy as np

def classificacao_conica(A, B, C, D, E, F):
    """
    Esta função será responsável por classificar a cônica.

    Podemos separar a equação geral em três partes:
    - Forma Quadrática: qxy = Ax² + Bxy + Cy²
    - Funcional Linear: φxy = Dx + Ey
    - Constante: F
    """
    #Primeiramente, precisamos saber se A = B = C = 0 (Que, neste contexto, não pode ocorrer)
    if ((A == 0) and (B == 0) and C == 0):
        raise ValueError("A, B e C não devem ser iguais a zero")
    
    if((A < 0) or ((C < 0) and (A == 0)) or ((A == C) and (A == 0) and (B < 0))):#Inverte sinal caso um dos requisitos sejam cumpridos..
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F
    
    #Definindo variáveis da matriz da forma quadrática:
    tracoM = A + C
    # -> detM = A*C - (B**2)/4 <- #
    sqr = np.sqrt((A - C)**2 + B**2)

    #Com isto, o polinômio característico será:
    #p(λ) = λ² - traxoM * λ + detM    (1)

    #Calculando autovalores baseados em |λ1| >= |λ2| que satisfaçam (1):
    λp = (0.5)*(tracoM + sqr) #λp é o λ+ (análogo ao λ1)
    λm = (0.5)*(tracoM - sqr) #λm é o λ_ (análogo ao λ2)

    λ1 = λ2 = 0.0 #Inicializando autovalores

    #Definindo uma variável booleana para detectar troca de autovalores (no caso, para λ1 = λm), uma vez que, a priori, λ1 = λp
    auto_reo = False
    
    if(abs(λp) >= abs(λm)):
        λ1 = λp
    else: #|λp| < |λm|
        λ1 = λm
        auto_reo = True #Houve uma reordenação de autovalores

    #Sabendo que λ1 + λ2 = tracoM:
    λ2 = tracoM - λ1

    print(f"Autovalores análogos: ({λp}, {λm})")
    print(f"Autovalores: ({λ1}, {λ2})")

    #Calculando os autovetores a partir dos casos:
    #inicializando autovetores
    V1 = V2 = [0.0, 0.0] #Vetores nulos

    if(B < 0):
        #Inicializando uma constante para não haver uma poluição visual:
        k = np.sqrt((B**2) + 4*((λ1 - A)**2))

        #Vetor 1:
        V1 = (-1/k) * np.array([[B],
                                [2*(λ1 - A)]], float)
        #Vetor 2:
        V2 = (1/k) * np.array([[2*(λ1 - A)], 
                                [-B]], float)

    elif(B > 0):
        #Inicializando uma constante para não haver uma poluição visual:
        k = np.sqrt((B**2) + 4*((λ1 - A)**2))

        #Vetor 1:
        V1 = (1/k) * np.array([[B],
                                [2*(λ1 - A)]], float)
        #Vetor 2:
        V2 = (1/k) * np.array([[-2*(λ1 - A)], 
                                [B]], float)
    else: #B == 0
        if(A >= C):
            #Vetor 1:
            V1 = np.array([[1],
                            [0]], float)
            #Vetor 2:
            V2 = np.array([[0], 
                            [1]], float)
        else: #A < C
            #Vetor 1:
            V1 = np.array([[0],
                            [1]], float)
            #Vetor 2:
            V2 = np.array([[-1], 
                            [0]], float)
            
    #Criando a matriz Q para a realização da primeira substituição:
    Q = np.array([[V1[0, 0], V2[0, 0]],
              [V1[1, 0], V2[1, 0]]], dtype = float)
    
    print(f"Matriz de rotação:\n {Q}")
    #Definindo equação após [x, y] = Q@[r,s], tal que
    #eq = λ1 * (r**2) + λ2 * (s**2) + d * r + e * s + F
    
    #Onde, definindo constantes:
    d = D * Q[0][0] + E * Q[1][0]
    e = D * Q[0][1] + E * Q[1][1]
    
    print(f"Valor de d: {d}\nValor de e: {e}")

    f = 0.0 #Inicializando constante da forma padrão
    if(λ1*λ2 != 0):
        #Definindo a constante f:
        f = F - (d**2)/(4*λ1) - (e**2)/(4*λ2)

    else: #λ1 != 0 e λ2 == 0
        if(auto_reo):
            #Definindo a constante f:
            f = F - (e**2)/(4*λ1)
        else:
            #Definindo a constante f:
            f = F - (d**2)/(4*λ1)

    #Classificando cônica:
    tipo = ""
    a = b = 0.0
    if(λ1*λ2 != 0):
        if((abs(λ1 - λ2) < 1e-10) and (f < 0)):
            tipo = "Circunferencia"
        
        if(λ1*λ2 > 0):
            if(λ1*λ2*f < 0): #Isso implica que λ1 e λ2 tem sinal oposto a f
                tipo = "Elipse"
            elif(abs(f) < 1e-6):
                tipo = "Ponto"
            elif((λ1*λ2)*f > 0):
                tipo = "Vazio"
        else:
            if(abs(f) > 1e-10):
                tipo = "Hiperbole"
            else:
                tipo = "Par de retas concorrentes"

        #Coeficientes da forma padrão
        a = λ1
        b = λ2

        return [
            tipo,
            Q,
            λ1,
            λ2,
            a,
            b,
            f,
            auto_reo
        ]
    else:
        if((e != 0 and not auto_reo) or (abs(d) > 1e-10 and auto_reo)):
            tipo = "Parabola"
            #Com s substituição, f será igual a 0
            f = 0.0 #Garantir que a parábola da forma padrão se localize no vértice.
        else:
            tipo = "Par de retas paralelas"

        if((((abs(e) < 1e-10) and (abs(f) < 1e-10)) and not auto_reo) or ((abs(d) < 1e-10) and (abs(f) < 1e-10) and auto_reo)):
            tipo = "Reta unica"
        
        elif((((abs(e) < 1e-10) and (λ1 * f > 0)) and not auto_reo) or ((abs(d) < 1e-10) and (λ2 * f > 0) and auto_reo)):
            tipo = "Vazio"
        
        #Coeficientes da forma padrão
        if(auto_reo):
            a = d
            b = λ1
        else:
            a = λ1
            b = e

        return [
            tipo,
            Q,
            λ1,
            λ2,
            a,
            b,
            f,
            auto_reo
        ]