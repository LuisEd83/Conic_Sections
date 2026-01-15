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

#Bibliotecas utilizadas
import sympy as sp
import numpy as np
from sympy import symbols, Matrix, pretty_print

def completa_quadrado(expr, var):
    #Coleta a expressão como polinômio em var
    phi = sp.collect(expr, var, evaluate=False)

    #Extração segura dos coeficientes
    a = phi.get(var**2, 0)
    b = phi.get(var, 0)
    resto = phi.get(1, 0)

    #Se não há termo quadrático, não há o que completar
    if(a == 0):
        return expr

    #Completar quadrado corretamente
    expr_completada = a*(var + b/(2*a))**2 + (resto - b**2/(4*a))

    return sp.simplify(expr_completada)

def completa_quadrado_conica(λ_1, λ_2, d, e, F):
    x1, y1 = symbols("x1 y1")

    #A expressão após a primeira Substituição é:
    expr_t1 = λ_1*(x1**2) + λ_2*(y1**2) + d*x1 + e*y1 + F
    #___Separando-a em três partes___#
    #expr_x = λ_1*(x1**2) + d*x1
    #expr_y = λ_2*(y1**2) + e*y1
    #F = F

    #Em relação a x
    expr = completa_quadrado(expr_t1, var = x1)
    #Em relação a y
    expr = completa_quadrado(expr_t1, var = y1)

    expr_t1 = expr
    return expr_t1

def normaliza_parabola(expr, x, y):
    expr = sp.expand(expr)

    ax2 = expr.coeff(x, 2)
    by2 = expr.coeff(y, 2)
    f   = expr.subs({x: 0, y: 0})

    #Caso y**2
    if((by2 != 0) and (ax2 == 0)):
        expr = expr - f
        expr = sp.solve(expr, x)[0]
        expr += -x

    #Caso x**2
    if((ax2 != 0) and (by2 == 0)):
        expr = expr - f
        expr = sp.solve(expr, y)[0]
        expr += -y
    
    return sp.simplify(expr)


def classificacao_conica(A,B,C,D,E,F):
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

    #Escrevendo as funções
    x, y = symbols("x y")
    expr = A*(x**2) + B*(x*y) + C*(y**2) + D*x + E*y + F

    #Criando a matriz relacionada a Forma Quadrática qxy
    X = Matrix([[A, B/2],
                [B/2, C]])
    X = X.evalf()
    
    """
    Encontrando os autovalores associados a X
    Seja λ um autovalor qualquer:
    """
    λ = symbols("λ")
    pλ = sp.det(X - λ*sp.eye(2)) #pλ é o polinômio característico
    sol = sp.solve(sp.Eq(pλ, 0), λ)

    #Se não houver solução real
    if not(sol):
        raise ValueError("Não há solução do polinômio característico!")

    # Garante dois autovalores reais
    sol = [s for s in sol if s.is_real]
    if(len(sol) == 1):
        λ_1 = λ_2 = sol[0]
    else:
        λ_1, λ_2 = sol[0], sol[1]

    #Arredondando os autovalores
    λ_1 = round(float(λ_1.evalf()), 6) 
    λ_2 = round(float(λ_2.evalf()), 6)

    #Ordenando os autovalores:
    if(λ_1 < λ_2): 
        t = λ_1
        λ_1 = λ_2
        λ_2 = t

    #Encontrando e normalizando os autovetores associados aos autovalores (usando a função autovetor_norm)
    if(abs(λ_1 - λ_2) < 1e-10):
        #Matriz é múltiplo da identidade — qualquer base ortonormal serve
        #Autovetores padrão associado a λ (quando λ_1 == λ_2)
        u1 = Matrix([[1], [0]]) 
        u2 = Matrix([[0], [1]])
    else:
        #Autovetores distintos
        eigvecs = X.eigenvects()
        for val, mult, vecs in eigvecs:
            if (abs(float(val) - float(λ_1)) < 1e-10):
                u1 = sp.Matrix(vecs[0])
                u1 = (1)*u1.evalf()
            if (abs(float(val) - float(λ_2)) < 1e-10):
                u2 = sp.Matrix(vecs[0])
                u2 = (1)*u2.evalf()

    #Criando a matriz Q para a realização da primeira substituição:
    Q = Matrix([[u1[0], u2[0]],
                [u1[1], u2[1]]])
    Q = np.array(Q.tolist(), dtype=float) #Convertendo Q para numpy, garantindo que Q seja numérico
    #Realizando a substituição
    x1, y1 = symbols("x1 y1") #Coordenadas da base de autovetores associados a λ_1, λ_2
    # v = Matrix([[x],
    #            [y]])
    w = Matrix([[x1],
                [y1]]) #Nova base
    Y = Q * w

    #Nova Expressão
    expr_transf1 = expr.subs({x : Y[0], y : Y[1]})
    expr_transf1 = sp.expand(expr_transf1)
    """
    Após a primeira substituição, a fóruma geral será reduzida para:
    λ_1*(x1)**2 + λ_2*(y1)**2 + d*x1 + e*y1 + F = 0

    Após isso, deve-se realizar uma série de análise:
    """
    #Extraindo dos coeficientes que faltam (d,e) da nova expressão: 
    d_simb = sp.expand(expr_transf1).coeff(x1, 1).subs(y1, 0)
    e_simb = sp.expand(expr_transf1).coeff(y1, 1).subs(x1, 0)

    d = float(d_simb.evalf())
    e = float(e_simb.evalf())
    
    #Completando quadrado
    expr_transf2 = completa_quadrado_conica(λ_1, λ_2, d, e, F)
    tipo = " "

    x2, y2 = symbols("x2 y2")
    if(λ_1*λ_2 != 0):
        #Realizando a segunda substituição:
        expr_transf2 = expr_transf2.subs({
            x1: (x2 - d/(2*λ_1)),
            y1: (y2 - e/(2*λ_2))
        })
        expr_transf2 = sp.simplify(expr_transf2)
        f = float(sp.N(F - (d**2)/(4*λ_1) - (e**2)/(4*λ_2))) #Feito de forma direa para garantir o valor numérico de f
        if((abs(λ_1 - λ_2) < 1e-10) and (f < 0)):
            return [
                "Circunferência",
                Q, 
                λ_1, 
                λ_2,
                float(sp.N(expr_transf2.coeff(x2, 2))),
                float(sp.N(expr_transf2.coeff(y2, 2))),
                f
            ]

        if(λ_1*λ_2 > 0):
            if(λ_1*λ_2*f < 0):
                #Isso implica que λ_1 e λ_2 tem sinal oposto a f
                tipo = "Elipse"
            elif(abs(f) < 1e-6):
                tipo = "Ponto"
            elif((λ_1*λ_2)*f > 0):
                tipo = "Vazio"
        else:
            if(abs(f) > 1e-10):
                tipo = "Hipérbole"
            else:
                tipo = "Par de retas concorrentes"

        return [
            tipo,
            Q,
            λ_1,
            λ_2,
            float(sp.N(expr_transf2.coeff(x2, 2))),
            float(sp.N(expr_transf2.coeff(y2, 2))),
            f
        ]

    elif((λ_1 == 0) and (λ_2 != 0)):
        #Realizando a segunda substituição:
        expr_transf2 = expr_transf2.subs({
            x1: x2,
            y1: (y2 - e/(2*λ_2))
        })
        expr_transf2 = sp.simplify(expr_transf2)

        f = float(sp.N(F - (e**2)/(4*λ_2))) #Feito de forma direa para garantir o valor numérico de f

        if(abs(d) > 1e-10):
            tipo = "Parábola"
            expr_transf2 = normaliza_parabola(expr_transf2, x2, y2)
            f = expr_transf2.subs({x2: 0, y2: 0})
            f = f.evalf()

            print(expr_transf2)
        elif((abs(d) < 1e-10) and (λ_2 * f < 0)):
            tipo = "Par de retas paralelas"
        elif((abs(d) < 1e-10) and (abs(f) < 1e-10)):
            tipo = "Reta única"
        
        if((abs(d) < 1e-10) and (λ_2 * f > 0)):
            tipo = "Vazio"
        
        return [
            tipo,
            Q,
            λ_1,
            λ_2,
            float(sp.N(expr_transf2.coeff(x2, 1))),
            float(sp.N(expr_transf2.coeff(y2, 2))),
            f
        ]

    elif((λ_1 != 0) and (λ_2 == 0)):
        #Realizando a segunda substituição:
        expr_transf2 = expr_transf2.subs({
            x1: (x2 - d/(2*λ_1)),
            y1: y2
        })
        expr_transf2 = sp.simplify(expr_transf2)

        f = float(sp.N(F - (d**2)/(4*λ_1))) #Feito de forma direa para garantir o valor numérico de f

        if(e != 0):
            tipo = "Parábola"
            expr_transf2 = normaliza_parabola(expr_transf2, x2, y2)
            f = expr_transf2.subs({x2: 0, y2: 0})
            f = f.evalf()
            
            print(expr_transf2)
        else:
            tipo = "Par de retas paralelas"
        if((abs(e) < 1e-10) and (abs(f) < 1e-10)):
            tipo = "Reta única"
        if((abs(e) < 1e-10) and (λ_1 * f > 0)):
            tipo = "Vazio"
        
        return [
            tipo,
            Q,
            λ_1,
            λ_2,
            float(sp.N(expr_transf2.coeff(x2, 2))),
            float(sp.N(expr_transf2.coeff(y2, 1))),
            f
        ]