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
- Forma quadrática.

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
from sympy import symbols, Matrix

def autovetor_norm(λ, X : Matrix): #Aparentemente é esta função que está causando problemas
    a, b = symbols("a b")
    v = Matrix([[a],
                [b]])
    
    M_λ = (X - λ * sp.eye(2)) * v
    eqs = [sp.Eq(M_λ[0], 0), sp.Eq(M_λ[1], 0)]

    # Força a = 1 e tenta resolver em relação à variável b
    sol = sp.solve([eqs[0].subs(a, 1), eqs[1].subs(a, 1)], [b], dict=True)

    var_a = None
    var_b = None
    if sol:
        var_a = 1
        var_b = sol[0][b]
    else:
        #Forçar b = 1 e tenta resolver em relação à variável a
        sol = sp.solve([eqs[0].subs(b, 1), eqs[1].subs(b, 1)], [a], dict=True)
        if (sol):
            var_a = sol[0][a]
            var_b = 1
        else:
            #Caso as formas acima não consiga resolver, tentaremos resolver usando a função que retorna os autovetores da matriz X
            eigvecs = X.eigenvects()
            for val, mult, vecs in eigvecs:
                if (abs(float(val) - float(λ)) < 1e-8): #Compara o autovalor encontrado ao autovalor posto na função autovetor_norm
                    v = sp.Matrix(vecs[0]) #Pega o primeiro autovetor associado ao autovalor encontrado em eigvecs
                    break
            else:
                return None
            var_a, var_b = float(v[0]), float(v[1])
    if not(sol):
        raise ValueError("Não foi possível encontrar os autovetores.")

    v = Matrix([[var_a],
                [var_b]])
    v = v.evalf()
    norm = sp.N(sp.sqrt(v.dot(v)))

    v_norm = sp.N(v/norm)
    if(float(abs(norm)) < 1e-10):
        return None
    else:
        return v_norm

def completa_quadrado(a, b, c, var):
    expr = a*(var**2) + b*var + c

    if(a == 0):
        expr = b*var + c
    elif((b == 0) and (c == 0)):
        expr = a*(var**2)
    elif((a == 0) and (b == 0) and (c == 0)):
        raise ValueError("Não é possível completar quadrado com esta expressão")
    else:
        expr = a*(var + b/(2*a))**2 + (c - (b**2)/(4*a))
    return expr

def completa_quadrado_conica(λ_1, λ_2, d, e, F):
    x1, y1 = symbols("x1 y1")

    #A expressão após a primeira Substituição é:
    expr_t1 = λ_1*(x1**2) + λ_2*(y1**2) + d*x1 + e*y1 + F
    #Separando-a em três partes
    expr_x = λ_1*(x1**2) + d*x1
    expr_y = λ_2*(y1**2) + e*y1
    #F = F

    #Em relação a x
    expr_x = completa_quadrado(λ_1, d, 0, var = x1)
    #Em relação a y
    expr_y = completa_quadrado(λ_2, e, 0, var = y1)

    #Juntando tudo em uma única expressão e simplificando:
    expr_t1 = sp.expand(expr_x + expr_y + F)

    return expr_t1

def classificacao_conica(A,B,C,D,E,F):
    """
    Esta função será responsável por classificar a cônica.

    Podemos separar a equação geral em três partes:
    - Forma Quadrática: qxy = Ax² + Bxy + Cy²
    - Funcional Linear: φxy = Dx + Ey
    - Constante: F
    """
    #Primeiramente, precisamos saber se A = B = C = 0 (Que, neste contexto, não pode ocorrer)
    if all(v == 0 for v in [A,B,C]):
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
    if len(sol) == 1:
        sol = [sol[0], sol[0]]
    elif len(sol) == 0:
        raise ValueError("Não há solução real do polinômio característico!")

    λ_1, λ_2 = sol[1], sol[0] #λ_1, λ_2 são os dois autovalores associados a X
    λ_1 = float(λ_1.evalf()) #Transforma em valor numérico 
    λ_2 = float(λ_2.evalf())
    
    if(λ_1 is None):
        λ_1 = sol[0]

    if all(v == 0 for v in [λ_1, λ_2]): #λ_1 = λ_2 = 0 não pode ocorrer
        raise ValueError("Ambos os autovalores são nulos!")

    #Encontrando e normalizando os autovetores associados aos autovalores (usando a função autovetor_norm)

    if (abs(λ_1 - λ_2) < 1e-10):
    # Matriz é múltiplo da identidade — qualquer base ortonormal serve
    #Autovetores padrão associado a λ (quando λ_1 == λ_2)
        u1 = Matrix([[1], [0]]) 
        u2 = Matrix([[0], [1]]) 
    else:
        # Autovetores distintos
        u1 = autovetor_norm(λ_1, X) 
        u2 = autovetor_norm(λ_2, X)

    #Criando a matriz Q para a realização da primeira substituição:
    Q = Matrix([[u1[0], u2[0]],
                [u1[1], u2[1]]])

    #Realizando a substituição
    x1,y1 = symbols("x1 y1") #Coordenadas da base de autovetores associados a λ_1, λ_2
    # v = Matrix([[x],
    #            [y]])
    w = Matrix([[x1],
                [y1]])
    Y = Q * w

    #Nova Expressão
    expr_transf1 = expr.subs({x : Y[0], y : Y[1]})
    expr_transf1 = sp.expand(expr_transf1)
    expr_transf1 = sp.sympify(expr_transf1)
    """
    Após a primeira substituição, a fóruma geral será reduzida para:
    λ_1*(x1)**2 + λ_2*(y1)**2 + d*x1 + e*y1 + F = 0

    Após isso, deve-se realizar uma série de análise:
    """
    #Extraindo dos coeficientes que faltam (d,e) da nova expressão: 
    d_simb = expr_transf1.coeff(x1, 1) 
    e_simb = expr_transf1.coeff(y1, 1)

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
        expr_transf2 = sp.expand(expr_transf2)

        f = float(sp.N(F - (d**2)/(4*λ_1) - (e**2)/(4*λ_2))) #Feito de forma direa para garantir o valor numérico de f

        if((abs(λ_1 - λ_2) < 1e-10) and (f < 0)):
            return ["Circunferência", Q, λ_1, λ_2, f]

        if(λ_1*λ_2 > 0):
            if(λ_1*λ_2*f < 0):
                #Isso implica que λ_1 e λ_2 tem sinal oposto a f
                tipo = "Elipse"
            if(abs(f) < 1e-10):
                tipo = "Ponto"
            if(λ_1*λ_2*f > 0):
                tipo = "Vazio"
        else:
            if(f != 0):
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
        expr_transf2 = sp.expand(expr_transf2)
        f = float(sp.N(F - (e**2)/(4*λ_2))) #Feito de forma direa para garantir o valor numérico de f

        if(d != 0):
            tipo = "Parábola"
        elif((d == 0) and (λ_2 * f < 0)):
            tipo = "Par de retas paralelas"
        elif((d == 0) and (abs(f) == 1e-10)):
            tipo = "Reta única"
        
        if((d == 0) and (λ_2 * f > 0)):
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
        expr_transf2 = sp.expand(expr_transf2)
        f = float(sp.N(F - (d**2)/(4*λ_1))) #Feito de forma direa para garantir o valor numérico de f

        if(e != 0):
            tipo = "Parábola"
        else:
            tipo = "Par de retas paralelas"
        if((e == 0) and (abs(f) < 1e-10)):
            tipo = "Reta única"
        if((e == 0) and (λ_1 * f > 0)):
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