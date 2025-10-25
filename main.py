"""
Criação de programa para calcular as Seções de Cônicas
"""
#Bibliotecas utilizadas
from Interface import extracao
from linear_algebra_conics import classificacao_conica
import matplotlib as mp


#inicialização dos coeficientes da Equação Geral das Cônicas
A, B, C, D, E, F = extracao()
print(f"Valores dos coeficientes: A = {A}, B = {B}, C = {C}, D = {D}, E = {E}, F = {F}")
try:
    tipo, a, b, f = classificacao_conica(A, B, C, D, E, F)
    print(f"Classificação: {tipo};\n Autovalor 1: {a}; \n Autovalor 2: {b};\n Constante: {f}")
except ValueError as VE:
    print(f"Erro: {VE}")

