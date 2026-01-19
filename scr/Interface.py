# --------------------------------------------
#|Criação da interface para extração de dados |
# --------------------------------------------

"""
Módulo: Interface

Este módulo implementa um algoritmo cuja função é criar uma interface para melhor 
extração de dados que serão utilizados futuramente para classificar cônicas.

Objetivos:
- Criar uma janela que irá conter campos para o usuário inserir os dados;
- Criar botões para auxiliar o usuário durante o processo de inserir os dados.

Com os objetivos alcançados, será possível dar valores aos coeficientes da equação 
geral Ax² + Bxy + Cy² + Dx + Ey + F = 0. 
"""

#Bibliotecas utilizadas
import customtkinter as ctk
from sympy import sympify, pi, E, sqrt, N
from tkinter import END, StringVar

def extracao():
    #configuração da janela
    ctk.set_appearance_mode('dark')
    window = ctk.CTk()
    window.title('Sistema de extração de dados')
    window.geometry('450x550')

    #verificação de campos
    def conversor(palavra : str):
        """
        O conversor é uma função que irá receber a string que contém a informação do campo
        e vai converter a string em uma expressão matemática e, após isso, irá o transformar em
        um valor real.
        """
        if not (palavra == ""):
            try:
                expr = sympify(palavra, locals ={'e': E, 'pi': pi,'sqrt':sqrt})
                valor = expr.evalf()

                if not(valor.is_real or valor.is_number):
                    raise ValueError("Valor não numérico!")
                
                return float(valor)
            except ValueError as VE:
                window.update_idletasks()
                verificacao.configure(text=f"{VE}", font = ("Times", 15))
                return None
        
        

    def validar_variaveis():
        #Esse trecho será utilizado para validar as entradas 'inputadas' pelo usuário
        """
        Esta função irá validar, com o auxílio da função 'conversor', as entradas
        digitadas pelo usuário.
        """
        try:
            A = conversor(campo_A.get())
            B = conversor(campo_B.get())
            C = conversor(campo_C.get())
            D = conversor(campo_D.get())
            E = conversor(campo_E.get())
            F = conversor(campo_F.get())

            if any(v is None for v in [A,B,C,D,E,F]):
                raise ValueError("Valor(es) incorreto(s)!")
            verificacao.configure(text="Variáveis aceitas!\n  Executando cálculo...", font = ("Times", 15))
            window.update_idletasks()
            window.after(2000)
            window.destroy()
            coef[:] = [A, B, C, D, E, F, False]
            
            #Algoritmo para salvar as variáveis em um arquivo
            linhas_defalt = []
            with open("scr/coefficients.txt", "r") as arq: #Lê os valores default antes da escrita
                for _ in range(8):
                    arq.readline()
                
                linhas_defalt = arq.readlines()

            with open("scr/coefficients.txt", "w") as arq: #Escreve os coeficientes e os valores default
                for c in coef:
                    arq.write(str(c) + '\n') #Escrevendo no arquivo coefficients.txt
                
                arq.write('\n') #Escreve uma linha vazia para não ocorrer algum problema com outras funções

                for l in linhas_defalt:
                    arq.write(str(l)) #Reescrevendo os valores default

        except ValueError as e:
            window.update_idletasks()
            verificacao.configure(text=f"{e}", font = ("Times", 15))
            return None

    def validar_clear():
        """
        Essa função apaga todos os valores digitados nos campos
        """
        campo_A.delete(0,END)
        campo_B.delete(0,END)
        campo_C.delete(0,END)
        campo_D.delete(0,END)
        campo_E.delete(0,END)
        campo_F.delete(0,END)

    def validar_sair():
        """
        Essa função te faz sair da interface
        """

        A = B = C = D = E = F = 0
        verificacao.configure(text="Saindo...", font = ("Times", 15))
        window.update_idletasks()
        window.after(2000)
        window.destroy()
        coef[:] = [A, B, C, D, E, F, True]

    def validar_repetir():
        """
        Essa função põe valores aos coeficientes de forma automática com valores default.
        """
        arq_def = open("scr/coefficients.txt", "r")
        if((arq_def.readable()) and (arq_def.read(1) != "")):

            arq_def.seek(0)
            entry_varA.set(arq_def.readline().strip())
            entry_varB.set(arq_def.readline().strip())
            entry_varC.set(arq_def.readline().strip())
            entry_varD.set(arq_def.readline().strip())
            entry_varE.set(arq_def.readline().strip())
            entry_varF.set(arq_def.readline().strip())
        else:
            verificacao.configure(text = "Não foi possível ler e extrair os valores do arquivo.", font = ("Times", 15))
        arq_def.close()
    
    def validar_default():
        """
        Essa função pôe valores aos coeficientes da última cônica testada.  
        """

        try:
            with open("scr/coefficients.txt", "r", encoding="utf-8") as arq:
                #Descarta valores entre as linhas 1 e 8
                for _ in range(8):
                    arq.readline()

                # Agora estamos exatamente na linha 9
                entry_varA.set(arq.readline().strip())
                entry_varB.set(arq.readline().strip())
                entry_varC.set(arq.readline().strip())
                entry_varD.set(arq.readline().strip())
                entry_varE.set(arq.readline().strip())
                entry_varF.set(arq.readline().strip())

        except (FileNotFoundError, ValueError, IndexError):
            verificacao.configure(text="Não foi testada nenhuma cônica.", font=("Times", 15))


    #Criando os campos:     
    #Há 3 campos : 
    # - Label : texto
    # - Entry : entrada
    # - Button : botão

    #Label
    #Essa variável auxiliar serve para dá um pequeno espaço no topo da janela
    texto_auxiliar1 = ctk.CTkLabel(window, text = '') 
    texto_auxiliar1.pack(pady = 5)


    texto_primario = ctk.CTkLabel(window, text = 'Fórmula Geral - Cônicas',
                                    font = ("Helvenica", 18))
    texto_primario.pack(pady = 10)

    formula_conica = "Ax² + Bxy + Cy² + Dx + Ey + F = 0"
    texto_secundario = ctk.CTkLabel(window, text = formula_conica,
                                    font = ("Times", 22, "italic"))
    texto_secundario.pack(pady = 10)

    texto_terciario = ctk.CTkLabel(window, text = 'Insira os valores corretamente.',
                                    font = ("Helvenica", 15))
    texto_terciario.pack(pady = 2)

    texto_observacao = ctk.CTkLabel(window, text = "Observações:\n Constantes → E, pi  |  Multiplicação → *  |  Exponenciação → **\nRaíz Quadrada → sqrt( )  |  Logaritmo Neperiano → ln( )",
                                    font = ("Times", 13))
    texto_observacao.pack(pady = 2)
    
    #Essa variável auxiliar serve para dá um pequeno espaço
    texto_auxiliar2 = ctk.CTkLabel(window, text = '') 
    texto_auxiliar2.pack(pady = 5)

    #--------------------------Campo (A,B,C)------------------------------#
    texto_titulo1 = ctk.CTkLabel(window, text = '  Coeficiente A:                 Coeficiente B:                 Coeficiente C:',
                                    font = ("Helvenica", 13))
    texto_titulo1.pack(pady = 0) #Para por títulos aos campos A, B e C

    #Entry
    #Utilizando a variável linha1 para auxiliar a posição dos campos de entrada A, B e C
    linha1 = ctk.CTkFrame(window)
    linha1.pack(pady = 5)

    #Iniciando variáveis
    entry_varA = StringVar()
    entry_varB = StringVar()
    entry_varC = StringVar()
    entry_varD = StringVar()
    entry_varE = StringVar()
    entry_varF = StringVar()

    campo_A = ctk.CTkEntry(linha1, fg_color = ("black"),
                        textvariable = entry_varA)
    campo_B = ctk.CTkEntry(linha1, fg_color = ("black"),
                        textvariable = entry_varB)
    campo_C = ctk.CTkEntry(linha1, fg_color = ("black"),
                        textvariable = entry_varC)

    campo_A.grid(row = 0, column = 0, padx = 5, pady = 5)
    campo_B.grid(row = 0, column = 1, padx = 5, pady = 5)
    campo_C.grid(row = 0, column = 2, padx = 5, pady = 5)

    #--------------------------Campo (D,E,F)------------------------------#
    texto_titulo2 = ctk.CTkLabel(window, text = '  Coeficiente D:                 Coeficiente E:                 Coeficiente F:',
                                    font = ("Helvenica", 13))
    texto_titulo2.pack(pady = 0)# Para por títulos aos campos D, E e F
    #Utilizando a variável linha2 para auxiliar a posição dos campos de entrada D, E e F
    linha2 = ctk.CTkFrame(window)
    linha2.pack(pady = 5)

    campo_D = ctk.CTkEntry(linha2, fg_color = ("black"),
                        textvariable = entry_varD)
    campo_E = ctk.CTkEntry(linha2, fg_color = ("black"),
                        textvariable = entry_varE)
    campo_F = ctk.CTkEntry(linha2, fg_color = ("black"),
                        textvariable = entry_varF)

    campo_D.grid(row = 0, column = 0, padx = 5, pady = 5)
    campo_E.grid(row = 0, column = 1, padx = 5, pady = 5)
    campo_F.grid(row = 0, column = 2, padx = 5, pady = 5)

    #Button

    #Utilizando as variável linha3 para auxiliar a posição dos 3 primeiros botões
    linha3 = ctk.CTkFrame(window)
    linha3.pack(pady = 5)

    #Utilizando as variável linha4 para auxiliar a posição dos 2 últimos botões
    linha4 = ctk.CTkFrame(window)
    linha4.pack(pady = 5)

    #Criando o botão Executar
    botao_executar = ctk.CTkButton(linha3, text = 'Executar',
                                    command = validar_variaveis,
                                    hover_color = 'green',
                                    corner_radius = 50)

    #Criando o botão Limpar
    botao_limpar = ctk.CTkButton (linha3, text = 'Limpar campos',
                                command = validar_clear,
                                hover_color = 'darkblue',
                                corner_radius = 50)
    
    #Criando o botão Default
    botao_default = ctk.CTkButton(linha3, text = "Default",
                                  command = validar_default,
                                  corner_radius = 50)
    

    #Criando o botão Repetir
    botao_repetir = ctk.CTkButton(linha4, text = "Repetir",
                                  command = validar_repetir,
                                  hover_color = 'darkgreen',
                                  corner_radius = 50)
    
    #Criando o botão Sair
    botao_sair = ctk.CTkButton (linha4, text = 'Sair',
                                command = validar_sair,
                                hover_color = 'red',
                                corner_radius = 50)
    
    #Linha superior
    botao_executar.grid(row = 0, column = 0, padx = 5, pady = 8)
    botao_limpar.grid(row = 0, column = 1, padx = 5, pady = 8)
    botao_default.grid(row = 0, column = 2, padx = 5, pady = 8)

    #Linha inferior
    botao_repetir.grid(row = 0, column = 0, padx = 5, pady = 5)
    botao_sair.grid(row = 0, column = 1, padx = 5, pady = 5)

    #Feedback dos botões botão_calcular e botao_sair
    verificacao = ctk.CTkLabel(window, text = '')
    verificacao.pack(pady = 5)
    coef = []

    #inicia a aplicação
    window.mainloop()

    #Esse trecho retorna todos os coeficientes necessários
    return tuple(coef)