# --------------------------------------------
#|Criação da interface para extração de dados |
# --------------------------------------------

#Bibliotecas utilizadas
import customtkinter as ctk
from sympy import sympify, E, pi, sqrt
from tkinter import END

def extracao():
    #configuração da janela
    ctk.set_appearance_mode('dark')
    window = ctk.CTk()
    window.title('Sistema de extração de dados')
    window.geometry('450x450')

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

            verificacao.configure(text="Variáveis aceitas!\n  Calculando Cônica...", font = ("Times", 15))
            window.update_idletasks()
            window.after(2000)
            window.destroy()
            coef[:] = [A, B, C, D, E, F]

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

    #Criando os campos: 
    #Há 3 campos : 
    # - Label : texto
    # - Entry : entrada
    # - Button : botão

    #Label

    #Essa variável auxiliar serve para dá um pequeno espaço no topo da janela
    texto_auxiliar = ctk.CTkLabel(window, text = '') 
    texto_auxiliar.pack(pady = 10)

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

    #Entry
    #Utilizando a variável linha1 para auxiliar a posição dos campos de entrada A, B e C
    linha1 = ctk.CTkFrame(window)
    linha1.pack(pady = 5)

    campo_A = ctk.CTkEntry(linha1, placeholder_text = 'Digite o valor de A:',
                        fg_color = ("black"))
    campo_B = ctk.CTkEntry(linha1, placeholder_text = 'Digite o valor de B:',
                        fg_color = ("black"))
    campo_C = ctk.CTkEntry(linha1, placeholder_text = 'Digite o valor de C:',
                        fg_color = ("black"))

    campo_A.grid(row = 0, column = 0, padx = 5, pady = 5)
    campo_B.grid(row = 0, column = 1, padx = 5, pady = 5)
    campo_C.grid(row = 0, column = 2, padx = 5, pady = 5)

    #Utilizando a variável linha2 para auxiliar a posição dos campos de entrada D, E e F
    linha2 = ctk.CTkFrame(window)
    linha2.pack(pady = 5)

    campo_D = ctk.CTkEntry(linha2, placeholder_text = 'Digite o valor de D:',
                        fg_color = ("black"))
    campo_E = ctk.CTkEntry(linha2, placeholder_text = 'Digite o valor de E:',
                        fg_color = ("black"))
    campo_F = ctk.CTkEntry(linha2, placeholder_text = 'Digite o valor de F:',
                        fg_color = ("black"))

    campo_D.grid(row = 0, column = 0, padx = 5, pady = 5)
    campo_E.grid(row = 0, column = 1, padx = 5, pady = 5)
    campo_F.grid(row = 0, column = 2, padx = 5, pady = 5)

    #Button
    #Utilizando a variável linha3 para auxiliar a posição dos botões

    linha3 = ctk.CTkFrame(window)
    linha3.pack(pady = 5)

    botao_calcular = ctk.CTkButton(linha3, text = 'Calcular cônica',
                                    command = validar_variaveis,
                                    hover_color = "green",
                                    corner_radius = 50)

    botao_limpar = ctk.CTkButton (linha3, text = 'Limpar campos',
                                command = validar_clear,
                                hover_color = "red",
                                corner_radius = 50)

    botao_calcular.grid(row = 0, column = 0, padx = 5, pady = 5)
    botao_limpar.grid(row = 0, column = 1, padx = 5, pady = 5)

    #Feedback do botão_calcular
    verificacao = ctk.CTkLabel(window, text = '')
    verificacao.pack(pady = 5)
    coef = []

    #inicia a aplicação
    window.mainloop()

    #Esse trecho retorna todos os coeficientes necessários
    return tuple(coef)