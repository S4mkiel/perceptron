"""
Perceptron para Controle de Forno IoT
Autor: Roberto Barreto Ferraz Filho
RU: 4277827
Descrição: Implementa um perceptron com 6 entradas (sensores) e 1 saída binária
(1 ou -1) para controlar esteiras e forno em uma indústria de cimento. Usa pesos
fixos obtidos do treinamento na fase 1. Desenvolvido para microcontrolador com
512 KB de memória e baixa capacidade de processamento, compilado com Python-Assembly
(1 comando Python = 12,7 opcodes, 4,3 bytes/operação).
Saída: Imprime "1" (manter produção ativa) ou "-1" (desativar esteiras e forno).
"""

# Pesos finais do perceptron, obtidos do treinamento na fase 1
pesos = [-63.7, 3.7, 1.7, 2.5, 5.1, -0.3, 1.9]

def soma_ponderada(entradas):
    """
    Calcula a soma ponderada das entradas com os pesos
    Entrada: Lista de 6 inteiros (leituras dos sensores)
    Saída: Float com o resultado da soma ponderada
    """
    soma = pesos[0]  # Bias (w0)
    for i in range(6):
        soma += pesos[i + 1] * entradas[i]  # Soma w[i] * x[i]
    return soma

def funcao_ativacao(soma):
    """
    Função de ativação degrau
    Entrada: Resultado da soma ponderada (float)
    Saída: 1 (manter ativo) se soma >= 0, -1 (desativar) caso contrário
    """
    return 1 if soma >= 0 else -1

def classificar(entradas):
    """
    Classifica uma amostra e imprime a decisão
    Entrada: Lista de 6 inteiros (leituras dos sensores)
    Saída: Imprime "1" ou "-1" na tela (simulação)
    """
    soma = soma_ponderada(entradas)
    saida = funcao_ativacao(soma)
    print(saida)  # Simula decisão; em hardware real, acionaria esteiras/forno

# Simulação com 5 amostras de teste
if __name__ == "__main__":
    # Amostras de teste (5 primeiras do conjunto da fase 1)
    amostras = [
        [2, 0, 3, 6, 9, 9],    # Esperado: -1
        [1, 9, 1, -1, 3, 9],   # Esperado: -1
        [2, 2, 3, 1, 1, 8],    # Esperado: -1
        [7, -1, 2, 7, 5, 9],   # Esperado: -1
        [-1, 5, -2, 9, 9, 8]   # Esperado: -1
    ]

    # Classifica cada amostra
    for amostra in amostras:
        classificar(amostra)
