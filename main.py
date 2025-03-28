"""
PERCEPTRON PARA CLASSIFICAÇÃO BINÁRIA

Autor: Roberto Barreto Ferraz Filho
RU: 4277827
Descrição: Implementação de um Perceptron com 6 entradas e 1 saída binária (-1 ou 1)
           usando a regra delta para aprendizado.
"""

# =============================================
# IMPORTAÇÕES
# =============================================
import sys
import time
from colorama import init, Fore, Back, Style

# Inicializa colorama para funcionar no Windows também
init()

# =============================================
# CONSTANTES E CONFIGURAÇÕES
# =============================================
CORES = {
    'titulo': Fore.BLUE + Style.BRIGHT,
    'sucesso': Fore.GREEN + Style.BRIGHT,
    'erro': Fore.RED + Style.BRIGHT,
    'alerta': Fore.YELLOW + Style.BRIGHT,
    'info': Fore.CYAN,
    'destaque': Fore.MAGENTA + Style.BRIGHT,
    'normal': Style.RESET_ALL
}

# =============================================
# DADOS DE TREINAMENTO
# =============================================
"""
Estrutura dos dados:
Cada amostra contém 6 valores de entrada (sensores) e 1 valor de saída (controle acionador)
Formato: [entrada1, entrada2, entrada3, entrada4, entrada5, entrada6, saida_esperada]
Saída esperada: -1 (não ativar) ou 1 (ativar)
"""
DADOS_TREINAMENTO = [
    [2, 0, 3, 6, 9, 9, -1], [1, 9, 1, -1, 3, 9, -1], [2, 2, 3, 1, 1, 8, -1], [7, -1, 2, 7, 5, 9, -1],
    [-1, 5, -2, 9, 9, 8, -1], [-2, 4, 1, -1, 5, 5, -1], [9, -1, 9, 3, 1, 2, -1], [7, 2, 0, -2, 2, 3, -1],
    [5, -2, 2, 9, 7, -2, -1], [8, -2, 4, 9, -1, 8, 1], [7, -2, 5, 4, 1, -1, -1], [8, 4, -1, 0, 9, 6, -1],
    [0, 1, 0, 3, 1, -2, -1], [5, 7, 8, 1, 8, 0, -1], [1, -2, 8, 3, 3, 9, -1], [7, 0, 7, 2, 2, 1, -1],
    [4, 8, -1, -2, 0, 4, -1], [-2, 1, 3, 1, 0, 2, -1], [9, 8, 8, 1, 7, 0, -1], [6, 7, -1, 9, -1, -1, -1],
    [4, 5, 4, 0, 4, 6, -1], [2, 3, 9, -1, -1, 2, -1], [8, 2, 7, 8, -2, -2, -1], [0, 9, 7, 3, -2, 7, -1],
    [1, -2, 5, -1, 5, 9, -1], [3, -2, 3, 3, -1, 5, -1], [-2, 0, 6, 5, 9, 2, -1], [5, 4, 3, 1, -1, 7, -1],
    [5, -1, 5, 6, 1, 7, -1], [2, 7, 7, 3, 8, -2, -1], [2, 1, -1, 2, 1, 4, -1], [8, -1, 2, -1, 4, 2, -1],
    [7, 9, -1, 4, 7, -1, -1], [0, 4, 8, 3, 3, 4, -1], [7, 8, 2, 9, 1, -1, -1], [0, 7, -1, 5, 3, 5, -1],
    [3, 4, 9, 4, -1, 5, -1], [3, 8, 1, 5, 2, 1, -1], [4, 4, 4, 0, -2, 4, -1], [9, 3, 1, 7, 5, 9, 1],
    [7, -2, 7, 8, -1, 3, -1], [-1, 2, 1, -1, 5, -1, -1], [6, 2, 6, 4, 0, 2, -1], [3, 7, 8, 0, 9, 0, -1],
    [5, 9, -1, 3, 9, 8, -1], [4, 1, 7, 0, 4, 3, -1], [8, 9, 1, 2, 8, 4, -1], [0, 8, 7, 5, 5, 0, -1],
    [5, 3, 8, 6, 9, 1, 1], [3, 6, 7, 0, -2, 3, -1], [-2, 4, 9, 6, 0, 3, -1], [0, 7, 0, 3, 0, 7, -1],
    [5, 3, 6, 6, 8, 2, -1], [1, 7, 0, 6, 0, 6, -1], [3, -2, 3, 7, -2, 2, -1], [0, 0, 4, 0, 9, 2, -1],
    [2, -1, -2, 6, -2, -2, -1], [-1, 6, 1, 1, 1, 1, -1], [8, 2, 9, -1, -1, 8, -1], [4, 0, 7, 8, -2, 5, -1],
    [-2, 7, 2, 3, 3, 4, -1], [4, 8, 7, -2, 4, 6, -1], [3, -1, 7, 3, -2, 7, -1], [0, 4, -1, -2, 3, 1, -1],
    [7, 8, 8, -1, 6, 6, -1], [5, -2, 7, 3, -1, -1, -1], [-2, 9, 3, 4, 9, 7, -1], [3, 0, 5, 8, 5, 8, 1],
    [1, 1, -2, 2, 1, 8, -1], [3, 6, 1, -1, 4, 7, -1], [8, 4, -2, 2, 2, -1, -1], [5, 0, 5, -2, 9, 9, -1],
    [6, 9, 5, 3, -2, 8, -1], [7, 6, 3, 1, 5, 8, -1], [5, -1, -1, 9, 0, 8, -1], [8, 6, 5, 8, 6, 9, 1],
    [6, 3, 9, 4, 7, 1, -1], [9, 6, 4, -2, -1, 4, -1], [3, -2, -1, 6, -1, 3, -1], [5, 8, 6, 4, 2, 7, -1],
    [7, 8, 3, 2, 8, -1, -1], [0, 5, -1, 2, 4, 5, -1], [-2, 2, -2, 1, 4, 7, -1], [9, 3, 9, 5, 3, 9, 1],
    [-2, 8, 1, 1, 4, 0, -1], [7, 3, -2, 8, 6, 3, -1], [3, -2, 8, 2, -2, -2, -1], [9, 8, -2, -1, 0, 9, -1],
    [-2, 4, 3, 6, -1, 4, -1], [-1, 0, 9, 6, 7, 8, -1], [8, 1, 3, 1, 5, 3, -1], [6, 0, 1, 0, 3, 7, -1],
    [6, 7, 9, 9, 9, 7, 1], [4, 2, 6, 4, 9, -2, -1], [8, 7, 8, -2, 8, 5, -1], [5, 5, 5, 0, 9, 2, -1],
    [6, 0, 3, 4, 7, -2, -1], [2, 4, 5, 3, 1, 9, -1], [5, 0, 6, 8, 1, 0, -1], [2, 2, 6, 2, 9, 9, -1],
    [9, 3, 8, 7, 0, 6, 1]
]

# =============================================
# FUNÇÕES AUXILIARES
# =============================================

def mostrar_cabecalho():
    """Exibe o cabeçalho do programa com informações básicas"""
    print(CORES['titulo'] + "\n" + "="*60)
    print("PERCEPTRON PARA CLASSIFICAÇÃO BINÁRIA".center(60))
    print("="*60 + CORES['normal'])
    print(f"{CORES['info']}• {len(DADOS_TREINAMENTO)} amostras de treinamento")
    print(f"• 6 entradas e 1 saída binária (-1 ou 1){CORES['normal']}\n")

def progress_bar(iteracao, total, tamanho=50):
    """Exibe uma barra de progresso colorida"""
    percentual = iteracao / total
    blocos = int(percentual * tamanho)
    barra = f"{CORES['sucesso']}{'█' * blocos}{CORES['erro']}{'░' * (tamanho - blocos)}{CORES['normal']}"
    porcentagem = f"{percentual:.0%}"
    sys.stdout.write(f"\r[{barra}] {porcentagem}")
    sys.stdout.flush()

def soma_ponderada(entradas, pesos):
    """
    Calcula a soma ponderada das entradas com os pesos
    
    Args:
        entradas (list): Lista com 6 valores de entrada
        pesos (list): Lista com 7 pesos (incluindo o bias)
    
    Returns:
        float: Resultado da soma ponderada
    """
    soma = pesos[0]  # Bias (w0)
    for i in range(6):
        soma += pesos[i + 1] * entradas[i]
    return soma

def funcao_ativacao(soma):
    """
    Função de ativação degrau (step function)
    
    Args:
        soma (float): Valor da soma ponderada
    
    Returns:
        int: 1 se soma >= 0, -1 caso contrário
    """
    return 1 if soma >= 0 else -1

# =============================================
# CONFIGURAÇÕES DO PERCEPTRON
# =============================================
TAXA_APRENDIZADO = 0.1
MAX_ITERACOES = 100
PESOS_INICIAIS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # [w0, w1, w2, w3, w4, w5, w6]

# =============================================
# TREINAMENTO DO PERCEPTRON
# =============================================
def treinar_perceptron():
    """Função principal para treinar o perceptron"""
    
    mostrar_cabecalho()
    
    pesos = PESOS_INICIAIS.copy()
    print(f"{CORES['info']}Pesos iniciais: {pesos}{CORES['normal']}\n")
    
    print(f"{CORES['titulo']}Iniciando treinamento...{CORES['normal']}")
    print(f"{CORES['info']}Taxa de aprendizado: {TAXA_APRENDIZADO}")
    print(f"Máximo de iterações: {MAX_ITERACOES}{CORES['normal']}\n")
    
    inicio_tempo = time.time()
    
    for iteracao in range(MAX_ITERACOES):
        erro_total = 0
        progress_bar(iteracao, MAX_ITERACOES)
        
        for amostra in DADOS_TREINAMENTO:
            entradas = amostra[0:6]       # Valores dos 6 sensores
            saida_esperada = amostra[6]   # Valor de controle (-1 ou 1)
            
            # Passo 1: Calcular a saída do perceptron
            soma = soma_ponderada(entradas, pesos)
            saida_calculada = funcao_ativacao(soma)
            
            # Passo 2: Calcular o erro
            erro = saida_esperada - saida_calculada
            erro_total += abs(erro)
            
            # Passo 3: Atualizar os pesos (Regra Delta)
            pesos[0] += TAXA_APRENDIZADO * erro  # Atualiza o bias (w0)
            for i in range(6):
                pesos[i + 1] += TAXA_APRENDIZADO * erro * entradas[i]
        
        # Verifica convergência (erro total = 0)
        if erro_total == 0:
            print(f"\n\n{CORES['sucesso']}★ Convergência alcançada na iteração {iteracao + 1}!{CORES['normal']}")
            break
    
    # Exibe resumo do treinamento
    tempo_treinamento = time.time() - inicio_tempo
    print(f"\n{CORES['titulo']}Resumo do Treinamento:{CORES['normal']}")
    print(f"{CORES['info']}• Iterações realizadas: {iteracao + 1}")
    print(f"• Tempo de treinamento: {tempo_treinamento:.2f} segundos")
    print(f"• Pesos finais: {pesos}{CORES['normal']}\n")
    
    return pesos

# =============================================
# TESTE DO PERCEPTRON
# =============================================
def testar_perceptron(pesos_finais, num_amostras=5):
    """Testa o perceptron com os pesos treinados"""
    
    print(f"{CORES['titulo']}\nTestando com {num_amostras} amostras:{CORES['normal']}")
    print(f"{CORES['info']}{'Entradas':<30} {'Esperado':<10} {'Calculado':<10} {'Status'}{CORES['normal']}")
    
    acertos = 0
    for amostra in DADOS_TREINAMENTO[:num_amostras]:
        entradas = amostra[0:6]
        saida_esperada = amostra[6]
        
        soma = soma_ponderada(entradas, pesos_finais)
        saida_calculada = funcao_ativacao(soma)
        
        status = "✔" if saida_calculada == saida_esperada else "✘"
        cor_status = CORES['sucesso'] if status == "✔" else CORES['erro']
        
        if status == "✔":
            acertos += 1
        
        print(f"{str(entradas):<30} {saida_esperada:<10} {saida_calculada:<10} {cor_status}{status}{CORES['normal']}")
    
    print(f"\n{CORES['titulo']}Acurácia nos testes:{CORES['normal']} {acertos}/{num_amostras} ({acertos/num_amostras:.0%})")

# =============================================
# EXECUÇÃO PRINCIPAL
# =============================================
if __name__ == "__main__":
    pesos_finais = treinar_perceptron()
    testar_perceptron(pesos_finais)
    
    print(f"\n{CORES['sucesso']}Treinamento concluído com sucesso!{CORES['normal']}\n")