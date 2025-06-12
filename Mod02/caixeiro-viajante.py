import itertools

#Pontos
pontos = ['A', 'B', 'C', 'D']

# Distâncias entre os pontos
distancias = {
    'G': {'A': 15, 'B': 8, 'C': 7, 'D': 12},
    'A': {'G': 15, 'B': 3, 'C': 4, 'D': 6},
    'B': {'G': 8, 'A': 3, 'C': 2, 'D': 5},
    'C': {'G': 7, 'A': 4, 'B': 2, 'D': 3},
    'D': {'G': 12, 'A': 6, 'B': 5, 'C': 3}
}

# Função para calcular a distância total de um caminho
def calcular_distancia_total(caminho):
    dist_total = 0
    ponto_atual = 'G'
    for proximo_ponto in caminho:
        dist_total += distancias[ponto_atual][proximo_ponto]
        ponto_atual = proximo_ponto
    dist_total += distancias[ponto_atual]['G']  # Volta ao ponto de partida
    return dist_total

# Gerar todas as permutações dos pontos
rotas = list(itertools.permutations(pontos))
melhor_rota = None
melhor_distancia = float('inf')

#verificar cada rota
for rota in rotas:
    distancia = calcular_distancia_total(rota)
    if distancia < melhor_distancia:
        melhor_distancia = distancia
        melhor_rota = rota
# Exibir o resultado
print(f"Melhor rota: Garagem -> {' -> '.join(melhor_rota)} -> Garagem")
print(f"Distância total: {melhor_distancia} km")