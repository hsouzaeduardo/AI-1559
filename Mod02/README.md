# Problema do Caixeiro-Viajante (TSP) — Solução em Python

Este projeto apresenta uma solução para o problema do caixeiro-viajante (Traveling Salesman Problem - TSP) utilizando Python. O objetivo é encontrar a rota de menor distância para um entregador que deve sair da garagem, visitar todos os pontos (A, B, C, D) exatamente uma vez e retornar à garagem.

## Como funciona a solução

1. **Definição dos Pontos e Distâncias**  
Na matriz abaixo representa as distancias do exércicio

| Cliente KM    | G  | A  | B  | C  | D  |
|-----|----|----|----|----|----|
| G   | 0  | 5  | 8  | 7  | 12 |
| A   | 5  | 0  | 3  | 4  | 6  |
| B   | 8  | 3  | 0  | 2  | 5  |
| C   | 7  | 4  | 2  | 0  | 3  |
| D   | 12 | 6  | 5  | 3  | 0  |



   Os pontos a serem visitados e as distâncias entre eles (incluindo a garagem) são definidos em um dicionário chamado `distancias` no arquivo `caixeiro-viajante.py`.

2. **Cálculo da Distância Total de uma Rota**  
   A função `calcular_distancia_total(rota)` recebe uma ordem de visita aos pontos e soma as distâncias, começando e terminando na garagem.

3. **Geração de Todas as Rotas Possíveis**  
   Usando `itertools.permutations`, o código gera todas as possíveis sequências de visita aos pontos.

4. **Busca pela Melhor Rota**  
   O programa verifica todas as rotas possíveis, calcula a distância total de cada uma e guarda a rota com a menor distância.

5. **Exibição do Resultado**  
   Ao final, o programa imprime a melhor rota encontrada e a distância total percorrida.

## Exemplo de Saída

```
Melhor rota: Garagem -> B -> C -> D -> A -> Garagem
Distância total: 21 km
```

## Observações

- Esta solução utiliza força bruta (testa todas as possibilidades), sendo eficiente apenas para poucos pontos.
- É uma ótima introdução ao TSP e ao uso de permutações em Python.

## Como executar

1. Certifique-se de ter o Python instalado.
2. Salve o código em `Mod02/caixeiro-viajante.py`.
3. Execute no terminal:
   ```
   python caixeiro-viajante.py
   ```

Assim, você verá a melhor rota e a distância total para o conjunto de pontos definido.

---

Se quiser explorar soluções mais eficientes para o TSP, pesquise sobre algoritmos como "vizinho mais próximo" ou "algoritmos genéticos".
