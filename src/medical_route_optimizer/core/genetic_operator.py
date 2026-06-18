import random
import copy
from typing import List
from data.delivery_points import PontoEntrega

# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

def pmx_crossover(parent1: List[PontoEntrega], parent2: List[PontoEntrega]) -> List[PontoEntrega]:
    """
    Partially Mapped Crossover (PMX) para permutações.
    Garante filhos válidos sem duplicação de genes.
    """
    length = len(parent1)
    start, end = sorted(random.sample(range(length), 2))
    child = [None] * length

    # Copia segmento do primeiro pai
    child[start:end+1] = parent1[start:end+1]

    # Mapeamento
    for i in range(start, end+1):
        if parent2[i] not in child:
            pos = i
            while child[pos] is not None:
                pos = parent2.index(parent1[pos])
            child[pos] = parent2[i]

    # Preenche posições restantes
    for i in range(length):
        if child[i] is None:
            child[i] = parent2[i]

    return child

def order_crossover(
    parent1: List[PontoEntrega],
    parent2: List[PontoEntrega]
) -> List[PontoEntrega]:
    """
    Operador de crossover por ordem (OX — Order Crossover) para permutações.

    Preserva a estrutura de permutação válida, evitando duplicidade de cidades.

    Parâmetros:
    - parent1: rota do primeiro pai
    - parent2: rota do segundo pai

    Retorno:
    - rota filho resultante da recombinação
    """
    length = len(parent1)
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    child = parent1[start_index:end_index]
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child

def mutate(
    rota: List[PontoEntrega],
    probabilidade_mutacao: float
) -> List[PontoEntrega]:
    """
    Operador de mutação: troca dois pontos adjacentes da rota com dada probabilidade.

    Preserva a validade da permutação.

    Parâmetros:
    - rota: sequência de pontos de entrega
    - probabilidade_mutacao: probabilidade de ocorrer mutação [0.0, 1.0]

    Retorno:
    - rota mutada (ou cópia da original se mutação não ocorrer)
    """
    rota_mutada = copy.deepcopy(rota)

    if random.random() < probabilidade_mutacao:
        if len(rota) < 2:
            return rota_mutada
        index = random.randint(0, len(rota) - 2)
        rota_mutada[index], rota_mutada[index + 1] = rota[index + 1], rota[index]

    return rota_mutada

def mutate_segment_inversion(
    rota: List[PontoEntrega],
    probabilidade_mutacao: float
) -> List[PontoEntrega]:
    """
    Operador de mutação por inversão de segmento:
    escolhe dois índices aleatórios e inverte o trecho entre eles.

    Parâmetros:
    - rota: sequência de pontos de entrega
    - probabilidade_mutacao: probabilidade de ocorrer mutação [0.0, 1.0]

    Retorno:
    - rota mutada (ou cópia da original se mutação não ocorrer)
    """
    rota_mutada = copy.deepcopy(rota)

    if random.random() < probabilidade_mutacao and len(rota) > 2:
        i, j = sorted(random.sample(range(len(rota)), 2))
        rota_mutada[i:j+1] = reversed(rota_mutada[i:j+1])

    return rota_mutada