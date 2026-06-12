"""
Modelagem dos pontos de entrega do domínio médico-hospitalar.

Cada ponto representa um destino de entrega de medicamentos ou insumos:
- pacientes domiciliares, UBS, clínicas, farmácias ou postos de saúde.
O primeiro ponto da lista (is_origin=True) é sempre o hospital base (ponto de partida e retorno).

Prioridade de atendimento:
    1 = Alta   (paciente crítico, medicamento urgente)
    2 = Média  (atendimento programado)
    3 = Baixa  (reposição de estoque, não urgente)
"""

from typing import List
from dataclasses import dataclass, field


@dataclass
class PontoEntrega:
    """Representa um ponto de entrega de medicamentos ou insumos."""
    nome: str
    coords: tuple          # (x, y) em pixels ou coordenadas normalizadas
    prioridade: int        # 1=Alta, 2=Média, 3=Baixa
    tempo_atendimento: int # minutos estimados no local
    is_origin: bool = False

    def __hash__(self):
        return hash((self.nome, self.coords))

    def __eq__(self, other):
        if isinstance(other, PontoEntrega):
            return self.coords == other.coords
        return False


# ---------------------------------------------------------------------------
# Dataset de exemplo: Hospital Base + 14 pontos de entrega
# ---------------------------------------------------------------------------
PONTOS_ENTREGA: List[PontoEntrega] = [
    PontoEntrega("Hospital Base",           coords=(512, 317), prioridade=0, tempo_atendimento=0,  is_origin=True),
    PontoEntrega("UBS Vila Nova",           coords=(741,  72), prioridade=1, tempo_atendimento=15),
    PontoEntrega("Clínica São Lucas",       coords=(552,  50), prioridade=2, tempo_atendimento=10),
    PontoEntrega("Posto Saúde Centro",      coords=(772, 346), prioridade=1, tempo_atendimento=20),
    PontoEntrega("Paciente - Rua das Flores",(637, 12),  prioridade=1, tempo_atendimento=10),
    PontoEntrega("Farmácia Popular Norte",  coords=(589, 131), prioridade=3, tempo_atendimento=5),
    PontoEntrega("UBS Jardim América",      coords=(732, 165), prioridade=2, tempo_atendimento=15),
    PontoEntrega("Paciente - Av. Brasil",   coords=(605,  15), prioridade=1, tempo_atendimento=10),
    PontoEntrega("Clínica Bem Estar",       coords=(730,  38), prioridade=2, tempo_atendimento=10),
    PontoEntrega("Posto Saúde Sul",         coords=(576, 216), prioridade=2, tempo_atendimento=20),
    PontoEntrega("UBS Parque Verde",        coords=(589, 381), prioridade=3, tempo_atendimento=15),
    PontoEntrega("Farmácia Central",        coords=(711, 387), prioridade=3, tempo_atendimento=5),
    PontoEntrega("Paciente - Rua do Lago",  coords=(563, 228), prioridade=1, tempo_atendimento=10),
    PontoEntrega("Posto Saúde Leste",       coords=(494,  22), prioridade=2, tempo_atendimento=20),
    PontoEntrega("UBS Vila Esperança",      coords=(787, 288), prioridade=3, tempo_atendimento=15),
]


def get_hospital_base() -> PontoEntrega:
    """Retorna o ponto de origem (hospital base)."""
    return next(p for p in PONTOS_ENTREGA if p.is_origin)


def get_pontos_entrega_sem_origem() -> List[PontoEntrega]:
    """Retorna apenas os pontos de entrega, excluindo o hospital base."""
    return [p for p in PONTOS_ENTREGA if not p.is_origin]


# Mapeamento reutilizável de prioridade para rótulo textual
PRIORIDADE_LABEL = {1: "Alta", 2: "Média", 3: "Baixa"}
