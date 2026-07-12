import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("streamlit", types.SimpleNamespace(columns=None, markdown=None, caption=None))

from visualizacao import maps


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_build_route_map_cria_tracos_e_anotacoes(hospital, pontos_tres):
    fig = maps.build_route_map(hospital, pontos_tres, [pontos_tres[:2], pontos_tres[2:]])

    assert len(fig.data) == 10
    assert fig.data[0].name == "Veículo 1"
    assert fig.data[-1].name == "Hospital Base"
    assert len(fig.layout.annotations) == 6
    assert "Legenda dos Pontos" in fig.layout.annotations[-1].text
    assert "Veículo 2" in fig.layout.annotations[-1].text
    assert fig.layout.title.text == "Mapa das Rotas — Veículos com Numeração, Direção e Legenda de Pontos"
    assert fig.layout.yaxis.autorange == "reversed"


def test_render_vrp_cards_renderiza_status(monkeypatch):
    markdowns = []
    captions = []

    fake_st = SimpleNamespace(
        columns=lambda n: [_FakeColumn() for _ in range(n)],
        markdown=lambda text: markdowns.append(text),
        caption=lambda text: captions.append(text),
    )
    monkeypatch.setattr(maps, "st", fake_st)

    maps.render_vrp_cards(
        [
            {
                "veiculo": 1,
                "capacidade_ok": True,
                "autonomia_ok": True,
                "n_pontos": 2,
                "peso_total": 3.5,
                "capacidade_veiculo": 10.0,
                "distancia_pixels": 42.0,
                "autonomia_veiculo": 100.0,
            },
            {
                "veiculo": 2,
                "capacidade_ok": False,
                "autonomia_ok": True,
                "n_pontos": 1,
                "peso_total": 9.0,
                "capacidade_veiculo": 8.0,
                "distancia_pixels": 70.0,
                "autonomia_veiculo": 120.0,
            },
        ]
    )

    assert markdowns == [
        "**Veículo 1 ✅**",
        "**Veículo 2 ❌**",
    ]
    assert captions == [
        "2 paradas | Carga: 3.5/10.0 | Dist: 42/100.0 px",
        "1 paradas | Carga: 9.0/8.0 | Dist: 70/120.0 px",
    ]
