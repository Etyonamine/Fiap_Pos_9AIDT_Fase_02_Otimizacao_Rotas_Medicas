import matplotlib.pyplot as plt

from visualizacao.animacao_vrp import AnimacaoVRP


def _build_animacao(monkeypatch, hospital, pontos_tres):
    plt.close("all")
    monkeypatch.setattr(plt, "ion", lambda: None)
    monkeypatch.setattr(plt, "pause", lambda *_: None)
    monkeypatch.setattr(plt, "ioff", lambda: None)
    return AnimacaoVRP(pontos_tres, hospital)


def test_init_configura_figuras(monkeypatch, hospital, pontos_tres):
    animacao = _build_animacao(monkeypatch, hospital, pontos_tres)

    assert animacao.hospital_base == hospital
    assert animacao.locais_entrega == pontos_tres
    assert animacao.historico_custos == []
    assert animacao.historico_media == []
    assert animacao.ax_custos.get_title() == "Evolução da Função Objetivo"
    assert animacao.ax_mapa.get_title() == "Mapa de Entregas NN + GA (dinâmico)"
    assert animacao.ax_vrp_split.get_title() == "Mapa de Entregas VRP SPLIT + TWO-2-OPT"


def test_desenhar_populacao_inicial_adiciona_legenda(monkeypatch, hospital, pontos_tres):
    animacao = _build_animacao(monkeypatch, hospital, pontos_tres)

    animacao.desenhar_populacao_inicial(animacao.ax_mapa, pontos_tres[:2])

    assert animacao.ax_mapa.get_title() == "Rota Inicial (Nearest Neighbor)"
    assert len(animacao.ax_mapa.lines) == 1
    assert animacao.ax_mapa.get_legend() is not None


def test_desenhar_rota_cria_setas_e_legenda_sem_duplicatas(monkeypatch, hospital, pontos_tres):
    animacao = _build_animacao(monkeypatch, hospital, pontos_tres)

    animacao.desenhar_rota(animacao.ax_mapa, pontos_tres[:2], "Rota Final")

    assert animacao.ax_mapa.get_title() == "Rota Final"
    assert len(animacao.ax_mapa.lines) == 1
    assert len(animacao.ax_mapa.texts) >= len(pontos_tres) + 1
    legend_labels = animacao.ax_mapa.get_legend_handles_labels()[1]
    assert len(legend_labels) == len(set(legend_labels))


def test_desenhar_vrp_split_cobre_multiplas_rotas(monkeypatch, hospital, pontos_tres):
    animacao = _build_animacao(monkeypatch, hospital, pontos_tres)

    animacao.desenhar_vrp_split(animacao.ax_vrp_split, [pontos_tres[:2], pontos_tres[2:]])

    assert animacao.ax_vrp_split.get_title() == "Mapa VRP Split + Two-Opt"
    assert len(animacao.ax_vrp_split.lines) == 2
    assert animacao.ax_vrp_split.get_legend() is not None


def test_desenhar_vrp_split_cobre_rota_unica(monkeypatch, hospital, pontos_tres):
    animacao = _build_animacao(monkeypatch, hospital, pontos_tres)

    animacao.desenhar_vrp_split(animacao.ax_vrp_split, pontos_tres[:2], titulo="Rota Única")

    assert animacao.ax_vrp_split.get_title() == "Rota Única"
    assert len(animacao.ax_vrp_split.lines) == 1


def test_registrar_atualiza_historicos_e_comparacao_nn(monkeypatch, hospital, pontos_tres):
    animacao = _build_animacao(monkeypatch, hospital, pontos_tres)

    animacao.registrar(
        geracao=3,
        melhor_custo=12.5,
        media_custos=15.0,
        melhor_rota=pontos_tres[:2],
        rota_nn=[pontos_tres[:2]],
    )

    assert animacao.historico_custos == [12.5]
    assert animacao.historico_media == [15.0]
    assert animacao.ax_mapa.get_title() == "Mapa de Entregas - Geração 3"
    labels = animacao.ax_mapa.get_legend_handles_labels()[1]
    assert "Rota NN (baseline)" in labels
    assert "Rota GA (otimizada)" in labels


def test_finalizar_desliga_modo_interativo(monkeypatch, hospital, pontos_tres):
    calls = {"ioff": 0, "pause": []}
    plt.close("all")
    monkeypatch.setattr(plt, "ion", lambda: None)
    monkeypatch.setattr(plt, "ioff", lambda: calls.__setitem__("ioff", calls["ioff"] + 1))
    monkeypatch.setattr(plt, "pause", lambda valor: calls["pause"].append(valor))

    animacao = AnimacaoVRP(pontos_tres, hospital)
    animacao.finalizar()

    assert calls["ioff"] == 1
    assert calls["pause"][-1] == 0.001
