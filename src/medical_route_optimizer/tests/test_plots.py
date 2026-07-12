from visualizacao.plots import build_fitness_chart


def test_build_fitness_chart_normaliza_valores():
    fig = build_fitness_chart([10, 5], [8, 4], normalizar=True)

    assert len(fig.data) == 2
    assert list(fig.data[0].x) == [0, 1]
    assert list(fig.data[0].y) == [1.0, 0.5]
    assert list(fig.data[1].y) == [0.8, 0.4]
    assert fig.layout.xaxis.title.text == "Geração"
    assert fig.layout.yaxis.title.text == "Fitness (normalizado)"


def test_build_fitness_chart_sem_normalizacao_evita_divisao_por_zero():
    fig = build_fitness_chart([0, 0], [0, 0], normalizar=False)

    assert list(fig.data[0].y) == [0.0, 0.0]
    assert list(fig.data[1].y) == [0.0, 0.0]
    assert fig.layout.yaxis.title.text == "Fitness (custo)"
