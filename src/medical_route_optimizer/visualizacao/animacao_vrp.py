# src/medical_route_optimizer/visualizacao/animacao_vrp.py
import matplotlib.pyplot as plt

class AnimacaoVRP:

    def __init__(self, locais_entrega, hospital_base):
        self.hospital_base = hospital_base
        self.locais_entrega = locais_entrega
        self.historico_custos = []
        self.historico_media = []

        # Figura 1 — Evolução da Função Objetivo
        self.fig_custos, self.ax_custos = plt.subplots(figsize=(6, 4))
        self.ax_custos.set_title("Evolução da Função Objetivo")
        self.ax_custos.set_xlabel("Geração")
        self.ax_custos.set_ylabel("Custo")

        # Figura 2 — Mapa de Entregas (dinâmico)
        self.fig_mapa, self.ax_mapa = plt.subplots(figsize=(6, 4))
        self.ax_mapa.set_title("Mapa de Entregas NN + GA (dinâmico)")
        self.ax_mapa.set_xlim(400, 820)
        self.ax_mapa.set_ylim(0, 420)

        # Figura 3 — Mapa de Entregas (dinâmico)
        self.fig_vrp_split, self.ax_vrp_split = plt.subplots(figsize=(6, 4))
        self.ax_vrp_split.set_title("Mapa de Entregas VRP SPLIT + TWO-2-OPT")
        self.ax_vrp_split.set_xlim(400, 820)
        self.ax_vrp_split.set_ylim(0, 420)


        plt.ion()  # modo interativo para atualização dinâmica


    def desenhar_populacao_inicial(self, ax, rota_inicial, titulo="Rota Inicial (Nearest Neighbor)"):

        ax.clear()
        rota_completa = [self.hospital_base] + rota_inicial + [self.hospital_base]

        # Coordenadas
        x_rota = [p.coords[0] for p in rota_completa]
        y_rota = [p.coords[1] for p in rota_completa]

        # Desenha rota inicial
        ax.plot(x_rota, y_rota, color="gray", lw=1.5, linestyle="--", label="Rota Inicial (NN)")

        # Clientes
        x = [p.coords[0] for p in self.locais_entrega]
        y = [p.coords[1] for p in self.locais_entrega]
        ax.scatter(x, y, c='red', s=50, label="Clientes")

        # Hospital base
        ax.scatter(self.hospital_base.coords[0], self.hospital_base.coords[1],
                c='blue', s=80, marker='s', label="Hospital Base")
       # ax.text(self.hospital_base.coords[0] + 5, self.hospital_base.coords[1] + 5,
       #         "Hospital Base", fontsize=9, color="blue")

        # Nomes dos pontos
        for p in self.locais_entrega:
            ax.text(p.coords[0] + 5, p.coords[1] + 5, p.nome, fontsize=8, color="gray")

        ax.set_title(titulo)
        ax.legend()
    
    def desenhar_rota(self, ax, rota, titulo):
        ax.clear()
        rota_completa = [self.hospital_base] + rota + [self.hospital_base]

        # Usa o atributo coords da classe
        x_rota = [p.coords[0] for p in rota_completa]
        y_rota = [p.coords[1] for p in rota_completa]

        # Clientes
        x = [p.coords[0] for p in self.locais_entrega]
        y = [p.coords[1] for p in self.locais_entrega]
        ax.scatter(x, y, c='red', s=50, label="Clientes")

        # Hospital base — desenha apenas uma vez com label
        ax.scatter(self.hospital_base.coords[0], self.hospital_base.coords[1],
                c='blue', s=80, marker='s', label="Hospital Base")

        # Texto do hospital sem label (não adiciona nova entrada)
        ax.text(self.hospital_base.coords[0] + 5, self.hospital_base.coords[1] + 5,
                "Hospital Base", fontsize=9, color="blue")

        # Rota
        ax.plot(x_rota, y_rota, color='green', lw=1.5)
        for i in range(len(rota_completa) - 1):
            x1, y1 = rota_completa[i].coords
            x2, y2 = rota_completa[i + 1].coords
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        # Nomes dos pontos
        for p in self.locais_entrega:
            ax.text(p.coords[0] + 5, p.coords[1] + 5, p.nome, fontsize=8)

        ax.set_title(titulo)

        # Remove duplicatas da legenda
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
   
    def desenhar_vrp_split(self, ax, rotas, titulo="Mapa VRP Split + Two-Opt"):
        ax.clear()

        # Paleta de cores para veículos
        cores = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

        # Se rotas for uma lista de várias rotas (um veículo por lista)
        if isinstance(rotas[0], list):
            for i, rota in enumerate(rotas):
                rota_completa = [self.hospital_base] + rota + [self.hospital_base]
                x_rota = [p.coords[0] for p in rota_completa]
                y_rota = [p.coords[1] for p in rota_completa]
                cor = cores[i % len(cores)]
                ax.plot(x_rota, y_rota, lw=1.5, color=cor, label=f"Veículo {i+1}")

                # Adiciona nomes dos pontos
                for p in rota:
                    ax.text(p.coords[0] + 5, p.coords[1] + 5, p.nome, fontsize=8, color=cor)
        else:
            # Caso seja uma única rota
            rota_completa = [self.hospital_base] + rotas + [self.hospital_base]
            x_rota = [p.coords[0] for p in rota_completa]
            y_rota = [p.coords[1] for p in rota_completa]
            ax.plot(x_rota, y_rota, lw=1.5, color="tab:blue", label="Rota Única")

            for p in rotas:
                ax.text(p.coords[0] + 5, p.coords[1] + 5, p.nome, fontsize=8, color="tab:blue")

        # Clientes
        x = [p.coords[0] for p in self.locais_entrega]
        y = [p.coords[1] for p in self.locais_entrega]
        ax.scatter(x, y, c='red', s=50, label="Clientes")

        # Hospital base
        ax.scatter(self.hospital_base.coords[0], self.hospital_base.coords[1],
                c='blue', s=80, marker='s', label="Hospital Base")
        ax.text(self.hospital_base.coords[0] + 5, self.hospital_base.coords[1] + 5,
                "Hospital Base", fontsize=9, color="blue")

        ax.set_title(titulo)
        ax.legend()

        self.fig_vrp_split.tight_layout()
        self.fig_vrp_split.canvas.draw()
        self.fig_vrp_split.canvas.flush_events()
        
        plt.pause(0.01)

    
    def registrar(self, geracao, melhor_custo, media_custos, melhor_rota, rota_nn=None):
        # Atualiza histórico
        self.historico_custos.append(melhor_custo)
        self.historico_media.append(media_custos)

        # --- Atualiza gráfico de evolução ---
        self.ax_custos.clear()
        self.ax_custos.plot(range(len(self.historico_custos)), self.historico_custos,
                            color='blue', label='Custo atual')
        self.ax_custos.plot(range(len(self.historico_media)), self.historico_media,
                            color='orange', linestyle='--', label='Média atual')
        self.ax_custos.set_title("Evolução da Função Objetivo")
        self.ax_custos.set_xlabel("Geração")
        self.ax_custos.set_ylabel("Custo")
        self.ax_custos.legend()
        self.fig_custos.canvas.draw()
        self.fig_custos.canvas.flush_events()

        # --- Atualiza mapa de entregas ---
        self.ax_mapa.clear()
        self.ax_mapa.set_title(f"Mapa de Entregas - Geração {geracao}")
        self.ax_mapa.set_xlim(400, 820)
        self.ax_mapa.set_ylim(0, 420)

        # --- Rota NN (comparação visual) ---
        if rota_nn is not None:
            # Se rota_nn for uma lista de listas, pega a primeira rota
            if isinstance(rota_nn[0], list):
                rota_nn = rota_nn[0]

            rota_nn_completa = [self.hospital_base] + rota_nn + [self.hospital_base]
            x_nn = [p.coords[0] for p in rota_nn_completa]
            y_nn = [p.coords[1] for p in rota_nn_completa]
            self.ax_mapa.plot(x_nn, y_nn, color='gray', lw=1.2, linestyle='--', label="Rota NN (baseline)")


        # --- Rota GA otimizada ---
        rota_completa = [self.hospital_base] + melhor_rota + [self.hospital_base]
        x_rota = [p.coords[0] for p in rota_completa]
        y_rota = [p.coords[1] for p in rota_completa]
        self.ax_mapa.plot(x_rota, y_rota, color='green', lw=1.8, label="Rota GA (otimizada)")

        # Setas indicando o fluxo da rota
        for i in range(len(rota_completa) - 1):
            x1, y1 = rota_completa[i].coords
            x2, y2 = rota_completa[i + 1].coords
            self.ax_mapa.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        # Clientes
        self.ax_mapa.scatter([p.coords[0] for p in self.locais_entrega],
                            [p.coords[1] for p in self.locais_entrega],
                            c='red', s=50, label="Clientes")

        # Hospital base
        self.ax_mapa.scatter(self.hospital_base.coords[0], self.hospital_base.coords[1],
                            c='blue', s=80, marker='s', label="Hospital Base")

        # Nomes diretamente nos pontos
        for p in self.locais_entrega:
            self.ax_mapa.text(p.coords[0] + 5, p.coords[1] + 5, p.nome,
                            fontsize=7, ha='center', va='bottom', color='black')

        # Nome do hospital
        self.ax_mapa.text(self.hospital_base.coords[0] + 5,
                        self.hospital_base.coords[1] + 5,
                        "Hospital Base", fontsize=8, color='blue')

        # Legenda e layout
        self.ax_mapa.legend(loc="upper left", fontsize=8)
        self.fig_mapa.tight_layout()
        self.fig_mapa.canvas.draw()
        self.fig_mapa.canvas.flush_events()

        plt.pause(0.01)
 
    def finalizar(self):
        """Finaliza a animação mantendo a janela aberta"""
        plt.ioff()
        plt.pause(0.001)  # mantém janela aberta sem travar o fluxo