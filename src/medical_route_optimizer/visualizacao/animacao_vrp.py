# src/medical_route_optimizer/visualizacao/animacao_vrp.py
import matplotlib.pyplot as plt

class AnimacaoVRP:

    def __init__(self,  locais_entrega, hospital_base):
        self.hospital_base = hospital_base
        self.locais_entrega = locais_entrega
        self.historico_custos = []
        self.historico_media = []

        # Cria uma figura com 3 painéis lado a lado
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(24, 12))
        plt.subplots_adjust(wspace=0.25, hspace=0.3)
        
        # Painel 1: evolução dos custos
        self.ax1.set_title("Evolução da Função Objetivo")
        self.ax1.set_xlabel("Geração")
        self.ax1.set_ylabel("Custo")

        # Painel 2: mapa GA dinâmico
        self.ax2.set_title("Mapa GA (dinâmico)")
        self.ax2.set_xlim(400, 820)
        self.ax2.set_ylim(0, 420)

        # Painel 3: mapa VRP Split + Two-Opt (será desenhado no main)
        self.ax3.set_title("Mapa VRP Split + Two-Opt")
        self.ax3.set_xlim(400, 820)
        self.ax3.set_ylim(0, 420)

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

    
    def registrar(self, geracao, melhor_custo, media_custos, melhor_rota):
        self.historico_custos.append(melhor_custo)
        self.historico_media.append(media_custos)

        # --- Gráfico de evolução ---
        self.ax1.clear()
        self.ax1.plot(range(len(self.historico_custos)), self.historico_custos,
                    color='blue', label='Custo atual')
        self.ax1.plot(range(len(self.historico_media)), self.historico_media,
                    color='orange', linestyle='--', label='Média atual')
        self.ax1.set_title("Evolução da Função Objetivo")
        self.ax1.set_xlabel("Geração")
        self.ax1.set_ylabel("Custo")

        # Linhas pontilhadas para custo e média anteriores
        if len(self.historico_custos) > 1:
            self.ax1.plot(
                [len(self.historico_custos) - 2, len(self.historico_custos) - 1],
                [self.historico_custos[-2], self.historico_custos[-1]],
                color='red', linestyle='--', linewidth=2, marker='o', markersize=5,
                alpha=0.8
            )
            self.ax1.plot(
                [len(self.historico_media) - 2, len(self.historico_media) - 1],
                [self.historico_media[-2], self.historico_media[-1]],
                color='green', linestyle='--', linewidth=2, marker='s', markersize=5,
                alpha=0.8
            )

        # Ponto atual destacado
        self.ax1.scatter(len(self.historico_custos) - 1, self.historico_custos[-1],
                        color='blue', s=80, edgecolors='black')
        self.ax1.scatter(len(self.historico_media) - 1, self.historico_media[-1],
                        color='orange', s=80, edgecolors='black')

        # Legenda dinâmica com valores
        labels = [
            f"Custo atual: {self.historico_custos[-1]:.2f}",
            f"Média atual: {self.historico_media[-1]:.2f}"
        ]
        if len(self.historico_custos) > 1:
            labels.append(f"Custo anterior: {self.historico_custos[-2]:.2f}")
            labels.append(f"Média anterior: {self.historico_media[-2]:.2f}")

        self.ax1.legend(labels, loc="upper right", fontsize=8)

        # --- Mapa VRP/TSP dinâmico ---
        self.ax2.set_title(f"Mapa de Entregas - Geração {geracao}")
        self.ax2.set_xlim(400, 820)
        self.ax2.set_ylim(0, 420)

        # Atualiza ou cria os elementos do mapa
        rota_completa = [self.hospital_base] + melhor_rota + [self.hospital_base]
        x_rota = [p.coords[0] for p in rota_completa]
        y_rota = [p.coords[1] for p in rota_completa]

        # Atualiza a linha da rota sem limpar o mapa
        if hasattr(self, 'linha_rota'):
            self.linha_rota.set_data(x_rota, y_rota)
            self.linha_rota.set_color('lime' if geracao % 2 == 0 else 'green')
        else:
            self.linha_rota, = self.ax2.plot(x_rota, y_rota,
                                            color='green', lw=1.8)

        # Atualiza ou cria os pontos
        # Atualiza ou cria os pontos
        if not hasattr(self, 'pontos_clientes'):
            x = [p.coords[0] for p in self.locais_entrega]
            y = [p.coords[1] for p in self.locais_entrega]
            self.pontos_clientes = self.ax2.scatter(x, y, c='red', s=50, label="Clientes")

            # Hospital base só uma vez com label
            self.ponto_hospital = self.ax2.scatter(
                self.hospital_base.coords[0],
                self.hospital_base.coords[1],
                c='blue', s=80, marker='s', label="Hospital Base"
            )

            # Texto do hospital sem label
            self.ax2.text(self.hospital_base.coords[0] + 5,
                        self.hospital_base.coords[1] + 5,
                        "Hospital Base", fontsize=9, color="blue")

            # Nomes dos pontos
            for p in self.locais_entrega:
                self.ax2.text(p.coords[0] + 5, p.coords[1] + 5, p.nome, fontsize=8)


        # Texto dinâmico da geração
        if hasattr(self, 'texto_geracao'):
            self.texto_geracao.set_text(f"Geração: {geracao}")
        else:
            self.texto_geracao = self.ax2.text(0.05, 0.95, f"Geração: {geracao}",
                                            transform=self.ax2.transAxes,
                                            fontsize=10, color='darkgreen',
                                            bbox=dict(facecolor='white', alpha=0.6))

        #self.ax2.legend()
        handles, labels = self.ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax2.legend(by_label.values(), by_label.keys())

        plt.pause(0.01)
 
    def finalizar(self):
        """Finaliza a animação mantendo a janela aberta"""
        plt.ioff()
        plt.pause(0.001)  # mantém janela aberta sem travar o fluxo