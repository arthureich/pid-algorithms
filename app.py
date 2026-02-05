"""Interface gráfica com Threading para evitar congelamento."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import cv2
from PIL import Image, ImageTk

# Importa seus algoritmos como módulos locais (execução direta do app.py)
from box_filter import box_filter
from canny import canny
from freeman import boundary_to_image
from freeman import connect_points_image
from freeman import largest_component_mask
from freeman import subsample_boundary
from freeman import freeman_chain_code
from marr_hildreth import marr_hildreth
from otsu import count_objects, otsu_threshold
from segmentation import intensity_segmentation
from text import comparison_text
from utils import to_list
from watershed import watershed_segment
# Constante para limitar o tamanho da imagem e acelerar o processamento (opcional)
MAX_DIMENSION = 600 

def resize_if_needed(image):
    h, w = image.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def grayscale_to_image(data: list[list[int]]) -> Image.Image:
    height = len(data)
    width = len(data[0])
    flat = bytes([int(max(0, min(255, pixel))) for row in data for pixel in row])
    return Image.frombytes("L", (width, height), flat)

def rgb_to_image(data: list[list[tuple[int, int, int]]]) -> Image.Image:
    height = len(data)
    width = len(data[0])
    # Flatten e garante que valores estão entre 0-255
    flat = bytes([int(max(0, min(255, val))) for row in data for pixel in row for val in pixel])
    return Image.frombytes("RGB", (width, height), flat)

class ImagePanel(ttk.Label):
    def __init__(self, master: tk.Widget, text: str) -> None:
        super().__init__(master, text=text, anchor="center")
        self.image = None

    def update_image(self, image: Image.Image) -> None:
        display_img = image.copy()
        display_img.thumbnail((350, 350), Image.Resampling.LANCZOS)
        self.image = ImageTk.PhotoImage(display_img)
        self.configure(image=self.image, text="")

    def clear(self) -> None:
        self.configure(image="", text="Sem imagem")

class PIDApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Processamento de Imagens")
        self.geometry("1200x850")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self._build_edge_tab()
        self._build_otsu_tab()
        self._build_watershed_tab()
        self._build_freeman_tab()
        self._build_box_tab()
        self._build_segmentation_tab()
        
        # Barra de status
        self.status_var = tk.StringVar(value="Pronto.")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_image(self) -> list[list[int]] | None:
        path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not path:
            return None
        
        # Carrega com OpenCV
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
            
        # Reduz tamanho para performance 
        image = resize_if_needed(image)
        
        return to_list(image)

    # --- UTILITÁRIO DE THREADING ---
    def _run_async(self, worker_func, update_func):
        """
        Executa worker_func em uma thread separada.
        Quando terminar, chama update_func na thread principal com o resultado.
        """
        self.config(cursor="watch") # Cursor de 'carregando'
        self.status_var.set("Processando... Aguarde (pode demorar em Python puro)...")
        self.update_idletasks() # Força atualização da UI

        def thread_target():
            try:
                print("Iniciando Worker...") # Debug no console
                result = worker_func()
                print("Worker finalizado com sucesso.")
                self.after(0, lambda: self._on_process_complete(update_func, result))
            except Exception as e:
                import traceback
                traceback.print_exc() # Isso vai mostrar exatamente ONDE o código parou no seu terminal
                self.after(0, lambda err=e: self._on_process_error(err))

        threading.Thread(target=thread_target, daemon=True).start()

    def _on_process_complete(self, update_func, result):
        self.config(cursor="") # Restaura cursor
        self.status_var.set("Concluído.")
        update_func(result) # Atualiza a tela

    def _on_process_error(self, error):
        self.config(cursor="")
        self.status_var.set(f"Erro: {error}")

    # --- ABAS ---

    def _build_edge_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="1 & 2: Marr-Hildreth x Canny")
        
        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        ttk.Button(control, text="Carregar e Processar", command=self._run_edges).pack(side="left", padx=5)

        panels = ttk.Frame(frame)
        panels.pack(fill="both", expand=True)
        
        self.edge_original = ImagePanel(panels, "Original")
        self.edge_marr = ImagePanel(panels, "Marr-Hildreth")
        self.edge_canny = ImagePanel(panels, "Canny")
        
        for p in (self.edge_original, self.edge_marr, self.edge_canny):
            p.pack(side="left", expand=True, fill="both", padx=2)

        text = tk.Text(frame, height=6, wrap="word", bg="#f0f0f0")
        text.insert("1.0", comparison_text())
        text.configure(state="disabled")
        text.pack(fill="x", padx=5, pady=5)

    def _run_edges(self) -> None:
        image = self._load_image()
        if image is None: return

        # Mostra a original imediatamente
        self.edge_original.update_image(grayscale_to_image(image))

        # Define o trabalho pesado
        def worker():
            marr = marr_hildreth(image).edges
            cny = canny(image).edges
            return marr, cny

        # Define o que fazer quando terminar
        def update_ui(result):
            marr, cny = result
            self.edge_marr.update_image(grayscale_to_image(marr))
            self.edge_canny.update_image(grayscale_to_image(cny))

        self._run_async(worker, update_ui)

    def _build_otsu_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="3: Otsu + Contagem")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        ttk.Button(control, text="Carregar e Processar", command=self._run_otsu).pack(side="left", padx=5)
        self.otsu_label = ttk.Label(control, text="Objetos encontrados: -", font=("Arial", 12, "bold"))
        self.otsu_label.pack(side="left", padx=20)

        self.otsu_original = ImagePanel(frame, "Original")
        self.otsu_binary = ImagePanel(frame, "Binária (Otsu)")
        self.otsu_original.pack(side="left", expand=True, fill="both", padx=5)
        self.otsu_binary.pack(side="left", expand=True, fill="both", padx=5)

    def _run_otsu(self) -> None:
        image = self._load_image()
        if image is None: return
        self.otsu_original.update_image(grayscale_to_image(image))

        def worker():
            _, binary = otsu_threshold(image)
            count = count_objects(binary)
            return binary, count

        def update_ui(result):
            binary, count = result
            self.otsu_binary.update_image(grayscale_to_image(binary))
            self.otsu_label.configure(text=f"Objetos encontrados: {count}")

        self._run_async(worker, update_ui)

    def _build_watershed_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="4: Watershed")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        ttk.Button(control, text="Carregar e Processar", command=self._run_watershed).pack(side="left", padx=5)

        self.watershed_original = ImagePanel(frame, "Original")
        self.watershed_result = ImagePanel(frame, "Resultado Watershed")
        self.watershed_original.pack(side="left", expand=True, fill="both", padx=5)
        self.watershed_result.pack(side="left", expand=True, fill="both", padx=5)

    def _run_watershed(self) -> None:
        image = self._load_image()
        if image is None: return
        self.watershed_original.update_image(grayscale_to_image(image))

        def worker():
            return watershed_segment(image)

        def update_ui(result):
            self.watershed_result.update_image(rgb_to_image(result))

        self._run_async(worker, update_ui)

    def _build_freeman_tab(self) -> None:
        # 1. Cria o frame principal da aba
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="5: Cadeia de Freeman")

        # 2. Área de Controle (Botões) - Fixa no topo
        control = ttk.Frame(tab_frame)
        control.pack(fill="x", pady=5)
        ttk.Button(control, text="Carregar e Processar", command=self._run_freeman).pack(side="left", padx=5)

        # --- INÍCIO DA LÓGICA DE SCROLL ---
        
        # 3. Container para o Canvas e a Barra de Rolagem
        container = ttk.Frame(tab_frame)
        container.pack(fill="both", expand=True)

        # Canvas onde o conteúdo será desenhado
        canvas = tk.Canvas(container)
        
        # Barra de rolagem vertical ligada ao Canvas
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Frame interno que ficará "dentro" do Canvas (é aqui que colocamos os widgets)
        scrollable_frame = ttk.Frame(canvas)

        # Configura o scrollable_frame para atualizar a área de rolagem quando mudar de tamanho
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Cria a janela dentro do canvas apontando para o frame
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Conecta o canvas à barra de rolagem
        canvas.configure(yscrollcommand=scrollbar.set)

        # Posiciona Canvas e Scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # (Opcional) Habilita scroll com a roda do mouse (Windows/Linux)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Liga o evento de scroll quando o mouse entra no canvas
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- FIM DA LÓGICA DE SCROLL ---

        # Agora adicionamos tudo dentro de 'scrollable_frame' em vez de 'tab_frame'
        
        images_frame = ttk.Frame(scrollable_frame)
        images_frame.pack(fill="both", expand=True)

        row_top = ttk.Frame(images_frame)
        row_top.pack(fill="both", expand=True, pady=5)
        row_bottom = ttk.Frame(images_frame)
        row_bottom.pack(fill="both", expand=True, pady=5)

        self.freeman_panels = []
        labels = [
            "(a) Original/ruidosa",
            "(b) Suavizada (Box 9x9)",
            "(c) Limiarizada (Otsu)",
            "(d) Fronteira maior",
            "(e) Fronteira subamostrada",
            "(f) Pontos conectados",
        ]
        
        for i, label in enumerate(labels):
            # Coloca os 3 primeiros na linha de cima, os outros na de baixo
            parent = row_top if i < 3 else row_bottom
            panel = ImagePanel(parent, label)
            panel.pack(side="left", expand=True, fill="both", padx=5, pady=5)
            self.freeman_panels.append(panel)

        self.freeman_text = tk.Text(scrollable_frame, height=12, wrap="word") # Aumentei um pouco a altura do texto
        self.freeman_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.freeman_text.insert("1.0", "Carregue uma imagem para gerar a cadeia.")
        self.freeman_text.configure(state="disabled")

    def _run_freeman(self) -> None:
        image = self._load_image()
        if image is None: return
        
        def progress(message: str) -> None:
            def apply_message() -> None:
                self.status_var.set(message)
                self.update_idletasks()

            self.after(0, apply_message)

        def worker():
            progress("Freeman: suavizando (Box 9x9)...")
            smoothed = box_filter(image, 9)
            progress("Freeman: limiarizando (Otsu)...")
            _, binary = otsu_threshold(smoothed)
            progress("Freeman: buscando maior componente...")
            largest = largest_component_mask(binary)
            progress("Freeman: extraindo fronteira...")
            chain_result = freeman_chain_code(largest)
            progress("Freeman: preparando imagens de saída...")
            boundary_image = boundary_to_image(chain_result.boundary, len(largest), len(largest[0]))
            progress("Freeman: preparando 1")
            subsampled = subsample_boundary(chain_result.boundary, max(1, len(chain_result.boundary) // 60))
            progress("Freeman: preparando 2")
            subsampled_image = boundary_to_image(subsampled, len(largest), len(largest[0]))
            progress("Freeman: preparando 3")
            connected_image = connect_points_image(subsampled, len(largest), len(largest[0]))
            progress("Freeman: preparado")
            return image, smoothed, binary, boundary_image, subsampled_image, connected_image, chain_result

        def update_ui(result):
            original, smoothed, binary, boundary_img, subsampled_img, connected_img, chain_result = result
            images = [
                original,
                smoothed,
                binary,
                boundary_img,
                subsampled_img,
                connected_img,
            ]
            for panel, img in zip(self.freeman_panels, images):
                panel.update_image(grayscale_to_image(img))

            if not chain_result.chain:
                txt = "Nenhum objeto detectado ou objeto sem contorno fechado encontrado."
            else:
                chain_text = "".join(map(str, chain_result.chain))
                normalized_text = "".join(map(str, chain_result.normalized_chain))
                first_diff_text = "".join(map(str, chain_result.first_difference))
                circular_diff_text = "".join(map(str, chain_result.circular_first_difference))
                start_text = (
                    f"({chain_result.start_point[0]}, {chain_result.start_point[1]})"
                    if chain_result.start_point
                    else "-"
                )
                txt = (
                    "Seguidor de fronteira (Moore):\n"
                    f"Ponto inicial b0 (topo-esquerda): {start_text}\n"
                    f"Total de pontos na fronteira: {len(chain_result.boundary)}\n"
                    f"Total de elos na cadeia: {len(chain_result.chain)}\n\n"
                    "Cadeia de Freeman (bruta):\n"
                    f"{chain_text}\n\n"
                    "Cadeia normalizada (menor inteiro por rotação):\n"
                    f"{normalized_text}\n\n"
                    "1ª diferença (invariante à rotação):\n"
                    f"{first_diff_text}\n\n"
                    "1ª diferença circular:\n"
                    f"{circular_diff_text}"
                )

            self.freeman_text.configure(state="normal")
            self.freeman_text.delete("1.0", tk.END)
            self.freeman_text.insert("1.0", txt)
            self.freeman_text.configure(state="disabled")

        self._run_async(worker, update_ui)

    def _build_box_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="5: Filtro Box")
        
        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        ttk.Button(control, text="Carregar e Processar", command=self._run_box).pack(side="left", padx=5)
        
        scroll_frame = ttk.Frame(frame)
        scroll_frame.pack(fill="both", expand=True)
        
        self.box_panels = []
        for lbl in ["Original", "Box 2x2", "Box 3x3", "Box 5x5", "Box 7x7"]:
            p = ImagePanel(scroll_frame, lbl)
            p.pack(side="left", expand=True, fill="both", padx=2)
            self.box_panels.append(p)

    def _run_box(self) -> None:
        image = self._load_image()
        if image is None: return
        self.box_panels[0].update_image(grayscale_to_image(image))

        def worker():
            # Seleciona tamanhos baseados na dimensão da imagem
            h, w = len(image), len(image[0])
            sizes = [2, 3, 5, 7]
            if max(h, w) > 1000:
                sizes = [5, 9, 13, 17] # Aumenta filtro se img for grande
            
            results = []
            for s in sizes:
                results.append(box_filter(image, s))
            return results

        def update_ui(results):
            for i, res in enumerate(results):
                self.box_panels[i+1].update_image(grayscale_to_image(res))

        self._run_async(worker, update_ui)

    def _build_segmentation_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="6: Segmentação (Tabela)")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        ttk.Button(control, text="Carregar e Processar", command=self._run_segmentation).pack(side="left", padx=5)

        self.seg_original = ImagePanel(frame, "Original Greyscale")
        self.seg_result = ImagePanel(frame, "Segmentada")
        self.seg_original.pack(side="left", expand=True, fill="both", padx=5)
        self.seg_result.pack(side="left", expand=True, fill="both", padx=5)

    def _run_segmentation(self) -> None:
        image = self._load_image()
        if image is None: return
        self.seg_original.update_image(grayscale_to_image(image))

        def worker():
            return intensity_segmentation(image)

        def update_ui(result):
            self.seg_result.update_image(grayscale_to_image(result))

        self._run_async(worker, update_ui)

if __name__ == "__main__":
    app = PIDApp()
    app.mainloop()

