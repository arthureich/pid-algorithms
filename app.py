from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import cv2
from PIL import Image, ImageTk
from box_filter import box_filter
from canny import canny
from freeman import boundary_to_image, subsample_boundary_grid, connect_points_image
from freeman import largest_component_mask, subsample_boundary, freeman_chain_code, get_freeman_from_points
from marr_hildreth import marr_hildreth
from otsu import count_objects, otsu_threshold
from segmentation import intensity_segmentation
from text import comparison_text
from utils import to_list
from watershed import watershed_segment
# Limitar o tamanho da imagem e acelerar o processamento 
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

    # --- UTHREADING ---
    def _run_async(self, worker_func, update_func):
        """
        Executa worker_func em uma thread separada.
        Quando terminar, chama update_func na thread principal com o resultado.
        """
        self.config(cursor="watch") 
        self.status_var.set("Processando...")
        self.update_idletasks() 

        def thread_target():
            try:
                print("Iniciando Worker...") 
                result = worker_func()
                print("Worker finalizado com sucesso.")
                self.after(0, lambda: self._on_process_complete(update_func, result))
            except Exception as e:
                import traceback
                traceback.print_exc() 
                self.after(0, lambda err=e: self._on_process_error(err))

        threading.Thread(target=thread_target, daemon=True).start()

    def _on_process_complete(self, update_func, result):
        self.config(cursor="") 
        self.status_var.set("Concluído.")
        update_func(result) 

    def _on_process_error(self, error):
        self.config(cursor="")
        self.status_var.set(f"Erro: {error}")

    # --- ABAS ---

    def _build_edge_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="1 & 2: Marr-Hildreth x Canny")
        
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill="x", pady=5, padx=5)
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(side="left", padx=5)
        ttk.Button(action_frame, text="Carregar e Processar", command=self._run_edges).pack(side="top", pady=2)

        # --- MARR-HILDRETH ---
        mh_group = ttk.LabelFrame(control_frame, text="Marr-Hildreth")
        mh_group.pack(side="left", padx=5, fill="y")

        ttk.Label(mh_group, text="Sigma:").grid(row=0, column=0, sticky="e")
        self.mh_sigma = tk.DoubleVar(value=1.4)
        ttk.Entry(mh_group, textvariable=self.mh_sigma, width=5).grid(row=0, column=1)

        ttk.Label(mh_group, text="Thresh:").grid(row=0, column=2, sticky="e")
        self.mh_threshold = tk.DoubleVar(value=0.5)
        ttk.Entry(mh_group, textvariable=self.mh_threshold, width=5).grid(row=0, column=3)

        ttk.Label(mh_group, text="Kernel:").grid(row=1, column=0, sticky="e")
        self.mh_kernel = tk.IntVar(value=0)
        ttk.Spinbox(mh_group, from_=0, to=51, increment=2, textvariable=self.mh_kernel, width=5).grid(row=1, column=1)
        ttk.Label(mh_group, text="(0=Auto)").grid(row=1, column=2, columnspan=2, sticky="w")

        # --- CANNY ---
        canny_group = ttk.LabelFrame(control_frame, text="Canny")
        canny_group.pack(side="left", padx=5, fill="y")

        ttk.Label(canny_group, text="Sigma:").grid(row=0, column=0, sticky="e")
        self.canny_sigma = tk.DoubleVar(value=1.0)
        ttk.Entry(canny_group, textvariable=self.canny_sigma, width=5).grid(row=0, column=1)

        ttk.Label(canny_group, text="Kernel:").grid(row=0, column=2, sticky="e")
        self.canny_kernel = tk.IntVar(value=5)
        ttk.Spinbox(canny_group, from_=3, to=31, increment=2, textvariable=self.canny_kernel, width=5).grid(row=0, column=3)

        ttk.Label(canny_group, text="T High:").grid(row=1, column=0, sticky="e")
        self.canny_th_high = tk.DoubleVar(value=0.15) 
        ttk.Entry(canny_group, textvariable=self.canny_th_high, width=5).grid(row=1, column=1)

        ttk.Label(canny_group, text="T Low:").grid(row=1, column=2, sticky="e")
        self.canny_th_low = tk.DoubleVar(value=0.05) 
        ttk.Entry(canny_group, textvariable=self.canny_th_low, width=5).grid(row=1, column=3)
        
        panels = ttk.Frame(frame)
        panels.pack(fill="both", expand=True)
        
        self.edge_original = ImagePanel(panels, "Original")
        self.edge_marr = ImagePanel(panels, "Marr-Hildreth")
        self.edge_canny = ImagePanel(panels, "Canny")
        
        for p in (self.edge_original, self.edge_marr, self.edge_canny):
            p.pack(side="left", expand=True, fill="both", padx=2)

        text = tk.Text(frame, height=3, wrap="word", bg="#f0f0f0")
        text.insert("1.0", "Marr-Hildreth: Kernel 0 = Automático. \nCanny: T High e T Low são porcentagens (0.0 a 1.0) da magnitude máxima da borda.")
        text.configure(state="disabled")
        text.pack(fill="x", padx=5, pady=5)

    def _run_edges(self) -> None:
        image = self._load_image()
        if image is None: return

        # Parâmetros Marr-Hildreth
        try:
            mh_s = self.mh_sigma.get()
            mh_t = self.mh_threshold.get()
            mh_k = self.mh_kernel.get()
            
            # Parâmetros Canny
            cn_s = self.canny_sigma.get()
            cn_k = self.canny_kernel.get()
            cn_th = self.canny_th_high.get()
            cn_tl = self.canny_th_low.get()
            
        except tk.TclError:
            self.status_var.set("Erro: Verifique se todos os parâmetros são números válidos.")
            return
        
        self.edge_original.update_image(grayscale_to_image(image))

        k_mh_text = "Auto" if mh_k == 0 else f"{mh_k}x{mh_k}"
        self.edge_marr.configure(text=f"Marr-Hildreth\n(σ={mh_s}, Th={mh_t}, K={k_mh_text})")
        self.edge_canny.configure(text=f"Canny\n(σ={cn_s}, K={cn_k}x{cn_k}, H={cn_th}, L={cn_tl})")

        def worker():
            marr = marr_hildreth(image, sigma=mh_s, threshold=mh_t, kernel_size=mh_k).edges
            
            cny = canny(
                image, 
                sigma=cn_s, 
                kernel_size=cn_k, 
                low_ratio=cn_tl, 
                high_ratio=cn_th
            ).edges
            
            return marr, cny

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
        
        ttk.Button(control, text="Carregar Imagem", command=self._load_otsu_image).pack(side="left", padx=5)
        
        ttk.Label(control, text="Threshold (0=Auto):").pack(side="left", padx=(15, 2))
        
        self.otsu_scale_var = tk.IntVar(value=0)
        self.otsu_scale = tk.Scale(
            control, 
            from_=0, 
            to=255, 
            orient=tk.HORIZONTAL, 
            variable=self.otsu_scale_var,
            length=200,
            command=self._on_otsu_slider_change 
        )
        self.otsu_scale.pack(side="left", padx=5)

        self.otsu_label = ttk.Label(control, text="Objetos: -", font=("Arial", 12, "bold"))
        self.otsu_label.pack(side="left", padx=20)

        panels = ttk.Frame(frame)
        panels.pack(fill="both", expand=True)
        
        self.otsu_original = ImagePanel(panels, "Original")
        self.otsu_binary = ImagePanel(panels, "Binária (Otsu)")
        self.otsu_original.pack(side="left", expand=True, fill="both", padx=5)
        self.otsu_binary.pack(side="left", expand=True, fill="both", padx=5)
        
        self.otsu_cached_image = None
        self.otsu_timer = None 

    def _load_otsu_image(self) -> None:
        """Carrega a imagem do disco e guarda em self.otsu_cached_image"""
        image = self._load_image()
        if image is None: return
        
        self.otsu_cached_image = image
        self.otsu_original.update_image(grayscale_to_image(image))
        
        self.otsu_scale.set(0)
        
        self._update_otsu_processing()

    def _on_otsu_slider_change(self, event=None):
        """Chamado toda vez que o slider move. Usa debounce para não travar."""
        if self.otsu_cached_image is None:
            return

        if self.otsu_timer:
            self.after_cancel(self.otsu_timer)
        
        self.otsu_timer = self.after(150, self._update_otsu_processing)

    def _update_otsu_processing(self) -> None:
        """Lógica de processamento (roda na Thread)"""
        if self.otsu_cached_image is None: 
            return
            
        manual_val = self.otsu_scale_var.get()
        image = self.otsu_cached_image 

        def worker():
            th_used, binary = otsu_threshold(image, manual_threshold=manual_val)
            
            h = len(binary)
            w = len(binary[0])
            corners = [binary[0][0], binary[0][w-1], binary[h-1][0], binary[h-1][w-1]]
            if corners.count(255) >= 3:
                binary = [[255 - val for val in row] for row in binary]

            # 2. Conta Objetos
            count = count_objects(binary, min_area=20)
            
            return binary, count, th_used

        def update_ui(result):
            binary, count, th_used = result
            self.otsu_binary.update_image(grayscale_to_image(binary))
            
            # Texto informativo
            status_txt = "Auto" if manual_val == 0 else "Manual"
            self.otsu_label.configure(text=f"Objetos: {count} | Th: {th_used} ({status_txt})")
            self.otsu_binary.configure(text=f"Binária (T={th_used})")

        self._run_async(worker, update_ui)

    def _build_watershed_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="4: Watershed")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        ttk.Button(control, text="Carregar e Processar", command=self._run_watershed).pack(side="left", padx=5)

        container = ttk.Frame(frame)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        v_scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        h_scroll = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        
        self.ws_scroll_frame = ttk.Frame(canvas)

        self.ws_scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.ws_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.ws_original = ImagePanel(self.ws_scroll_frame, "1. Original")
        self.ws_dist = ImagePanel(self.ws_scroll_frame, "2. Distância")
        self.ws_markers = ImagePanel(self.ws_scroll_frame, "3. Marcadores")
        self.ws_result = ImagePanel(self.ws_scroll_frame, "4. Watershed Final")
        
        for p in (self.ws_original, self.ws_dist, self.ws_markers, self.ws_result):
            p.pack(side="left", padx=10, pady=10)

    def _run_watershed(self) -> None:
        image = self._load_image()
        if image is None: return
        self.ws_original.update_image(grayscale_to_image(image))

        def worker():
            # Resultado, Distância e Marcadores
            return watershed_segment(image)

        def update_ui(result):
            final_img, dist_img, markers_img = result
            
            self.ws_dist.update_image(grayscale_to_image(dist_img))
            
            self.ws_markers.update_image(rgb_to_image(markers_img))
            self.ws_result.update_image(rgb_to_image(final_img))

        self._run_async(worker, update_ui)

    def _build_freeman_tab(self) -> None:
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="5: Cadeia de Freeman")

        control = ttk.Frame(tab_frame)
        control.pack(fill="x", pady=5)
        ttk.Button(control, text="Carregar e Processar", command=self._run_freeman).pack(side="left", padx=5)
        ttk.Label(control, text="Subamostragem do grid (px):").pack(side="left", padx=(15, 2))
        self.grid_step_var = tk.IntVar(value=20)

        spin = ttk.Spinbox(
            control, 
            from_=5, 
            to=100, 
            textvariable=self.grid_step_var, 
            width=5,
            increment=5 
        )
        spin.pack(side="left", padx=2)
        
        container = ttk.Frame(tab_frame)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
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
            "(d) Fronteira maior (Moore)",
            "(e) Fronteira subamostrada",
            "(f) Pontos conectados",
        ]
        
        for i, label in enumerate(labels):
            parent = row_top if i < 3 else row_bottom
            panel = ImagePanel(parent, label)
            panel.pack(side="left", expand=True, fill="both", padx=5, pady=5)
            self.freeman_panels.append(panel)

        self.freeman_text = tk.Text(scrollable_frame, height=14, wrap="word") 
        self.freeman_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.freeman_text.insert("1.0", "Selecione o tamanho do grid e carregue uma imagem.")
        self.freeman_text.configure(state="disabled")

    def _run_freeman(self) -> None:
        image = self._load_image()
        if image is None: return
        current_grid_step = self.grid_step_var.get()

        def progress(message: str) -> None:
            def apply_message() -> None:
                self.status_var.set(message)
                self.update_idletasks()

            self.after(0, apply_message)

        def worker():
            smoothed = box_filter(image, 9)
            _, binary = otsu_threshold(smoothed)
            largest = largest_component_mask(binary)
            raw_result = freeman_chain_code(largest)
            boundary_image = boundary_to_image(raw_result.boundary, len(largest), len(largest[0]))
            simplified = subsample_boundary_grid(raw_result.boundary, current_grid_step)
            short_chain = get_freeman_from_points(simplified)
            subsampled_image = boundary_to_image(simplified, len(largest), len(largest[0]))
            connected_image = connect_points_image(simplified, len(largest), len(largest[0]))
            return image, smoothed, binary, boundary_image, subsampled_image, connected_image, raw_result, short_chain, simplified, current_grid_step

        def update_ui(result):
            original, smoothed, binary, boundary_img, subsampled_img, connected_img, raw_result, short_chain, sparse_points, step_used = result
            images = [original, smoothed, binary, boundary_img, subsampled_img, connected_img]
            for panel, img in zip(self.freeman_panels, images):
                panel.update_image(grayscale_to_image(img))

            if not raw_result.chain:
                txt = "Nenhum objeto detectado ou objeto sem contorno fechado encontrado."
            else:
                chain_text = "".join(map(str, raw_result.chain))
                short_chain_str = "".join(map(str, short_chain)) if short_chain else "N/A"
                first_diff_text = "".join(map(str, raw_result.first_difference))
                circular_diff_text = "".join(map(str, raw_result.circular_first_difference))
                start_text = (
                    f"({raw_result.start_point[0]}, {raw_result.start_point[1]})"
                    if raw_result.start_point
                    else "-"
                )
                txt = (
                    "Seguidor de fronteira (Moore):\n"
                    f"Ponto inicial b0 (topo-meio): {start_text}\n"
                    f"Pontos Resumidos (Grade): {len(sparse_points)}\n\n"
                    f"Total de pontos na fronteira: {len(raw_result.boundary)}\n"
                    "Cadeia de Freeman (bruta):\n"
                    f"{chain_text}\n\n"
                    f"CADEIA DE FREEMAN (Simplificada):\n"
                    f"{short_chain}\n\n"
                    "1ª diferença:\n"
                    f"{first_diff_text}\n\n"
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

        self.box_info_label = ttk.Label(control, text="Tamanhos originais: Filtro 1: 2x2, Filtro 2: 3x3, Filtro 3: 5x5, Filtro 4: 7x7. Ajustados para imagens grandes.", font=("Arial", 10, "italic"))
        self.box_info_label.pack(side="left", padx=15)
        
        scroll_frame = ttk.Frame(frame)
        scroll_frame.pack(fill="both", expand=True)
        
        self.box_panels = []
        for lbl in ["Original", "Filtro 1", "Filtro 2", "Filtro 3", "Filtro 4"]:
            p = ImagePanel(scroll_frame, lbl)
            p.pack(side="left", expand=True, fill="both", padx=2)
            self.box_panels.append(p)

    def _run_box(self) -> None:
        image = self._load_image()
        if image is None: return
        self.box_panels[0].update_image(grayscale_to_image(image))
        self.box_panels[0].configure(text="Original")

        def worker():
            # Seleciona tamanhos baseados na dimensão da imagem
            h, w = len(image), len(image[0])

            sizes = [2, 3, 5, 7]
            # Verifica a resolução da imagem. Se a imagem for muito grande, o ruído é muito
            # fino em relação aos pixels, então filtros pequenos tornam-se imperceptíveis. 
            # Neste caso, usa kernels maiores para garantir que a suavização seja visível.
            is_large = max(h, w) > 1200
            
            if is_large:
                sizes = [9, 15, 25, 35]
            
            results = []
            for s in sizes:
                results.append(box_filter(image, s))
            
            return results, sizes, is_large

        def update_ui(payload):
            results, sizes, is_large = payload
            if is_large:
                self.box_info_label.configure(text="Imagem grande detectada: Usando filtros maiores (9, 15, 25, 35).", foreground="blue")
            else:
                self.box_info_label.configure(text="Imagem Padrão: Usando filtros normais (2, 3, 5, 7).", foreground="black")
            for i, res in enumerate(results):
                panel = self.box_panels[i+1]
                panel.update_image(grayscale_to_image(res))
                panel.configure(text=f"Box {sizes[i]}x{sizes[i]}", compound="top")

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

