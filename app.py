"""Interface gráfica """
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from algorithms import (
    box_filter,
    canny,
    comparison_text,
    count_objects,
    freeman_chain_code,
    intensity_segmentation,
    marr_hildreth,
    otsu_threshold,
    to_list,
    watershed_segment,
)


def grayscale_to_image(data: list[list[int]]) -> Image.Image:
    height = len(data)
    width = len(data[0])
    flat = bytes([pixel for row in data for pixel in row])
    return Image.frombytes("L", (width, height), flat)


def rgb_to_image(data: list[list[tuple[int, int, int]]]) -> Image.Image:
    height = len(data)
    width = len(data[0])
    flat = bytes([value for row in data for pixel in row for value in pixel])
    return Image.frombytes("RGB", (width, height), flat)


class ImagePanel(ttk.Label):
    def __init__(self, master: tk.Widget, text: str) -> None:
        super().__init__(master, text=text, anchor="center")
        self.image = None

    def update_image(self, image: Image.Image) -> None:
        self.image = ImageTk.PhotoImage(image)
        self.configure(image=self.image, text="")

    def clear(self) -> None:
        self.configure(image="", text="Sem imagem")


class PIDApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Processamento de Imagens - PID")
        self.geometry("1200x800")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self._build_edge_tab()
        self._build_otsu_tab()
        self._build_watershed_tab()
        self._build_freeman_tab()
        self._build_box_tab()
        self._build_segmentation_tab()

    def _load_image(self) -> list[list[int]] | None:
        path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not path:
            return None
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return to_list(image)

    def _build_edge_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Marr-Hildreth x Canny")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)

        load_button = ttk.Button(control, text="Carregar imagem", command=self._run_edges)
        load_button.pack(side="left", padx=5)

        self.edge_original = ImagePanel(frame, "Original")
        self.edge_marr = ImagePanel(frame, "Marr-Hildreth")
        self.edge_canny = ImagePanel(frame, "Canny")

        panels = ttk.Frame(frame)
        panels.pack(fill="both", expand=True)
        for panel in (self.edge_original, self.edge_marr, self.edge_canny):
            panel.pack(in_=panels, side="left", expand=True, fill="both", padx=5, pady=5)

        text = tk.Text(frame, height=8, wrap="word")
        text.insert("1.0", comparison_text())
        text.configure(state="disabled")
        text.pack(fill="x", padx=5, pady=5)

    def _run_edges(self) -> None:
        image = self._load_image()
        if image is None:
            return
        self.edge_original.update_image(grayscale_to_image(image))
        marr = marr_hildreth(image).edges
        canny_edges = canny(image).edges
        self.edge_marr.update_image(grayscale_to_image(marr))
        self.edge_canny.update_image(grayscale_to_image(canny_edges))

    def _build_otsu_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Otsu + Contagem")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        load_button = ttk.Button(control, text="Carregar imagem", command=self._run_otsu)
        load_button.pack(side="left", padx=5)
        self.otsu_label = ttk.Label(control, text="Objetos: -")
        self.otsu_label.pack(side="left", padx=10)

        self.otsu_original = ImagePanel(frame, "Original")
        self.otsu_binary = ImagePanel(frame, "Binária")
        self.otsu_original.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        self.otsu_binary.pack(side="left", expand=True, fill="both", padx=5, pady=5)

    def _run_otsu(self) -> None:
        image = self._load_image()
        if image is None:
            return
        self.otsu_original.update_image(grayscale_to_image(image))
        _, binary = otsu_threshold(image)
        count = count_objects(binary)
        self.otsu_label.configure(text=f"Objetos: {count}")
        self.otsu_binary.update_image(grayscale_to_image(binary))

    def _build_watershed_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Watershed")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        load_button = ttk.Button(control, text="Carregar imagem", command=self._run_watershed)
        load_button.pack(side="left", padx=5)

        self.watershed_original = ImagePanel(frame, "Original")
        self.watershed_result = ImagePanel(frame, "Watershed")
        self.watershed_original.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        self.watershed_result.pack(side="left", expand=True, fill="both", padx=5, pady=5)

    def _run_watershed(self) -> None:
        image = self._load_image()
        if image is None:
            return
        self.watershed_original.update_image(grayscale_to_image(image))
        result = watershed_segment(image)
        self.watershed_result.update_image(rgb_to_image(result))

    def _build_freeman_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Cadeia de Freeman")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        load_button = ttk.Button(control, text="Carregar imagem", command=self._run_freeman)
        load_button.pack(side="left", padx=5)

        self.freeman_original = ImagePanel(frame, "Binária")
        self.freeman_original.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        self.freeman_text = tk.Text(frame, height=10, wrap="word")
        self.freeman_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _run_freeman(self) -> None:
        image = self._load_image()
        if image is None:
            return
        _, binary = otsu_threshold(image)
        self.freeman_original.update_image(grayscale_to_image(binary))
        chain = freeman_chain_code(binary)
        self.freeman_text.configure(state="normal")
        self.freeman_text.delete("1.0", tk.END)
        self.freeman_text.insert("1.0", "Cadeia: " + ", ".join(map(str, chain)))
        self.freeman_text.configure(state="disabled")

    def _build_box_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Filtro Box")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        load_button = ttk.Button(control, text="Carregar imagem", command=self._run_box)
        load_button.pack(side="left", padx=5)

        self.box_panels: list[ImagePanel] = []
        for label in ("Original", "2x2", "3x3", "5x5", "7x7"):
            panel = ImagePanel(frame, label)
            panel.pack(side="left", expand=True, fill="both", padx=5, pady=5)
            self.box_panels.append(panel)

    def _run_box(self) -> None:
        image = self._load_image()
        if image is None:
            return
        self.box_panels[0].update_image(grayscale_to_image(image))
        sizes = [2, 3, 5, 7]
        if max(len(image), len(image[0])) > 1500:
            # Para imagens muito grandes, usamos filtros maiores para aumentar o efeito da suavização.
            sizes = [5, 9, 13, 17]
        for panel, size in zip(self.box_panels[1:], sizes, strict=False):
            filtered = box_filter(image, size)
            panel.update_image(grayscale_to_image(filtered))

    def _build_segmentation_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Segmentação por Intensidade")

        control = ttk.Frame(frame)
        control.pack(fill="x", pady=5)
        load_button = ttk.Button(control, text="Carregar imagem", command=self._run_segmentation)
        load_button.pack(side="left", padx=5)

        self.seg_original = ImagePanel(frame, "Original")
        self.seg_result = ImagePanel(frame, "Segmentada")
        self.seg_original.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        self.seg_result.pack(side="left", expand=True, fill="both", padx=5, pady=5)

    def _run_segmentation(self) -> None:
        image = self._load_image()
        if image is None:
            return
        self.seg_original.update_image(grayscale_to_image(image))
        segmented = intensity_segmentation(image)
        self.seg_result.update_image(grayscale_to_image(segmented))


if __name__ == "__main__":
    app = PIDApp()
    app.mainloop()
