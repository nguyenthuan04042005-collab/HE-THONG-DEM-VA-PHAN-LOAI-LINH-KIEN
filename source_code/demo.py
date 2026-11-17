"""Desktop demo with Tkinter; computation runs in a worker thread."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from classifier import CLASS_NAMES, DISPLAY_NAMES
from config import load_config
from pipeline import COLORS, analyze_image, save_result, render_overlay

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
PREVIEW_MAX = 1600  # previews are downscaled once, then fitted to the canvas
FONT = "Segoe UI"
THEME = {
    "bg": "#f1f5f9", "card": "#ffffff", "border": "#e2e8f0", "text": "#0f172a",
    "muted": "#64748b", "header": "#0f172a", "header_sub": "#94a3b8",
    "accent": "#2563eb", "accent_hover": "#1d4ed8", "canvas": "#1e293b",
    "ok": "#15803d", "warn": "#b45309", "error": "#b91c1c", "select": "#facc15",
}
STAGES = [("rgb", "Gốc"), ("gray", "Xám"), ("magnitude", "Gradient"), ("nms", "NMS"),
          ("edges", "Biên"), ("closed", "Closing"), ("labels", "Vùng"),
          ("holes", "Lỗ"), ("overlay", "Kết quả")]
# name, label, hint, minimum, maximum, step, display format, integer
PARAMETERS = [
    ("sigma", "Sigma", "Độ làm trơn Gaussian (≥ 0.5)", 0.5, 10, 0.1, "%.1f", False),
    ("low_threshold", "Ngưỡng thấp", "Ngưỡng hysteresis dưới (0–255)", 1, 255, 5, "%.0f", False),
    ("high_threshold", "Ngưỡng cao", "Ngưỡng hysteresis trên (0–255)", 1, 255, 5, "%.0f", False),
    ("closing_iterations", "Closing", "Số lần giãn/co để nối biên hở", 0, 20, 1, "%.0f", True),
    ("min_area", "Diện tích tối thiểu", "Bỏ vùng nhỏ hơn (pixel)", 1, 1000000, 50, "%.0f", True),
]
TABLE_COLUMNS = [  # key, title, width, anchor, stretch
    ("id", "ID", 46, "center", False), ("label", "Loại", 110, "w", False),
    ("elongation", "Độ kéo dài", 95, "center", False), ("holes", "Số lỗ", 60, "center", False),
    ("radial_cv", "Độ lệch bán kính", 125, "center", False),
    ("hex_score", "Đặc trưng lục giác", 130, "center", False),
    ("reason", "Lý do phân loại", 260, "w", True),
]


def hex_color(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


def natural_key(path):
    """Sort anh2 before anh10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def format_value(value):
    return str(value) if isinstance(value, int) else f"{value:g}"


class Demo:
    def __init__(self, root, config, output_dir):
        self.root = root
        self.config = config
        self.output_dir = Path(output_dir)
        self.messages = queue.Queue()
        self.image_path = None
        self.folder_images = []
        self.previews = {}
        self.preview_factor = 1.0
        self.report = None
        self.last_target = None
        self.busy = False
        self.photo = None
        self.view = None  # (offset_x, offset_y, scale) from original pixels to canvas
        self._resize_job = None
        self.ui_scale = max(1.0, root.winfo_fpixels("1i") / 96)
        self._setup_style()
        self._build()
        self._bind_keys()
        self._update_stage_buttons()
        root.after(100, self.poll)

    def px(self, value):
        return int(round(value * self.ui_scale))

    # ------------------------------------------------------------------ layout
    def _setup_style(self):
        t = THEME
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.root.configure(bg=t["bg"])
        style.configure(".", background=t["bg"], foreground=t["text"], font=(FONT, 10))
        style.configure("Card.TFrame", background=t["card"])
        style.configure("Card.TLabel", background=t["card"], foreground=t["text"])
        style.configure("CardTitle.TLabel", background=t["card"], foreground=t["text"],
                        font=(FONT, 11, "bold"))
        style.configure("File.TLabel", background=t["card"], foreground=t["text"],
                        font=(FONT, 10, "bold"))
        style.configure("Hint.TLabel", background=t["card"], foreground=t["muted"], font=(FONT, 8))
        style.configure("Muted.TLabel", background=t["bg"], foreground=t["muted"], font=(FONT, 8))
        style.configure("Status.TLabel", background=t["card"], foreground=t["muted"], font=(FONT, 9))
        style.configure("Card.TCheckbutton", background=t["card"], foreground=t["text"])
        style.map("Card.TCheckbutton", background=[("active", t["card"])])
        style.configure("TButton", padding=(10, 6), background="#e2e8f0", foreground=t["text"],
                        bordercolor=t["border"], lightcolor="#e2e8f0", darkcolor="#e2e8f0")
        style.map("TButton", background=[("disabled", "#f1f5f9"), ("active", "#cbd5e1")],
                  foreground=[("disabled", "#94a3b8")])
        style.configure("Accent.TButton", background=t["accent"], foreground="white",
                        bordercolor=t["accent"], lightcolor=t["accent"], darkcolor=t["accent"],
                        font=(FONT, 11, "bold"), padding=(10, 10))
        style.map("Accent.TButton",
                  background=[("disabled", "#93c5fd"), ("active", t["accent_hover"])],
                  lightcolor=[("active", t["accent_hover"])], darkcolor=[("active", t["accent_hover"])],
                  foreground=[("disabled", "#eff6ff")])
        style.configure("Link.TButton", background=t["card"], foreground=t["accent"], padding=(0, 2),
                        borderwidth=0, bordercolor=t["card"], lightcolor=t["card"],
                        darkcolor=t["card"], font=(FONT, 9, "underline"))
        style.map("Link.TButton", background=[("active", t["card"])],
                  foreground=[("active", t["accent_hover"])])
        style.configure("Stage.Toolbutton", background="#f1f5f9", foreground=t["text"],
                        padding=(10, 5), bordercolor=t["border"], font=(FONT, 9))
        style.map("Stage.Toolbutton",
                  background=[("selected", t["accent"]), ("disabled", t["card"]), ("active", "#e2e8f0")],
                  foreground=[("selected", "white"), ("disabled", "#cbd5e1")])
        style.configure("TSpinbox", padding=4, fieldbackground="white", bordercolor=t["border"],
                        arrowsize=self.px(12))
        style.configure("Treeview", background="white", fieldbackground="white",
                        rowheight=self.px(26), bordercolor=t["border"], font=(FONT, 10))
        style.configure("Treeview.Heading", background="#f8fafc", foreground=t["muted"],
                        font=(FONT, 9, "bold"), relief="flat", padding=5)
        style.map("Treeview", background=[("selected", "#dbeafe")],
                  foreground=[("selected", t["text"])])
        style.configure("Horizontal.TProgressbar", background=t["accent"], troughcolor="#e2e8f0",
                        bordercolor="#e2e8f0", lightcolor=t["accent"], darkcolor=t["accent"])

    def card(self, parent, title=None):
        """White bordered panel; returns (outer, content) so content may use grid or pack."""
        outer = tk.Frame(parent, bg=THEME["card"], highlightthickness=1,
                         highlightbackground=THEME["border"])
        inner = ttk.Frame(outer, style="Card.TFrame", padding=self.px(12))
        inner.pack(fill="both", expand=True)
        if title:
            ttk.Label(inner, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, self.px(8)))
        content = ttk.Frame(inner, style="Card.TFrame")
        content.pack(fill="both", expand=True)
        return outer, content

    def _build(self):
        t, root, px = THEME, self.root, self.px
        root.title("Đếm và phân loại linh kiện · Manual Canny")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        width, height = min(px(1280), sw - 60), min(px(860), sh - 100)
        root.geometry(f"{width}x{height}+{(sw - width) // 2}+{max(0, (sh - height) // 2 - 20)}")
        root.minsize(min(px(1000), width), min(px(680), height))
        try:  # use the whole screen so the image preview gets as much room as possible
            root.state("zoomed")
        except tk.TclError:
            pass

        header = tk.Frame(root, bg=t["header"])
        header.pack(fill="x")
        tk.Label(header, text="Hệ thống đếm & phân loại linh kiện", bg=t["header"], fg="white",
                 font=(FONT, 15, "bold")).pack(anchor="w", padx=px(18), pady=(px(12), 0))
        tk.Label(header, text="Canny tự cài đặt  ·  Tách vùng  ·  Phân loại bu-lông, vòng đệm, đai ốc",
                 bg=t["header"], fg=t["header_sub"], font=(FONT, 9)).pack(
                     anchor="w", padx=px(18), pady=(px(2), px(12)))

        statusbar = tk.Frame(root, bg=t["card"], highlightthickness=1, highlightbackground=t["border"])
        statusbar.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(statusbar, mode="indeterminate", length=px(140))
        self.status = tk.StringVar()
        self.status_label = ttk.Label(statusbar, textvariable=self.status, style="Status.TLabel")
        self.status_label.pack(side="left", fill="x", expand=True, padx=px(12), pady=px(6))
        self._set_status("Chọn ảnh linh kiện để bắt đầu. Kết quả gồm ảnh từng bước, CSV đặc trưng và JSON.")

        body = ttk.Frame(root, padding=px(12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, minsize=px(290))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, px(12)))
        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky="nsew")
        self._build_sidebar(sidebar)
        self._build_main(main)

    def _build_sidebar(self, parent):
        px = self.px
        outer, box = self.card(parent, "Ảnh đầu vào")
        outer.pack(fill="x")
        self.choose_button = ttk.Button(box, text="Chọn ảnh…   (Ctrl+O)", command=self.choose_image)
        self.choose_button.pack(fill="x")
        self.file_text = tk.StringVar(value="Chưa chọn ảnh")
        ttk.Label(box, textvariable=self.file_text, style="File.TLabel",
                  wraplength=px(250)).pack(anchor="w", pady=(px(10), 0))
        self.position_text = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.position_text, style="Hint.TLabel").pack(anchor="w")
        nav = ttk.Frame(box, style="Card.TFrame")
        nav.pack(fill="x", pady=(px(8), 0))
        nav.columnconfigure((0, 1), weight=1, uniform="nav")
        self.prev_button = ttk.Button(nav, text="‹  Ảnh trước", command=lambda: self.step(-1))
        self.prev_button.grid(row=0, column=0, sticky="ew", padx=(0, px(4)))
        self.next_button = ttk.Button(nav, text="Ảnh sau  ›", command=lambda: self.step(1))
        self.next_button.grid(row=0, column=1, sticky="ew", padx=(px(4), 0))
        self.auto_run = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Tự phân tích khi đổi ảnh", variable=self.auto_run,
                        style="Card.TCheckbutton").pack(anchor="w", pady=(px(8), 0))

        outer, box = self.card(parent, "Tham số")
        outer.pack(fill="x", pady=(px(12), 0))
        box.columnconfigure(0, weight=1)
        self.parameters = {}
        for row, (name, label, hint, low, high, step, fmt, _) in enumerate(PARAMETERS):
            ttk.Label(box, text=label, style="Card.TLabel").grid(
                row=2 * row, column=0, sticky="w", pady=(px(6) if row else 0, 0))
            ttk.Label(box, text=hint, style="Hint.TLabel").grid(row=2 * row + 1, column=0, sticky="w")
            value = tk.StringVar(value=format_value(getattr(self.config, name)))
            ttk.Spinbox(box, textvariable=value, from_=low, to=high, increment=step, format=fmt,
                        width=8, justify="right").grid(row=2 * row, column=1, rowspan=2, sticky="e",
                                                       pady=(px(6) if row else 0, 0))
            self.parameters[name] = value
        ttk.Button(box, text="Khôi phục mặc định", style="Link.TButton",
                   command=self.reset_parameters).grid(row=2 * len(PARAMETERS), column=0,
                                                       sticky="w", pady=(px(10), 0))

        self.run_button = ttk.Button(parent, text="Phân tích   (F5)", style="Accent.TButton",
                                     command=self.run)
        self.run_button.pack(fill="x", pady=(px(12), 0))
        self.open_button = ttk.Button(parent, text="Mở thư mục kết quả", command=self.open_results)
        self.open_button.pack(fill="x", pady=(px(8), 0))
        self.open_button.state(["disabled"])
        ttk.Label(parent, text="Phím tắt: ← → đổi ảnh  ·  1–9 đổi bước xem\n"
                               "Bấm vào vật thể trên ảnh để xem chi tiết",
                  style="Muted.TLabel", justify="left").pack(side="bottom", anchor="w")

    def _build_main(self, parent):
        t, px = THEME, self.px
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        counts = ttk.Frame(parent)
        counts.grid(row=0, column=0, sticky="ew")
        self.count_vars = {}
        items = [(name, DISPLAY_NAMES[name], hex_color(COLORS[name])) for name in CLASS_NAMES]
        items.append(("total", "Tổng số", t["text"]))
        for column, (key, label, color) in enumerate(items):
            counts.columnconfigure(column, weight=1, uniform="count")
            box = tk.Frame(counts, bg=t["card"], highlightthickness=1, highlightbackground=t["border"])
            box.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else px(10), 0))
            tk.Frame(box, bg=color, width=px(5)).pack(side="left", fill="y")
            inner = tk.Frame(box, bg=t["card"])
            inner.pack(side="left", fill="both", expand=True, padx=px(12), pady=px(8))
            tk.Label(inner, text=label, bg=t["card"], fg=t["muted"], font=(FONT, 9)).pack(anchor="w")
            value = tk.StringVar(value="–")
            tk.Label(inner, textvariable=value, bg=t["card"], fg=color,
                     font=(FONT, 22, "bold")).pack(anchor="w")
            self.count_vars[key] = value

        outer, box = self.card(parent)
        outer.grid(row=1, column=0, sticky="nsew", pady=px(12))
        bar = ttk.Frame(box, style="Card.TFrame")
        bar.pack(fill="x", pady=(0, px(8)))
        self.stage = tk.StringVar(value="rgb")
        self.stage_buttons = {}
        for index, (key, label) in enumerate(STAGES, start=1):
            button = ttk.Radiobutton(bar, text=f"{index}. {label}", value=key, variable=self.stage,
                                     style="Stage.Toolbutton", command=self.show_preview)
            button.pack(side="left", padx=(0, px(4)))
            self.stage_buttons[key] = button
        self.canvas = tk.Canvas(box, bg=t["canvas"], highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        outer, box = self.card(parent, "Chi tiết từng vật thể")
        outer.grid(row=2, column=0, sticky="ew")
        self.table = ttk.Treeview(box, columns=[c[0] for c in TABLE_COLUMNS],
                                  show="headings", height=4)
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        for key, title, width, anchor, stretch in TABLE_COLUMNS:
            self.table.heading(key, text=title, anchor=anchor)
            self.table.column(key, width=px(width), minwidth=px(40), anchor=anchor, stretch=stretch)
        for name in CLASS_NAMES:
            self.table.tag_configure(name, foreground=hex_color(COLORS[name]))
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", lambda event: self.show_preview())

    def _bind_keys(self):
        self.root.bind("<Control-o>", lambda event: self.choose_image())
        self.root.bind("<F5>", lambda event: self.run())
        self.root.bind("<Left>", lambda event: self._key_step(event, -1))
        self.root.bind("<Right>", lambda event: self._key_step(event, 1))
        for index in range(1, len(STAGES) + 1):
            self.root.bind(str(index), self._key_stage)

    @staticmethod
    def _typing(event):
        return isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Spinbox))

    def _key_step(self, event, delta):
        if not self._typing(event):
            self.step(delta)

    def _key_stage(self, event):
        if self._typing(event):
            return
        key = STAGES[int(event.char) - 1][0]
        if key in self.previews:
            self.stage.set(key)
            self.show_preview()

    # ------------------------------------------------------------------ state
    def _set_status(self, text, kind="normal"):
        colors = {"normal": THEME["muted"], "ok": THEME["ok"], "warn": THEME["warn"],
                  "error": THEME["error"]}
        self.status.set(text)
        self.status_label.configure(foreground=colors[kind])

    def _set_busy(self, busy):
        self.busy = busy
        flag = ["disabled"] if busy else ["!disabled"]
        for button in (self.run_button, self.choose_button, self.prev_button, self.next_button):
            button.state(flag)
        if busy:
            self.progress.pack(side="right", padx=self.px(12))
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()
        self.root.configure(cursor="watch" if busy else "")

    def _update_stage_buttons(self):
        for key, button in self.stage_buttons.items():
            button.state(["!disabled"] if key in self.previews else ["disabled"])

    def _set_previews(self, images):
        width, height = next(iter(images.values())).size
        factor = min(1.0, PREVIEW_MAX / max(width, height))
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        self.preview_factor = factor
        self.previews = {key: (image.resize(size, Image.Resampling.BILINEAR) if factor < 1 else image)
                         for key, image in images.items()}

    def _clear_results(self):
        self.report = None
        self.table.delete(*self.table.get_children())
        for value in self.count_vars.values():
            value.set("–")

    # ------------------------------------------------------------------ actions
    def choose_image(self):
        if self.busy:
            return
        initial = Path(self.image_path).parent if self.image_path else Path("input")
        path = filedialog.askopenfilename(
            initialdir=str(initial) if initial.is_dir() else None,
            filetypes=[("Ảnh", " ".join(f"*{ext}" for ext in IMAGE_TYPES)), ("Tất cả", "*.*")])
        if path:
            self.load_image(path)

    def load_image(self, path):
        path = Path(path)
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
        except OSError as error:
            messagebox.showerror("Không mở được ảnh", str(error))
            return
        self.image_path = str(path)
        self.folder_images = sorted((p for p in path.parent.iterdir()
                                     if p.suffix.lower() in IMAGE_TYPES), key=natural_key)
        names = [p.name for p in self.folder_images]
        position = f"Ảnh {names.index(path.name) + 1} / {len(names)}" if path.name in names else ""
        self.file_text.set(path.name)
        self.position_text.set(f"{position}  ·  {rgb.width} × {rgb.height} px  ·  {path.parent.name}/")
        self._set_previews({"rgb": rgb})
        self._clear_results()
        self.stage.set("rgb")
        self._update_stage_buttons()
        target = self.output_dir / path.stem
        self.last_target = target if target.is_dir() else None
        self.open_button.state(["!disabled"] if self.last_target else ["disabled"])
        self.show_preview()
        if self.auto_run.get():
            self.run()
        else:
            self._set_status("Nhấn Phân tích (F5) để đếm và phân loại.")

    def step(self, delta):
        if self.busy or not self.folder_images or not self.image_path:
            return
        names = [p.name for p in self.folder_images]
        current = Path(self.image_path).name
        index = names.index(current) if current in names else -delta
        self.load_image(self.folder_images[(index + delta) % len(names)])

    def reset_parameters(self):
        for name, value in self.parameters.items():
            value.set(format_value(getattr(self.config, name)))

    def _read_parameters(self):
        values = {}
        for name, label, *_, integer in PARAMETERS:
            text = self.parameters[name].get().strip().replace(",", ".")
            try:
                number = float(text)
            except ValueError:
                raise ValueError(f"{label}: '{text}' không phải là số") from None
            if integer:
                if not number.is_integer():
                    raise ValueError(f"{label} phải là số nguyên")
                number = int(number)
            values[name] = number
        return values

    def run(self):
        if self.busy:
            return
        if not self.image_path:
            messagebox.showinfo("Chọn ảnh", "Hãy chọn ảnh trước khi phân tích.")
            return
        try:
            config = replace(self.config, **self._read_parameters())
        except ValueError as error:
            messagebox.showerror("Tham số không hợp lệ", str(error))
            return
        self._set_busy(True)
        self._set_status(f"Đang xử lý {Path(self.image_path).name}: phát hiện biên, tách vùng, phân loại…")
        # Capture the path so choosing a different file cannot alter this job.
        threading.Thread(target=self.worker, args=(self.image_path, config), daemon=True).start()

    def worker(self, image_path, config):
        try:
            report, stages = analyze_image(image_path, config)
            target = self.output_dir / Path(image_path).stem
            save_result(report, stages, target)
            self.messages.put((True, (report, stages, target)))
        except Exception as error:
            self.messages.put((False, str(error)))

    def poll(self):
        try:
            ok, payload = self.messages.get_nowait()
        except queue.Empty:
            pass
        else:
            self._set_busy(False)
            if not ok:
                self._set_status("Phân tích không thành công", "error")
                messagebox.showerror("Lỗi phân tích", payload)
            else:
                self._show_result(*payload)
        self.root.after(100, self.poll)

    def _show_result(self, report, stages, target):
        self.report, self.last_target = report, target
        images = {key: Image.fromarray(stages[key]) for key, _ in STAGES if key in stages}
        images["overlay"] = render_overlay(stages["rgb"], report)
        self._set_previews(images)
        for name in CLASS_NAMES:
            self.count_vars[name].set(str(report["counts"][name]))
        self.count_vars["total"].set(str(report["total"]))
        self.table.delete(*self.table.get_children())
        for obj in report["objects"]:
            features = obj["features"]
            self.table.insert("", "end", iid=str(obj["id"]), tags=(obj["label"],), values=(
                obj["id"], obj["display_name"], f"{features['elongation']:.3f}",
                features["hole_count"], f"{features['radial_cv']:.4f}",
                f"{features['hex_score']:.4f}", obj["reason"]))
        self.stage.set("overlay")
        self._update_stage_buttons()
        self.open_button.state(["!disabled"])
        message = f"Xong trong {report['processing_seconds']:.2f} s  ·  Đã lưu vào {target}"
        if report["warnings"]:
            self._set_status(message + "  ·  Cảnh báo: " + "; ".join(report["warnings"]), "warn")
        else:
            self._set_status(message, "ok")
        self.show_preview()

    def open_results(self):
        if not (self.last_target and self.last_target.is_dir()):
            return
        try:
            os.startfile(self.last_target.resolve())
        except (AttributeError, OSError):
            messagebox.showinfo("Thư mục kết quả", str(self.last_target.resolve()))

    # ------------------------------------------------------------------ preview
    def _on_resize(self, event=None):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self.show_preview)

    def show_preview(self, event=None):
        self._resize_job = None
        canvas = self.canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        image = self.previews.get(self.stage.get())
        if image is None:
            self.view = None
            canvas.create_text(width / 2, height / 2, fill="#94a3b8", font=(FONT, 12),
                               text="Chọn ảnh để bắt đầu  (Ctrl+O)")
            return
        scale = min(width / image.width, height / image.height, 1.0 / self.preview_factor)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        self.photo = ImageTk.PhotoImage(image.resize(size, Image.Resampling.BILINEAR))
        left, top = (width - size[0]) // 2, (height - size[1]) // 2
        canvas.create_image(left, top, image=self.photo, anchor="nw")
        self.view = (left, top, scale * self.preview_factor)
        self._draw_selection()

    def _draw_selection(self):
        if not (self.report and self.view):
            return
        left, top, scale = self.view
        objects = {str(obj["id"]): obj for obj in self.report["objects"]}
        for item in self.table.selection():
            obj = objects.get(item)
            if obj is None:
                continue
            x0, y0, x1, y1 = obj["bbox"]
            box = (left + x0 * scale - 4, top + y0 * scale - 4, left + x1 * scale + 4, top + y1 * scale + 4)
            self.canvas.create_rectangle(*box, outline=THEME["select"], width=3, dash=(6, 3))
            if self.stage.get() != "overlay":  # the overlay image already carries its own labels
                self.canvas.create_text(box[0], box[1] - 4, anchor="sw", fill=THEME["select"],
                                        font=(FONT, 10, "bold"),
                                        text=f"#{obj['id']} {obj['display_name']}")

    def _on_canvas_click(self, event):
        if not (self.report and self.view):
            return
        left, top, scale = self.view
        x, y = (event.x - left) / scale, (event.y - top) / scale
        hits = [obj for obj in self.report["objects"]
                if obj["bbox"][0] <= x < obj["bbox"][2] and obj["bbox"][1] <= y < obj["bbox"][3]]
        if hits:
            obj = min(hits, key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
            self.table.selection_set(str(obj["id"]))
            self.table.see(str(obj["id"]))
        else:
            self.table.selection_set(())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="results/demo")
    args = parser.parse_args()
    try:  # crisp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError, ImportError):
        pass
    root = tk.Tk()
    Demo(root, load_config(args.config), args.output)
    root.mainloop()


if __name__ == "__main__":
    main()
