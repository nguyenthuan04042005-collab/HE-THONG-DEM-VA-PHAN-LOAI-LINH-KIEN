"""Desktop demo with Tkinter; computation runs in a worker thread."""

import argparse
from dataclasses import replace
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from classifier import DISPLAY_NAMES
from config import load_config
from pipeline import analyze_image, save_result, render_overlay


class Demo:
    def __init__(self, root, config, output_dir):
        self.root, self.config = root, config
        self.output_dir = Path(output_dir)
        self.messages = queue.Queue()
        self.image_path = None
        self.previews = {}
        root.title("Đếm và phân loại linh kiện · Manual Canny")
        root.geometry("1180x820")
        root.minsize(950, 700)
        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x")
        self.choose_button = ttk.Button(toolbar, text="Chọn ảnh", command=self.choose_image)
        self.choose_button.pack(side="left")
        self.run_button = ttk.Button(toolbar, text="Phân tích", command=self.run)
        self.run_button.pack(side="left", padx=8)
        self.path_text = tk.StringVar(value="Chọn ảnh linh kiện để bắt đầu")
        ttk.Label(toolbar, textvariable=self.path_text).pack(side="left")
        controls = ttk.LabelFrame(frame, text="Tham số", padding=8)
        controls.pack(fill="x", pady=10)
        self.parameters = {}
        for name, label in [("sigma", "Sigma"), ("low_threshold", "Ngưỡng thấp"),
                            ("high_threshold", "Ngưỡng cao"),
                            ("closing_iterations", "Closing"), ("min_area", "Diện tích tối thiểu")]:
            ttk.Label(controls, text=label).pack(side="left", padx=(8, 3))
            value = tk.StringVar(value=str(getattr(config, name)))
            ttk.Entry(controls, textvariable=value, width=7).pack(side="left")
            self.parameters[name] = value
        self.count_text = tk.StringVar(value="Chưa có kết quả")
        ttk.Label(frame, textvariable=self.count_text, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        choices = ttk.Frame(frame)
        choices.pack(fill="x", pady=8)
        ttk.Label(choices, text="Xem bước:").pack(side="left")
        self.stage = tk.StringVar(value="overlay")
        selector = ttk.Combobox(choices, textvariable=self.stage, state="readonly", width=18,
                               values=("rgb", "gray", "magnitude", "nms", "edges", "closed",
                                       "labels", "holes", "overlay"))
        selector.pack(side="left", padx=8)
        selector.bind("<<ComboboxSelected>>", self.show_preview)
        self.preview = ttk.Label(frame, anchor="center")
        self.preview.pack(fill="both", expand=True)
        columns = ("id", "label", "elongation", "holes", "radial_cv", "hex_score")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", height=5)
        titles = ("ID", "Loại", "Độ kéo dài", "Số lỗ", "Độ lệch bán kính", "Đặc trưng lục giác")
        for name, title in zip(columns, titles):
            self.table.heading(name, text=title)
            self.table.column(name, width=140, anchor="center")
        self.table.pack(fill="x", pady=8)
        self.status = tk.StringVar(value="Kết quả sẽ gồm ảnh, CSV đặc trưng và JSON tham số.")
        ttk.Label(frame, textvariable=self.status, wraplength=1100).pack(anchor="w")
        root.after(100, self.poll)
        self.preview.bind("<Configure>", self.show_preview)

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.image_path = path
            self.path_text.set(Path(path).name)
            with Image.open(path) as image:
                self.previews = {"rgb": image.convert("RGB")}
            self.table.delete(*self.table.get_children())
            self.count_text.set("Chưa phân tích ảnh này")
            self.stage.set("rgb")
            self.show_preview()

    def run(self):
        if not self.image_path:
            messagebox.showinfo("Chọn ảnh", "Hãy chọn ảnh trước khi phân tích.")
            return
        try:
            values = {name: (int(value.get()) if name in ("closing_iterations", "min_area")
                             else float(value.get())) for name, value in self.parameters.items()}
            config = replace(self.config, **values)
        except ValueError as error:
            messagebox.showerror("Tham số không hợp lệ", str(error))
            return
        self.run_button.configure(state="disabled")
        self.choose_button.configure(state="disabled")
        self.status.set("Đang phát hiện biên, tách vùng và phân loại...")
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
            self.run_button.configure(state="normal")
            self.choose_button.configure(state="normal")
            if not ok:
                self.status.set("Phân tích không thành công")
                messagebox.showerror("Lỗi phân tích", payload)
            else:
                report, stages, target = payload
                self.path_text.set(Path(report["image"]).name)
                self.count_text.set("  |  ".join(f"{DISPLAY_NAMES[k]}: {v}" for k, v in report["counts"].items())
                                    + f"  |  Tổng: {report['total']}")
                self.previews = {name: Image.fromarray(array) for name, array in stages.items()}
                self.previews["overlay"] = render_overlay(stages["rgb"], report)
                self.table.delete(*self.table.get_children())
                for obj in report["objects"]:
                    features = obj["features"]
                    self.table.insert("", "end", values=(obj["id"], obj["display_name"],
                                      f"{features['elongation']:.3f}", features["hole_count"],
                                      f"{features['radial_cv']:.4f}", f"{features['hex_score']:.4f}"))
                self.stage.set("overlay")
                warnings = " | ".join(report["warnings"])
                self.status.set(f"{report['processing_seconds']:.2f}s · Đã lưu: {target}" + (f" · {warnings}" if warnings else ""))
                self.show_preview()
        self.root.after(100, self.poll)

    def show_preview(self, event=None):
        if self.stage.get() not in self.previews:
            return
        image = self.previews[self.stage.get()].copy()
        image.thumbnail((max(200, self.preview.winfo_width()),
                         max(200, self.preview.winfo_height())))
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="results/demo")
    args = parser.parse_args()
    root = tk.Tk()
    Demo(root, load_config(args.config), args.output)
    root.mainloop()


if __name__ == "__main__":
    main()
