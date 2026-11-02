"""Standalone Tkinter GUI for OpenCV feature matching and object localization."""

from __future__ import annotations

import base64
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


class ORBMatcherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OpenCV Feature Matcher")
        self.root.geometry("1280x820")
        self.root.minsize(980, 650)

        self.template: np.ndarray | None = None
        self.scene: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.template_path = ""
        self.scene_path = ""
        self._photos: dict[str, tk.PhotoImage] = {}

        self.orb_nfeatures = tk.IntVar(value=1500)
        self.orb_scale_factor = tk.DoubleVar(value=1.2)
        self.orb_nlevels = tk.IntVar(value=8)
        self.orb_fast_threshold = tk.IntVar(value=20)
        self.sift_nfeatures = tk.IntVar(value=1500)
        self.sift_contrast_threshold = tk.DoubleVar(value=0.04)
        self.sift_edge_threshold = tk.DoubleVar(value=10.0)
        self.sift_sigma = tk.DoubleVar(value=1.6)
        self.akaze_threshold = tk.DoubleVar(value=0.001)
        self.akaze_n_octaves = tk.IntVar(value=4)
        self.akaze_n_octave_layers = tk.IntVar(value=4)
        self.akaze_diffusivity = tk.StringVar(value="PM_G2")
        self.ratio = tk.DoubleVar(value=0.75)
        self.min_matches = tk.IntVar(value=10)
        self.algorithm = tk.StringVar(value="ORB")
        self.use_clahe = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Load a reference image and a scene image, then click Find Match.")
        self._worker_queue: queue.Queue = queue.Queue()
        self._busy = False

        self._build_ui()
        self.root.after(100, self._poll_worker)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("TButton", font=("Segoe UI", 12), padding=(8, 6))
        style.configure("TLabel", font=("Segoe UI", 12))
        style.configure("TLabelframe.Label", font=("Segoe UI", 12, "bold"))
        style.configure("TSpinbox", font=("Segoe UI", 12), padding=4)
        style.configure("TNotebook.Tab", font=("Segoe UI", 12), padding=(12, 7))
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.match_tab = ttk.Frame(self.notebook)
        self.about_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.match_tab, text="Matching")
        self.notebook.add(self.about_tab, text="About Algorithms")

        controls = ttk.Frame(self.match_tab, padding=(12, 10))
        controls.pack(fill="x")
        top_row = ttk.Frame(controls)
        top_row.pack(fill="x", pady=(0, 8))

        self.load_template_button = ttk.Button(top_row, text="1. Load Reference Image", command=self.load_template)
        self.load_template_button.pack(side="left", padx=(0, 8))
        self.load_scene_button = ttk.Button(top_row, text="2. Load Scene Image", command=self.load_scene)
        self.load_scene_button.pack(side="left", padx=8)
        ttk.Separator(top_row, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(top_row, text="Feature Method").pack(side="left", padx=(0, 6))
        self.algorithm_box = ttk.Combobox(
            top_row,
            textvariable=self.algorithm,
            values=("ORB", "SIFT", "AKAZE"),
            state="readonly",
            width=9,
        )
        self.algorithm_box.pack(side="left", padx=(0, 12))
        self.algorithm_box.bind("<<ComboboxSelected>>", self._on_algorithm_changed)

        self.algorithm_params = ttk.LabelFrame(self.match_tab, text="Algorithm Parameters", padding=(10, 8))
        self.algorithm_params.pack(fill="x", padx=12, pady=(0, 8))
        self.algorithm_parameter_widgets: list[ttk.Widget] = []
        self._rebuild_algorithm_parameters()

        shared = ttk.Frame(self.match_tab, padding=(12, 0, 12, 8))
        shared.pack(fill="x")
        ttk.Label(shared, text="Lowe Match Ratio").pack(side="left", padx=(0, 4))
        ttk.Spinbox(
            shared, from_=0.50, to=0.95, increment=0.05, width=7, textvariable=self.ratio, format="%.2f"
        ).pack(side="left", padx=(0, 16))
        ttk.Label(shared, text="Min. RANSAC Inliers").pack(side="left", padx=(0, 4))
        ttk.Spinbox(shared, from_=4, to=100, increment=1, width=6, textvariable=self.min_matches).pack(
            side="left", padx=(0, 16)
        )
        self.clahe_check = ttk.Checkbutton(shared, text="Enhance contrast (CLAHE)", variable=self.use_clahe)
        self.clahe_check.pack(side="left", padx=(0, 16))
        self.match_button = ttk.Button(shared, text="3. Find Match", command=self.match_images)
        self.match_button.pack(side="left", padx=4)
        self.save_button = ttk.Button(shared, text="Save Result Image", command=self.save_result)
        self.save_button.pack(side="left", padx=4)
        self.save_button.configure(state="disabled")

        self.template_title = ttk.Label(self.match_tab, text="Reference: No image selected", anchor="w")
        self.template_title.pack(fill="x", padx=14, pady=(0, 2))
        self.scene_title = ttk.Label(self.match_tab, text="Scene: No image selected", anchor="w")
        self.scene_title.pack(fill="x", padx=14, pady=(0, 8))

        views = ttk.Frame(self.match_tab, padding=(10, 0, 10, 8))
        views.pack(fill="both", expand=True)
        views.columnconfigure((0, 1, 2), weight=1, uniform="view")
        views.rowconfigure(0, weight=1)

        self.canvases: dict[str, tk.Canvas] = {}
        for col, (key, title) in enumerate(
            (("template", "Reference Image"), ("scene", "Scene Image"), ("result", "Matches and Detected Object"))
        ):
            panel = ttk.LabelFrame(views, text=title, padding=5)
            panel.grid(row=0, column=col, sticky="nsew", padx=4)
            canvas = tk.Canvas(panel, background="#202124", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda _event, name=key: self._render(name))
            self.canvases[key] = canvas

        ttk.Label(self.match_tab, textvariable=self.status, relief="sunken", anchor="w", padding=(8, 7)).pack(
            fill="x", side="bottom"
        )

        self._build_about_tab()

    def _on_algorithm_changed(self, _event: tk.Event | None = None) -> None:
        self._rebuild_algorithm_parameters()

    def _rebuild_algorithm_parameters(self) -> None:
        for child in self.algorithm_params.winfo_children():
            child.destroy()
        self.algorithm_parameter_widgets.clear()
        row = ttk.Frame(self.algorithm_params)
        row.pack(fill="x")
        method = self.algorithm.get()
        parameter_sets = {
            "ORB": [
                ("Max Features", self.orb_nfeatures, 100, 10000, 100, None),
                ("Scale Factor", self.orb_scale_factor, 1.01, 2.0, 0.05, "%.2f"),
                ("Pyramid Levels", self.orb_nlevels, 1, 16, 1, None),
                ("FAST Threshold", self.orb_fast_threshold, 1, 100, 1, None),
            ],
            "SIFT": [
                ("Max Features", self.sift_nfeatures, 100, 10000, 100, None),
                ("Contrast Threshold", self.sift_contrast_threshold, 0.001, 1.0, 0.01, "%.3f"),
                ("Edge Threshold", self.sift_edge_threshold, 1.0, 50.0, 1.0, "%.1f"),
                ("Sigma", self.sift_sigma, 0.5, 5.0, 0.1, "%.1f"),
            ],
            "AKAZE": [
                ("Detection Threshold", self.akaze_threshold, 0.0001, 0.1, 0.001, "%.4f"),
                ("Octaves", self.akaze_n_octaves, 1, 8, 1, None),
                ("Layers per Octave", self.akaze_n_octave_layers, 1, 8, 1, None),
            ],
        }
        for column, (label, variable, minimum, maximum, step, number_format) in enumerate(parameter_sets[method]):
            ttk.Label(row, text=label).grid(row=0, column=column * 2, sticky="w", padx=(0, 5))
            options = {
                "from_": minimum,
                "to": maximum,
                "increment": step,
                "width": 9,
                "textvariable": variable,
            }
            if number_format:
                options["format"] = number_format
            spinbox = ttk.Spinbox(row, **options)
            spinbox.grid(row=0, column=column * 2 + 1, sticky="w", padx=(0, 16))
            self.algorithm_parameter_widgets.append(spinbox)

        if method == "AKAZE":
            next_col = len(parameter_sets[method]) * 2
            ttk.Label(row, text="Diffusivity").grid(row=0, column=next_col, sticky="w", padx=(0, 5))
            diffusivity_box = ttk.Combobox(
                row,
                textvariable=self.akaze_diffusivity,
                values=("PM_G1", "PM_G2", "WEICKERT", "CHARBONNIER"),
                state="readonly",
                width=14,
            )
            diffusivity_box.grid(row=0, column=next_col + 1, sticky="w")
            self.algorithm_parameter_widgets.append(diffusivity_box)

    def _build_about_tab(self) -> None:
        about = ttk.Frame(self.about_tab, padding=16)
        about.pack(fill="both", expand=True)
        text = tk.Text(about, wrap="word", font=("Segoe UI", 12), padx=12, pady=12, spacing1=3, spacing3=7)
        scrollbar = ttk.Scrollbar(about, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        text.insert(
            "1.0",
            "PROGRAM OVERVIEW\n"
            "This application finds a reference object inside a scene image. It detects local image features, "
            "matches their descriptors, rejects inconsistent matches with RANSAC, and projects the reference "
            "image corners into the scene. Matching runs in a background thread so the interface stays responsive.\n\n"
            "CORE MATCHING PIPELINE\n"
            "1. Convert both images to grayscale (optionally enhance local contrast with CLAHE).\n"
            "2. Detect keypoints and calculate a descriptor for each keypoint using the selected method.\n"
            "3. Find the two nearest scene descriptors for each reference descriptor (KNN), then apply the Lowe ratio test.\n"
            "4. Estimate a perspective homography with RANSAC. Geometrically consistent matches are inliers.\n"
            "5. Draw inlier connections. If there are enough inliers, project the reference image boundary and draw it in yellow.\n\n"
            "FEATURE ALGORITHMS\n"
            "ORB — Fast, free-to-use binary descriptors. A good default for general matching. Uses Hamming distance. "
            "Scale Factor controls the image pyramid step; Pyramid Levels controls how many scales are searched; "
            "FAST Threshold affects keypoint detection sensitivity.\n\n"
            "SIFT — Scale- and rotation-aware floating-point descriptors. Often more robust to size and viewpoint changes, "
            "but usually slower than ORB. Uses L2 distance. Max Features limits the number of strongest keypoints; "
            "Contrast Threshold filters weak points; Edge Threshold filters edge-like responses; Sigma sets the initial blur.\n\n"
            "AKAZE — Nonlinear scale-space detector with compact binary descriptors. A useful alternative when ORB is not robust enough. "
            "Uses Hamming distance. Detection Threshold controls sensitivity; Octaves and Layers per Octave control the scale space; "
            "Diffusivity selects the nonlinear diffusion model.\n\n"
            "SHARED SETTINGS\n"
            "Lowe Match Ratio: lower values are stricter and retain fewer, usually cleaner matches. A value around 0.75 is a practical start.\n"
            "Min. RANSAC Inliers: minimum geometrically consistent matches required before drawing the yellow object outline.\n"
            "CLAHE: optional local contrast enhancement. It can help low-contrast images, but may add noisy keypoints.\n\n"
            "TIPS\n"
            "Use a clear reference crop showing the object, and a scene image where the object is visible at a reasonably similar appearance. "
            "This method works best for mostly planar objects. Repeated patterns, blur, severe occlusion, or large perspective changes "
            "can cause incorrect or missing detections. The match result can still be saved when a reliable outline is not found.\n",
        )
        text.configure(state="disabled")

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        # imdecode + fromfile supports non-ASCII paths on Windows.
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read this image. Please choose a valid image file.")
        return image

    def load_template(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Reference Image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.template = self._read_image(path)
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Image Load Error", str(exc))
            return
        self.template_path = path
        self.result = None
        self._set_busy(False)
        self.template_title.configure(text=f"Reference: {os.path.basename(path)}  ({self.template.shape[1]} × {self.template.shape[0]})")
        self._render("template")
        self._render("result")
        self.status.set("Reference image loaded. Now load a scene image." if self.scene is None else "Images ready. Click Find Match.")

    def load_scene(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Scene Image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.scene = self._read_image(path)
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Image Load Error", str(exc))
            return
        self.scene_path = path
        self.result = None
        self._set_busy(False)
        self.scene_title.configure(text=f"Scene: {os.path.basename(path)}  ({self.scene.shape[1]} × {self.scene.shape[0]})")
        self._render("scene")
        self._render("result")
        self.status.set("Images ready. Click Find Match." if self.template is not None else "Scene image loaded. Now load a reference image.")

    def match_images(self) -> None:
        if self._busy:
            return
        if self.template is None or self.scene is None:
            messagebox.showinfo("Images Required", "Please load both a reference image and a scene image first.")
            return
        try:
            ratio = float(self.ratio.get())
            min_matches = max(4, int(self.min_matches.get()))
            if not 0.1 < ratio < 1.0:
                raise ValueError("Match ratio must be between 0.1 and 1.0.")
            algorithm = self.algorithm.get()
            if algorithm == "ORB":
                algorithm_options = {
                    "nfeatures": max(100, int(self.orb_nfeatures.get())),
                    "scale_factor": float(self.orb_scale_factor.get()),
                    "nlevels": int(self.orb_nlevels.get()),
                    "fast_threshold": int(self.orb_fast_threshold.get()),
                }
                if algorithm_options["scale_factor"] <= 1.0:
                    raise ValueError("ORB Scale Factor must be greater than 1.0.")
            elif algorithm == "SIFT":
                algorithm_options = {
                    "nfeatures": max(100, int(self.sift_nfeatures.get())),
                    "contrast_threshold": float(self.sift_contrast_threshold.get()),
                    "edge_threshold": float(self.sift_edge_threshold.get()),
                    "sigma": float(self.sift_sigma.get()),
                }
                if algorithm_options["contrast_threshold"] <= 0 or algorithm_options["edge_threshold"] <= 0 or algorithm_options["sigma"] <= 0:
                    raise ValueError("SIFT thresholds and Sigma must be greater than zero.")
            else:
                algorithm_options = {
                    "threshold": float(self.akaze_threshold.get()),
                    "n_octaves": int(self.akaze_n_octaves.get()),
                    "n_octave_layers": int(self.akaze_n_octave_layers.get()),
                    "diffusivity": self.akaze_diffusivity.get(),
                }
                if algorithm_options["threshold"] <= 0:
                    raise ValueError("AKAZE Detection Threshold must be greater than zero.")
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            return

        self._set_busy(True)
        self.result = None
        self._render("result")
        use_clahe = self.use_clahe.get()
        self.status.set(f"Running {algorithm} matching in the background... The window will remain responsive.")
        template = self.template.copy()
        scene = self.scene.copy()
        worker = threading.Thread(
            target=self._match_worker,
            args=(template, scene, algorithm_options, ratio, min_matches, algorithm, use_clahe),
            daemon=True,
        )
        worker.start()

    @staticmethod
    def _compute_match(
        template: np.ndarray,
        scene: np.ndarray,
        algorithm_options: dict,
        ratio: float,
        min_matches: int,
        algorithm: str,
        use_clahe: bool,
    ) -> tuple[np.ndarray, int, int, int, bool, str]:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_template = clahe.apply(gray_template)
            gray_scene = clahe.apply(gray_scene)

        if algorithm == "SIFT":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("SIFT is unavailable in this OpenCV installation. Install opencv-contrib-python.")
            detector = cv2.SIFT_create(
                nfeatures=algorithm_options["nfeatures"],
                contrastThreshold=algorithm_options["contrast_threshold"],
                edgeThreshold=algorithm_options["edge_threshold"],
                sigma=algorithm_options["sigma"],
            )
        elif algorithm == "AKAZE":
            create_akaze = getattr(cv2, "AKAZE_create", None)
            if create_akaze is None:
                # Some contrib builds expose the generated class instead of the legacy factory.
                akaze_class = getattr(cv2, "xfeatures2d_AKAZE", None)
                if akaze_class is None:
                    raise RuntimeError("AKAZE is unavailable in this OpenCV installation.")
                create_akaze = akaze_class.create
            diffusivity = getattr(cv2, f"KAZE_DIFF_{algorithm_options['diffusivity']}", 1)
            detector = create_akaze(
                threshold=algorithm_options["threshold"],
                nOctaves=algorithm_options["n_octaves"],
                nOctaveLayers=algorithm_options["n_octave_layers"],
                diffusivity=diffusivity,
            )
        else:
            detector = cv2.ORB_create(
                nfeatures=algorithm_options["nfeatures"],
                scaleFactor=algorithm_options["scale_factor"],
                nlevels=algorithm_options["nlevels"],
                fastThreshold=algorithm_options["fast_threshold"],
            )
        norm_type = detector.defaultNorm()

        kp1, des1 = detector.detectAndCompute(gray_template, None)
        kp2, des2 = detector.detectAndCompute(gray_scene, None)
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            raise ValueError(
                f"Not enough {algorithm} features found. Try clearer images with more visible texture or detail."
            )

        matcher = cv2.BFMatcher(norm_type)
        pairs = matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
        if not good:
            raise ValueError("No reliable feature matches found. Try different images or adjust the Match Ratio.")

        homography = None
        inlier_mask = None
        if len(good) >= 4:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
        shown_matches = (
            [m for m, keep in zip(good, inlier_mask.ravel()) if keep]
            if homography is not None and inlier_mask is not None
            else good
        )
        output = cv2.drawMatches(
            template,
            kp1,
            scene,
            kp2,
            shown_matches,
            None,
            matchColor=(70, 220, 70),
            singlePointColor=(0, 150, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        located = homography is not None and inlier_count >= min_matches
        if located:
            height, width = gray_template.shape
            corners = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, homography)
            projected[:, 0, 0] += template.shape[1]
            cv2.polylines(output, [np.int32(projected)], True, (0, 255, 255), 4, cv2.LINE_AA)
        return output, len(good), inlier_count, min_matches, located, algorithm

    def _match_worker(
        self,
        template: np.ndarray,
        scene: np.ndarray,
        algorithm_options: dict,
        ratio: float,
        min_matches: int,
        algorithm: str,
        use_clahe: bool,
    ) -> None:
        try:
            result = self._compute_match(
                template, scene, algorithm_options, ratio, min_matches, algorithm, use_clahe
            )
            self._worker_queue.put(("success", result))
        except Exception as exc:
            self._worker_queue.put(("error", str(exc)))

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self._worker_queue.get_nowait()
                if kind == "error":
                    self._set_busy(False)
                    self.status.set(str(payload))
                    messagebox.showwarning("Matching Could Not Complete", str(payload))
                    continue
                self.result, good_count, inlier_count, min_matches, located, algorithm = payload
                self._set_busy(False)
                self._render("result")
                if located:
                    self.status.set(
                        f"{algorithm}: object detected. {good_count} good matches; {inlier_count} RANSAC inliers. "
                        "The yellow outline marks the reference image in the scene. You can now save the result."
                    )
                else:
                    self.status.set(
                        f"{algorithm}: matches drawn, but the object could not be located reliably: "
                        f"{good_count} good matches, "
                        f"{inlier_count} inliers (at least {min_matches} required). You can still save the result."
                    )
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.load_template_button.configure(state=state)
        self.load_scene_button.configure(state=state)
        self.algorithm_box.configure(state="disabled" if busy else "readonly")
        self.clahe_check.configure(state=state)
        for widget in self.algorithm_parameter_widgets:
            normal_state = "readonly" if isinstance(widget, ttk.Combobox) else "normal"
            widget.configure(state="disabled" if busy else normal_state)
        self.match_button.configure(state=state)
        if busy:
            self.save_button.configure(state="disabled")
        else:
            self.save_button.configure(state="normal" if self.result is not None else "disabled")

    def _render(self, key: str) -> None:
        canvas = self.canvases.get(key)
        if canvas is None:
            return
        image = {"template": self.template, "scene": self.scene, "result": self.result}.get(key)
        canvas.delete("all")
        if image is None:
            canvas.create_text(
                max(1, canvas.winfo_width() // 2),
                max(1, canvas.winfo_height() // 2),
                text="Load an image to preview" if key != "result" else "Your match result will appear here",
                fill="#b0b0b0",
            )
            return
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        scale = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        header = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255\n".encode("ascii")
        photo = tk.PhotoImage(data=base64.b64encode(header + rgb.tobytes()), format="PPM")
        self._photos[key] = photo
        canvas.create_image(width // 2, height // 2, image=photo, anchor="center")

    def save_result(self) -> None:
        if self.result is None:
            messagebox.showinfo("No Result", "Run a match before saving the result image.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Match Result",
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png"), ("BMP image", "*.bmp")],
        )
        if not path:
            return
        extension = os.path.splitext(path)[1] or ".jpg"
        success, encoded = cv2.imencode(extension, self.result)
        if not success:
            messagebox.showerror("Save Error", "OpenCV could not encode this image format.")
            return
        try:
            encoded.tofile(path)
        except OSError as exc:
            messagebox.showerror("Save Error", str(exc))
            return
        self.status.set(f"Result saved: {path}")


def main() -> None:
    root = tk.Tk()
    ORBMatcherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
