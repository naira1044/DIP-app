import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy import ndimage
from scipy import stats

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Understanding - Digital Image Processing")
        self.root.geometry("1000x700")
        
        self.original_image = None
        self.image_loaded = False

        # --- GUI Layout ---
        control_frame = tk.Frame(self.root, width=300, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.canvas_frame = tk.Frame(self.root, bg="white")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Load Button
        tk.Button(control_frame, text="1. Load Image", command=self.load_image, width=25, font=("Arial", 10, "bold"), bg="#4CAF50", fg="white").pack(pady=10)

        # Task Categories & Combobox
        tk.Label(control_frame, text="2. Select Task Category:", bg="#f0f0f0").pack(pady=5)
        
        self.tasks = {
            "1. Point Operation: Addition": self.point_add,
            "1. Point Operation: Subtraction": self.point_sub,
            "1. Point Operation: Division": self.point_div,
            "1. Point Operation: Complement": self.point_complement,
            "2. Color: Change Red Lighting": self.color_change_red,
            "2. Color: Swap R to G": self.color_swap_rg,
            "2. Color: Eliminate Red": self.color_eliminate_red,
            "3. Histogram: Stretching (Gray)": self.hist_stretching,
            "3. Histogram: Equalization (Gray)": self.hist_equalization,
            "4. Linear: Average Filter": self.filter_average,
            "4. Linear: Laplacian Filter": self.filter_laplacian,
            "4. Non-linear: Maximum Filter": self.filter_max,
            "4. Non-linear: Minimum Filter": self.filter_min,
            "4. Non-linear: Median Filter": self.filter_median,
            "4. Non-linear: Mode Filter": self.filter_mode,
            "5. Restoration: Salt & Pepper (Average)": self.rest_sp_avg,
            "5. Restoration: Salt & Pepper (Median)": self.rest_sp_median,
            "5. Restoration: Gaussian (Average)": self.rest_gaussian_avg,
            "6. Segmentation: Global Threshold": self.seg_global,
            "6. Segmentation: Otsu (Auto) Threshold": self.seg_otsu,
            "6. Segmentation: Adaptive Threshold": self.seg_adaptive,
            "7. Edge Detection: Sobel": self.edge_sobel,
            "8. Morphology: Dilation": self.morph_dilation,
            "8. Morphology: Erosion": self.morph_erosion,
            "8. Morphology: Opening": self.morph_opening,
            "8. Boundary: Internal": self.bound_internal,
            "8. Boundary: External": self.bound_external,
            "8. Boundary: Morphological Gradient": self.bound_gradient
        }
        
        self.task_var = tk.StringVar()
        self.task_combo = ttk.Combobox(control_frame, textvariable=self.task_var, values=list(self.tasks.keys()), width=35, state="readonly")
        self.task_combo.pack(pady=5)
        self.task_combo.current(0)

        # Apply Button
        tk.Button(control_frame, text="3. Apply & Show", command=self.apply_task, width=25, font=("Arial", 10, "bold"), bg="#2196F3", fg="white").pack(pady=20)

        self.figure, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            # OpenCV reads in BGR format
            self.original_image = cv2.imread(file_path)
            self.image_loaded = True
            messagebox.showinfo("Success", "Image loaded successfully!")
            self.show_images(self.original_image, self.original_image, "Loaded Image")

    def apply_task(self):
        if not self.image_loaded:
            messagebox.showerror("Error", "Please load an image first!")
            return
        
        task_name = self.task_var.get()
        func = self.tasks.get(task_name)
        if func:
            result_img = func()
            self.show_images(self.original_image, result_img, task_name)

    # --- Helper to plot in GUI ---
    def show_images(self, img1, img2, title):
        self.ax[0].clear()
        self.ax[1].clear()

        # Convert BGR to RGB for matplotlib display
        if len(img1.shape) == 3:
            img1_disp = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        else:
            img1_disp = img1
            
        if len(img2.shape) == 3:
            img2_disp = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        else:
            img2_disp = img2

        cmap1 = 'gray' if len(img1.shape) == 2 else None
        cmap2 = 'gray' if len(img2.shape) == 2 else None

        self.ax[0].imshow(img1_disp, cmap=cmap1)
        self.ax[0].set_title("Original Image")
        self.ax[0].axis('off')

        self.ax[1].imshow(img2_disp, cmap=cmap2)
        self.ax[1].set_title(title)
        self.ax[1].axis('off')

        self.figure.tight_layout()
        self.canvas.draw()

    # ================= TASK FUNCTIONS =================

    # 1. Point Operations
    def point_add(self):
        return cv2.add(self.original_image, np.array([50.0])) # Adding constant brightness
    def point_sub(self):
        return cv2.subtract(self.original_image, np.array([50.0]))
    def point_div(self):
        return (self.original_image / 2).astype(np.uint8)
    def point_complement(self):
        return 255 - self.original_image

    # 2. Color Image Operations (Remember OpenCV is BGR, Red is index 2)
    def color_change_red(self):
        img = self.original_image.copy()
        img[:,:,2] = cv2.add(img[:,:,2], 50)
        return img
    def color_swap_rg(self):
        img = self.original_image.copy()
        img[:,:,1], img[:,:,2] = img[:,:,2], img[:,:,1].copy()
        return img
    def color_eliminate_red(self):
        img = self.original_image.copy()
        img[:,:,2] = 0
        return img

    # 3. Image Histogram
    def hist_stretching(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # Using normalize to apply stretching formula: 255 * (x - min) / (max - min)
        stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return stretched
    def hist_equalization(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    # 4. Neighborhood processing
    def filter_average(self):
        return cv2.blur(self.original_image, (5, 5))
    def filter_laplacian(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)
    def filter_max(self):
        # Dilation acts as a maximum filter
        kernel = np.ones((5,5), np.uint8)
        return cv2.dilate(self.original_image, kernel)
    def filter_min(self):
        # Erosion acts as a minimum filter
        kernel = np.ones((5,5), np.uint8)
        return cv2.erode(self.original_image, kernel)
    def filter_median(self):
        return cv2.medianBlur(self.original_image, 5)
    def filter_mode(self):
        # Using SciPy for mode filter (computationally heavy, done on grayscale)
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        mode_filtered = ndimage.generic_filter(gray, lambda x: stats.mode(x, keepdims=True).mode[0], size=3)
        return mode_filtered

    # 5. Image Restoration
    def _add_salt_and_pepper(self, img):
        noisy = np.copy(img)
        salt_mask = np.random.rand(*img.shape[:2]) > 0.95
        pepper_mask = np.random.rand(*img.shape[:2]) < 0.05
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
        return noisy
    def rest_sp_avg(self):
        noisy = self._add_salt_and_pepper(self.original_image)
        return cv2.blur(noisy, (5,5))
    def rest_sp_median(self):
        noisy = self._add_salt_and_pepper(self.original_image)
        return cv2.medianBlur(noisy, 5)
    def rest_gaussian_avg(self):
        # Add gaussian noise then smooth
        gauss = np.random.normal(0, 25, self.original_image.shape).astype('uint8')
        noisy = cv2.add(self.original_image, gauss)
        return cv2.blur(noisy, (5,5))

    # 6. Image Segmentation
    def seg_global(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    def seg_otsu(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    def seg_adaptive(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 7. Edge Detection
    def edge_sobel(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        return cv2.convertScaleAbs(sobel_combined)

    # 8. Mathematical Morphology
    def morph_dilation(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        return cv2.dilate(thresh, kernel, iterations=1)
    def morph_erosion(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        return cv2.erode(thresh, kernel, iterations=1)
    def morph_opening(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    def bound_internal(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        return cv2.subtract(thresh, erosion)
    def bound_external(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        return cv2.subtract(dilation, thresh)
    def bound_gradient(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop() # Execute Tkinter [cite: 289]