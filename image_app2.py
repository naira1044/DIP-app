import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Image Processing - Comprehensive Tool")
        self.root.geometry("600x300")
        
        self.image_path = None
        self.original_img = None
        
        # --- Task Dictionary Hierarchy ---
        self.tasks = {
            "1. Point Operations": ["Addition", "Subtraction", "Division", "Complement"],
            "2. Color Image Operations": ["Change Red Lighting", "Swap R to G", "Eliminate Red"],
            "3. Image Histogram": ["Histogram Stretching (Gray)", "Histogram Equalization (Gray)"],
            "4. Neighborhood Processing": ["Linear: Average Filter", "Linear: Laplacian Filter", 
                                           "Non-linear: Maximum", "Non-linear: Minimum", 
                                           "Non-linear: Median", "Non-linear: Mode (Most Frequent)"],
            "5. Image Restoration": ["Salt & Pepper: Average", "Salt & Pepper: Median", "Salt & Pepper: Outlier Method",
                                     "Gaussian: Image Averaging", "Gaussian: Average Filter"],
            "6. Image Segmentation": ["Basic Global Thresholding", "Automatic Thresholding (Otsu)", "Adaptive Thresholding"],
            "7. Edge Detection": ["Sobel Detector"],
            "8. Mathematical Morphology": ["Image Dilation", "Image Erosion", "Image Opening", 
                                           "Boundary: Internal", "Boundary: External", "Boundary: Morphological Gradient"]
        }
        
        self.setup_gui()

    def setup_gui(self):
        # Load Image Button
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="1. Load Image", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", 
                  command=self.load_image, width=20).pack()
        
        self.lbl_img_path = tk.Label(self.root, text="No image loaded", fg="gray")
        self.lbl_img_path.pack()

        # Category Dropdown
        tk.Label(self.root, text="Select Category:").pack(pady=(10,0))
        self.combo_category = ttk.Combobox(self.root, values=list(self.tasks.keys()), width=40, state="readonly")
        self.combo_category.pack()
        self.combo_category.bind("<<ComboboxSelected>>", self.update_tasks)
        
        # Task Dropdown
        tk.Label(self.root, text="Select Task:").pack(pady=(10,0))
        self.combo_task = ttk.Combobox(self.root, width=40, state="readonly")
        self.combo_task.pack()

        # Apply Button
        tk.Button(self.root, text="2. Apply & Show Result", font=("Arial", 12, "bold"), bg="#2196F3", fg="white", 
                  command=self.apply_operation, width=20).pack(pady=20)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="Select an Image", 
                                                     filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if self.image_path:
            self.original_img = cv2.imread(self.image_path)
            self.lbl_img_path.config(text=self.image_path.split("/")[-1])

    def update_tasks(self, event):
        category = self.combo_category.get()
        self.combo_task.config(values=self.tasks[category])
        if self.tasks[category]:
            self.combo_task.current(0)

    def apply_operation(self):
        if self.original_img is None:
            messagebox.showerror("Error", "Please load an image first!")
            return
        
        task = self.combo_task.get()
        if not task:
            messagebox.showerror("Error", "Please select a task!")
            return

        img = self.original_img.copy()
        result = None
        input_display = img.copy()
        
        # ---------------------------------------------------------
        # 1. Point Operations
        # ---------------------------------------------------------
        if task == "Addition":
            val = np.array([50.0, 50.0, 50.0]) # Add constant 50
            result = cv2.add(img, val, dtype=cv2.CV_8U)
        elif task == "Subtraction":
            val = np.array([50.0, 50.0, 50.0])
            result = cv2.subtract(img, val, dtype=cv2.CV_8U)
        elif task == "Division":
            result = cv2.divide(img, 2.0)
        elif task == "Complement":
            result = cv2.bitwise_not(img)
            
        # ---------------------------------------------------------
        # 2. Color Image Operations
        # ---------------------------------------------------------
        elif task == "Change Red Lighting":
            result = img.copy()
            # In OpenCV, Red is channel 2
            result[:, :, 2] = cv2.add(result[:, :, 2], 50)
        elif task == "Swap R to G":
            result = img.copy()
            result[:, :, 1] = result[:, :, 2] # Put Red into Green
        elif task == "Eliminate Red":
            result = img.copy()
            result[:, :, 2] = 0
            
        # ---------------------------------------------------------
        # 3. Image Histogram
        # ---------------------------------------------------------
        elif task == "Histogram Stretching (Gray)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input_display = gray.copy()
            result = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        elif task == "Histogram Equalization (Gray)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input_display = gray.copy()
            result = cv2.equalizeHist(gray)
            
        # ---------------------------------------------------------
        # 4. Neighborhood Processing
        # ---------------------------------------------------------
        elif task == "Linear: Average Filter":
            result = cv2.blur(img, (7, 7))
        elif task == "Linear: Laplacian Filter":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input_display = gray.copy()
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            result = cv2.convertScaleAbs(laplacian)
        elif task == "Non-linear: Maximum":
            kernel = np.ones((5,5), np.uint8)
            result = cv2.dilate(img, kernel, iterations=1)
        elif task == "Non-linear: Minimum":
            kernel = np.ones((5,5), np.uint8)
            result = cv2.erode(img, kernel, iterations=1)
        elif task == "Non-linear: Median":
            result = cv2.medianBlur(img, 7)
        elif task == "Non-linear: Mode (Most Frequent)":
            # Approximating Mode filter using a combination or using scipy if available
            # Since strict OpenCV is preferred, median is often used as a substitute for smoothing, 
            # but to explicitly show mode, we use scipy.ndimage or simulate it.
            # Here we apply a quantize & mode approach. (Warning: can be slow on large images)
            messagebox.showinfo("Note", "Mode filter might take a few seconds to compute.")
            from scipy.ndimage import generic_filter
            from scipy import stats
            def mode_func(x):
                return stats.mode(x, keepdims=False)[0]
            # Convert to gray for speed
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input_display = gray.copy()
            # Downscale slightly for speed if image is too large
            small = cv2.resize(gray, (200, 200))
            res_small = generic_filter(small, mode_func, size=3)
            result = cv2.resize(res_small, (gray.shape[1], gray.shape[0]))
            
        # ---------------------------------------------------------
        # 5. Image Restoration
        # ---------------------------------------------------------
        elif task.startswith("Salt & Pepper"):
            # Add artificial S&P noise first to demonstrate
            noisy = img.copy()
            prob = 0.05
            thres = 1 - prob
            rand_matrix = np.random.rand(img.shape[0], img.shape[1])
            noisy[rand_matrix < prob] = 0
            noisy[rand_matrix > thres] = 255
            input_display = noisy.copy()
            
            if "Average" in task:
                result = cv2.blur(noisy, (5, 5))
            elif "Median" in task:
                result = cv2.medianBlur(noisy, 5)
            elif "Outlier" in task:
                # Custom Outlier Method: Compare pixel to local median
                median = cv2.medianBlur(noisy, 5)
                diff = cv2.absdiff(noisy, median)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                result = np.where(mask_bgr == 255, median, noisy)
                
        elif task.startswith("Gaussian"):
            if "Averaging" in task:
                # Generate 10 noisy images and average them
                noisy_images = []
                for _ in range(10):
                    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
                    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    noisy_images.append(noisy_img)
                input_display = noisy_images[0] # Show one noisy sample
                result = np.mean(noisy_images, axis=0).astype(np.uint8)
            else:
                # Add single Gaussian noise
                noise = np.random.normal(0, 25, img.shape).astype(np.int16)
                noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                input_display = noisy.copy()
                result = cv2.blur(noisy, (5, 5))
                
        # ---------------------------------------------------------
        # 6. Image Segmentation
        # ---------------------------------------------------------
        elif task == "Basic Global Thresholding":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input_display = gray.copy()
            _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif task == "Automatic Thresholding (Otsu)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input_display = gray.copy()
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif task == "Adaptive Thresholding":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input_display = gray.copy()
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            
        # ---------------------------------------------------------
        # 7. Edge Detection
        # ---------------------------------------------------------
        elif task == "Sobel Detector":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input_display = gray.copy()
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            result = cv2.convertScaleAbs(magnitude)
            
        # ---------------------------------------------------------
        # 8. Mathematical Morphology
        # ---------------------------------------------------------
        elif "Boundary" in task or "Image Dilation" in task or "Image Erosion" in task or "Image Opening" in task:
            # Morphology works best on binary shapes
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            input_display = binary.copy()
            kernel = np.ones((5,5), np.uint8)
            
            if task == "Image Dilation":
                result = cv2.dilate(binary, kernel, iterations=1)
            elif task == "Image Erosion":
                result = cv2.erode(binary, kernel, iterations=1)
            elif task == "Image Opening":
                result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            elif task == "Boundary: Internal":
                erosion = cv2.erode(binary, kernel, iterations=1)
                result = cv2.subtract(binary, erosion)
            elif task == "Boundary: External":
                dilation = cv2.dilate(binary, kernel, iterations=1)
                result = cv2.subtract(dilation, binary)
            elif task == "Boundary: Morphological Gradient":
                result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

        # Plotting the results
        self.show_subplot(task, input_display, result)

    def show_subplot(self, title, img1, img2):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.canvas.manager.set_window_title(title)
        
        # Convert colors for Matplotlib if the images have 3 channels
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            axes[0].imshow(img1)
        else:
            axes[0].imshow(img1, cmap='gray')
            
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            axes[1].imshow(img2)
        else:
            axes[1].imshow(img2, cmap='gray')

        axes[0].set_title("Input Image")
        axes[0].axis("off")
        axes[1].set_title("Result Image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()