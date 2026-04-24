import streamlit as st
import cv2
import numpy as np

# إعدادات صفحة الويب
st.set_page_config(page_title="Digital Image Processing Tool", layout="wide")
st.title("Digital Image Processing - Web App")
st.markdown("### Comprehensive Blueprint Tool using Streamlit & OpenCV")

# --- Task Dictionary Hierarchy ---
tasks = {
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

# --- Sidebar Controls (الشريط الجانبي للتحكم) ---
st.sidebar.header("Control Panel")

# 1. Load Image
uploaded_file = st.sidebar.file_uploader("1. Upload an Image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # تحويل الصورة من Byte Stream لـ OpenCV Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1) # 1 means load as color (BGR)
    
    # 2. Select Category & Task
    category = st.sidebar.selectbox("2. Select Category:", list(tasks.keys()))
    task = st.sidebar.selectbox("3. Select Task:", tasks[category])
    
    # 3. Apply Button
    if st.sidebar.button("Apply & Show Result", type="primary"):
        
        # رسالة تحميل تفاعلية
        with st.spinner(f'Applying {task}...'):
            img = original_img.copy()
            result = None
            input_display = img.copy()
            
            # =========================================================
            # CORE LOGIC (نفس العمليات بالظبط بدون أي تغيير في الخوارزميات)
            # =========================================================
            
            if task == "Addition":
                val = np.array([50.0, 50.0, 50.0]) 
                result = cv2.add(img, val, dtype=cv2.CV_8U)
            elif task == "Subtraction":
                val = np.array([50.0, 50.0, 50.0])
                result = cv2.subtract(img, val, dtype=cv2.CV_8U)
            elif task == "Division":
                result = cv2.divide(img, 2.0)
            elif task == "Complement":
                result = cv2.bitwise_not(img)
                
            elif task == "Change Red Lighting":
                result = img.copy()
                result[:, :, 2] = cv2.add(result[:, :, 2], 50)
            elif task == "Swap R to G":
                result = img.copy()
                result[:, :, 1] = result[:, :, 2] 
            elif task == "Eliminate Red":
                result = img.copy()
                result[:, :, 2] = 0
                
            elif task == "Histogram Stretching (Gray)":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                input_display = gray.copy()
                result = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            elif task == "Histogram Equalization (Gray)":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                input_display = gray.copy()
                result = cv2.equalizeHist(gray)
                
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
                st.info("Mode filter is mathematically intensive and might take a few seconds...")
                from scipy.ndimage import generic_filter
                from scipy import stats
                def mode_func(x):
                    return stats.mode(x, keepdims=False)[0]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                input_display = gray.copy()
                small = cv2.resize(gray, (200, 200))
                res_small = generic_filter(small, mode_func, size=3)
                result = cv2.resize(res_small, (gray.shape[1], gray.shape[0]))
                
            elif task.startswith("Salt & Pepper"):
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
                    median = cv2.medianBlur(noisy, 5)
                    diff = cv2.absdiff(noisy, median)
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    result = np.where(mask_bgr == 255, median, noisy)
                    
            elif task.startswith("Gaussian"):
                if "Averaging" in task:
                    noisy_images = []
                    for _ in range(10):
                        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
                        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        noisy_images.append(noisy_img)
                    input_display = noisy_images[0] 
                    result = np.mean(noisy_images, axis=0).astype(np.uint8)
                else:
                    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
                    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    input_display = noisy.copy()
                    result = cv2.blur(noisy, (5, 5))
                    
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
                
            elif task == "Sobel Detector":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                input_display = gray.copy()
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = cv2.magnitude(sobelx, sobely)
                result = cv2.convertScaleAbs(magnitude)
                
            elif "Boundary" in task or "Image Dilation" in task or "Image Erosion" in task or "Image Opening" in task:
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

            # =========================================================
            # DISPLAY RESULTS (عرض النتائج باستخدام Streamlit Columns)
            # =========================================================
            st.markdown(f"### Results for: `{task}`")
            col1, col2 = st.columns(2)
            
            # Helper function لضبط الألوان قبل العرض لأن Streamlit بيقرأ RGB
            def prep_for_display(image):
                if len(image.shape) == 3: # Color image
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image # Grayscale image
                
            with col1:
                st.image(prep_for_display(input_display), caption="Input Image", use_container_width=True)
                
            with col2:
                st.image(prep_for_display(result), caption="Result Image", use_container_width=True)

else:
    st.info("👈 Please upload an image from the sidebar to get started.")