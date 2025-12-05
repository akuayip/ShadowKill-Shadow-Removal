import cv2
import numpy as np

def shadow_removal(image):
    # 1. Simpan resolusi asli & Denoising awal
    denoised = cv2.medianBlur(image, 3)
    
    # 2. Proses di ruang warna HSV untuk mempertahankan warna
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 3. Estimasi Background Dinamis
    h_img, w_img = v.shape
    kernel_size = int(min(h_img, w_img) * 0.03) 
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(15, kernel_size)    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Morphological Close untuk mendapatkan estimasi background
    background = cv2.morphologyEx(v, cv2.MORPH_CLOSE, kernel)
    
    # Safety measure: hindari pembagian dengan nol
    background = np.where(background == 0, 1, background).astype(np.float32)
    v_float = v.astype(np.float32)
    
    # 4. Illumination Correction (Division)
    corrected_v = (v_float / background) * 255
    corrected_v = np.clip(corrected_v, 0, 255).astype(np.uint8)
    
    # 5. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_v = clahe.apply(corrected_v)
    
    # 6. Gabungkan kembali ke citra berwarna
    merged_hsv = cv2.merge([h, s, enhanced_v])
    result_color = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
    
    corrected_gray = enhanced_v
    
    # 7. Binarization 
    binary_result = cv2.adaptiveThreshold(
        enhanced_v,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, # Block size
        10  # C
    )

    return image, corrected_gray, result_color