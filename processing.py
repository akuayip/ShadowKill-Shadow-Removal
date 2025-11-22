import cv2
import numpy as np

def shadow_removal(image):
    # 1. Resize (Tetap sama)
    image = cv2.resize(image, (600, 800))
    
    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Estimasi Background (Ditingkatkan)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51)) 
    
    # Gunakan MORPH_CLOSE (Dilation -> Erosion) untuk menutup tulisan hitam
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Hindari pembagian dengan nol (safety measure)
    background = np.where(background == 0, 1, background)
    
    # 4. Illumination Correction
    corrected = cv2.divide(gray, background, scale=255)
        
    # 5. Post-Processing
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    corrected = cv2.filter2D(corrected, -1, sharpen_kernel)

    # 6. Binarization 
    binary_result = cv2.adaptiveThreshold(
        corrected,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 
        15  
    )

    return image, corrected, binary_result