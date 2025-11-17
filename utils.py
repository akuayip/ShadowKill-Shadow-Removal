import os
import cv2
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

def ensure_result_folder():
    if not os.path.exists("result"):
        os.makedirs("result")

def cv_to_pixmap(img, max_width=450, max_height=280):
    """
    Convert OpenCV image (BGR/GRAY) to QPixmap dengan scaling untuk preview.
    
    Args:
        img: Gambar OpenCV (numpy array)
        max_width: Lebar maksimal preview (default: 450)
        max_height: Tinggi maksimal preview (default: 280)
    
    Returns:
        QPixmap yang sudah di-scale untuk preview (gambar asli tidak berubah)
    """
    if len(img.shape) == 2:  # grayscale
        h, w = img.shape
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
    else:  # BGR
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)
    
    # Scale hanya untuk preview, menjaga aspect ratio agar tidak terzoom
    # Gambar asli (img) tetap tidak berubah
    scaled_pixmap = pixmap.scaled(
        max_width, 
        max_height, 
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    
    return scaled_pixmap

def save_image(img, name):
    ensure_result_folder()
    
    path = os.path.join("result", name)
    cv2.imwrite(path, img)

    return path
