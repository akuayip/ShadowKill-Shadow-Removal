import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt


from bedsrmodel import BEDSRNet

# ==========================================
# Helper: numpy RGB -> QPixmap
# ==========================================
def np_to_pixmap(img_np):
    h, w, c = img_np.shape
    qimg = QImage(img_np.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ==========================================
# Inference Engine (pakai 512x512)
# ==========================================
class ShadowRemovalEngine:
    def __init__(self, checkpoint_path, img_size=1024, device="cpu"):
        self.img_size = img_size          # HARUS sama dengan training: 512
        self.device = torch.device(device)

        self.model = BEDSRNet().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def run(self, img_input):
        # 1. Load gambar asli
        if isinstance(img_input, str):
            img_bgr = cv2.imread(img_input, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {img_input}")
        elif isinstance(img_input, np.ndarray):
            img_bgr = img_input
        else:
            raise ValueError("Input must be a file path (str) or numpy array")

        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize ke 512x512 (MATCH training)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_AREA)

        inp = img_resized.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 3. Forward ke model
        pred, bg, att = self.model(inp)

        # 4. Balik ke numpy RGB 0–255
        out = pred[0].permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).clip(0, 255).astype(np.uint8)

        # 5. Resize output balik ke resolusi asli
        out_full = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # Return RGB (karena main.py expect RGB/BGR handling, tapi processing.py return BGR/HSV converted)
        # processing.py return: image, corrected_gray, result_color (BGR)
        # Kita return BGR agar konsisten
        out_full_bgr = cv2.cvtColor(out_full, cv2.COLOR_RGB2BGR)

        return out_full_bgr


# ==========================================
# PyQt6 GUI
# ==========================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BEDSRNet – Document Shadow Removal (512)")
        self.setGeometry(200, 200, 800, 700)

        # Load model (cek path checkpoint-nya!)
        self.engine = ShadowRemovalEngine(
            checkpoint_path="./best_model/bedsrnet_jung_best.pth",
            img_size=2048,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Widgets
        self.input_label = QLabel("Input Image")
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_label.setStyleSheet("border: 1px solid gray; padding: 6px;")

        self.output_label = QLabel("Output Image")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setStyleSheet("border: 1px solid gray; padding: 6px;")

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)

        self.btn_process = QPushButton("Process Shadow Removal")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)

        self.btn_save = QPushButton("Save Output")
        self.btn_save.clicked.connect(self.save_output)
        self.btn_save.setEnabled(False)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.output_label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_process)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)

        # State
        self.img_path = None
        self.output_img = None

    # -----------------------------
    # Load image dari file
    # -----------------------------
    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.jpeg *.png)"
        )
        if file:
            self.img_path = file
            pix = QPixmap(file).scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio)
            self.input_label.setPixmap(pix)
            self.btn_process.setEnabled(True)

    # -----------------------------
    # Proses shadow removal
    # -----------------------------
    def process_image(self):
        if not self.img_path:
            return

        out = self.engine.run(self.img_path)
        self.output_img = out

        pix = np_to_pixmap(out).scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio)
        self.output_label.setPixmap(pix)
        self.btn_save.setEnabled(True)

    # -----------------------------
    # Save output ke file
    # -----------------------------
    def save_output(self):
        if self.output_img is None:
            return

        file, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "result.png", "Images (*.png *.jpg)"
        )

        if file:
            out_bgr = cv2.cvtColor(self.output_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file, out_bgr)
            QMessageBox.information(self, "Saved", "Output image saved!")


# ==========================================
# Run app
# ==========================================
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
