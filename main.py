import sys
import cv2
import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from processing import shadow_removal
from evaluation import evaluate_quality
from utils import cv_to_pixmap, save_image, ensure_result_folder


class ShadowKillApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ShadowKill – Shadow Removal (Conventional)")
        self.setGeometry(200, 100, 1400, 850)

        ensure_result_folder()

        self.original = None
        self.corrected = None
        self.result = None
        self.ground_truth = None

        self.build_ui()

    # =================================================================
    # BUILD UI
    # =================================================================
    def build_ui(self):
        main_layout = QHBoxLayout(self)

        # ======================================================
        # SIDEBAR
        # ======================================================
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #1f2937;
                border-right: 2px solid #111827;
            }
            QPushButton {
                background-color: #374151;
                color: white;
                border: none;
                padding: 14px;
                border-radius: 8px;
                margin-top: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # TITLE
        title = QLabel("ShadowKill")
        title.setStyleSheet("color: white; font-size: 22px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sidebar_layout.addWidget(title)
        sidebar_layout.addSpacing(20)

        # Buttons
        btn_load = QPushButton("Load Shadow Image")
        btn_load.clicked.connect(self.load_shadow)

        btn_gt = QPushButton("Load Ground Truth")
        btn_gt.clicked.connect(self.load_ground_truth)

        btn_process = QPushButton("Run Shadow Removal")
        btn_process.clicked.connect(self.process_shadow)

        btn_eval = QPushButton("Evaluate PSNR & SSIM")
        btn_eval.clicked.connect(self.evaluate_psnr_ssim)

        btn_save = QPushButton("Save Results")
        btn_save.clicked.connect(self.save_outputs)

        for btn in [btn_load, btn_gt, btn_process, btn_eval, btn_save]:
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # ======================================================
        # PREVIEW AREA (Cards)
        # ======================================================
        preview_area = QFrame()
        preview_area.setStyleSheet("""
            QFrame { background-color: #f3f4f6; }
        """)
        preview_layout = QVBoxLayout(preview_area)

        preview_title = QLabel("Image Preview")
        preview_title.setFont(QFont("Arial", 18))
        preview_title.setStyleSheet("color: #1f2937; font-weight: bold;")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        preview_layout.addWidget(preview_title)
        preview_layout.addSpacing(10)

        # GRID
        grid = QHBoxLayout()

        column1 = QVBoxLayout()
        column2 = QVBoxLayout()

        # Cards
        self.card_original, self.label_original = self.make_preview_card("Original")
        self.card_corrected, self.label_corrected = self.make_preview_card("Illumination Corrected")
        self.card_result, self.label_result = self.make_preview_card("Shadow Removed")
        self.card_gt, self.label_gt = self.make_preview_card("Ground Truth")

        column1.addWidget(self.card_original)
        column1.addWidget(self.card_result)

        column2.addWidget(self.card_corrected)
        column2.addWidget(self.card_gt)

        grid.addLayout(column1)
        grid.addLayout(column2)

        preview_layout.addLayout(grid)

        # STATUS
        self.status = QLabel("Ready.")
        self.status.setStyleSheet("""
            background-color: #e5e7eb;
            padding: 10px;
            font-size: 14px;
            border-radius: 6px;
            color: #111827;
        """)
        preview_layout.addSpacing(15)
        preview_layout.addWidget(self.status)

        # ADD TO MAIN LAYOUT
        main_layout.addWidget(sidebar)
        main_layout.addWidget(preview_area)

    # =================================================================
    # CREATE PREVIEW CARD
    # =================================================================
    def make_preview_card(self, title):
        frame = QFrame()
        frame.setFixedSize(500, 350)
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 2px solid #d1d5db;
            }
        """)

        layout = QVBoxLayout(frame)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 16px; color: #374151; margin-bottom: 4px;")

        img_label = QLabel()
        img_label.setFixedSize(450, 280)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setStyleSheet("""
            QLabel {
                background-color: #f9fafb;
                border: 1px dashed #d1d5db;
                border-radius: 8px;
            }
        """)

        layout.addWidget(lbl_title)
        layout.addWidget(img_label)

        return frame, img_label

    # =================================================================
    # LOAD SHADOW IMAGE
    # =================================================================
    def load_shadow(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg)")
        if path:
            self.original = cv2.imread(path)
            self.label_original.setPixmap(cv_to_pixmap(self.original))
            self.status.setText("Shadow image loaded.")

    # LOAD GROUND TRUTH
    def load_ground_truth(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth", "", "Images (*.png *.jpg)")
        if path:
            self.ground_truth = cv2.imread(path)
            self.label_gt.setPixmap(cv_to_pixmap(self.ground_truth))
            self.status.setText("Ground truth loaded.")

    # =================================================================
    # RUN SHADOW REMOVAL
    # =================================================================
    def process_shadow(self):
        if self.original is None:
            self.status.setText("Error: Load shadow image first.")
            return

        orig, corrected, result = shadow_removal(self.original)

        self.corrected = corrected
        self.result = result

        self.label_corrected.setPixmap(cv_to_pixmap(cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)))
        self.label_result.setPixmap(cv_to_pixmap(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)))

        self.status.setText("Shadow removal completed.")

    # =================================================================
    # EVALUATE PSNR + SSIM
    # =================================================================
    def evaluate_psnr_ssim(self):
        if self.ground_truth is None or self.result is None:
            self.status.setText("Error: Load ground truth & run processing first.")
            return

        psnr_val, ssim_val = evaluate_quality(self.ground_truth, self.result)

        QMessageBox.information(self, "Evaluation Result",
                                f"PSNR: {psnr_val:.4f} dB\nSSIM: {ssim_val:.4f}")

        self.status.setText("Evaluation completed.")

    # =================================================================
    # SAVE OUTPUTS
    # =================================================================
    def save_outputs(self):
        if self.corrected is None or self.result is None:
            self.status.setText("Error: Run shadow removal first.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        save_image(self.corrected, f"corrected_{timestamp}.png")
        save_image(self.result, f"shadow_removed_{timestamp}.png")

        QMessageBox.information(self, "Saved", "Images saved inside result/ folder.")
        self.status.setText("Output saved.")


# =================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShadowKillApp()
    window.show()
    sys.exit(app.exec())
