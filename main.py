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
        self.setWindowTitle("ShadowKill – Ultimate Shadow Removal")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set Application Font
        self.setFont(QFont("Segoe UI", 10))

        ensure_result_folder()

        self.original = None
        self.corrected = None
        self.result = None
        self.ground_truth = None

        self.apply_styles()
        self.build_ui()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QFrame#Sidebar {
                background-color: #181825;
                border-right: 1px solid #313244;
            }
            QFrame#Card {
                background-color: #313244;
                border-radius: 15px;
                border: 1px solid #45475a;
            }
            QLabel#CardTitle {
                color: #bac2de;
                font-weight: bold;
                font-size: 14px;
                background-color: transparent;
            }
            QLabel#ImagePlaceholder {
                background-color: #1e1e2e;
                border: 2px dashed #45475a;
                border-radius: 10px;
                color: #585b70;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #585b70;
                color: #ffffff;
            }
            QPushButton#PrimaryBtn {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton#PrimaryBtn:hover {
                background-color: #b4befe;
            }
            QPushButton#SuccessBtn {
                background-color: #a6e3a1;
                color: #1e1e2e;
            }
            QPushButton#SuccessBtn:hover {
                background-color: #94e2d5;
            }
            QLabel#Title {
                font-size: 24px;
                font-weight: bold;
                color: #cba6f7;
                padding: 10px;
            }
            QLabel#Status {
                background-color: #313244;
                color: #a6adc8;
                padding: 8px 15px;
                border-radius: 6px;
                font-style: italic;
            }
        """)

    def build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ======================================================
        # SIDEBAR
        # ======================================================
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(15)

        # TITLE
        title = QLabel("ShadowKill 🚀")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(title)
        sidebar_layout.addSpacing(20)

        # BUTTONS
        self.btn_load = self.create_button("📂  Load Shadow Image", self.load_shadow)
        self.btn_gt = self.create_button("🎯  Load Ground Truth", self.load_ground_truth)
        
        self.btn_process = self.create_button("✨  Run Shadow Removal", self.process_shadow, is_primary=True)
        
        self.btn_eval = self.create_button("📊  Evaluate Quality", self.evaluate_psnr_ssim)
        self.btn_save = self.create_button("Cc  Save Results", self.save_outputs, is_success=True)

        sidebar_layout.addWidget(self.btn_load)
        sidebar_layout.addWidget(self.btn_gt)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.btn_process)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.btn_eval)
        sidebar_layout.addWidget(self.btn_save)
        
        sidebar_layout.addStretch()
        
        # STATUS BAR
        self.status = QLabel("Ready to process.")
        self.status.setObjectName("Status")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(self.status)

        # ======================================================
        # MAIN AREA
        # ======================================================
        main_area = QWidget()
        main_layout_area = QVBoxLayout(main_area)
        main_layout_area.setContentsMargins(30, 30, 30, 30)
        main_layout_area.setSpacing(20)

        # Header
        header_lbl = QLabel("Workspace Preview")
        header_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #a6adc8;")
        main_layout_area.addWidget(header_lbl)

        # GRID PREVIEW
        grid = QHBoxLayout()
        grid.setSpacing(20)

        col1 = QVBoxLayout()
        col1.setSpacing(20)
        col2 = QVBoxLayout()
        col2.setSpacing(20)

        self.card_original, self.label_original = self.make_preview_card("Original Input")
        self.card_result, self.label_result = self.make_preview_card("Shadow Removed (Result)")
        
        self.card_corrected, self.label_corrected = self.make_preview_card("Illumination Map")
        self.card_gt, self.label_gt = self.make_preview_card("Ground Truth (Target)")

        col1.addWidget(self.card_original)
        col1.addWidget(self.card_corrected)
        
        col2.addWidget(self.card_gt)
        col2.addWidget(self.card_result)

        grid.addLayout(col1)
        grid.addLayout(col2)
        
        main_layout_area.addLayout(grid)

        # ADD TO MAIN
        main_layout.addWidget(sidebar)
        main_layout.addWidget(main_area)

    def create_button(self, text, callback, is_primary=False, is_success=False):
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(callback)
        if is_primary:
            btn.setObjectName("PrimaryBtn")
        elif is_success:
            btn.setObjectName("SuccessBtn")
        return btn

    def make_preview_card(self, title):
        card = QFrame()
        card.setObjectName("Card")
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        lbl_title = QLabel(title)
        lbl_title.setObjectName("CardTitle")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        img_label = QLabel("No Image")
        img_label.setObjectName("ImagePlaceholder")
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setScaledContents(False)
        from PyQt6.QtWidgets import QSizePolicy
        img_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(lbl_title)
        layout.addWidget(img_label)

        return card, img_label

    # =================================================================
    # LOGIC (UNCHANGED)
    # =================================================================
    def load_shadow(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.JPG *.JPEG *.PNG)"
        )
        if path:
            self.original = cv2.imread(path)
            self.label_original.setPixmap(cv_to_pixmap(self.original))
            self.status.setText("Shadow image loaded successfully.")

    def load_ground_truth(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Ground Truth", "", "Images (*.png *.jpg *.jpeg *.JPG *.JPEG *.PNG)"
        )
        if path:
            self.ground_truth = cv2.imread(path)
            self.label_gt.setPixmap(cv_to_pixmap(self.ground_truth))
            self.status.setText("Ground truth loaded.")

    def process_shadow(self):
        if self.original is None:
            self.status.setText("⚠️ Please load a shadow image first!")
            return
        
        self.status.setText("Processing... Please wait.")
        QApplication.processEvents() # Force UI update

        try:
            orig, corrected, result = shadow_removal(self.original)

            self.corrected = corrected
            self.result = result

            # Handle corrected image (might be grayscale or color)
            if len(corrected.shape) == 2:
                self.label_corrected.setPixmap(cv_to_pixmap(cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)))
            else:
                self.label_corrected.setPixmap(cv_to_pixmap(corrected))

            # Handle result image (might be grayscale or color)
            if len(result.shape) == 2:
                self.label_result.setPixmap(cv_to_pixmap(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)))
            else:
                self.label_result.setPixmap(cv_to_pixmap(result))

            self.status.setText("✅ Shadow removal completed!")
        except Exception as e:
            self.status.setText(f"❌ Error: {str(e)}")
            print(e)

    def evaluate_psnr_ssim(self):
        if self.ground_truth is None or self.result is None:
            self.status.setText("⚠️ Load GT & Run Process first!")
            return

        psnr_val, ssim_val = evaluate_quality(self.ground_truth, self.result)

        QMessageBox.information(self, "Evaluation Result",
                                f"📊 Evaluation Metrics:\n\n"
                                f"PSNR: {psnr_val:.4f} dB\n"
                                f"SSIM: {ssim_val:.4f}")

        self.status.setText(f"Eval Done: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.3f}")

    def save_outputs(self):
        if self.corrected is None or self.result is None:
            self.status.setText("⚠️ Nothing to save!")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        save_image(self.corrected, f"corrected_{timestamp}.png")
        save_image(self.result, f"shadow_removed_{timestamp}.png")

        QMessageBox.information(self, "Saved", "✅ Images saved inside 'result/' folder.")
        self.status.setText("Output saved successfully.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShadowKillApp()
    window.show()
    sys.exit(app.exec())
