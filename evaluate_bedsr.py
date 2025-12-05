import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import re
import torch
import sys

# Import BEDSRNet model
# Assuming bedsrmodel2.py is in the same directory
try:
    from bedsrmodel2 import BEDSRNet
except ImportError:
    print("Error: Could not import BEDSRNet from bedsrmodel2.py. Make sure the file exists in the same directory.")
    sys.exit(1)

# ==========================================
# 1. FUNGSI SORTING ALAMI
# ==========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# ==========================================
# 2. INFERENCE ENGINE (Adapted from inferenceDL.py)
# ==========================================
class ShadowRemovalEngine:
    def __init__(self, checkpoint_path, img_size=2048, device="cuda"):
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = BEDSRNet().to(self.device)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
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

        # 2. Resize ke img_size (MATCH training/inference config)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_AREA)

        inp = img_resized.astype(np.float32) / 255.0      # [0,1]
        inp = inp * 2.0 - 1.0                             # [-1,1]
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 3. Forward ke model
        pred, bg, att = self.model(inp)

        # 4. Balik ke numpy RGB 0–255
        out = pred[0].permute(1, 2, 0).cpu().numpy()      # [-1,1]
        out = (out + 1.0) / 2.0                           # [0,1]
        out = (out * 255.0).clip(0, 255).astype(np.uint8) # [0,255] uint8

        # 5. Resize output balik ke resolusi asli
        out_full = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # Return BGR for consistency with cv2 and evaluate.py pipeline
        out_bgr = cv2.cvtColor(out_full, cv2.COLOR_RGB2BGR)
        return out_bgr

# ==========================================
# 3. FUNGSI EVALUASI
# ==========================================
def calculate_metrics(gt_img, pred_img):
    # Resize prediksi jika ukuran beda sedikit (safety)
    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
    
    # Hitung PSNR & SSIM
    psnr_val = psnr_metric(gt_img, pred_img, data_range=255)
    ssim_val = ssim_metric(gt_img, pred_img, channel_axis=2, win_size=3, data_range=255)
    return psnr_val, ssim_val

# ==========================================
# 4. LOOPING UTAMA
# ==========================================
def main():
    # KONFIGURASI PATH FOLDER
    folder_input = 'data/test/img'  # Folder input (shadow)
    folder_gt    = 'data/test/gt'   # Folder ground truth
    model_path   = './best_model/training3/best_model.pth'
    
    # Cek folder
    if not os.path.exists(folder_input) or not os.path.exists(folder_gt):
        print("Error: Folder input atau gt tidak ditemukan.")
        return

    # Inisialisasi Model
    try:
        # Menggunakan img_size=2048 sesuai dengan inferenceDL.py
        # Jika ingin lebih ringan (laptop kentang), ubah ke 512
        engine = ShadowRemovalEngine(checkpoint_path=model_path, img_size=2048)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Ambil list file dan urutkan secara natural
    files = sorted(os.listdir(folder_input), key=natural_sort_key)
    results = []

    print(f"Mencari gambar di: {folder_input}")
    print(f"Mencari ground truth di: {folder_gt}\n")
    print(f"{'Filename':<20} | {'PSNR (dB)':<10} | {'SSIM':<10}")
    print("-" * 45)

    processed_count = 0

    for filename in files:
        # Filter hanya file gambar
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
            
        path_in = os.path.join(folder_input, filename)
        path_gt = os.path.join(folder_gt, filename) # Asumsi nama file SAMA PERSIS
        
        # Cek ketersediaan file GT
        if not os.path.exists(path_gt):
            print(f"[SKIP] {filename}: Tidak ditemukan di folder gt.")
            continue
            
        # Load Citra GT
        img_gt = cv2.imread(path_gt)
        
        if img_gt is None:
            print(f"[ERROR] Gagal membaca GT {filename}")
            continue
            
        # --- PROSES (METODE DEEP LEARNING) ---
        try:
            img_prediction = engine.run(path_in)
        except Exception as e:
            print(f"[ERROR] Gagal memproses {filename}: {e}")
            continue
        
        # --- HITUNG METRIK ---
        try:
            psnr, ssim = calculate_metrics(img_gt, img_prediction)
            
            results.append({
                'Filename': filename,
                'PSNR': round(psnr, 4),
                'SSIM': round(ssim, 4)
            })
            
            print(f"{filename:<20} | {psnr:.4f}     | {ssim:.4f}")
            processed_count += 1
        except Exception as e:
            print(f"[ERROR] Gagal menghitung metrik untuk {filename}: {e}")

    # ==========================================
    # 5. EXPORT HASIL
    # ==========================================
    if processed_count > 0:
        df = pd.DataFrame(results)
        
        # Hitung Rata-rata
        avg_psnr = df['PSNR'].mean()
        avg_ssim = df['SSIM'].mean()
        
        # Tambah baris rata-rata
        new_row = pd.DataFrame([{'Filename': 'RATA-RATA', 'PSNR': avg_psnr, 'SSIM': avg_ssim}])
        df = pd.concat([df, new_row], ignore_index=True)
        
        print("\n" + "="*45)
        print(f"TOTAL DIPROSES: {processed_count} Citra")
        print(f"RATA-RATA PSNR: {avg_psnr:.4f} dB")
        print(f"RATA-RATA SSIM: {avg_ssim:.4f}")
        print("="*45)
        
        # Simpan CSV
        output_file = 'hasil_evaluasi_bedsr.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ File Excel/CSV tersimpan: {output_file}")
    else:
        print("\n❌ Tidak ada gambar yang berhasil diproses.")

if __name__ == "__main__":
    main()
