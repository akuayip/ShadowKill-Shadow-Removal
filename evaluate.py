import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import re

# ==========================================
# 1. FUNGSI SORTING ALAMI (Agar urutan rect_1, rect_2, ... rect_10)
# ==========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# ==========================================
# 2. DEFINISI METODE KONVENSIONAL (KODE KAMU)
# ==========================================
def shadow_removal(image):
    # Pre-processing
    denoised = cv2.medianBlur(image, 3)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Estimasi Background Dinamis
    h_img, w_img = v.shape
    kernel_size = int(min(h_img, w_img) * 0.03) 
    if kernel_size % 2 == 0: kernel_size += 1
    kernel_size = max(15, kernel_size)    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    background = cv2.morphologyEx(v, cv2.MORPH_CLOSE, kernel)
    background = np.where(background == 0, 1, background).astype(np.float32)
    v_float = v.astype(np.float32)
    
    # Illumination Correction
    corrected_v = (v_float / background) * 255
    corrected_v = np.clip(corrected_v, 0, 255).astype(np.uint8)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_v = clahe.apply(corrected_v)
    
    # Merge kembali
    merged_hsv = cv2.merge([h, s, enhanced_v])
    result_color = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
    corrected_gray = enhanced_v

    return image, corrected_gray, result_color

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
# KONFIGURASI PATH FOLDER (SESUAI REQUEST)
folder_input = 'data/test/img'  # Folder input (shadow)
folder_gt    = 'data/test/gt'   # Folder ground truth

# Ambil list file dan urutkan secara natural (1, 2, ..., 10, bukan 1, 10, 2)
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
    
    # Cek ketersediaan file
    if not os.path.exists(path_gt):
        print(f"[SKIP] {filename}: Tidak ditemukan di folder gt.")
        continue
        
    # Load Citra
    img_input = cv2.imread(path_in)
    img_gt = cv2.imread(path_gt)
    
    if img_input is None or img_gt is None:
        print(f"[ERROR] Gagal membaca {filename}")
        continue
        
    # --- PROSES (METODE KONVENSIONAL) ---
    _, _, img_prediction = shadow_removal(img_input)
    
    # --- HITUNG METRIK ---
    psnr, ssim = calculate_metrics(img_gt, img_prediction)
    
    results.append({
        'Filename': filename,
        'PSNR': round(psnr, 4),
        'SSIM': round(ssim, 4)
    })
    
    print(f"{filename:<20} | {psnr:.4f}     | {ssim:.4f}")
    processed_count += 1

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
    output_file = 'hasil_evaluasi_rect_1_20.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ File Excel/CSV tersimpan: {output_file}")
else:
    print("\n❌ Tidak ada gambar yang berhasil diproses. Pastikan nama file di folder 'img' dan 'gt' sama persis.")