import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_quality(ground_truth, result_image):
    """
    Menghitung PSNR dan SSIM setelah memastikan ukuran sama.
    """

    # Pastikan grayscale
    if len(ground_truth.shape) == 3:
        gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    else:
        gt_gray = ground_truth

    if len(result_image.shape) == 3:
        res_gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    else:
        res_gray = result_image

    # Resize GT agar sama persis dengan result
    if gt_gray.shape != res_gray.shape:
        gt_gray = cv2.resize(gt_gray, (res_gray.shape[1], res_gray.shape[0]))

    # Hitung metrik
    psnr_value = psnr(gt_gray, res_gray)
    ssim_value = ssim(gt_gray, res_gray)

    return psnr_value, ssim_value
