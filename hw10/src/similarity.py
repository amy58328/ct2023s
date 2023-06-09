import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import warnings
from tqdm import tqdm
import heapq
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def load_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0
    return img, img_tensor

# 計算圖像相似度
def calculate_similarity(img1_tensor, img2_tensor, loss_fn):
    # MSE 相似度
    mse = np.mean((img1_tensor.squeeze().numpy() - img2_tensor.squeeze().numpy()) ** 2)
    # SSIM 相似度
    ssim_score = ssim(img1_tensor.squeeze().numpy(), img2_tensor.squeeze().numpy(), win_size=7, data_range=1.0)
    # LPIPS 相似度
    lpips_distance = loss_fn(img1_tensor, img2_tensor)
    return mse, ssim_score, lpips_distance.item()

def save_results(save_file, mse_scores, ssim_scores, lpips_scores):
    with open(save_file, 'wb') as f:
        pickle.dump((mse_scores, ssim_scores, lpips_scores), f)

def load_results(save_file):
    with open(save_file, 'rb') as f:
        return pickle.load(f)

def main(folder1, folder2, save_file):
    loss_fn = lpips.LPIPS(net='alex')

    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    common_files = list(set(files1).intersection(set(files2)))

    if os.path.exists(save_file):
        mse_scores, ssim_scores, lpips_scores = load_results(save_file)
        total_mse = sum(mse_scores.values())
        total_ssim = sum(ssim_scores.values())
        total_lpips = sum(lpips_scores.values())
    else:
        mse_scores, ssim_scores, lpips_scores = {}, {}, {}

        total_mse, total_ssim, total_lpips = 0, 0, 0

        for file in tqdm(common_files):
            gt, gt_tensor = load_image(os.path.join(folder1, file))
            gradient, gradient_tensor = load_image(os.path.join(folder2, file))

            mse, ssim_score, lpips_distance = calculate_similarity(gt_tensor, gradient_tensor, loss_fn)
            total_mse += mse
            total_ssim += ssim_score
            total_lpips += lpips_distance

            mse_scores[file] = mse
            ssim_scores[file] = ssim_score
            lpips_scores[file] = lpips_distance

        save_results(save_file, mse_scores, ssim_scores, lpips_scores)

    average_mse = total_mse / len(common_files)
    average_ssim = total_ssim / len(common_files)
    average_lpips = total_lpips / len(common_files)

    print("Average MSE score:", "{:.5f}".format(average_mse))
    print("Average SSIM score:", "{:.5f}".format(average_ssim))
    print("Average LPIPS distance:", "{:.5f}".format(average_lpips))

    top10_mse_files = heapq.nsmallest(10, mse_scores, key=mse_scores.get)
    top10_ssim_files = heapq.nlargest(10, ssim_scores, key=ssim_scores.get)
    top10_lpips_files = heapq.nsmallest(10, lpips_scores, key=lpips_scores.get)

    print("Top 10 files with smallest MSE scores:", top10_mse_files)
    print("Top 10 files with largest SSIM scores:", top10_ssim_files)
    print("Top 10 files with smallest LPIPS distances:", top10_lpips_files)


if __name__ == '__main__':
    target_sid = input("輸入欲比對學號：")
    folder1 = "C:/Users/annie/Downloads/111598012"
    folder2 = "C:/Users/annie/Downloads/"+target_sid
    main(folder1, folder2, 'similarity_scores'+target_sid+'.pkl')
