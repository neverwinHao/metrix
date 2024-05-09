import os
import numpy as np
import torch
from torchvision import transforms
import util
from option import args
# 您提供的calc_metrics函数和其他辅助函数
from PIL import Image

# 定义加载图像的函数
def load_image(img_path):
    # 读取图像并转换为Tensor
    img = Image.open(img_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img)
    return img_tensor

# 比较两个文件夹中的图像
def compare_folders(folder1, folder2):
    image_pairs = []  # 存储图像对的路径

    psnr_value = []
    ssim_value = []
    cnt = 0
    val_psnr = 0
    val_ssim = 0
    # 获取两个文件夹中的文件列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 找出共同的文件名
    common_files = files1.intersection(files2)

    # 为每个共同的文件名创建图像对
    for filename in common_files:
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)
        image_pairs.append((img1_path, img2_path))

    # 计算每个图像对的PSNR和SSIM
    for img1_path, img2_path in image_pairs:
        # 加载图像
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        # 将图像转换为NumPy数组并移动到CPU
        img1 = img1.cpu().numpy()
        img2 = img2.cpu().numpy()

        # 计算评价指标
        psnr, ssim = util.calc_metrics(img1, img2)
        val_psnr += psnr
        val_ssim += ssim
        psnr_value.append(psnr)
        ssim_value.append(ssim)
        # 打印结果
        cnt += 1
        if cnt % 100 == 0:
            print(f'已检测到 {cnt:<10}' + f' | cur psnr {val_psnr / cnt:<10.4f}' + f' | cur ssim {val_ssim / cnt:.4f}')
    print('PSNR:',np.mean(psnr_value))
    print('SSIM:',np.mean(ssim_value))

if args.test_dataset == 'CelebA':
    HR = './CelebA/HR'
    print(args.test_model)
    if 'SISN' in args.test_model:
        SR = './CelebA'+'/SR_SISN'
        print('SR Path',SR)
    elif 'DMD' in args.test_model:
        HR = './CelebA/HR_DMD'
        SR = './CelebA'+'/SR_DMD'
        print('SR Path',SR)
    else:
        SR = './CelebA'+'/SR_Ours'
        print('SR Path',SR)
elif args.test_dataset == 'Helen':
    HR = './Helen/HR'
    print(args.test_model)
    if 'SISN' in args.test_model:
        SR = './Helen'+'/SR_SISN'
        print('SR Path',SR)
    elif 'DMD' in args.test_model:
        SR = './Helen'+'/SR_DMD'
        print('SR Path',SR)
    else:
        SR = './Helen'+'/SR_Ours'
        print('SR Path',SR)
elif args.test_dataset == 'FFHQ':
    HR = './FFHQ/HR'
    print(args.test_model)
    if 'SISN' in args.test_model:
        SR = './FFHQ'+'/SR_SISN'
        print('SR Path',SR)
    elif 'DMD' in args.test_model:
        SR = './FFHQ'+'/SR_DMD'
        print('SR Path',SR)
    else:
        SR = './FFHQ'+'/SR_Ours'
        print('SR Path',SR)

# 执行比较
compare_folders(HR, SR)
