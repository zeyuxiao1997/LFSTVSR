import numpy as np
import os
from PIL import Image
import random
from os.path import exists, join, split, realpath, dirname
import cv2
import os
import torch
from PIL import Image
import math
import glob

def cubic(x):
    absx = torch.abs(x)
    absx16 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx16 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx16 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()


OrigData = '/disk3/zeyuxData/LFVideo/Sintel_LFV_9x9/Original'  
# saveRootBIx16 = '/gdata1/xiaozy/VSRBenchmark/TrainingSet/vimeo_septuplet/Damaged/BIx16'
saveRootBIx4 = '/disk3/zeyuxData/LFVideo/Sintel_LFV_9x9/BIx4'
saveRootBIx2 = '/disk3/zeyuxData/LFVideo/Sintel_LFV_9x9/BIx2'
saveRootCrop = '/disk3/zeyuxData/LFVideo/Sintel_LFV_9x9/Crop'
# # saveRootBDx16 = '/gdata1/xiaozy/VSRBenchmark/TrainingSet/vimeo_septuplet/Damaged/BDx16'
# saveRootBDx4 = '/gdata1/xiaozy/VSRBenchmark/TrainingSet/vimeo_septuplet/Damaged/BDx4'
# # saveRootBDx8 = '/gdata1/xiaozy/VSRBenchmark/TrainingSet/vimeo_septuplet/Damaged/BDx8'

# FileList = '/gdata1/xiaozy/VSRBenchmark/TrainingSet/vimeo_septuplet/all.txt'
# VideoFolderList = [line.rstrip() for line in open(FileList)]
# VideoFolderPath = [os.path.join(OrigData,x) for x in VideoFolderList]

# dst=cv2.GaussianBlur(src,(7,7),1.6)

VideoFolderList = sorted(os.listdir(OrigData))
print(VideoFolderList)
for video in VideoFolderList:
    index = os.listdir(os.path.join(OrigData, video))
    # print(index)
    for id in index:
        folderName = os.listdir(os.path.join(OrigData, video, id))
        # print(folderName)
        img = os.listdir(os.path.join(OrigData, video, id))
        # print(img)
        for i in img:
            imgPath = os.path.join(OrigData, video, id, i)
            if imgPath.endswith(".png"):
                # print(imgPath)

                saveCrop = os.path.join(saveRootCrop, video, id)
                if not os.path.exists(saveCrop):
                    os.makedirs(saveCrop)

                saveBIx2 = os.path.join(saveRootBIx2, video, id)
                # print(saveBIx2)
                if not os.path.exists(saveBIx2):
                    os.makedirs(saveBIx2)

                saveBIx4 = os.path.join(saveRootBIx4, video, id)
                # print(saveBIx4)
                if not os.path.exists(saveBIx4):
                    os.makedirs(saveBIx4)

                image = cv2.imread(imgPath)
                image = image[10:426, 0:1024]

                imageBI_LRx2 = imresize_np(image, 1/2, True)
                imageBI_LRx4 = imresize_np(image, 1/4, True)

                savePathBIx4 = os.path.join(saveBIx4, i)
                savePathBIx2 = os.path.join(saveBIx2, i)
                savePathCrop = os.path.join(saveCrop, i)
                
                cv2.imwrite(savePathBIx4, imageBI_LRx4)
                cv2.imwrite(savePathBIx2, imageBI_LRx2)
                cv2.imwrite(savePathCrop, image)
                
                print(savePathBIx4)
                print(savePathBIx2)
                print(savePathCrop)


            # print(imgPath)

# print(VideoFolderList)

# for item in VideoFolderPath:
#     videoID = item[66:]
#     print(videoID)
    
#     # saveBIx16 = os.path.join(saveRootBIx16,videoID)
#     # if not os.path.exists(saveBIx16):
#     #     os.makedirs(saveBIx16)
#     saveBIx4 = os.path.join(saveRootBIx4,videoID)
#     if not os.path.exists(saveBIx4):
#         os.makedirs(saveBIx4)
#     # saveBIx8 = os.path.join(saveRootBIx8,videoID)
#     # if not os.path.exists(saveBIx8):
#     #     os.makedirs(saveBIx8)

#     # saveBDx16 = os.path.join(saveRootBDx16,videoID)
#     # if not os.path.exists(saveBDx16):
#     #     os.makedirs(saveBDx16)
#     saveBDx4 = os.path.join(saveRootBDx4,videoID)
#     if not os.path.exists(saveBDx4):
#         os.makedirs(saveBDx4)
#     # saveBDx8 = os.path.join(saveRootBDx8,videoID)
#     # if not os.path.exists(saveBDx8):
#     #     os.makedirs(saveBDx8)


    # for i in range(1,8,1):
    #     imgPath = os.path.join(OrigData,videoID,'im'+str(i)+'.png')
    #     # savePathBIx16 = os.path.join(saveBIx16,'im'+str(i)+'.png')
    #     savePathBIx4 = os.path.join(saveBIx4,'im'+str(i)+'.png')
    #     # savePathBIx8 = os.path.join(saveBIx8,'im'+str(i)+'.png')
    #     # savePathBDx16 = os.path.join(saveBDx16,'im'+str(i)+'.png')
    #     savePathBDx4 = os.path.join(saveBDx4,'im'+str(i)+'.png')
    #     # savePathBDx8 = os.path.join(saveBDx8,'im'+str(i)+'.png')
    #     print(imgPath)
    #     print(savePathBIx4)
    #     image = cv2.imread(imgPath)
    #     # imageBI_LRx16 = imresize_np(image, 1/16, True)
    #     imageBI_LRx4 = imresize_np(image, 1/4, True)
    #     # imageBI_LRx8 = imresize_np(image, 1/8, True)
    #     # cv2.imwrite(savePathBIx16, imageBI_LRx16)
    #     cv2.imwrite(savePathBIx4, imageBI_LRx4)
    #     # cv2.imwrite(savePathBIx8, imageBI_LRx8)
    #     # print(savePathBIx16)
    #     print(savePathBIx4)
    #     # print(savePathBIx8)

    #     image = cv2.imread(imgPath)
    #     image = cv2.GaussianBlur(image,(7,7),1.6)
    #     # imageBD_LRx16 = imresize_np(image, 1/16, True)
    #     imageBD_LRx4 = imresize_np(image, 1/4, True)
    #     # imageBD_LRx8 = imresize_np(image, 1/8, True)
    #     # cv2.imwrite(savePathBDx16, imageBD_LRx16)
    #     cv2.imwrite(savePathBDx4, imageBD_LRx4)
    #     # cv2.imwrite(savePathBDx8, imageBD_LRx8)
    #     # print(savePathBDx16)
    #     print(savePathBDx4)
    #     # print(savePathBDx8)
        











# if __name__ == '__main__':
#     # getPatch(scale=4)
#     # getPatch(scale=3)
#     getPatch(scale=2)
