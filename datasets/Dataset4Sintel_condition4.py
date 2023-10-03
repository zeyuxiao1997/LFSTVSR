
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt, patch_size, scale=4, ix=-1, iy=-1):
    (ih, iw) = one1[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    centerView1 = centerView1.crop((iy, ix, iy + ip, ix + ip))
    one1 = [i.crop((iy, ix, iy + ip, ix + ip)) for i in one1]
    two1 = [i.crop((iy, ix, iy + ip, ix + ip)) for i in two1]
    three1 = [i.crop((iy, ix, iy + ip, ix + ip)) for i in three1]

    centerView3 = centerView3.crop((iy, ix, iy + ip, ix + ip))
    one3 = [i.crop((iy, ix, iy + ip, ix + ip)) for i in one3]
    two3 = [i.crop((iy, ix, iy + ip, ix + ip)) for i in two3]
    three3 = [i.crop((iy, ix, iy + ip, ix + ip)) for i in three3]

    gt = [i.crop((ty, tx, ty + tp, tx + tp)) for i in gt]  # [:, iy:iy + ip, ix:ix + ip]

    return  centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt


def augment(centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        centerView1 = ImageOps.flip(centerView1)
        one1 = [ImageOps.flip(i) for i in one1]
        two1 = [ImageOps.flip(i) for i in two1]
        three1 = [ImageOps.flip(i) for i in three1]
        centerView3 = ImageOps.flip(centerView3)
        one3 = [ImageOps.flip(i) for i in one3]
        two3 = [ImageOps.flip(i) for i in two3]
        three3 = [ImageOps.flip(i) for i in three3]
        gt = [ImageOps.flip(i) for i in gt]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            centerView1 = ImageOps.mirror(centerView1)
            one1 = [ImageOps.mirror(i) for i in one1]
            two1 = [ImageOps.mirror(i) for i in two1]
            three1 = [ImageOps.mirror(i) for i in three1]
            centerView3 = ImageOps.mirror(centerView3)
            one3 = [ImageOps.mirror(i) for i in one3]
            two3 = [ImageOps.mirror(i) for i in two3]
            three3 = [ImageOps.mirror(i) for i in three3]
            gt = [ImageOps.mirror(i) for i in gt]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            centerView1 = centerView1.rotate(180)
            one1 = [i.rotate(180) for i in one1]
            two1 = [i.rotate(180) for i in two1]
            three1 = [i.rotate(180) for i in three1]
            centerView3 = centerView3.rotate(180)
            one3 = [i.rotate(180) for i in one3]
            two3 = [i.rotate(180) for i in two3]
            three3 = [i.rotate(180) for i in three3]
            gt = [i.rotate(180) for i in gt]
            info_aug['trans'] = True

    return centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt


def get_image(img):
    img = Image.open(img).convert('RGB')
    return img


def load_image(group): # 8 5 3
    images = [get_image(img) for img in group]

    centerView1 = images[0]
    one1 = images[1:9]
    two1 = images[9:14]
    three1 = images[14:17]

    centerView3 = images[17]
    one3 = images[18:26]
    two3 = images[26:31]
    three3 = images[31:34]

    gt = images[34:]
    return centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt


def transform():
    return Compose([
        ToTensor(),
    ])


class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, opt, transform=transform()):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(opt.group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = opt.upscale_factor
        self.transform = transform
        self.data_augmentation = opt.augmentation
        self.patch_size = opt.patch_size
        self.hflip = opt.hflip
        self.rot = opt.rot

    def __getitem__(self, index):

        centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = load_image(self.image_filenames[index])

        if self.patch_size != 0:
            centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = get_patch(centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = augment(centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt, self.hflip, self.rot)

        if self.transform:
            centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt
            centerView1 = self.transform(centerView1)
            one1 = [self.transform(i) for i in one1]
            two1 = [self.transform(i) for i in two1]
            three1 = [self.transform(i) for i in three1]
            centerView3 = self.transform(centerView3)
            one3 = [self.transform(i) for i in one3]
            two3 = [self.transform(i) for i in two3]
            three3 = [self.transform(i) for i in three3]
            gt = [self.transform(i) for i in gt]

        one1 = torch.cat((torch.unsqueeze(one1[0], 0), torch.unsqueeze(one1[1], 0),
                             torch.unsqueeze(one1[2], 0), torch.unsqueeze(one1[3], 0),
                             torch.unsqueeze(one1[4], 0), torch.unsqueeze(one1[5], 0),
                             torch.unsqueeze(one1[6], 0), torch.unsqueeze(one1[7], 0)))

        two1 = torch.cat((torch.unsqueeze(two1[0], 0), torch.unsqueeze(two1[1], 0),
                             torch.unsqueeze(two1[2], 0), torch.unsqueeze(two1[3], 0),
                             torch.unsqueeze(two1[4], 0)))

        three1 = torch.cat((torch.unsqueeze(three1[0], 0), torch.unsqueeze(three1[1], 0),
                             torch.unsqueeze(three1[2], 0)))

        one3 = torch.cat((torch.unsqueeze(one3[0], 0), torch.unsqueeze(one3[1], 0),
                             torch.unsqueeze(one3[2], 0), torch.unsqueeze(one3[3], 0),
                             torch.unsqueeze(one3[4], 0), torch.unsqueeze(one3[5], 0),
                             torch.unsqueeze(one3[6], 0), torch.unsqueeze(one3[7], 0)))

        two3 = torch.cat((torch.unsqueeze(two3[0], 0), torch.unsqueeze(two3[1], 0),
                             torch.unsqueeze(two3[2], 0), torch.unsqueeze(two3[3], 0),
                             torch.unsqueeze(two3[4], 0)))

        three3 = torch.cat((torch.unsqueeze(three3[0], 0), torch.unsqueeze(three3[1], 0),
                             torch.unsqueeze(three3[2], 0)))

        gt = torch.cat((torch.unsqueeze(gt[0], 0), torch.unsqueeze(gt[1], 0),
                             torch.unsqueeze(gt[2], 0)))

        return centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt

    def __len__(self):
        return len(self.image_filenames)




class DatasetFromFolderValid(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, opt, transform=transform()):
        super(DatasetFromFolderValid, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(opt.group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = opt.upscale_factor
        self.transform = transform
        self.data_augmentation = opt.augmentation
        self.patch_size = opt.patch_size
        self.hflip = opt.hflip
        self.rot = opt.rot

    def __getitem__(self, index):

        centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = load_image(self.image_filenames[index])

        # if self.patch_size != 0:
        #     centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = get_patch(centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt, self.patch_size, self.upscale_factor)

        # if self.data_augmentation:
        #     centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = augment(centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt, self.hflip, self.rot)

        if self.transform:
            centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt
            centerView1 = self.transform(centerView1)
            one1 = [self.transform(i) for i in one1]
            two1 = [self.transform(i) for i in two1]
            three1 = [self.transform(i) for i in three1]
            centerView3 = self.transform(centerView3)
            one3 = [self.transform(i) for i in one3]
            two3 = [self.transform(i) for i in two3]
            three3 = [self.transform(i) for i in three3]
            gt = [self.transform(i) for i in gt]

        one1 = torch.cat((torch.unsqueeze(one1[0], 0), torch.unsqueeze(one1[1], 0),
                             torch.unsqueeze(one1[2], 0), torch.unsqueeze(one1[3], 0),
                             torch.unsqueeze(one1[4], 0), torch.unsqueeze(one1[5], 0),
                             torch.unsqueeze(one1[6], 0), torch.unsqueeze(one1[7], 0)))

        two1 = torch.cat((torch.unsqueeze(two1[0], 0), torch.unsqueeze(two1[1], 0),
                             torch.unsqueeze(two1[2], 0), torch.unsqueeze(two1[3], 0),
                             torch.unsqueeze(two1[4], 0)))

        three1 = torch.cat((torch.unsqueeze(three1[0], 0), torch.unsqueeze(three1[1], 0),
                             torch.unsqueeze(three1[2], 0)))

        one3 = torch.cat((torch.unsqueeze(one3[0], 0), torch.unsqueeze(one3[1], 0),
                             torch.unsqueeze(one3[2], 0), torch.unsqueeze(one3[3], 0),
                             torch.unsqueeze(one3[4], 0), torch.unsqueeze(one3[5], 0),
                             torch.unsqueeze(one3[6], 0), torch.unsqueeze(one3[7], 0)))

        two3 = torch.cat((torch.unsqueeze(two3[0], 0), torch.unsqueeze(two3[1], 0),
                             torch.unsqueeze(two3[2], 0), torch.unsqueeze(two3[3], 0),
                             torch.unsqueeze(two3[4], 0)))

        three3 = torch.cat((torch.unsqueeze(three3[0], 0), torch.unsqueeze(three3[1], 0),
                             torch.unsqueeze(three3[2], 0)))

        gt = torch.cat((torch.unsqueeze(gt[0], 0), torch.unsqueeze(gt[1], 0),
                             torch.unsqueeze(gt[2], 0)))

        return centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt

    def __len__(self):
        return len(self.image_filenames)





if __name__ == '__main__':
    output = 'visualize'
    if not os.path.exists(output):
        os.mkdir(output)
    dataset = DatasetFromFolder(4, True, 'dataset/groups.txt', 64, True, True, True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
    for i, (inputs, target) in enumerate(dataloader):
        if i > 10:
            break
        if not os.path.exists(os.path.join(output, 'group{}'.format(i))):
            os.mkdir(os.path.join(output, 'group{}'.format(i)))
        input0, input1, input2, input3, input4 = inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3], inputs[0][4]
        vutils.save_image(input0, os.path.join(output, 'group{}'.format(i), 'input0.png'))
        vutils.save_image(input1, os.path.join(output, 'group{}'.format(i), 'input1.png'))
        vutils.save_image(input2, os.path.join(output, 'group{}'.format(i), 'input2.png'))
        vutils.save_image(input3, os.path.join(output, 'group{}'.format(i), 'input3.png'))
        vutils.save_image(input4, os.path.join(output, 'group{}'.format(i), 'input4.png'))
        vutils.save_image(target, os.path.join(output, 'group{}'.format(i), 'target.png'))


