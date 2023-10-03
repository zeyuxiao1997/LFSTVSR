import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    vids = sorted(os.listdir(inputdir))
    for vid in vids:
        for idx in range(0, 177):
            groups = ''
            for i in range(idx, idx+5):
                groups += os.path.join(inputdir, vid, '{:08d}'.format(2*i) + ext) + '|'
            groups = groups[:-1]
            print(groups)
            with open(os.path.join(outputdir, 'Valsgroups.txt'), 'a') as f:
                f.write(groups + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/userhome/xiaozeyuData/AIM2020/ALLVals', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/userhome/xiaozeyuData/AIM2020/ALLVals', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/userhome/xiaozeyuCode/AIM2020/v2', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()