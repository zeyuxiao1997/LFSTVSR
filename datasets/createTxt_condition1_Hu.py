#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   createTxt_condition1.py
@Time    :   2021/08/18 14:01:00
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

import os
import argparse

vidsTest = ['roadside_1','roadside_2','stone_tablet_fixed']

condition1 = ['03_03','03_04','03_05',\
              '04_03','04_04','04_05',\
              '05_03','05_04','05_05']

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    # vids = sorted(os.listdir(inputdir))
    for vid in vidsTest:
        foldername = os.path.join(inputdir,vid)
        # print(foldername)
        frameNum = len(sorted(os.listdir(os.path.join(foldername,'00_00'))))
        
        # one = []
        for view in condition1:
            one = []
            two = []
            three = []
            hang = view[1:2]
            lie = view[4:6]
            # print(hang,lie)
            one.append(str(int(hang)-1).zfill(2)+'_'+str(int(int(lie))-1).zfill(2))
            one.append(str(int(hang)-1).zfill(2)+'_'+str(int(lie)).zfill(2))
            one.append(str(int(hang)-1).zfill(2)+'_'+str(int(lie)+1).zfill(2))
            one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)-1).zfill(2))
            # one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
            one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+1).zfill(2))
            one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)-1).zfill(2))
            one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)).zfill(2))
            one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)+1).zfill(2))

            two.append(str(int(hang)-2).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))
            two.append(str(int(hang)-2).zfill(2)+'_'+str(int(lie)).zfill(2))
            two.append(str(int(hang)-2).zfill(2)+'_'+str(int(lie)+2).zfill(2))
            two.append(str(int(hang)).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))
            # two.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
            two.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+2).zfill(2))
            two.append(str(int(hang)+2).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))
            two.append(str(int(hang)+2).zfill(2)+'_'+str(int(lie)).zfill(2))
            two.append(str(int(hang)+2).zfill(2)+'_'+str(int(lie)+2).zfill(2))

            three.append(str(int(hang)-3).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            three.append(str(int(hang)-3).zfill(2)+'_'+str(int(lie)).zfill(2))
            three.append(str(int(hang)-3).zfill(2)+'_'+str(int(lie)+3).zfill(2))
            three.append(str(int(hang)).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            # three.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
            three.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+3).zfill(2))
            three.append(str(int(hang)+3).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            three.append(str(int(hang)+3).zfill(2)+'_'+str(int(lie)).zfill(2))
            three.append(str(int(hang)+3).zfill(2)+'_'+str(int(lie)+3).zfill(2))

            # print(view,one)

            for idx in range(1, frameNum-2,1):
                groups = ''
                print(os.path.join(inputdir, vid, view, str(idx).zfill(6)+'.jpg'))

                ## first frame
                # center view
                groups += os.path.join(inputdir, vid, view, str(idx).zfill(6)+'.jpg') + '|'
                # 1x view
                groups += os.path.join(inputdir, vid, one[0], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[1], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[2], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[3], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[4], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[5], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[6], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[7], str(idx).zfill(6)+'.jpg') + '|'
                # 2x view
                groups += os.path.join(inputdir, vid, two[0], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[1], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[2], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[3], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[4], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[5], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[6], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[7], str(idx).zfill(6)+'.jpg') + '|'
                # 4x view
                groups += os.path.join(inputdir, vid, three[0], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[1], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[2], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[3], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[4], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[5], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[6], str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[7], str(idx).zfill(6)+'.jpg') + '|'

                ## third frame
                # center view
                groups += os.path.join(inputdir, vid, view, str(idx+2).zfill(6)+'.jpg') + '|'
                # 1x view
                groups += os.path.join(inputdir, vid, one[0], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[1], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[2], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[3], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[4], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[5], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[6], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, one[7], str(idx+2).zfill(6)+'.jpg') + '|'
                # 2x view
                groups += os.path.join(inputdir, vid, two[0], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[1], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[2], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[3], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[4], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[5], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[6], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, two[7], str(idx+2).zfill(6)+'.jpg') + '|'
                # 4x view
                groups += os.path.join(inputdir, vid, three[0], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[1], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[2], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[3], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[4], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[5], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[6], str(idx+2).zfill(6)+'.jpg') + '|'
                groups += os.path.join(inputdir, vid, three[7], str(idx+2).zfill(6)+'.jpg') + '|'

                ## second frame
                # center view
                groups += os.path.join(targetdir, vid, view, str(idx).zfill(6)+'.jpg') + '|'
                groups += os.path.join(targetdir, vid, view, str(idx+1).zfill(6)+'.jpg') + '|'
                groups += os.path.join(targetdir, vid, view, str(idx+2).zfill(6)+'.jpg') 

                print(groups)
                with open(os.path.join(outputdir, 'TestGroupsCondition1.txt'), 'a') as f:
                    f.write(groups + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/disk3/zeyuxData/LFVideo/released_dataset/LF_video_processed_BIx4', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/disk3/zeyuxData/LFVideo/released_dataset/LF_video_processed_Crop', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/disk3/zeyuxData/LFVideo/released_dataset/LFSTVSRGroups', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.jpg', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()