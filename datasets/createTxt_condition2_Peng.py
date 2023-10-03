# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   createTxt_condition2.py
@Time    :   2021/09/21 20:51:11
@Author  :   ZeyuXiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   None
"""

import os
import argparse


vidsTest = ['sub4_MotionBlur17', 'sub3_Bush_2', 'sub3_Tree4', 'sub4_Occlusion1', 'sub3_Stone', 'sub4_MotionBlur13', 'sub3_Flower_1', 'sub3_Department_1', 'sub3_Tree5', 'sub4_Occlusion4', 'sub4_MotionBlur11', 'sub4_MotionBlur6', 'sub3_Bush_1', 'sub4_MotionBlur20', 'sub3_Tree_2', 'sub4_Occlusion8', 'sub3_Tree2', 'sub4_MotionBlur_Reflect2', 'sub3_Tree3', 'sub3_Libaray_1', 'sub3_Tree1', 'sub4_Occlusion2', 'sub3_Lawn_lamp', 'sub4_MotionBlur19', 'sub3_Bamboo', 'sub4_MotionBlur_Reflect1', 'sub4_MotionBlur9', 'sub3_Flower_2', 'sub3_Tree_1', 'sub4_Occlusion3', 'sub4_MotionBlur8', 'sub3_Tree_Road', 'sub3_Root1', 'sub4_MotionBlur18', 'sub4_MotionBlur2', 'sub3_Path', 'sub4_MotionBlur15', 'sub4_MotionBlur4', 'sub4_MotionBlur12', 'sub4_MotionBlur7', 'sub4_Reflect2', 'sub4_Reflect3', 'sub3_Root4', 'sub4_MotionBlur5', 'sub4_Reflect1', 'sub4_Occlusion7', 'sub3_Department_2', 'sub4_MotionBlur16', 'sub4_MotionBlur14', 'sub4_Occlusion5', 'sub4_Occlusion6', 'sub4_MotionBlur1', 'sub3_Flower_3', 'sub3_Root2', 'sub3_Root3']

condition2 = ['02_03','02_04','02_05',\
              '03_02','03_06',\
              '04_02','04_06',\
              '05_02','05_06',\
              '06_03','06_04','06_05']

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
        for view in condition2:
            one = []
            two = []
            three = []
            hang = view[1:2]
            lie = view[4:6]
            # print(hang,lie)
            if (int(hang)-1 >=0 and int(hang)-1 <= 8) and (int(lie)-1 >=0 and int(lie)-1 <= 8):
                one.append(str(int(hang)-1).zfill(2)+'_'+str(int(int(lie))-1).zfill(2))
            if (int(hang)-1 >=0 and int(hang)-1 <= 8) and (int(lie) >=0 and int(lie) <= 8):
                one.append(str(int(hang)-1).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)-1 >=0 and int(hang)-1 <= 8) and (int(lie)+1 >=0 and int(lie)+1 <= 8):
                one.append(str(int(hang)-1).zfill(2)+'_'+str(int(lie)+1).zfill(2))
            if (int(hang) >=0 and int(hang) <= 8) and (int(lie)-1 >=0 and int(lie)-1 <= 8):
                one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)-1).zfill(2))
            if (int(hang) >=0 and int(hang) <= 8) and (int(lie)+1 >=0 and int(lie)+1 <= 8):
                # one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
                one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+1).zfill(2))
            if (int(hang)+1 >=0 and int(hang)+1 <= 8) and (int(lie)-1 >=0 and int(lie)-1 <= 8):
                one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)-1).zfill(2))
            if (int(hang)+1 >=0 and int(hang)+1 <= 8) and (int(lie) >=0 and int(lie) <= 8):
                one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)+1 >=0 and int(hang)+1 <= 8) and (int(lie)+1 >=0 and int(lie)+1 <= 8):
                one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)+1).zfill(2))

            if (int(hang)-2 >=0 and int(hang)-2 <= 8) and (int(lie)-2 >=0 and int(lie)-2 <= 8):
                two.append(str(int(hang)-2).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))
            if (int(hang)-2 >=0 and int(hang)-2 <= 8) and (int(lie) >=0 and int(lie) <= 8):
                two.append(str(int(hang)-2).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)-2 >=0 and int(hang)-2 <= 8) and (int(lie)+2 >=0 and int(lie)+2 <= 8):
                two.append(str(int(hang)-2).zfill(2)+'_'+str(int(lie)+2).zfill(2))
            if (int(hang) >=0 and int(hang) <= 8) and (int(lie)-2 >=0 and int(lie)-2 <= 8):
                two.append(str(int(hang)).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))

                # two.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang) >=0 and int(hang) <= 8) and (int(lie)+2 >=0 and int(lie)+2 <= 8):
                two.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+2).zfill(2))
            if (int(hang)+2 >=0 and int(hang)+2 <= 8) and (int(lie)-2 >=0 and int(lie)-2 <= 8):
                two.append(str(int(hang)+2).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))
            if (int(hang)+2 >=0 and int(hang)+2 <= 8) and (int(lie) >=0 and int(lie) <= 8):
                two.append(str(int(hang)+2).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)+2 >=0 and int(hang)+2 <= 8) and (int(lie)+2 >=0 and int(lie)+2 <= 8):
                two.append(str(int(hang)+2).zfill(2)+'_'+str(int(lie)+2).zfill(2))
            if (int(hang)-3 >=0 and int(hang)-3 <= 8) and (int(lie)-3 >=0 and int(lie)-3 <= 8):
                three.append(str(int(hang)-3).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            if (int(hang)-3 >=0 and int(hang)-3 <= 8) and (int(lie) >=0 and int(lie) <= 8):
                three.append(str(int(hang)-3).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)-3 >=0 and int(hang)-3 <= 8) and (int(lie)+3 >=0 and int(lie)+3 <= 8):
                three.append(str(int(hang)-3).zfill(2)+'_'+str(int(lie)+3).zfill(2))
            if (int(hang) >=0 and int(hang) <= 8) and (int(lie)-3 >=0 and int(lie)-3 <= 8):
                three.append(str(int(hang)).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            if (int(hang) >=0 and int(hang) <= 8) and (int(lie)+3 >=0 and int(lie)+3 <= 8):
                # three.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
                three.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+3).zfill(2))
            if (int(hang)+3 >=0 and int(hang)+3 <= 8) and (int(lie)-3 >=0 and int(lie)-3 <= 8):
                three.append(str(int(hang)+3).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            if (int(hang)+3 >=0 and int(hang)+3 <= 8) and (int(lie) >=0 and int(lie) <= 8):
                three.append(str(int(hang)+3).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)+3 >=0 and int(hang)+3 <= 8) and (int(lie)+3 >=0 and int(lie)+3 <= 8):
                three.append(str(int(hang)+3).zfill(2)+'_'+str(int(lie)+3).zfill(2))

            # print(view,one)

            for idx in range(0, frameNum-2,1):
                groups = ''
                print(os.path.join(inputdir, vid, view, list[idx]))

                print('length of one', len(one))
                print('length of two', len(two))
                print('length of three', len(three))
                ## first frame
                # center view
                groups += os.path.join(inputdir, vid, view, list[idx]) + '|'
                # 1x view
                groups += os.path.join(inputdir, vid, one[0], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, one[1], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, one[2], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, one[3], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, one[4], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, one[5], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, one[6], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, one[7], list[idx]) + '|'
                # 2x view
                groups += os.path.join(inputdir, vid, two[0], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, two[1], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, two[2], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, two[3], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, two[4], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, two[5], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, two[6], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, two[7], list[idx]) + '|'
                # 4x view
                groups += os.path.join(inputdir, vid, three[0], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, three[1], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, three[2], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, three[3], list[idx]) + '|'
                groups += os.path.join(inputdir, vid, three[4], list[idx]) + '|'
                # groups += os.path.join(inputdir, vid, three[5], list[idx]) + '|'
                # groups += os.path.join(inputdir, vid, three[6], list[idx]) + '|'
                # groups += os.path.join(inputdir, vid, three[7], list[idx]) + '|'

                ## third frame
                # center view
                groups += os.path.join(inputdir, vid, view, list[idx+2]) + '|'
                # 1x view
                groups += os.path.join(inputdir, vid, one[0], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, one[1], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, one[2], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, one[3], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, one[4], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, one[5], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, one[6], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, one[7], list[idx+2]) + '|'
                # 2x view
                groups += os.path.join(inputdir, vid, two[0], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, two[1], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, two[2], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, two[3], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, two[4], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, two[5], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, two[6], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, two[7], list[idx+2]) + '|'
                # 4x view
                groups += os.path.join(inputdir, vid, three[0], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, three[1], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, three[2], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, three[3], list[idx+2]) + '|'
                groups += os.path.join(inputdir, vid, three[4], list[idx+2]) + '|'
                # groups += os.path.join(inputdir, vid, three[5], list[idx+2]) + '|'
                # groups += os.path.join(inputdir, vid, three[6], list[idx+2]) + '|'
                # groups += os.path.join(inputdir, vid, three[7], list[idx+2]) + '|'

                ## second frame
                # center view
                groups += os.path.join(targetdir, vid, view, list[idx]) + '|'
                groups += os.path.join(targetdir, vid, view, list[idx+1]) + '|'
                groups += os.path.join(targetdir, vid, view, list[idx+2]) 

                print(groups)
                with open(os.path.join(outputdir, 'TestGroupsCondition2.txt'), 'a') as f:
                    f.write(groups + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/disk3/zeyuxData/LFVideo/Peng/Image_BIx4', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/disk3/zeyuxData/LFVideo/Peng/Image_Crop', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/disk3/zeyuxData/LFVideo/Peng/LFSTVSRGroups', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()