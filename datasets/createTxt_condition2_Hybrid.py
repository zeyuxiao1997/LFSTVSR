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


vidsTest = ['seq01','seq02','seq03','seq04','seq05','seq06']

condition2 = ['02_03',\
              '03_02','03_04',\
              '04_03']

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
            if (int(hang)-1 >=0 and int(hang)-1 <= 6) and (int(lie)-1 >=0 and int(lie)-1 <= 6):
                one.append(str(int(hang)-1).zfill(2)+'_'+str(int(int(lie))-1).zfill(2))
            if (int(hang)-1 >=0 and int(hang)-1 <= 6) and (int(lie) >=0 and int(lie) <= 6):
                one.append(str(int(hang)-1).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)-1 >=0 and int(hang)-1 <= 6) and (int(lie)+1 >=0 and int(lie)+1 <= 6):
                one.append(str(int(hang)-1).zfill(2)+'_'+str(int(lie)+1).zfill(2))
            if (int(hang) >=0 and int(hang) <= 6) and (int(lie)-1 >=0 and int(lie)-1 <= 6):
                one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)-1).zfill(2))
            if (int(hang) >=0 and int(hang) <= 6) and (int(lie)+1 >=0 and int(lie)+1 <= 6):
                # one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
                one.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+1).zfill(2))
            if (int(hang)+1 >=0 and int(hang)+1 <= 6) and (int(lie)-1 >=0 and int(lie)-1 <= 6):
                one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)-1).zfill(2))
            if (int(hang)+1 >=0 and int(hang)+1 <= 6) and (int(lie) >=0 and int(lie) <= 6):
                one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)+1 >=0 and int(hang)+1 <= 6) and (int(lie)+1 >=0 and int(lie)+1 <= 6):
                one.append(str(int(hang)+1).zfill(2)+'_'+str(int(lie)+1).zfill(2))

            if (int(hang)-2 >=0 and int(hang)-2 <= 6) and (int(lie)-2 >=0 and int(lie)-2 <= 6):
                two.append(str(int(hang)-2).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))
            if (int(hang)-2 >=0 and int(hang)-2 <= 6) and (int(lie) >=0 and int(lie) <= 6):
                two.append(str(int(hang)-2).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)-2 >=0 and int(hang)-2 <= 6) and (int(lie)+2 >=0 and int(lie)+2 <= 6):
                two.append(str(int(hang)-2).zfill(2)+'_'+str(int(lie)+2).zfill(2))
            if (int(hang) >=0 and int(hang) <= 6) and (int(lie)-2 >=0 and int(lie)-2 <= 6):
                two.append(str(int(hang)).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))

                # two.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang) >=0 and int(hang) <= 6) and (int(lie)+2 >=0 and int(lie)+2 <= 6):
                two.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+2).zfill(2))
            if (int(hang)+2 >=0 and int(hang)+2 <= 6) and (int(lie)-2 >=0 and int(lie)-2 <= 6):
                two.append(str(int(hang)+2).zfill(2)+'_'+str(int(int(lie))-2).zfill(2))
            if (int(hang)+2 >=0 and int(hang)+2 <= 6) and (int(lie) >=0 and int(lie) <= 6):
                two.append(str(int(hang)+2).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)+2 >=0 and int(hang)+2 <= 6) and (int(lie)+2 >=0 and int(lie)+2 <= 6):
                two.append(str(int(hang)+2).zfill(2)+'_'+str(int(lie)+2).zfill(2))
            if (int(hang)-3 >=0 and int(hang)-3 <= 6) and (int(lie)-3 >=0 and int(lie)-3 <= 6):
                three.append(str(int(hang)-3).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            if (int(hang)-3 >=0 and int(hang)-3 <= 6) and (int(lie) >=0 and int(lie) <= 6):
                three.append(str(int(hang)-3).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)-3 >=0 and int(hang)-3 <= 6) and (int(lie)+3 >=0 and int(lie)+3 <= 6):
                three.append(str(int(hang)-3).zfill(2)+'_'+str(int(lie)+3).zfill(2))
            if (int(hang) >=0 and int(hang) <= 6) and (int(lie)-3 >=0 and int(lie)-3 <= 6):
                three.append(str(int(hang)).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            if (int(hang) >=0 and int(hang) <= 6) and (int(lie)+3 >=0 and int(lie)+3 <= 6):
                # three.append(str(int(hang)).zfill(2)+'_'+str(int(lie)).zfill(2))
                three.append(str(int(hang)).zfill(2)+'_'+str(int(lie)+3).zfill(2))
            if (int(hang)+3 >=0 and int(hang)+3 <= 6) and (int(lie)-3 >=0 and int(lie)-3 <= 6):
                three.append(str(int(hang)+3).zfill(2)+'_'+str(int(int(lie))-3).zfill(2))
            if (int(hang)+3 >=0 and int(hang)+3 <= 6) and (int(lie) >=0 and int(lie) <= 6):
                three.append(str(int(hang)+3).zfill(2)+'_'+str(int(lie)).zfill(2))
            if (int(hang)+3 >=0 and int(hang)+3 <= 6) and (int(lie)+3 >=0 and int(lie)+3 <= 6):
                three.append(str(int(hang)+3).zfill(2)+'_'+str(int(lie)+3).zfill(2))

            # print(view,one)

            for idx in range(0, frameNum-2,1):
                groups = ''
                print(os.path.join(inputdir, vid, view, str(idx).zfill(3)+'.png'))

                print('length of one', len(one))
                print('length of two', len(two))
                print('length of three', len(three))
                ## first frame
                # center view
                groups += os.path.join(inputdir, vid, view, str(idx).zfill(3)+'.png') + '|'
                # 1x view
                groups += os.path.join(inputdir, vid, one[0], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[1], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[2], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[3], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[4], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[5], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[6], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[7], str(idx).zfill(3)+'.png') + '|'
                # 2x view
                groups += os.path.join(inputdir, vid, two[0], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[1], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[2], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[3], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[4], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[5], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[6], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[7], str(idx).zfill(3)+'.png') + '|'
                # 4x view
                groups += os.path.join(inputdir, vid, three[0], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, three[1], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, three[2], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, three[3], str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, three[4], str(idx).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[5], str(idx).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[6], str(idx).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[7], str(idx).zfill(3)+'.png') + '|'

                ## third frame
                # center view
                groups += os.path.join(inputdir, vid, view, str(idx+2).zfill(3)+'.png') + '|'
                # 1x view
                groups += os.path.join(inputdir, vid, one[0], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[1], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[2], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[3], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[4], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[5], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[6], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, one[7], str(idx+2).zfill(3)+'.png') + '|'
                # 2x view
                groups += os.path.join(inputdir, vid, two[0], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[1], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[2], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[3], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[4], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[5], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[6], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, two[7], str(idx+2).zfill(3)+'.png') + '|'
                # 4x view
                groups += os.path.join(inputdir, vid, three[0], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, three[1], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, three[2], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, three[3], str(idx+2).zfill(3)+'.png') + '|'
                groups += os.path.join(inputdir, vid, three[4], str(idx+2).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[5], str(idx+2).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[6], str(idx+2).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[7], str(idx+2).zfill(3)+'.png') + '|'

                ## second frame
                # center view
                groups += os.path.join(targetdir, vid, view, str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(targetdir, vid, view, str(idx+1).zfill(3)+'.png') + '|'
                groups += os.path.join(targetdir, vid, view, str(idx+2).zfill(3)+'.png') 

                print(groups)
                with open(os.path.join(outputdir, 'TestGroupsCondition2.txt'), 'a') as f:
                    f.write(groups + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/disk3/zeyuxData/LFVideo/hybrid/BIx4', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/disk3/zeyuxData/LFVideo/hybrid/Crop', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/disk3/zeyuxData/LFVideo/hybrid/LFSTVSRGroups', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()