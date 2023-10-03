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

vidsAll = ['ambushfight_1','ambushfight_2','ambushfight_3','ambushfight_4',\
            'ambushfight_5','ambushfight_6','bamboo_1','bamboo_2',\
            'bamboo_3','chickenrun_1','chickenrun_2','chickenrun_3',\
            'foggyrocks_1','foggyrocks_2','questbegins_1','shaman_1',\
            'shaman_2','shaman_3','shaman_b_1','shaman_b_2',\
            'thebigfight_1','thebigfight_2','thebigfight_3','thebigfight_4']

vidsTrain = ['ambushfight_1','ambushfight_2','ambushfight_3',\
                'ambushfight_4','ambushfight_5','bamboo_1',\
                'bamboo_2','chickenrun_1','chickenrun_2',\
                'foggyrocks_1','questbegins_1','shaman_1',\
                'shaman_2','shaman_b_1','shaman_b_2',\
                'thebigfight_1','thebigfight_2','thebigfight_3']

vidsValid = ['ambushfight_6','chickenrun_3','thebigfight_4']

vidsTest = ['bamboo_3','foggyrocks_2','thebigfight_4']

condition3 = ['02_02','02_06',\
              '06_02','06_06']

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
        for view in condition3:
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
            print('length of one', len(one))
            print('length of two', len(two))
            print('length of three', len(three))
            
            for idx in range(0, frameNum-2,1):
                groups = ''
                print(os.path.join(inputdir, vid, view, str(idx).zfill(3)+'.png'))

                
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
                # groups += os.path.join(inputdir, vid, three[3], str(idx).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[4], str(idx).zfill(3)+'.png') + '|'
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
                # groups += os.path.join(inputdir, vid, three[3], str(idx+2).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[4], str(idx+2).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[5], str(idx+2).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[6], str(idx+2).zfill(3)+'.png') + '|'
                # groups += os.path.join(inputdir, vid, three[7], str(idx+2).zfill(3)+'.png') + '|'

                ## second frame
                # center view
                groups += os.path.join(targetdir, vid, view, str(idx).zfill(3)+'.png') + '|'
                groups += os.path.join(targetdir, vid, view, str(idx+1).zfill(3)+'.png') + '|'
                groups += os.path.join(targetdir, vid, view, str(idx+2).zfill(3)+'.png') 

                print(groups)
                with open(os.path.join(outputdir, 'TestGroups.txt'), 'a') as f:
                    f.write(groups + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/disk3/zeyuxData/LFVideo/Sintel_LFV_9x9/BIx4', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/disk3/zeyuxData/LFVideo/Sintel_LFV_9x9/Original', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/disk3/zeyuxData/LFVideo/Sintel_LFV_9x9/LFSTVSRGroupsCondition3', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()