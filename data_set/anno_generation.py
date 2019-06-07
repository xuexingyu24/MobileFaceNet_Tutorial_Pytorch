#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:36:12 2019
Generate the annotation file for webface_align_112

@author: AIRocker
"""
import os
import argparse

parser = argparse.ArgumentParser(description='Annotation file generator')
parser.add_argument('--root', type=str, default='faces_emore_images', help='image data folder path')
parser.add_argument('--file', type=str, default='faces_emore_images/faces_emore_align_112.txt', help='anno file path')
args = parser.parse_args()

imgdir = args.root
list_txt_file = args.file
docs = [f for f in os.listdir(imgdir) if not f.startswith('.')]
docs.sort()

label = 0
for name in docs:

    print('writing name:', name)
    image_folder = imgdir+'/'+name
    L=''
    files = [f for f in os.listdir(image_folder) if not f.startswith('.')]
    files.sort()

    for file in files:
        txt_name = os.path.join(name, file)

        with open(list_txt_file, 'a') as f:
            f.write(txt_name+' '+str(label)+'\n')
        f.close()

    label+= 1

print('writing finished')


