

import os
import shutil


ROOT_DIR = '/data/datasets/UCF101_DVS/UCF101_DVS'
train_path = os.path.join(ROOT_DIR, 'train')
val_path = os.path.join(ROOT_DIR, 'val')
val_fname = 'testlist01.txt'

cls_path = os.listdir(train_path)

if not os.path.exists(val_path):
    os.mkdir(val_path)
    for cls_name in cls_path:
        os.mkdir(os.path.join(val_path, cls_name))

f = open(val_fname, 'r')

for fname in f.readlines():
    fname = fname[:-4] + 'mat'
    fname.replace('Billards', 'Billiards')
    src = os.path.join(train_path, fname)
    dst = os.path.join(val_path, fname)
    try:
        shutil.move(src, dst)
    except:
        print('[Warning] Cannot find {}.'.format(src))
    print('[Moving] {} -> {}.'.format(src, dst))
