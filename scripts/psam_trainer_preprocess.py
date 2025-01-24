# script selects individual lesions from each BraTS 2024 volume
# for use in fine-tuning the pointsam
# optionally also auto-creates a point prompt, but this could also 
# be a form in augmentation in real-time

import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
import nibabel as nb
import shutil
import copy
import matplotlib.pyplot as plt
from skimage.io import imsave
import cc3d
import random

# load a single nifti file
def loadnifti(t1_file,dir,type=None):
    img_arr_t1 = None
    try:
        img_nb_t1 = nb.load(os.path.join(dir,t1_file))
    except IOError as e:
        print('Can\'t import {}'.format(t1_file))
        return None,None
    nb_header = img_nb_t1.header.copy()
    # nibabel convention will be transposed to sitk convention
    img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
    if type is not None:
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine
    return img_arr_t1,affine

# write a single nifti file. use uint8 for masks 
def writenifti(img_arr,filename,header=None,norm=False,type='float64',affine=None):
    img_arr_cp = copy.deepcopy(img_arr)
    if norm:
        img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
    # using nibabel nifti coordinates
    img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
    nb.save(img_nb,filename)
    if True:
        os.system('gzip --force "{}"'.format(filename))

def load_dataset(cpath,type='t1c'):
    ifiles = os.listdir(cpath)
    t1c_file = [f for f in ifiles if type in f][0]
    seg_file  = [f for f in ifiles if 'seg' in f][0]    
    img_arr_t1c,_ = loadnifti(t1c_file,cpath,type='float64')
    img_arr_seg,_ = loadnifti(seg_file,cpath,type='uint8')

    return img_arr_t1c,img_arr_seg

# main

if os.name == 'posix':
    datadir = "/media/jbishop/WD4/brainmets/brats2024/raw"

with open(os.path.join(datadir,'BraTS_2024_testcases.txt'),'r') as fp:
    tlist = [c for c in fp.readlines()]
testcases = [re.search('([0-9]{5}-[0-9]{3})',c).group(1) for c in tlist]

# note. still have additional brats validation cases, without ground truth, in 'validation' dir
casedirs = os.listdir(os.path.join(datadir,'training'))
casenumbers = [re.search('([0-9]{5}-[0-9]{3})',c).group(1) for c in casedirs]
traincases = [c for c in casenumbers if c not in testcases]
traincasedirs = [d for c,d in zip(casenumbers,casedirs) if c in traincases]
cases_train,cases_val,casedirs_train,casedirs_val = train_test_split(traincases,traincasedirs,test_size=0.2,random_state=4)
cases_train = sorted(cases_train)
cases_val = sorted(cases_val)
casedirs_train = sorted(casedirs_train)
casedirs_val = sorted(casedirs_val)

casedirs_test = sorted([d.split('.')[0] for d in tlist])
cases_test = sorted(testcases)

tv_path = ['psam_training','psam_validation','psam_test']
if True: # for training images
    tv_set = {'psam_training':{'casedirs':casedirs_train,'cases':cases_train},
          'psam_validation':{'casedirs':casedirs_val,'cases':cases_val}}
else: # for test images
    tv_set = {'psam_test':{'casedirs':casedirs_test,'cases':cases_test}}

# t1c,t2f, a tag to pick filename using brats nomenclature
img_type = 't2f'

for tv in tv_set.keys():

    try:
        shutil.rmtree(os.path.join(datadir,tv))
    except FileNotFoundError:
        pass
    os.makedirs(os.path.join(datadir,tv,'images'),exist_ok=True)
    os.makedirs(os.path.join(datadir,tv,'labels'),exist_ok=True)
    os.makedirs(os.path.join(datadir,tv,'prompts'),exist_ok=True)

    img_idx = 0
    for case_idx,C in enumerate(tv_set[tv]['casedirs']):
        case = tv_set[tv]['cases'][case_idx]
        if False: #debugging
            if case != '00019-000':
                continue
        print('case = {}'.format(case))
        cpath = os.path.join(datadir,'training',C)
        opath = os.path.join(datadir,tv)
        img_arr_t1c,img_arr_seg = load_dataset(cpath,type=img_type)
        # converting to uint8
        img_arr_t1c = (img_arr_t1c / np.max(img_arr_t1c) * 255).astype('uint8')

        if np.max(img_arr_t1c) == 0:
            print('Invalid image, skipping...')
            continue

        if np.shape(img_arr_t1c)[1] != np.shape(img_arr_t1c)[2]:
            print('Rectangular matrix in case {}, skipping.'.format(C))
            continue
        
        # take entire tumor mask for now
        mask = ((img_arr_seg > 0)).astype('uint8')*255
        CC_mask = cc3d.connected_components(mask,connectivity=6)
        nCC_mask = len(np.unique(CC_mask))
        # select a component. occasionally, the mask of a single large and 
        # complicated lesion will have a few disconnected pixels in a particular slice,
        # so want to check for unusually small selections. 
        if nCC_mask > 1:
            lset = list(range(1,nCC_mask+1))
            try:
                for isel,sel in enumerate(lset):
                    print('lesion #{}'.format(sel))
                    mask = (CC_mask == sel).astype('uint8') * 255
                    if len(np.where(mask==255)[0]) > 9:

                        ofile = 'img_' + str(img_idx).zfill(5) + '_case_' + case + '.nii'
                        writenifti(img_arr_t1c,os.path.join(opath,'images',ofile))

                        ofile = 'lbl_' + str(img_idx).zfill(5) + '_case_' + case + '.nii' 
                        mask = (CC_mask == sel).astype('uint8') * 255
                        writenifti(mask,os.path.join(opath,'labels',ofile))

                        img_idx += 1

            except ValueError:
                continue

        elif nCC_mask == 1:
            print('No suitable mask found, skipping...')
            continue


