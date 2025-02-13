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


# control points option. single point positive point, centroid of a mask, or multiple labelled points 
def generate_input_points(mask=None,pts=None,dynamic_distance=True,erode=True):
    input_points = []
    if mask is not None:
        slice,row,col = map(int,np.mean(np.where(mask),axis=1))
        input_points.append([slice,row,col]) 
        input_points = np.array(input_points)
        input_labels = np.array([1])
    elif pts is not None:
        # add dummy dimension for batch
        input_points = np.array([(x,y) for x,y in zip(pts['x'],pts['y'])])
        # dummy batch dimension? or samprocessor adds automatically
        # input_labels = np.expand_dims(np.array(pts['fg']),0)
        input_labels = np.array(pts['fg'])

    return  input_points,input_labels  


# default cropdim for approx 10000 points
def crop3d(img_arr,mask_arr,pts,lbls,ndim=[20,20,20],pdim=[20,20,20]):
    # centroid of foreground prompt points
    # this might cut off some prompt points, fg or bg, needs to be improved.

    fg_pts = np.array([p for p,l in zip(pts,lbls) if l==1])
    cpt = np.round(np.mean(fg_pts,axis=0)).astype(int)

    dim = img_arr.shape
    for i in range(3):
        if cpt[i]-ndim[i] < 0:
            ndim[i] = cpt[i]
            pdim[i] += ndim[i]-cpt[i]
        elif cpt[i]+pdim[i] >= dim[i]:
            pdim[i] = dim[i]-cpt[i]-1
            ndim[i] += (pdim[i] - (dim[i]-cpt[i]-1))

    crop_img_arr = img_arr[cpt[0]-ndim[0]:cpt[0]+pdim[0],
                        cpt[1]-ndim[1]:cpt[1]+pdim[1],
                        cpt[2]-ndim[2]:cpt[2]+pdim[2]]
    crop_mask_arr = mask_arr[cpt[0]-ndim[0]:cpt[0]+pdim[0],
                        cpt[1]-ndim[1]:cpt[1]+pdim[1],
                        cpt[2]-ndim[2]:cpt[2]+pdim[2]]
    # adjust the prompts accordingly
    crop_pts = []
    crop_lbls = []
    if False: # not using these for now
        for pt,lbl in zip(pts,lbls):
            if all(pt-cpt) >= 0:
                crop_pts.append(pt-cpt+cropdim)
                crop_lbls.append(lbl)
    return crop_img_arr,crop_mask_arr,crop_pts,crop_lbls

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
    skip_idx = 0

    for case_idx,C in enumerate(tv_set[tv]['casedirs']):
        case = tv_set[tv]['cases'][case_idx]
        if False: #debugging
            if case != '00019-000':
                continue
        print('case = {}'.format(case))
        cpath = os.path.join(datadir,'training',C)
        opath = os.path.join(datadir,tv)
        img_arr,img_arr_seg = load_dataset(cpath,type=img_type)
        # converting to uint8
        img_arr = (img_arr / np.max(img_arr) * 255).astype('uint8')

        if np.max(img_arr) == 0:
            print('Invalid image, skipping...')
            continue

        if np.shape(img_arr)[1] != np.shape(img_arr)[2]:
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
                    print('lesion #{}'.format(sel),end='')
                    mask = (CC_mask == sel).astype('uint8') * 255
                    npixel = len(np.where(mask==255)[0])
                    if npixel > 99 and npixel < 64000:

                        # precrop to the locality of the control point prompt. this the approach
                        # to limit point cloud size for gpu memory, since random down-sampling is not 
                        # appropriate in this context. 
                        # minimum crop must check for some background pixels otherwise the auto-prompt point
                        # gen will throw an error
                        # can also be done in realtime with __getitem__() and for augmenting, but file reads are taking too 
                        # much time so start with this.
                        if True:
                            # auto-gen prompt points is also available from p'sam
                            input_points, input_labels = generate_input_points(  
                                pts=None,
                                mask=mask
                            )  
                            img_arr_c,mask_c,input_points,input_labels = crop3d(img_arr,mask,input_points,input_labels)

                        ofile = 'img_' + str(img_idx).zfill(5) + '_case_' + case + '.nii'
                        writenifti(img_arr_c,os.path.join(opath,'images',ofile))

                        ofile = 'lbl_' + str(img_idx).zfill(5) + '_case_' + case + '.nii' 
                        mask = (CC_mask == sel).astype('uint8') * 255
                        writenifti(mask_c,os.path.join(opath,'labels',ofile))

                        img_idx += 1
                        print('')
                    else:
                        skip_idx += 1
                        print(' skipped')

            except ValueError:
                continue

        elif nCC_mask == 1:
            print('No suitable mask found, skipping...')
            continue


    print('{} lesions, {} skipped'.format(img_idx,skip_idx))