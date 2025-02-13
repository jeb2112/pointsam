import numpy as np
import skimage
from skimage.io import imread,imsave
from skimage.transform import resize
import nibabel as nb
import glob
import copy
import os
import time
import pickle
import random
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone, timedelta
import gc
import cc3d
from torch.utils.data import Dataset

from pc_sam.datasets.transforms import normalize_points

class PromptType:  
    CONTROL_POINTS = "pts"  
    BOUNDING_BOX = "bbox"

class pointSAMDataset(Dataset):  
    def __init__(  
        self,   
        datadir,
        dataset,   
        transform = None,
        prompt_type = PromptType.CONTROL_POINTS,  
        perturbation = 0,
        padding = 3,  
        image_size = (1024, 1024),  
        mask_size = (256, 256),  
    ):  
        # Asign all values to self  
        self.datadir = datadir
        self.dataset = dataset
        self.prompt_type = prompt_type
        self.image_size = image_size
        self.mask_size = mask_size

        # general image processing
        self.transform = transform

        # This is only to be used for BOUNDING_BOX prompts.
        self.perturbation = perturbation
        self.padding = padding

    def __len__(self):  
        return len(self.dataset)

    def __getitem__(self, idx):  

        # images are duplicated for multiple lesions, need to avoid re-loading from disk
        # but if images have been shuffled this isn't possible
        idxdir,ifile = os.path.split(self.dataset[idx]['image'])
        lbldir,lfile = os.path.split(self.dataset[idx]['label'])
        time1 = time.time()
        input_image,affine = self.loadnifti(ifile,idxdir)
        label_image,affine = self.loadnifti(lfile,lbldir)
        etime1 = time.time() - time1
        if False:
            print('file read time = {:.2f}'.format(etime1))

        label_image = label_image.astype(bool)

        # some general processing could done here explicitly since no longer using huggingface processor
        # initially all that is being implemented is the 3d crop to the locality of the prompt
        # not sure if the crop can be implemented with torchvision transforms and the cfg yaml, becuase
        # it requires the current point prompt, so it will be implemented internally. 
        # but other more generic transforms could be added here later.
        if False:
            if self.transform is not None:
                input_image1 = self.transform(input_image)
        else:
            input_image1 = input_image

        # tile for rgb. for now there is no actual colorization
        if len(np.shape(input_image1)) == 3:
            input_image1 = np.tile(input_image1[:,:,:,np.newaxis],(1,1,1,3))
            
        # subdir 'prompts' is hard-coded here
        # currently bounding box prompts are being stored as mask images, but this should be changed to be json or other text.
        # bbox not implemented yet
        if self.prompt_type == PromptType.BOUNDING_BOX:
            ifile = glob.glob(os.path.join(self.datadir,'prompts','img_' + str(idx).zfill(5) + '_case_*_slice_???.png'))[0]
            # again, masks prepared by viewer code are currently rgba, float. 
            ground_truth_mask = imread(ifile,as_gray=True).astype('uint8') * 255
            # note. skimage resize is one of the many resize algorithms that has an easy option for
            # nearest neighbour. so many others do not.
            ground_truth_mask = skimage.transform.resize(ground_truth_mask,self.mask_size,order=0)
            inputs = self._getitem_bbox(input_image1, ground_truth_mask)
            # this is a dummy mask which is just an image of the bbox
            inputs["ground_truth_mask"] = ground_truth_mask  
        # working on controlpoints first
        # control point prompts are stored as json dict
        elif self.prompt_type == PromptType.CONTROL_POINTS:
            try:
                ifile = glob.glob(os.path.join(self.datadir,'prompts','pts_' + str(idx).zfill(5) + '_case_*_slice_???.json'))[0]
                with open(ifile,'r') as fp:
                    ground_truth_pts = json.load(fp)
            except IndexError as e:
                # no ground truth control points use auto-generated
                ground_truth_pts = None
            try:
                inputs = self._getitem_ctrlpts(input_image1, ground_truth_mask=label_image,ground_truth_pts=ground_truth_pts)  
            except ValueError:
                with open(os.path.join(os.path.expanduser('~'),'dump'+str(idx)+'.pkl'),'wb') as fp:
                    pickle.dump((input_image1,label_image,idx),fp)
                self.writenifti(np.max(input_image1,axis=3),os.path.join(os.path.expanduser('~'),'img'+str(idx)+'.nii'))
                self.writenifti(label_image,os.path.join(os.path.expanduser('~'),'lbl'+str(idx)+'.nii'),type='uint8')

        # debug plotting
        if False:

            # note matplotlib 3.9.0 vs code 1.92.2 plts stopped showing in debug console,
            # until using plt.show(block=True), which is the default corresponding to
            # plt.show(), but neither plt.show() nor plt.show(block=False work
            if self.prompt_type == PromptType.BOUNDING_BOX:
                check_image = np.copy(inputs['pixel_values'])[0]
                bbox = np.array(inputs['input_boxes'][0]).astype('int')
                check_mask = np.copy(ground_truth_mask).astype('uint8')

                if True: # additionally save to file. 
                    # 32 bit tiffs
                    if False:
                        vmax = np.max(check_image)
                        rr,cc = skimage.draw.rectangle_perimeter(bbox[1::-1],end=bbox[:1:-1],shape=check_image.shape)
                        check_image[rr,cc] = vmax
                        check_mask = resize(check_mask,np.shape(check_image),order=0).astype('float32') * vmax
                        check_comb = np.concatenate((check_image,check_mask),axis=1)
                        ofile = os.path.join(self.datadir,'datacheck','comb_' + str(idx).zfill(4) + '.tiff')
                        cv2.imwrite(ofile,check_comb)
                    else:
                    # 8 bit pngs
                        check_image2 = (skimage.transform.resize(input_image,np.shape(check_image))*255).astype('uint8')
                        check_mask = resize(check_mask,np.shape(check_image2),order=0).astype('uint8') * 255
                        rr,cc = skimage.draw.rectangle_perimeter(bbox[1::-1],end=bbox[:1:-1],shape=check_mask.shape)
                        check_image2[rr,cc] = 255
                        check_comb = np.concatenate((check_image2,check_mask),axis=1).astype('uint8')
                        ofile = os.path.join(self.datadir,'datacheck','comb_' + str(idx).zfill(4) + '.png')
                        imsave(ofile,check_comb)

            elif self.prompt_type == PromptType.CONTROL_POINTS:
                check_image = np.moveaxis(np.array(inputs['pixel_values'][0].cpu()),0,-1)
                pts = np.array(inputs['input_points'][0]).astype('int')

                # 8 bit pngs
                # check_image2 = (skimage.transform.resize(input_image,np.shape(check_image))*255).astype('uint8')
                for p in pts[0]:
                    # check_image2[p[1]-4:p[1]+4,p[0]-4:p[0]+4] = np.array([255,0,0,255],dtype='uint8')
                    check_image[p[1]-4:p[1]+4,p[0]-4:p[0]+4] = np.array([255,0,0],dtype='uint8')
                ofile = os.path.join(self.datadir,'datacheck','comb_' + str(idx).zfill(4) + '.png')
                plt.figure(8)
                plt.imshow(check_image)
                plt.show(block=False)
                # imsave(ofile,check_image2)

        return inputs
    
    # control points option. single point positive point, centroid of a mask, or multiple labelled points 
    def generate_input_points(self,mask=None,pts=None,dynamic_distance=True,erode=True):
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

    def _getitem_ctrlpts(self, input_image, ground_truth_mask=None,ground_truth_pts=None,crop=True):  
        # Get control points prompt. See the GitHub for the source  
        # of this function, or replace with your own point selection algorithm.  

        # for now, only coding for a ground truth mask
        if ground_truth_mask is None:
            raise ValueError
        
        img = np.copy(input_image)
        mask = np.copy(ground_truth_mask)
        
        # auto-gen prompt points is also available from p'sam
        input_points, input_labels = self.generate_input_points(  
            pts=ground_truth_pts,
            mask=ground_truth_mask
        )  

        # crop to the locality of the control point prompt. this is needed
        # for now to limit point cloud size, rather than random sampling, for gpu memory
        # minimum crop must check for some background pixels otherwise the auto-prompt point
        # gen will throw an error
        # currently the crop3d does not check for an out of bounds on any dimension. 
        if crop:
            img,mask,input_points,input_labels = self.crop3d(img,mask,input_points,input_labels)

        # form point cloud
        # 
        points = np.where(np.max(img,axis=3))
        if len(points[0]) == 0:
            raise ValueError
        assert len(points[0]), 'no points detected'
        # coords generally have to be floats as would be the case in arbitrary 3d parts clouds
        # somehow need specifically float32 for torch, not python float=64? 
        xyzcoords = np.array(points).T.astype(np.float32)
        xyzcoords = normalize_points(xyzcoords)
        # form rgb
        rgbfeatures = img[points]
        rgbfeatures /= np.max(rgbfeatures)
        rgbfeatures = rgbfeatures.astype(np.float32)

        if ground_truth_mask is not None:
            gt_mask = mask[points[:3]].astype(bool)
            # may need a dummy singleton dim here, because multiple masks are supported in the psam trainer
            gt_mask = np.tile(gt_mask[np.newaxis,:],(1,1))

        # Prepare the image and prompt for the model.  
        # not using sam processor but may need similar processing
        if False:
            inputs = self.processor(  
                input_image,  
                input_points=input_points,  
                input_labels=input_labels,  
                return_tensors="pt"  
            )
        else:
            inputs = {'coords':xyzcoords, 'features':rgbfeatures , 'gt_masks':gt_mask}

        # pre-Remove batch dimension because it gets added back later.  
        # inputs = {k: v for k, v in inputs.items()}
        if False:
            for k in inputs.keys(): 
                inputs[k] = inputs[k].squeeze(0)

        return inputs

    # bbox option
    def get_input_bbox(self,mask, perturbation=0):
        # Find minimum mask bounding all included mask points.
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add padding
        ydim,xdim = np.shape(mask)
        x_min = max(0,x_min - self.padding)
        y_min = max(0,y_min - self.padding)
        x_max = min(xdim-1,x_max + self.padding)
        y_max = min(ydim-1,y_max + self.padding)

        if perturbation:  # Add perturbation to bounding box coordinates.
            H, W = mask.shape
            x_min = max(0, x_min + np.random.randint(-perturbation, perturbation))
            x_max = min(W, x_max + np.random.randint(-perturbation, perturbation))
            y_min = max(0, y_min + np.random.randint(-perturbation, perturbation))
            y_max = min(H, y_max + np.random.randint(-perturbation, perturbation))

        bbox = [x_min, y_min, x_max, y_max]
        
        return bbox

    def _getitem_bbox(self, input_image, ground_truth_mask):  
        # Get bounding box prompt.  
        bbox = self.get_input_bbox(ground_truth_mask, perturbation=self.perturbation)

        # Prepare the image and prompt for the model.  
        inputs = self.processor(input_image, input_boxes=[[bbox]], return_tensors="pt")  
        inputs = {k: v.squeeze(0) for k, v in inputs.items()} # Remove batch dimension which the processor adds by default.

        return inputs
    

    # load a single nifti file
    def loadnifti(self,t1_file,dir,type=None):
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
    def writenifti(self,img_arr,filename,header=None,norm=False,type='float64',affine=None):
        img_arr_cp = copy.deepcopy(img_arr)
        if norm:
            img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
        # using nibabel nifti coordinates
        img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
        nb.save(img_nb,filename)
        if True:
            os.system('gzip --force "{}"'.format(filename))


    # select a single lesion from a BraTS segmentation
    # selection will start from cc3d idx #1, unless idx arg is provided > 1
    # index of selected mask is returned as second return arg
    def find_lesion(self,img_arr_seg,idx=1):
        # taking combined tumor here.
        mask = ((img_arr_seg > 0)).astype('uint8')*255
        CC_mask = cc3d.connected_components(mask,connectivity=4)
        nCC_mask = len(np.unique(CC_mask))
        # select a component. occasionally, the mask of a single large and 
        # complicated lesion will have a few disconnected pixels in a particular slice,
        # so want to check for unusually small selections. 
        if nCC_mask > 2:
            cset = list(range(idx,nCC_mask+1))
            try:
                for isel,sel in enumerate(cset):
                    mask = (CC_mask == sel).astype('uint8') * 255
                    if len(np.where(mask==255)[0]) > 9:
                        return mask,isel
                    if isel == nCC_mask-1:
                        print('No suitable mask found, skipping...')
                        return None
            except ValueError:
                return None
            # sel = 1
        elif nCC_mask == 2:
            sel = 1
            mask = (CC_mask == sel).astype('uint8') * 255
            if len(np.where(mask==255)[0]) < 10:
                print('No suitable mask found, skipping...')
                return None
        elif nCC_mask == 1:
            print('No suitable mask found, skipping...')
            return None
        
    # default cropdim for approx 10000 points
    def crop3d(self,img_arr,mask_arr,pts,lbls,ndim=[20,20,20],pdim=[20,20,20]):
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