import numpy as np
import skimage
from skimage.io import imread,imsave
from skimage.transform import resize
import monai
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone, timedelta
import gc
from torch.utils.data import Dataset

class PromptType:  
    CONTROL_POINTS = "pts"  
    BOUNDING_BOX = "bbox"

class SAMDataset(Dataset):  
    def __init__(  
        self,   
        datadir,   
        processor,   
        prompt_type = PromptType.BOUNDING_BOX,  
        num_positive = 3,  
        num_negative = 0,  
        erode = True,  
        multi_mask = "mean",  
        perturbation = 0,
        padding = 3,  
        image_size = (1024, 1024),  
        mask_size = (256, 256),  
    ):  
        # Asign all values to self  
        # self.dataset = dataset
        self.datadir = datadir
        self.dataset = os.listdir(os.path.join(self.datadir,'images'))
        self.processor = processor
        self.prompt_type = prompt_type
        self.image_size = image_size
        self.mask_size = mask_size

        # These should only be used for CONTROL_POINTS prompts.
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.erode = erode
        self.multi_mask = multi_mask

        # This is only to be used for BOUNDING_BOX prompts.
        self.perturbation = perturbation
        self.padding = padding

    def __len__(self):  
        return len(self.dataset)

    def __getitem__(self, idx):  

        if False: # former syntax
            ifile = glob.glob(os.path.join(self.datadir,'images','img_' + str(idx).zfill(5) + '_case_???_slice_???.png'))[0]
        else: # new syntax
            ifile = glob.glob(os.path.join(self.datadir,'images','img_' + str(idx).zfill(5) + '_case_*_slice_???.png'))[0]

        input_image = imread(ifile)
        input_image1 = skimage.transform.resize(input_image,self.image_size)
        # huggingface transformers SamImageProcessor claims to support channel dim
        # first or last and infer from shape of data, but last didn't work and 
        # first did. 
        input_image1 = np.tile(input_image1,(3,1,1))
        ifile = glob.glob(os.path.join(self.datadir,'labels','img_' + str(idx).zfill(5) + '_case_*_slice_???.png'))[0]
        ground_truth_mask = imread(ifile)
        ground_truth_mask = skimage.transform.resize(ground_truth_mask,self.mask_size)

        if self.prompt_type == PromptType.CONTROL_POINTS:  
            inputs = self._getitem_ctrlpts(input_image1, ground_truth_mask)  
        elif self.prompt_type == PromptType.BOUNDING_BOX:  
            inputs = self._getitem_bbox(input_image1, ground_truth_mask)

        inputs["ground_truth_mask"] = ground_truth_mask  

        # debug plotting
        if False:
            # note matplotlib 3.9.0 vs code 1.92.2 plts stopped showing in debug console,
            # until using plt.show(block=True), which is the default corresponding to
            # plt.show(), but neither plt.show() nor plt.show(block=False work
            check_image = np.copy(inputs['pixel_values'])[0]
            bbox = np.array(inputs['input_boxes'][0]).astype('int')
            check_mask = np.copy(ground_truth_mask).astype('uint8')

            if False: # additionally save to file. 
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

        return inputs
    
    # won't do control points to begin with
    if False: 
        def _getitem_ctrlpts(self, input_image, ground_truth_mask):  
            # Get control points prompt. See the GitHub for the source  
            # of this function, or replace with your own point selection algorithm.  
            input_points, input_labels = generate_input_points(  
                num_positive=self.num_positive,  
                num_negative=self.num_negative,  
                mask=ground_truth_mask,  
                dynamic_distance=True,  
                erode=self.erode,  
            )  
            input_points = input_points.astype(float).tolist()  
            input_labels = input_labels.tolist()  
            input_labels = [[x] for x in input_labels]

            # Prepare the image and prompt for the model.  
            inputs = self.processor(  
                input_image,  
                input_points=input_points,  
                input_labels=input_labels,  
                return_tensors="pt"  
            )

            # Remove batch dimension which the processor adds by default.  
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}  
            inputs["input_labels"] = inputs["input_labels"].squeeze(1)

            return inputs

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

        if True: # original code
            bbox = [x_min, y_min, x_max, y_max]
        else: # apparent correction for sam bbox convention is needed
            bbox = [y_min, x_min, y_max, x_max]
        return bbox

    def _getitem_bbox(self, input_image, ground_truth_mask):  
        # Get bounding box prompt.  
        bbox = self.get_input_bbox(ground_truth_mask, perturbation=self.perturbation)

        # Prepare the image and prompt for the model.  
        inputs = self.processor(input_image, input_boxes=[[bbox]], return_tensors="pt")  
        inputs = {k: v.squeeze(0) for k, v in inputs.items()} # Remove batch dimension which the processor adds by default.

        return inputs