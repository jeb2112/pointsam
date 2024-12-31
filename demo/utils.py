import numpy as np
import nibabel as nb
import os
import copy


def load_ply(filename):
    with open(filename, "r") as rf:
        while True:
            try:
                line = rf.readline()
            except:
                raise NotImplementedError
            if "end_header" in line:
                break
            if "element vertex" in line:
                arr = line.split()
                num_of_points = int(arr[2])

        # print("%d points in ply file" %num_of_points)
        points = np.zeros([num_of_points, 6])
        for i in range(points.shape[0]):
            point = rf.readline().split()
            assert len(point) == 6
            points[i][0] = float(point[0])
            points[i][1] = float(point[1])
            points[i][2] = float(point[2])
            points[i][3] = float(point[3])
            points[i][4] = float(point[4])
            points[i][5] = float(point[5])
    rf.close()
    del rf
    return points

# load a single nifti file
def loadnifti(t1_file,dir,type='uint8'):
    img_arr_t1 = None

    try:
        img_nb_t1 = nb.load(os.path.join(dir,t1_file))
    except IOError as e:
        print('Can\'t import {}'.format(t1_file))
        return None,None
    affine = copy.copy(img_nb_t1.affine)
    img_arr_t1 = np.array(img_nb_t1.dataobj)
    # modify the affine to match itksnap convention
    for i in range(2):
        if affine[i,i] > 0:
            affine[i,3] += (img_nb_t1.shape[i]-1) * affine[i,i]
            affine[i,i] = -1*(affine[i,i])
            # will use flips for now for speed
            img_arr_t1 = np.flip(img_arr_t1,axis=i)
    # this takes too long and requires re-masking
    if False:
        img_nb_t1 = nb.processing.resample_from_to(img_nb_t1,(img_nb_t1.shape,affine))

    nb_header = img_nb_t1.header.copy()
    # nibabel convention will be transposed to sitk convention
    if False:
        img_arr_t1 = np.transpose(img_arr_t1,axes=(2,1,0))
    if type is not None:
        img_arr_t1 = img_arr_t1.astype(type)

    return img_arr_t1,affine
