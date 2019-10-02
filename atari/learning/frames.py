import numpy as np
import torch
from skimage.transform import rescale, resize, downscale_local_mean

def gray_conversion(image):
    image = np.array(image)
    gray_value = 0.07 * image[2,:,:] + 0.72 * image[1,:,:] + 0.21 * image[0,:,:]
    return gray_value.astype(np.uint8)

def add_frame(input_frame,frames):
    f = gray_conversion(input_frame)
    f = resize(f, (110, 84),anti_aliasing=False)
    f = np.array(f)

    if len(frames) < 3:
        frames.append(f)
    else:
        frames[2] = frames[1]
        frames[1] = frames[0]
        frames[0] = f

def concat_frames(frames):
    return torch.from_numpy(np.concatenate((np.array([frames[2]]),np.array([frames[1]]),np.array([frames[0]])),axis=0))

def concat_next_frames(next_f,frames):
    next_frames = frames.copy()
    add_frame(next_f,next_frames)
    return concat_frames(next_frames)
