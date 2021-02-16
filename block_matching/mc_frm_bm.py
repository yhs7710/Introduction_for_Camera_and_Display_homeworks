import argparse
import os
import itertools
import math
import sys

import numpy as np
import cv2

import matplotlib.pyplot as plt

block_size = 10
search_range = 3
expand_pixels = 2
expand_pix = block_size>>1

def main():

    # read images
    # earlier_frm_rgb = cv2.imread('/database/IVS/동해과제/other-color-twoframes/other-data/DogDance/frame10.png')
    # real_frm_rgb = cv2.imread('/database/IVS/동해과제/other-gt-interp/other-gt-interp/DogDance/frame10i11.png')
    # later_frm_rgb = cv2.imread('/database/IVS/동해과제/other-color-twoframes/other-data/DogDance/frame11.png')
    earlier_frm_rgb = cv2.imread('/database/IVS/동해과제/test2/frame_image/478.jpg')
    real_frm_rgb = cv2.imread('/database/IVS/동해과제/test2/frame_image/479.jpg')
    later_frm_rgb = cv2.imread('/database/IVS/동해과제/test2/frame_image/480.jpg')
    
    # convert rgb images to grayscale images
    # earlier_frm_gray = cv2.cvtColor(earlier_frm_rgb[0:400,0:1000], cv2.COLOR_RGB2GRAY)
    # real_frm_gray = cv2.cvtColor(real_frm_rgb[0:400,0:1000], cv2.COLOR_RGB2GRAY)
    # later_frm_gray = cv2.cvtColor(later_frm_rgb[0:400,0:1000], cv2.COLOR_RGB2GRAY)
    earlier_frm_gray = cv2.cvtColor(earlier_frm_rgb, 1) #color
    real_frm_gray = cv2.cvtColor(real_frm_rgb, 1) #color
    later_frm_gray = cv2.cvtColor(later_frm_rgb, 1) #color
    
    # save original frames
    cv2.imwrite('1_earlier_frame.png',earlier_frm_gray)
    cv2.imwrite('4_later_frame.png',later_frm_gray)
    cv2.imwrite('2_real_frame.png',real_frm_gray)
    
    # search motion vectors
    print("Computing motion vectors...")
    MVS = Motion_Vector_Searcher(block_size, search_range)
    # predicted_frm, motion_field = MVS.run(earlier_frm_gray, later_frm_gray)
    motion_field = MVS.run(earlier_frm_gray, later_frm_gray)
    
    # # open motion flow
    # with open('out.flo', 'r') as flo:
        # tag = np.fromfile(flo, np.float32, count=1)[0]
        # width = np.fromfile(flo, np.int32, count=1)[0]
        # height = np.fromfile(flo, np.int32, count=1)[0]
        # print('tag', tag, 'width', width, 'height', height)
        # nbands = 2
        # tmp = np.fromfile(flo, np.float32, count= nbands * width * height)
        # flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    # motion_field = np.flip(flow,2)
    
    # Interpolate with BBMC
    print("Interpolating frame with BBMC...")
    BBMC = Basic_Block_Motion_Compensation(block_size)
    predicted_frm_BBMC = BBMC.run(motion_field, earlier_frm_gray, later_frm_gray)
    
    # store predicted frame
    cv2.imwrite('3_predicted_frame_BBMC.png',predicted_frm_BBMC)
    
    # store error image
    error_image_BBMC = abs(np.array(predicted_frm_BBMC[expand_pixels:-expand_pixels, expand_pixels:-expand_pixels], dtype=float) - np.array(real_frm_gray[expand_pixels:-expand_pixels, expand_pixels:-expand_pixels], dtype=float))
    error_image_BBMC = np.array(error_image_BBMC, dtype=np.uint8)
    cv2.imwrite('5_error_image_BBMC.png', error_image_BBMC)

    # Peak Signal-to-Noise Ratio of the predicted image
    mse_BBMC = (np.array(error_image_BBMC, dtype=float) ** 2).mean()
    psnr_BBMC = 10 * math.log10((255 ** 2) / mse_BBMC)
    print('PSNR_BBMC: %s dB' % psnr_BBMC)
    
    
    # Interpolate with SOBMC
    print("Interpolating frame with SOBMC...")
    SOBMC = Simple_Overlapped_Block_Motion_Compensation(block_size)
    predicted_frm_pad_SOBMC = SOBMC.run(motion_field, earlier_frm_gray, later_frm_gray)
    
    # remove padded area
    predicted_frm_SOBMC = predicted_frm_pad_SOBMC[2*expand_pixels:-2*expand_pixels, 2*expand_pixels:-2*expand_pixels]
    
    # store predicted frame
    cv2.imwrite('3_predicted_frame_SOBMC.png',predicted_frm_SOBMC)
    
    # store error image
    error_image_SOBMC = abs(np.array(predicted_frm_SOBMC, dtype=float) - np.array(real_frm_gray[expand_pixels:-expand_pixels, expand_pixels:-expand_pixels], dtype=float))
    error_image_SOBMC = np.array(error_image_SOBMC, dtype=np.uint8)
    cv2.imwrite('5_error_image_SOBMC.png', error_image_SOBMC)

    # Peak Signal-to-Noise Ratio of the predicted image
    mse_SOBMC = (np.array(error_image_SOBMC, dtype=float) ** 2).mean()
    psnr_SOBMC = 10 * math.log10((255 ** 2) / mse_SOBMC)
    print('PSNR_SOBMC: %s dB' % psnr_SOBMC)


    # Interpolate with FOBMC
    print("Interpolating frame with FOBMC...")
    FOBMC = Full_Overlapped_Block_Motion_Compensation(block_size)
    predicted_frm_pad_FOBMC = FOBMC.run(motion_field, earlier_frm_gray, later_frm_gray)
    
    # remove padded area
    predicted_frm_FOBMC = predicted_frm_pad_FOBMC[block_size:-block_size, block_size:-block_size]
    
    # store predicted frame
    cv2.imwrite('3_predicted_frame_FOBMC.png',predicted_frm_FOBMC)
    
    # store error image
    error_image_FOBMC = abs(np.array(predicted_frm_FOBMC, dtype=float) - np.array(real_frm_gray[expand_pix:-expand_pix, expand_pix:-expand_pix], dtype=float))
    error_image_FOBMC = np.array(error_image_FOBMC, dtype=np.uint8)
    cv2.imwrite('5_error_image_FOBMC.png', error_image_FOBMC)

    # Peak Signal-to-Noise Ratio of the predicted image
    mse_FOBMC = (np.array(error_image_FOBMC, dtype=float) ** 2).mean()
    psnr_FOBMC = 10 * math.log10((255 ** 2) / mse_FOBMC)
    print('PSNR_FOBMC: %s dB' % psnr_FOBMC)


    motion_field_x = cv2.flip(motion_field[:,:,1],0)
    motion_field_y = -cv2.flip(motion_field[:,:,0],0)

    plt.figure()
    plt.title('Motion Vectors')
    plt.quiver(motion_field_x, motion_field_y, scale=0.1, units='xy', width=0.1)
    plt.savefig("6_Motion_Vector.png")
    plt.show()

    
class Motion_Vector_Searcher():
    """
    Estimates motion vectors of predicted frame block
    Minimizes the norm of the Displaced Frame Difference
    """

    def __init__(self, block_size, search_range):
        """
        :param block_size: size of block
        :param search_range: range of search in the image
        """
        self.block_size = block_size
        self.search_range = search_range
        
    def run(self, earlier_frame, later_frame):
        """
        :param earlier_frame: Image of earlier frame
        :param later_frame: Image of later frame
        :return: motion vector map of frame to estimate
        """
        block_size = self.block_size
        search_range = self.search_range
        height = earlier_frame.shape[0]
        width = earlier_frame.shape[1]
        height_mf = int(height/block_size)
        width_mf = int(width/block_size)
        
        # # predicted frame. anchor_frame is predicted from target_frame
        # predicted_frame = np.empty((height, width), dtype=np.uint8)
        
        # motion field consisting in the displacement of each block in vertical and horizontal
        motion_field = np.empty((height_mf, width_mf, 2))
        
        # loop through every blocks of motion field
        for (blk_row, blk_col) in itertools.product(range(0, height-(block_size-1), block_size), range(0, width-(block_size-1), block_size)):
            
            # minimum norm of the DFD norm found so far
            dfd_n_min = np.infty
            matching_blk = np.empty((block_size,block_size),dtype=np.int32)
            dx = 0
            dy = 0
            # search which block in a surrounding search_range minimizes the norm of the DFD. Blocks overlap.
            for(r_row, r_col) in itertools.product(range(-search_range, search_range), range(-search_range, search_range)):
                
                # candidate block upper left vertex and lower right vertex position as (row,col)
                up_l_candidate_blk_earlier = [blk_row-r_row, blk_col-r_col]
                up_l_candidate_blk_later = [blk_row+r_row, blk_col+r_col]
                low_r_candidate_blk_earlier = [blk_row-r_row+block_size, blk_col-r_col+block_size]
                low_r_candidate_blk_later = [blk_row+r_row+block_size, blk_col+r_col+block_size]
            
                # don't search outside the frame
                if (up_l_candidate_blk_earlier[0]<0) or (up_l_candidate_blk_earlier[1]<0) or \
                    (up_l_candidate_blk_later[0]<0) or (up_l_candidate_blk_later[1]<0) or \
                    (low_r_candidate_blk_earlier[0]>height) or (low_r_candidate_blk_earlier[1]>width) or \
                    (low_r_candidate_blk_later[0]>height) or (low_r_candidate_blk_later[1]>width):
                    continue
                    
                # compare earlier and later blocks from frame
                earlier_frame_blk = earlier_frame[up_l_candidate_blk_earlier[0]:low_r_candidate_blk_earlier[0], up_l_candidate_blk_earlier[1]:low_r_candidate_blk_earlier[1]]
                later_frame_blk = later_frame[up_l_candidate_blk_later[0]:low_r_candidate_blk_later[0], up_l_candidate_blk_later[1]:low_r_candidate_blk_later[1]]
                
                # compute dfd
                dfd = np.sum(np.absolute(np.array(earlier_frame_blk, dtype=np.float32)-np.array(later_frame_blk,dtype=np.float32)))
                
                # better matching block has been found, save it.
                if dfd < dfd_n_min:
                    dfd_n_min = dfd
                    # matching_blk = (earlier_frame_blk/2**1)+(later_frame_blk/2**1)
                    # matching_blk = earlier_frame_blk
                    dx = r_row
                    dy = r_col
                    
            # # construct the predicted image with the block that matches this block
            # predicted_frame[blk_row:blk_row+block_size, blk_col:blk_col+block_size] = matching_blk
            
            # displacement of this block in each direction
            motion_field[int(blk_row/block_size), int(blk_col/block_size), 0] = dx/block_size
            motion_field[int(blk_row/block_size), int(blk_col/block_size), 1] = dy/block_size
            
            
        # return predicted_frame, motion_field
        return motion_field
        
class Basic_Block_Motion_Compensation():
    """
    Interpolation with simple algorithm from motion vectors
    """
    def __init__(self, block_size):
        """
        :param block_size: size of block made for motion field
        """
        self.block_size = block_size
        
    def run(self, motion_field, earlier_frame, later_frame):
        """
        :param motion_field: Motion field computed from previeous stage
        :param earlier_frame: Image of earlier frame
        :param later_frame: Image of later frame
        :return: Interpolated frame by basic motion compensation
        """
        block_size = self.block_size
        height = earlier_frame.shape[0]
        width = earlier_frame.shape[1]
    
        # predicted_frame, gonna be interpolated from motion vector and original frames
        predicted_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # loop through every blocks
        for (blk_row, blk_col) in itertools.product(range(0, height-block_size+1, block_size), range(0, width-block_size+1, block_size)):
            
            # extract motion vector from wanted field
            [dx, dy] = (motion_field[int(blk_row/block_size), int(blk_col/block_size), :]*block_size).astype(int)
            
            # candidate block upper left vertex and lower right vertex position as (row,col)
            up_l_candidate_blk_earlier = [blk_row-dx, blk_col-dy]
            low_r_candidate_blk_earlier = [blk_row-dx+block_size, blk_col-dy+block_size]
            up_l_candidate_blk_later = [blk_row+dx, blk_col+dy]
            low_r_candidate_blk_later = [blk_row+dx+block_size, blk_col+dy+block_size]
            
            # check candidate block frame
            if (up_l_candidate_blk_earlier[0]<0) or (up_l_candidate_blk_earlier[1]<0) or \
                (low_r_candidate_blk_earlier[0]>height) or (low_r_candidate_blk_earlier[1]>width) or \
                (up_l_candidate_blk_later[0]<0) or (up_l_candidate_blk_later[1]<0) or \
                (low_r_candidate_blk_later[0]>height) or (low_r_candidate_blk_later[1]>width):
                print("Warning : reference block gone outside")
                continue
                
            # set block to reference
            earlier_frame_blk = earlier_frame[up_l_candidate_blk_earlier[0]:low_r_candidate_blk_earlier[0], up_l_candidate_blk_earlier[1]:low_r_candidate_blk_earlier[1]]
            later_frame_blk = later_frame[up_l_candidate_blk_later[0]:low_r_candidate_blk_later[0], up_l_candidate_blk_later[1]:low_r_candidate_blk_later[1]]
               
            # 100% overlapped area
            predicted_frame[blk_row:blk_row+block_size, blk_col:blk_col+block_size] = (earlier_frame_blk/2**1)+(later_frame_blk/2**1)
            # predicted_frame[blk_row:blk_row+block_size, blk_col:blk_col+block_size] = (earlier_frame_blk/2**0)
        
        return predicted_frame

    
class Simple_Overlapped_Block_Motion_Compensation():
    """
    Interpolation with overlapping from motion vectors
    """
    def __init__(self, block_size):
        """
        :param block_size: size of block made for motion field
        """
        self.block_size = block_size
    
    def run(self, motion_field, earlier_frame, later_frame):
        """
        :param motion_field: Motion field computed from previeous stage
        :param earlier_frame: Image of earlier frame
        :param later_frame: Image of later frame
        :return: Interpolated frame by OBMC
        """
        block_size = self.block_size
        height = earlier_frame.shape[0]
        width = earlier_frame.shape[1]
        height_mf = int(height/block_size)
        width_mf = int(width/block_size)
        height_pad = height+2*expand_pixels
        width_pad = width+2*expand_pixels
        
        # pad reference frames
        # earlier_frame_pad = np.pad(earlier_frame, [(expand_pixels, expand_pixels), (expand_pixels, expand_pixels)], mode='constant', constant_values=0) #color
        # later_frame_pad = np.pad(later_frame, [(expand_pixels, expand_pixels), (expand_pixels, expand_pixels)], mode='constant', constant_values=0) #color
        earlier_frame_pad = np.pad(earlier_frame, [(expand_pixels, expand_pixels), (expand_pixels, expand_pixels), (0, 0)], mode='constant', constant_values=0)
        later_frame_pad = np.pad(later_frame, [(expand_pixels, expand_pixels), (expand_pixels, expand_pixels), (0, 0)], mode='constant', constant_values=0)
        
        # predicted_frame, gonna be interpolated from motion vector and original frames
        # predicted_frame = np.zeros((height_pad, width_pad), dtype=np.float32)
        predicted_frame = np.zeros((height_pad, width_pad, 3), dtype=np.float32) # color
        
        # loop through every blocks
        for (blk_row, blk_col) in itertools.product(range(0, height-block_size+1, block_size), range(0, width-block_size+1, block_size)):
            
            # extract motion vector from wanted field
            [dx, dy] = (motion_field[int(blk_row/block_size), int(blk_col/block_size), :]*block_size).astype(int)
            
            # candidate block upper left vertex and lower right vertex position as (row,col)
            up_l_candidate_blk_earlier = [blk_row-dx, blk_col-dy]
            low_r_candidate_blk_earlier = [blk_row-dx+block_size+2*expand_pixels, blk_col-dy+block_size+2*expand_pixels]
            up_l_candidate_blk_later = [blk_row+dx, blk_col+dy]
            low_r_candidate_blk_later = [blk_row+dx+block_size+2*expand_pixels, blk_col+dy+block_size+2*expand_pixels]
            
            # check candidate block frame
            if (up_l_candidate_blk_earlier[0]<0) or (up_l_candidate_blk_earlier[1]<0) or \
                (low_r_candidate_blk_earlier[0]>height_pad) or (low_r_candidate_blk_earlier[1]>width_pad) or \
                (up_l_candidate_blk_later[0]<0) or (up_l_candidate_blk_later[1]<0) or \
                (low_r_candidate_blk_later[0]>height_pad) or (low_r_candidate_blk_later[1]>width_pad):
                print("Warning : reference block gone outside")
                continue
                
            # set block to reference
            earlier_frame_blk = earlier_frame_pad[up_l_candidate_blk_earlier[0]:low_r_candidate_blk_earlier[0], up_l_candidate_blk_earlier[1]:low_r_candidate_blk_earlier[1]]
            later_frame_blk = later_frame_pad[up_l_candidate_blk_later[0]:low_r_candidate_blk_later[0], up_l_candidate_blk_later[1]:low_r_candidate_blk_later[1]]
              
            # 25% overlapped area
            predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col:blk_col+2*expand_pixels] += (earlier_frame_blk[0:2*expand_pixels, 0:2*expand_pixels]/2**3)+(later_frame_blk[0:2*expand_pixels, 0:2*expand_pixels]/2**3)
            predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col:blk_col+2*expand_pixels] += (earlier_frame_blk[block_size:block_size+2*expand_pixels, 0:2*expand_pixels]/2**3)+(later_frame_blk[block_size:block_size+2*expand_pixels, 0:2*expand_pixels]/2**3)
            predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (earlier_frame_blk[0:2*expand_pixels, block_size:block_size+2*expand_pixels]/2**3)+(later_frame_blk[0:2*expand_pixels, block_size:block_size+2*expand_pixels]/2**3)
            predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (earlier_frame_blk[block_size:block_size+2*expand_pixels, block_size:block_size+2*expand_pixels]/2**3)+(later_frame_blk[block_size:block_size+2*expand_pixels, block_size:block_size+2*expand_pixels]/2**3)
            
            # 50% overlapped area
            predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col:blk_col+2*expand_pixels] += (earlier_frame_blk[2*expand_pixels:block_size, 0:2*expand_pixels]/2**2)+(later_frame_blk[2*expand_pixels:block_size, 0:2*expand_pixels]/2**2)
            predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col+2*expand_pixels:blk_col+block_size] += (earlier_frame_blk[0:2*expand_pixels, 2*expand_pixels:block_size]/2**2)+(later_frame_blk[0:2*expand_pixels, 2*expand_pixels:block_size]/2**2)
            predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (earlier_frame_blk[2*expand_pixels:block_size, block_size:block_size+2*expand_pixels]/2**2)+(later_frame_blk[2*expand_pixels:block_size, block_size:block_size+2*expand_pixels]/2**2)
            predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col+2*expand_pixels:blk_col+block_size] += (earlier_frame_blk[block_size:block_size+2*expand_pixels, 2*expand_pixels:block_size]/2**2)+(later_frame_blk[block_size:block_size+2*expand_pixels, 2*expand_pixels:block_size]/2**2)
            
            # 100% overlapped area
            predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col+2*expand_pixels:blk_col+block_size] += (earlier_frame_blk[2*expand_pixels:block_size, 2*expand_pixels:block_size]/2**1)+(later_frame_blk[2*expand_pixels:block_size, 2*expand_pixels:block_size]/2**1)
            
            # # 25% overlapped area
            # predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col:blk_col+2*expand_pixels] += (earlier_frame_blk[0:2*expand_pixels, 0:2*expand_pixels]/2**2)
            # predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col:blk_col+2*expand_pixels] += (earlier_frame_blk[block_size:block_size+2*expand_pixels, 0:2*expand_pixels]/2**2)
            # predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (earlier_frame_blk[0:2*expand_pixels, block_size:block_size+2*expand_pixels]/2**2)
            # predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (earlier_frame_blk[block_size:block_size+2*expand_pixels, block_size:block_size+2*expand_pixels]/2**2)
            
            # # 50% overlapped area
            # predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col:blk_col+2*expand_pixels] += (earlier_frame_blk[2*expand_pixels:block_size, 0:2*expand_pixels]/2**1)
            # predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col+2*expand_pixels:blk_col+block_size] += (earlier_frame_blk[0:2*expand_pixels, 2*expand_pixels:block_size]/2**1)
            # predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (earlier_frame_blk[2*expand_pixels:block_size, block_size:block_size+2*expand_pixels]/2**1)
            # predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col+2*expand_pixels:blk_col+block_size] += (earlier_frame_blk[block_size:block_size+2*expand_pixels, 2*expand_pixels:block_size]/2**1)
            
            # # 100% overlapped area
            # predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col+2*expand_pixels:blk_col+block_size] += (earlier_frame_blk[2*expand_pixels:block_size, 2*expand_pixels:block_size])

            
            # # 25% overlapped area
            # predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col:blk_col+2*expand_pixels] += (later_frame_blk[0:2*expand_pixels, 0:2*expand_pixels]/2**2)
            # predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col:blk_col+2*expand_pixels] += (later_frame_blk[block_size:block_size+2*expand_pixels, 0:2*expand_pixels]/2**2)
            # predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (later_frame_blk[0:2*expand_pixels, block_size:block_size+2*expand_pixels]/2**2)
            # predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (later_frame_blk[block_size:block_size+2*expand_pixels, block_size:block_size+2*expand_pixels]/2**2)
            
            # # 50% overlapped area
            # predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col:blk_col+2*expand_pixels] += (later_frame_blk[2*expand_pixels:block_size, 0:2*expand_pixels]/2**1)
            # predicted_frame[blk_row:blk_row+2*expand_pixels, blk_col+2*expand_pixels:blk_col+block_size] += (later_frame_blk[0:2*expand_pixels, 2*expand_pixels:block_size]/2**1)
            # predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col+block_size:blk_col+block_size+2*expand_pixels] += (later_frame_blk[2*expand_pixels:block_size, block_size:block_size+2*expand_pixels]/2**1)
            # predicted_frame[blk_row+block_size:blk_row+block_size+2*expand_pixels, blk_col+2*expand_pixels:blk_col+block_size] += (later_frame_blk[block_size:block_size+2*expand_pixels, 2*expand_pixels:block_size]/2**1)
            
            # # 100% overlapped area
            # predicted_frame[blk_row+2*expand_pixels:blk_row+block_size, blk_col+2*expand_pixels:blk_col+block_size] += (later_frame_blk[2*expand_pixels:block_size, 2*expand_pixels:block_size])

        return predicted_frame.astype(np.uint8)
        
        
class Full_Overlapped_Block_Motion_Compensation():
    """
    Interpolation with overlapping from motion vectors
    """
    def __init__(self, block_size):
        """
        :param block_size: size of block made for motion field
        """
        self.block_size = block_size
    
    def run(self, motion_field, earlier_frame, later_frame):
        """
        :param motion_field: Motion field computed from previeous stage
        :param earlier_frame: Image of earlier frame
        :param later_frame: Image of later frame
        :return: Interpolated frame by OBMC
        """
        block_size = self.block_size
        height = earlier_frame.shape[0]
        width = earlier_frame.shape[1]
        height_mf = int(height/block_size)
        width_mf = int(width/block_size)
        height_pad = height+block_size
        width_pad = width+block_size
        
        # pad reference frames
        # earlier_frame_pad = np.pad(earlier_frame, [(expand_pix, expand_pix), (expand_pix, expand_pix)], mode='constant', constant_values=0) #color
        # later_frame_pad = np.pad(later_frame, [(expand_pix, expand_pix), (expand_pix, expand_pix)], mode='constant', constant_values=0) #color
        earlier_frame_pad = np.pad(earlier_frame, [(expand_pix, expand_pix), (expand_pix, expand_pix), (0, 0)], mode='constant', constant_values=0)
        later_frame_pad = np.pad(later_frame, [(expand_pix, expand_pix), (expand_pix, expand_pix), (0, 0)], mode='constant', constant_values=0)
        
        # predicted_frame, gonna be interpolated from motion vector and original frames
        # predicted_frame = np.zeros((height_pad, width_pad), dtype=np.float32)
        predicted_frame = np.zeros((height_pad, width_pad, 3), dtype=np.float32) # color
        
        # loop through every blocks
        for (blk_row, blk_col) in itertools.product(range(0, height-block_size+1, block_size), range(0, width-block_size+1, block_size)):
            
            # extract motion vector from wanted field
            [dx, dy] = (motion_field[int(blk_row/block_size), int(blk_col/block_size), :]*block_size).astype(int)
            
            # candidate block upper left vertex and lower right vertex position as (row,col)
            up_l_candidate_blk_earlier = [blk_row-dx, blk_col-dy]
            low_r_candidate_blk_earlier = [blk_row-dx+2*block_size, blk_col-dy+2*block_size]
            up_l_candidate_blk_later = [blk_row+dx, blk_col+dy]
            low_r_candidate_blk_later = [blk_row+dx+2*block_size, blk_col+dy+2*block_size]
            
            # check candidate block frame
            if (up_l_candidate_blk_earlier[0]<0) or (up_l_candidate_blk_earlier[1]<0) or \
                (low_r_candidate_blk_earlier[0]>height_pad) or (low_r_candidate_blk_earlier[1]>width_pad) or \
                (up_l_candidate_blk_later[0]<0) or (up_l_candidate_blk_later[1]<0) or \
                (low_r_candidate_blk_later[0]>height_pad) or (low_r_candidate_blk_later[1]>width_pad):
                print("Warning : reference block gone outside")
                continue
                
            # set block to reference
            earlier_frame_blk = earlier_frame_pad[up_l_candidate_blk_earlier[0]:low_r_candidate_blk_earlier[0], up_l_candidate_blk_earlier[1]:low_r_candidate_blk_earlier[1]]
            later_frame_blk = later_frame_pad[up_l_candidate_blk_later[0]:low_r_candidate_blk_later[0], up_l_candidate_blk_later[1]:low_r_candidate_blk_later[1]]
            
            # form block with filtering
            for (pix_row, pix_col) in itertools.product(range(0,int(block_size)), range(0,int(block_size))):
                predicted_frame[blk_row+pix_row, blk_col+pix_col, 0] += earlier_frame_blk[pix_row, pix_col, 0]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[pix_row, pix_col, 0]*(pix_row+pix_col)/(8*block_size-8)
                predicted_frame[blk_row+pix_row, blk_col+pix_col, 1] += earlier_frame_blk[pix_row, pix_col, 1]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[pix_row, pix_col, 1]*(pix_row+pix_col)/(8*block_size-8)
                predicted_frame[blk_row+pix_row, blk_col+pix_col, 2] += earlier_frame_blk[pix_row, pix_col, 2]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[pix_row, pix_col, 2]*(pix_row+pix_col)/(8*block_size-8)
                # predicted_frame[blk_row+2*block_size-pix_row-1, blk_col+pix_col, 0] += earlier_frame_blk[2*block_size-pix_row-1, pix_col, 0]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[2*block_size-pix_row-1, pix_col, 0]*(pix_row+pix_col)/(8*block_size-8)
                # predicted_frame[blk_row+2*block_size-pix_row-1, blk_col+pix_col, 1] += earlier_frame_blk[2*block_size-pix_row-1, pix_col, 1]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[2*block_size-pix_row-1, pix_col, 1]*(pix_row+pix_col)/(8*block_size-8)
                # predicted_frame[blk_row+2*block_size-pix_row-1, blk_col+pix_col, 2] += earlier_frame_blk[2*block_size-pix_row-1, pix_col, 2]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[2*block_size-pix_row-1, pix_col, 2]*(pix_row+pix_col)/(8*block_size-8)
                # predicted_frame[blk_row+pix_row, blk_col+2*block_size-pix_col-1, 0] += earlier_frame_blk[pix_row, 2*block_size-pix_col-1, 0]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[pix_row, 2*block_size-pix_col-1, 0]*(pix_row+pix_col)/(8*block_size-8)
                # predicted_frame[blk_row+pix_row, blk_col+2*block_size-pix_col-1, 1] += earlier_frame_blk[pix_row, 2*block_size-pix_col-1, 0]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[pix_row, 2*block_size-pix_col-1, 1]*(pix_row+pix_col)/(8*block_size-8)
                # predicted_frame[blk_row+pix_row, blk_col+2*block_size-pix_col-1, 2] += earlier_frame_blk[pix_row, 2*block_size-pix_col-1, 0]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[pix_row, 2*block_size-pix_col-1, 2]*(pix_row+pix_col)/(8*block_size-8)
                predicted_frame[blk_row+2*block_size-pix_row-1, blk_col+2*block_size-pix_col-1, 0] += earlier_frame_blk[2*block_size-pix_row-1, 2*block_size-pix_col-1, 0]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[2*block_size-pix_row-1, 2*block_size-pix_col-1, 0]*(pix_row+pix_col)/(8*block_size-8)
                predicted_frame[blk_row+2*block_size-pix_row-1, blk_col+2*block_size-pix_col-1, 1] += earlier_frame_blk[2*block_size-pix_row-1, 2*block_size-pix_col-1, 1]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[2*block_size-pix_row-1, 2*block_size-pix_col-1, 1]*(pix_row+pix_col)/(8*block_size-8)
                predicted_frame[blk_row+2*block_size-pix_row-1, blk_col+2*block_size-pix_col-1, 2] += earlier_frame_blk[2*block_size-pix_row-1, 2*block_size-pix_col-1, 2]*(pix_row+pix_col)/(8*block_size-8)+later_frame_blk[2*block_size-pix_row-1, 2*block_size-pix_col-1, 2]*(pix_row+pix_col)/(8*block_size-8)

        return predicted_frame.astype(np.uint8)*2
        
if __name__ == "__main__":
    main()