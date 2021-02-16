import argparse
import os
import itertools
import math
import sys

import numpy as np
import cv2

import matplotlib.pyplot as plt

block_size = 32
search_range = 4


def main():

    # read images
    earlier_frm_rgb = cv2.imread('514.jpg')
    real_frm_rgb = cv2.imread('515.jpg')
    later_frm_rgb = cv2.imread('516.jpg')
    
    # convert rgb images to grayscale images
    earlier_frm_gray = cv2.cvtColor(earlier_frm_rgb, cv2.COLOR_RGB2GRAY)
    real_frm_gray = cv2.cvtColor(real_frm_rgb, cv2.COLOR_RGB2GRAY)
    later_frm_gray = cv2.cvtColor(later_frm_rgb, cv2.COLOR_RGB2GRAY)
    
    # save original frames
    cv2.imwrite('1_earlier_frame.png',earlier_frm_gray)
    cv2.imwrite('4_later_frame.png',later_frm_gray)
    cv2.imwrite('2_real_frame.png',real_frm_gray)
    
    # search motion vectors
    MVS = Motion_Vector_Searcher(block_size, search_range)
    predicted_frm, motion_field = MVS.run(earlier_frm_gray, later_frm_gray)
    
    # store predicted frame
    cv2.imwrite('3_predicted_frame.png',predicted_frm)
    
    # store error image
    error_image = abs(np.array(predicted_frm, dtype=float) - np.array(real_frm_gray, dtype=float))
    error_image = np.array(error_image, dtype=np.uint8)
    cv2.imwrite('5_error_image.png', error_image)

    # Peak Signal-to-Noise Ratio of the predicted image
    mse = (np.array(error_image, dtype=float) ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)
    print('PSNR: %s dB' % psnr)
   
    # show motion field
    motion_field_x = cv2.flip(motion_field[:, :, 0],0)
    motion_field_y = -cv2.flip(motion_field[:, :, 1],0)
    
    a = np.ones(motion_field_x.shape)
    
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
        :return: predicted frame, motion vector map of frame to estimate
        """
        block_size = self.block_size
        search_range = self.search_range
        height = earlier_frame.shape[0]
        width = earlier_frame.shape[1]
        mf_height = int(height/block_size)
        mf_width = int(width/block_size)
        
        # predicted frame. anchor_frame is predicted from target_frame
        predicted_frame = np.empty((height, width), dtype=np.uint8)
        
        # motion field consisting in the displacement of each block in vertical and horizontal
        motion_field = np.empty((mf_height, mf_width, 2))
        
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
                    matching_blk = (earlier_frame_blk>>1)+(later_frame_blk>>1)
                    # matching_blk = earlier_frame_blk
                    dx = r_row/block_size
                    dy = r_col/block_size
                    
            # construct the predicted image with the block that matches this block
            predicted_frame[blk_row:blk_row+block_size, blk_col:blk_col+block_size] = matching_blk
            
            print (str((blk_row/block_size, blk_col/block_size)) + '--- Displacement: ' + str((dx, dy)))
            
            # displacement of this block in each direction
            motion_field[int(blk_row/block_size), int(blk_col/block_size), 1] = dx
            motion_field[int(blk_row/block_size), int(blk_col/block_size), 0] = dy
            
        return predicted_frame, motion_field
        
if __name__ == "__main__":
    main()