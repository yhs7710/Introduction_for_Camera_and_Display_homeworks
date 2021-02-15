import numpy as np
import cv2
import matplotlib.pyplot as plt

file_name = "chicago"
image = cv2.imread('./%s.png'%file_name).astype(np.float32)/256
image = np.expand_dims(image.mean(axis=2), axis=2)
print (image.shape)

#  'gauss'     Gaussian-distributed additive noise.
#  's&p'       Replaces random pixels with 0 or 1.
# 'poisson'   Poisson-distributed noise generated from the data.
# 'speckle'   Multiplicative noise using out = image + n*image,where
#                 n is uniform noise with specified mean & variance.
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.03  # 분산 0.1
      sigma = var**0.5   # root(var)
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy.clip(min=0)
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5  # 0.5
      amount = 0.05   # 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i, int(num_salt))
              for i in image.shape]
      out[coords] = 1.0-1/256

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i, int(num_pepper))
              for i in image.shape]
      out[coords] = 0.0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ == "speckle":
      row, col, ch = image.shape
      gauss = np.random.randn(row, col, ch)
      gauss = gauss.reshape(row, col, ch)
      noisy = image + image * gauss
      return noisy

#plt.imshow(image[:,:, [2,1,0]])
#plt.show()

noise_img1 = noisy("gauss", image).astype(np.float32)
noise_img2 = noisy("s&p", image).astype(np.float32)
# print(noise_img1)
# print(noise_img2)
#plt.imshow(noise_img[:, :, [2,1,0]])
# plt.savefig('dog1.png', dpi=300)
#plt.show()

#plt.imshow(noise_img2[:, :, [2,1,0]])
# plt.savefig('dog2.png', dpi=300)
#plt.show()

# image1 = cv2.imread('dog1.png').astype(np.float32)/255
# image2 = cv2.imread('dog2.png').astype(np.float32)/255


# bilat1 = cv2.bilateralFilter(image1, -1, 5, 5)
# median_blur1 = cv2.medianBlur((image1 * 255).astype(np.uint8), 9)
# avg_blur1 = cv2.blur(image1, (13,13))
# bilat2 = cv2.bilateralFilter(image2, -1, 5, 5)
# median_blur2 = cv2.medianBlur((image2 * 255).astype(np.uint8), 9)
# avg_blur2 = cv2.blur(image2, (13,13))

bilat1 = cv2.bilateralFilter(noise_img1, -1, 5, 5)
median_blur1 = cv2.medianBlur((noise_img1 * 255).astype(np.uint8), 9)
avg_blur1 = cv2.blur(noise_img1, (13,13))
bilat2 = cv2.bilateralFilter(noise_img2, -1, 5, 5)
median_blur2 = cv2.medianBlur((noise_img2 * 255).astype(np.uint8), 9)
avg_blur2 = cv2.blur(noise_img2, (13,13))

cv2.imwrite("./%s_gauss.png"%file_name, noise_img1*256)
cv2.imwrite("./%s_gauss_bilat.png"%file_name, bilat1*256)
cv2.imwrite("./%s_gauss_median.png"%file_name, median_blur1*256)
cv2.imwrite("./%s_gauss_avg.png"%file_name, avg_blur1*256)
cv2.imwrite("./%s_s&p.png"%file_name, noise_img2*256)
cv2.imwrite("./%s_s&p_bilat.png"%file_name, bilat2*256)
cv2.imwrite("./%s_s&p_median.png"%file_name, median_blur2*256)
cv2.imwrite("./%s_s&p_avg.png"%file_name, avg_blur2*256)