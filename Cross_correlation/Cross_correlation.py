import numpy as np
import cv2


img1 = np.mean(cv2.imread('a1.png'),axis=2) # image 1 read
img2 = np.mean(cv2.imread('a2.png'),axis=2) # image 2 read

cv2.imwrite('a1_bw.png',abs(img1))
cv2.imwrite('a2_bw.png',abs(img2))

# img1 = cv2.imread('a1.png') # image 1 read
# img2 = cv2.imread('a2.png') # image 2 read

img1_ft = np.fft.fft2(img1) # image 1 fft
img2_ft = np.fft.fft2(img2) # image 2 fft
img2_ft_conj = np.conj(img2_ft) # image 2 fft conjugate
print(np.shape(img1_ft)[1])
print(np.shape(img2_ft_conj))
cv2.imwrite('a1_tf.png',np.angle(img1_ft))
cv2.imwrite('a2_tf.png',np.angle(img2_ft))

out_freq = img1_ft*img2_ft_conj/np.abs(img1_ft*img2_ft_conj)
out = np.fft.ifft2(out_freq)

print(abs(out))
out_freq_im = (np.angle(out_freq)-np.min(np.angle(out_freq))/(np.max(np.angle(out_freq))-np.min(np.angle(out_freq))))*255
out_im = (np.abs(out)-np.min(np.abs(out))/(np.max(np.abs(out))-np.min(np.abs(out))))*1000

cv2.imwrite('out_freq.png',out_freq_im)
cv2.imwrite('out.png',out_im)