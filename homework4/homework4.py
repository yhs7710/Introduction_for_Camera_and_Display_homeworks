import numpy as np
import cv2

img = cv2.imread("abundance.png", 0)
img_200 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
img_300 = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("input.png", img)
cv2.imwrite("output_200.png", img_200)
cv2.imwrite("output_300.png", img_300)

img_dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
img_200_dft = cv2.dft(np.float32(img_200), flags = cv2.DFT_COMPLEX_OUTPUT)
img_300_dft = cv2.dft(np.float32(img_300), flags = cv2.DFT_COMPLEX_OUTPUT)
img_dft_shift = np.fft.fftshift(img_dft, axes=[0,1])
img_200_dft_shift = np.fft.fftshift(img_200_dft, axes=[0,1])
img_300_dft_shift = np.fft.fftshift(img_300_dft, axes=[0,1])
magnitude_spectrum = 20*np.log(cv2.magnitude(img_dft_shift[:,:,0], img_dft_shift[:,:,1])+1)
magnitude_spectrum_200 = 20*np.log(cv2.magnitude(img_200_dft_shift[:,:,0], img_200_dft_shift[:,:,1])+1)
magnitude_spectrum_300 = 20*np.log(cv2.magnitude(img_300_dft_shift[:,:,0], img_300_dft_shift[:,:,1])+1)
cv2.imwrite("input_dft.png", magnitude_spectrum)
cv2.imwrite("output_200_dft.png", magnitude_spectrum_200)
cv2.imwrite("output_300_dft.png", magnitude_spectrum_300)