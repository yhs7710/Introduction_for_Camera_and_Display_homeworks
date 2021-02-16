import numpy as np

# Prepare data

dataset = np.loadtxt("./homework1_dataset.txt")#, dtype=np.dtype(float)) # raw data

w1 = dataset[0:10,1:]
w2 = dataset[10:20,1:]
w3 = dataset[20:30,1:]


# Process

print("\n*Consider Gaussian density models in different dimensions.")


# For a)

w1x1 = w1[:,0]
w1x2 = w1[:,1]
w1x3 = w1[:,2]

mu_w1x1 = (np.sum(w1x1))/(np.shape(w1x1)[0])
sigma_w1x1 = (np.sum(np.square(w1x1-mu_w1x1)))/(np.shape(w1x1)[0])
mu_w1x2 = (np.sum(w1x2))/(np.shape(w1x2)[0])
sigma_w1x2 = (np.sum(np.square(w1x2-mu_w1x2)))/(np.shape(w1x2)[0])
mu_w1x3 = (np.sum(w1x3))/(np.shape(w1x3)[0])
sigma_w1x3 = (np.sum(np.square(w1x3-mu_w1x3)))/(np.shape(w1x3)[0])

print("\n(a) Write a program to find the maximum-likelihood values mu and sigma. Apply your program individually to each of the three features x_i of category w_i in the table above.")
print("mu of the feature x_1: \n", mu_w1x1)
print("sigma of the feature x_1: \n", sigma_w1x1)
print("mu of the feature x_2: \n", mu_w1x2)
print("sigma of the feature x_2: \n", sigma_w1x2)
print("mu of the feature x_3: \n", mu_w1x3)
print("sigma of the feature x_3: \n", sigma_w1x3)


# For b)

w1x12 = w1[:,[0,1]]
w1x23 = w1[:,[1,2]]
w1x31 = w1[:,[2,0]]

mu_w1x12 = (np.sum(w1x12, axis=0))/(np.shape(w1x12)[0])
sigma_w1x12 = (np.matmul(np.transpose(w1x12-mu_w1x12), w1x12-mu_w1x12))/(np.shape(w1x12)[0])
mu_w1x23 = (np.sum(w1x23, axis=0))/(np.shape(w1x23)[0])
sigma_w1x23 = (np.matmul(np.transpose(w1x23-mu_w1x23), w1x23-mu_w1x23))/(np.shape(w1x23)[0])
mu_w1x31 = (np.sum(w1x31, axis=0))/(np.shape(w1x31)[0])
sigma_w1x31 = (np.matmul(np.transpose(w1x31-mu_w1x31), w1x31-mu_w1x31))/(np.shape(w1x31)[0])

print("\n(b) Modify your program to apply to two-dimensional Gaussian data p(x)~N(u,sigma). Apply your data to each of the three possible pairings of two features for w_1")
print("mu of the feature x_1 and x_2: \n", mu_w1x12)
print("sigma of the feature x_1 and x_2: \n", sigma_w1x12)
print("mu of the feature x_2 and x_3: \n", mu_w1x23)
print("sigma of the feature x_2 and x_3: \n", sigma_w1x23)
print("mu of the feature x_3 and x_1: \n", mu_w1x31)
print("sigma of the feature x_3 and x_1: \n", sigma_w1x31)


# For c)

mu_w1 = (np.sum(w1, axis=0))/(np.shape(w1)[0])
sigma_w1 = (np.matmul(np.transpose(w1-mu_w1), w1-mu_w1))/(np.shape(w1)[0])

print("\n(c) Modify your program to apply to three-dimensional Gaussian data. Apply your data to the full three-dimensional data for w_1")
print("mu of all the features for w_1: \n", mu_w1)
print("sigma of all the features for w_1: \n", sigma_w1)


# For d)

def suggested_program(wx):
    wxx1 = wx[:,0]
    wxx2 = wx[:,1]
    wxx3 = wx[:,2]
    mu_wxx1 = (np.sum(wxx1))/(np.shape(wxx1)[0])
    sigma_wxx1 = (np.sum(np.square(wxx1-mu_wxx1)))/(np.shape(wxx1)[0])
    mu_wxx2 = (np.sum(wxx2))/(np.shape(wxx2)[0])
    sigma_wxx2 = (np.sum(np.square(wxx2-mu_wxx2)))/(np.shape(wxx2)[0])
    mu_wxx3 = (np.sum(wxx3))/(np.shape(wxx3)[0])
    sigma_wxx3 = (np.sum(np.square(wxx3-mu_wxx3)))/(np.shape(wxx3)[0])
    return [mu_wxx1, mu_wxx2, mu_wxx3], [sigma_wxx1, sigma_wxx2, sigma_wxx3]
expected_mu, expected_sigma = suggested_program(w2)

print("\n(d) Assume your three-dimensional model is separable, so that sigma = diag(sigma_1^2, sigma_2^2, sigma_3^2). Write a program to estimate the mean and the diagonal components of sigma. Apply your program to the data in w_2")
print("Expected mu: ", expected_mu)
print("Expected sigma: ", expected_sigma)


# e) and f) were written in words

