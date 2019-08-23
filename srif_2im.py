# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:36:05 2019

@author: Nikolina Mileva
"""

import time
import scipy
import rasterio
import numpy as np
import skimage as sk
from sklearn.decomposition import PCA
from sklearn.feature_extraction import image
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning
import matplotlib.pyplot as plt

from sporco.dictlrn import bpdndl


def extract_patch(array, step):
    #array_w = sk.util.view_as_windows(array, (9,9), step=2) # same functionality 
#    step = 1
    windowSize = 9
    result = image.extract_patches(array, windowSize, step)
    d1,d2,d3,d4 = result.shape
    result = np.reshape(result, (d1*d2, d3, d4))
    result = result.reshape(result.shape[0], -1)
    
    return result


###############################################################
############# PREPROCESSING AND FEATURE EXTRACTION ############
###############################################################
    

# Load high resolution image
product = rasterio.open('C:/Users/Nikolina Mileva/Desktop/text_tr_crop.tif')
#profile = product.profile
y_h = product.read(1)
#y_h = np.mean(plt.imread('C:/Users/Nikolina Mileva/Desktop/text_tr_crop.png'), axis=2)

# Blur
blur_filter_hor = np.array(([1/12, 3/12, 4/12, 3/12, 1/12]))
blur_filter_ver = np.array(([1/12], [3/12], [4/12], [3/12], [1/12]))

y_h_hor_blur = scipy.ndimage.convolve1d(y_h, blur_filter_hor, mode='reflect')
y_h_blur = scipy.ndimage.convolve(y_h_hor_blur, blur_filter_ver, mode='reflect')

# Downscale by a factor of 3
z_l = sk.transform.rescale(y_h_blur, scale=1/3, anti_aliasing=False, multichannel=False)

# Upscale by a factor of 3
y_l = sk.transform.resize(z_l, y_h.shape, order=3, anti_aliasing=False)
#product_l = rasterio.open('C:/Users/Nikolina Mileva/Documents/Data/subset_S2_S3_2017_2018_2019_coreg.tif')
#y_l = product_l.read(8)
#y_l = rasterio.open('C:/Users/Nikolina Mileva/Desktop/text_tr_l.tif').read(1)

# Filtering
filter_1 = np.array(([0, 0, 1, 0, 0, -1]))
filter_2 = np.array(([0], [0], [1], [0], [0], [-1]))
filter_3 = np.array(([1, 0, 0, -2, 0, 0, 1]))
filter_4 = np.array(([1], [0], [0], [-2], [0], [0], [1]))

im_fil_1_l = scipy.ndimage.convolve1d(y_l, filter_1, mode='reflect')
im_fil_2_l = scipy.ndimage.convolve(y_l, filter_2, mode='reflect')
im_fil_3_l = scipy.ndimage.convolve1d(y_l, filter_3, mode='reflect')
im_fil_4_l = scipy.ndimage.convolve(y_l, filter_4, mode='reflect')

# Extract low resolution patches  
print('Extracting patches...')
t0 = time.time()

patches_1_l = extract_patch(im_fil_1_l, 2)
patches_2_l = extract_patch(im_fil_2_l, 2)
patches_3_l = extract_patch(im_fil_3_l, 2)
patches_4_l = extract_patch(im_fil_4_l, 2)


print('done in %.2fs.' % (time.time() - t0))

# Concatenate patches from the different filters into one vector
patches_l = np.hstack((patches_1_l, patches_2_l, patches_3_l, patches_4_l))


# Normalize the patches
#patches_l -= np.mean(patches_l, axis=0)
#patches_l /= np.std(patches_l, axis=0)


# Feature extraction from high resolution image
e_h = y_h - y_l

# Extract high resolution patches
patches_h = extract_patch(e_h, 2)

# Normalize the patches
#patches_h -= np.mean(patches_h, axis=0)
#patches_h /= np.std(patches_h, axis=0)

plt.imshow(y_h)
plt.gray()
plt.show()

#plt.imshow(z_l)
#plt.gray()
#plt.show()

plt.imshow(y_l)
plt.gray()
plt.show()


###########################################################
############### DIMENSIONALITY REDUCTION ##################
###########################################################

## Fine tuning of PCA number of components
#scaler = MinMaxScaler(feature_range=[0, 1])
#data_rescaled = scaler.fit_transform(patches_l)
##Fitting the PCA algorithm with our Data
#pca = PCA().fit(data_rescaled)
##Plotting the Cumulative Summation of the Explained Variance
#plt.figure()
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)') #for each component
#plt.title('Explained Variance')
#plt.show()

# Dimensionality reduction with PCA
pca = PCA(n_components=49) # 47 changed to 49 for display purposes
pca.fit(patches_l)
patches_l = pca.transform(patches_l)

########################################################
################## DICTIONARY LEARNING #################
########################################################

# Dictionary learning on low resolution patches
print('Learning the low resolution dictionary...')
t0 = time.time()

#dico = DictionaryLearning(n_components=1000, alpha=0.1, max_iter=40)
dico = MiniBatchDictionaryLearning(n_components=1000, alpha=0.1, n_iter=40) # try with less iterations
#np.save('C:/Users/Nikolina Mileva/Documents/Sparseland/dico.npy', dico)
#V = dico.fit(patches_l).components_
#np.save('C:/Users/Nikolina Mileva/Documents/Sparseland/dictionary_l.npy', V)

# Load and visualize the low resolution dictionary
#V = np.load('C:/Users/Nikolina Mileva/Documents/Sparseland/dictionary_l.npy')

#patch_size = (7,7) # patch size after PCA
#plt.figure(figsize=(4.2, 4))
#for i, comp in enumerate(V[:100]):
#    plt.subplot(10, 10, i + 1)
#    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
#               interpolation='nearest')
#    plt.xticks(())
#    plt.yticks(())
#
#plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
#
#plt.imshow(V)
#plt.gray()
#plt.show()

transform_algorithms = [('omp',{'transform_n_nonzero_coefs': 3})]

for transform_algorithm, kwargs in transform_algorithms:
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    Q = dico.fit_transform(patches_l) # sparse representation

dt = time.time() - t0
print('done in %.2fs.' % dt)

# Dictionary learning on high resolution patches
print('Solving the pseudo-inverse...')
t0 = time.time()

Q_t = np.transpose(Q)
P_t = np.transpose(patches_h)
print('Patches: ', P_t.shape)
print('Sparse rep.: ', Q_t.shape)
dictionary_h = P_t @ np.linalg.pinv(Q_t)
#dictionary_h = P_t @ Q @ (np.linalg.pinv(Q_t @ Q))

dt = time.time() - t0
print('done in %.2fs.' % dt)
#np.save('C:/Users/Nikolina Mileva/Documents/Sparseland/dictionary_h.npy', dictionary_h)
#np.save('C:/Users/Nikolina Mileva/Documents/Sparseland/sparse_representation.npy', Q)

## Visualize high resolution dictionary
#plt.figure(figsize=(4.2, 4))
#for i, comp in enumerate(dictionary_h[:324]):
#    plt.subplot(18, 18, i + 1)
#    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
#               interpolation='nearest')
#    plt.xticks(())
#    plt.yticks(())
#
#plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
#
#plt.imshow(dictionary_h)
#plt.gray()
#plt.show()

#####################################################################
###################### RECONSTRUCTION PHASE #########################
#####################################################################

print('Reconstruction...')
t0 = time.time()

# Load image to be reconstructed
#z_l = rasterio.open('C:/Users/Nikolina Mileva/Documents/Sparseland/Images/NIR/subset_S2A_S3A_20180901_20180924_NIR_z_l.tif').read(1)
y_re = rasterio.open('C:/Users/Nikolina Mileva/Desktop/text_h_crop.tif').read(1)

# Blur
y_re_hor_blur = scipy.ndimage.convolve1d(y_re, blur_filter_hor, mode='reflect')
y_re_blur = scipy.ndimage.convolve(y_re_hor_blur, blur_filter_ver, mode='reflect')

# Downscale by a factor of 3
z_l = sk.transform.rescale(y_re_blur, scale=1/3, anti_aliasing=False, multichannel=False)

# Up-sampling by a factor of 2 with bi-cubic interpolation
#y_l = product_l.read(12)
#y_l = sk.transform.rescale(z_l, scale=3, anti_aliasing=False, multichannel=False)
y_l = sk.transform.resize(z_l, y_re.shape, order=3, anti_aliasing=False)
#product = rasterio.open('C:/Users/Nikolina Mileva/Desktop/text_l_crop.tif')
#profile = product.profile
#y_l = product.read(1)

# Filtering
im_fil_1 = scipy.ndimage.convolve1d(y_l, filter_1, mode='reflect')
im_fil_2 = scipy.ndimage.convolve(y_l, filter_2, mode='reflect')
im_fil_3 = scipy.ndimage.convolve1d(y_l, filter_3, mode='reflect')
im_fil_4 = scipy.ndimage.convolve(y_l, filter_4, mode='reflect')

# Patch extraction
patches_1_h = extract_patch(im_fil_1, 1)
patches_2_h = extract_patch(im_fil_2, 1)
patches_3_h = extract_patch(im_fil_3, 1)
patches_4_h = extract_patch(im_fil_4, 1)

# Concatenate patches from the different filters into one vector
patches = np.hstack((patches_1_h, patches_2_h, patches_3_h, patches_4_h))

# Dimensionality reduction with PCA
pca = PCA(n_components=49) # 47 changed to 49 for display purposes
pca.fit(patches)
patches = pca.transform(patches)

# Deriving the sparse representation
#dico = np.load('C:/Users/Nikolina Mileva/Documents/Sparseland/dico.npy', allow_pickle=True)
#dictionary_h = np.load('C:/Users/Nikolina Mileva/Documents/Sparseland/dictionary_h.npy')

transform_algorithms = [('omp',{'transform_n_nonzero_coefs': 3})]

for transform_algorithm, kwargs in transform_algorithms:
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    Q = dico.transform(patches) # sparse representation

# Reconstruct high resolution patches
dictionary_h = np.transpose(dictionary_h)
patches = np.dot(Q, dictionary_h)
#intercept = np.mean(patches, axis=0)
#patches += intercept
patch_size = (9,9)
patches = patches.reshape(len(patches), *patch_size)


# Reconstruct the final image from the patches
y_r_diff = image.reconstruct_from_patches_2d(patches, image_size=y_l.shape)
#y_r_diff = sk.transform.resize(z_l, y_h.shape, order=3, anti_aliasing=False)
y_r = y_l + y_r_diff

dt = time.time() - t0
print('done in %.2fs.' % dt)

#plt.imshow(z_l)
#plt.gray()
#plt.show()

plt.imshow(y_l)
plt.gray()
plt.show()

plt.imshow(y_r)
plt.gray()
plt.show()


# Write the results to a .tif file   
print ('Writing product...')
profile = product.profile
profile.update(dtype='float64', count=1) # number of bands
file_name = 'output_2.tif'

result = rasterio.open(file_name, 'w', **profile)#rasterio.open(file_name, 'w', **defaults(count=1))# 
result.write(y_r, 1)
result.close()



#array_up = sk.transform.rescale(array, scale = 2, order=3, anti_aliasing=False, multichannel=False)
#print(array_up)
#patch_size = (5,5)
#array = np.arange(100).reshape(10,10)
#array_p = image.extract_patches(array, (5,5), 1)
#d1,d2,d3,d4 = array_p.shape
#array_p = np.reshape(array_p, (d1*d2, d3, d4))
#array_p = array_p.reshape(array_p.shape[0], -1)
#
#patches = array_p.reshape(len(array_p), *patch_size)
#
#print(array) 
#print(array_p.shape)
#array_r = image.reconstruct_from_patches_2d(array_p, image_size=array.shape)
#print(array_r.shape)
#print(array_r)






    

















