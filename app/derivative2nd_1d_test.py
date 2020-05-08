
import cv2
import numpy as np

def gaussfilter_2d_test():


    img = cv2.imread('../data/scaling_test.png')

    sigmas = [5, 12.5, 25, 50, 100, 180]
    filtered_imgs = list()

    for sigma in sigmas:
        gauss = cv2.getGaussianKernel(ksize = 300, sigma=sigma)
        gauss_filt = np.outer(gauss, gauss.transpose())
        filtered_img = cv2.filter2D(img, -1, gauss_filt)
        filtered_imgs.append(filtered_img)

    cv2.imshow('Original Image', img)

    for i in range(len(sigmas)):
        filtered_img = filtered_imgs[i]
        sigma = sigmas[i]
        cv2.imshow('Filtered Image with Sigma = ' + str(sigma), filtered_img)
        cv2.imwrite('filtered_sigma' + str(sigma) + '.png', filtered_img)

    cv2.waitKey(0)




if __name__ == "__main__":
    
    gaussfilter_2d_test()