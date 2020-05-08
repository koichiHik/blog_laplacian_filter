
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def derivative_filter():

    print('Derivative Filter Test')

    # Create Sample Image.
    height = 61
    width = 501
    band_width = 16
    peak = 255
    org_img = np.zeros((height, width), dtype=np.uint8)
    org_img[:, int(width / 2 - band_width / 2):int(width / 2 + band_width / 2 + 1)] = peak

    # Prepare figure.
    fig = plt.figure(figsize=(10, 12))

    # Preapare Buffers.
    unit = np.zeros((width, 1))
    unit[int(width / 2), :] = 1.0

    # Genearate 1st Deriv Kernel
    deriv_1st_kernel = np.array([[-1.0], [1.0]], dtype=np.float)

    # Genearate 2nd Deriv Kernel
    deriv_2nd_kernel = np.array([[1.0], [-2.0], [1.0]], dtype=np.float)

    # Apply 1st Derivative Filter
    filtered_1st = cv2.filter2D(
        org_img, cv2.CV_32FC1, deriv_1st_kernel.transpose())

    # Apply Laplacian Filter
    filtered_2nd = cv2.filter2D(
        org_img, cv2.CV_32FC1, deriv_2nd_kernel.transpose())

    # Show Input Image
    ax = fig.add_subplot(4, 1, 1)
    ax.set_title("Input Image")
    ax.imshow(org_img, cmap='gray', vmin=0, vmax=255)

    # Plot Original Profile
    ax = fig.add_subplot(4, 1, 2)
    ax.set_title("Profile of Input Image")
    ax.set_ylim([0, 300])
    line_1d = org_img[int(height / 2.0), :]
    ax.plot(line_1d)

    # Plot Filtered Profile
    ax = fig.add_subplot(4, 1, 3)
    ax.set_title("Responce against 1st Derivative Filter")
    ax.set_ylim([-300, 300])
    line_1d = filtered_1st[int(height / 2.0), :]
    ax.plot(line_1d)

    # Plot Filtered Profile
    ax = fig.add_subplot(4, 1, 4)
    ax.set_title("Responce against Laplacian Filter")
    ax.set_ylim([-300, 300])
    line_1d = filtered_2nd[int(height / 2.0), :]
    ax.plot(line_1d)

    fig.tight_layout()
    plt.show()


def derivative_filter_with_gaussian(band, init_sigma, k, normalize=False):

    print('Derivative Filter with Gaussian Test')

    # Create Sample Image.
    height = 101
    width = 501
    band_width = band
    peak = 255
    org_img = np.zeros((height, width), dtype=np.uint8)
    org_img[:, int(width / 2 - band_width):int(width / 2 + band_width)] = peak

    # Prepare figure.
    fig = plt.figure(figsize=(20, 8))

    # Preapare Buffers.
    unit = np.zeros((width, 1))
    unit[int(width / 2), :] = 1.0
    sigmas = list()
    gauss_kernels = list()
    filtered_images = list()
    filtered_1st_images = list()
    filtered_2nd_images = list()

    # Genearate 1st Deriv Kernel
    deriv_1st_kernel = np.array([[-0.5], [0.0], [0.5]], dtype=np.float)

    # Genearate 2nd Deriv Kernel
    deriv_2nd_kernel = np.array([[1.0], [-2.0], [1.0]], dtype=np.float)

    # Generate Gaussian Kernel
    sigma_num = 5
    sigma = init_sigma
    for i in range(sigma_num):
        # Generate Gaussian Kernel
        sigmas.append(sigma)
        gauss_kernels.append(cv2.getGaussianKernel(ksize=width, sigma=sigma))

        # Generate Images.
        filtered_images.append(cv2.filter2D(
            org_img, cv2.CV_32FC1, gauss_kernels[-1].transpose()))
        filtered_1st_images.append(cv2.filter2D(
            filtered_images[-1], cv2.CV_32FC1, deriv_1st_kernel.transpose()) * (sigma if normalize else 1.0))
        filtered_2nd_images.append(cv2.filter2D(
            filtered_images[-1], cv2.CV_32FC1, deriv_2nd_kernel.transpose()) * (sigma**2 if normalize else 1.0))

        sigma = k * sigma

    for i in range(sigma_num):
        # Show Input Image
        ax = fig.add_subplot(4, sigma_num, i + 1)
        # ax.set_title("Filtered Image with Sigma : " + str(sigmas[i]))
        ax.set_title('sigma=' + str('{:.2f}'.format(sigmas[i])), fontsize=9)
        ax.imshow(filtered_images[i], cmap='gray', vmin=0, vmax=255)

    for i in range(sigma_num):
        # Plot Original Profile
        ax = fig.add_subplot(4, sigma_num, i + 1 + sigma_num)
        ax.set_title("Profile of Input Image", fontsize=9)
        ax.set_xlim([0, 500])
        ax.set_ylim([0, 300])
        line_1d = filtered_images[i][int(height / 2.0), :]
        ax.plot(line_1d)

    for i in range(sigma_num):
        # Plot Filtered Profile
        ax = fig.add_subplot(4, sigma_num, i + 1 + 2 * sigma_num)
        ax.set_title("Responce against 1st Derivative Filter", fontsize=9)
        ax.set_xlim([0, 500])
        ax.set_ylim([-150, 150])
        # ax.set_ylim([-2000, 2000])
        line_1d = filtered_1st_images[i][int(height / 2.0), :]
        ax.plot(line_1d)

    for i in range(sigma_num):
        # Plot Filtered Profile
        ax = fig.add_subplot(4, sigma_num, i + 1 + 3 * sigma_num)
        ax.set_title("Responce against Laplacian Filter", fontsize=9)
        ax.set_xlim([0, 500])
        ax.set_ylim([-150, 150])
        # ax.set_ylim([-2000, 2000])
        line_1d = filtered_2nd_images[i][int(height / 2.0), :]
        ax.plot(line_1d)

    fig.tight_layout()
    plt.show()


def filter_sample_image():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = cur_dir + '/../data/Titech2.jpg'
    img = cv2.resize(cv2.imread(path), None, fx=0.5, fy=0.5)
    img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=0.5)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Genearate 1st Deriv Kernel
    deriv_1st_kernel_x = np.array([[-1/2.0, 0.0, 1/2.0]], dtype=np.float)
    gray_img_1st_x = cv2.filter2D(
        gray_img, cv2.CV_32FC1, deriv_1st_kernel_x) + 127.0
    gray_img_1st_x = np.uint8(gray_img_1st_x)

    deriv_1st_kernel_y = np.array([[-1/2.0], [0.0], [1/2.0]], dtype=np.float)
    gray_img_1st_y = cv2.filter2D(
        gray_img, cv2.CV_32FC1, deriv_1st_kernel_y) + 127.0
    gray_img_1st_y = np.uint8(gray_img_1st_y)

    # Genearate 2nd Deriv Kernel
    deriv_2nd_kernel = np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float)
    gray_img_2nd = cv2.filter2D(
        gray_img, cv2.CV_32FC1, deriv_2nd_kernel) + 127.0
    gray_img_2nd = np.uint8(gray_img_2nd)

    cv2.imshow('Original Image', gray_img)
    cv2.imshow('Image with 1st Derivative Filter in x', gray_img_1st_x)
    cv2.imshow('Image with 2nd Derivative Filter in y', gray_img_1st_y)
    cv2.imshow('Image with Laplacian Filter', gray_img_2nd)
    cv2.waitKey(0)


def filter_himawari_pics():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = cur_dir + '/../data/himawari.jpg'
    img = cv2.imread(path)
    #img = cv2.resize(cv2.imread(path), None, fx=0.5, fy=0.5)
    #img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=0.5)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sigma = 15.0
    gauss_kernel = cv2.getGaussianKernel(300, sigma, cv2.CV_32F)
    gauss_filter = np.outer(gauss_kernel, gauss_kernel.transpose())
    filtered = cv2.filter2D(gray_img, cv2.CV_32FC1, gauss_filter)

    deriv_2nd_kernel = np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float)
    gray_img_2nd = cv2.filter2D(
        filtered, cv2.CV_32FC1, deriv_2nd_kernel) * sigma * sigma
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_img_2nd)
    gray_img_2nd = -gray_img_2nd

    _, gray_img_2nd = cv2.threshold(
        gray_img_2nd, 10.0, 0.0, cv2.THRESH_TOZERO)

    print(min_val)
    print(max_val)

    gray_img_2nd = np.uint8(gray_img_2nd)

    cv2.imshow("Laplacian Responce", gray_img_2nd)
    cv2.waitKey(0)


if __name__ == "__main__":

    filter_sample_image()

    derivative_filter()

    derivative_filter_with_gaussian(
        band=16, init_sigma=2.0, k=2, normalize=True)

    derivative_filter_with_gaussian(
        band=16, init_sigma=2.0, k=2, normalize=False)

    # filter_himawari_pics()
