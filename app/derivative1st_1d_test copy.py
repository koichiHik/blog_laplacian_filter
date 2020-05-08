
import cv2
import matplotlib.pyplot as plt
import numpy as np

def gaussian_filter2d():

    print('Gaussian Filter 2D Test')

    # Create Sample Image.
    height = 61
    width = 501
    band_width = 11
    peak = 255
    org_img = np.zeros((height, width), dtype=np.uint8)
    org_img[:, int(width / 2 - band_width / 2):int(width / 2 + band_width / 2 + 1)] = peak

    # Prepare figure.
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.0,hspace=0.0)
    fig3 = plt.figure(figsize=(6, 12))
    fig3.subplots_adjust(left=0.1,right=0.9,bottom=0.05,top=0.95,wspace=0.5,hspace=1.0)

    # Preapare Buffers.
    filtered_imgs = [org_img]
    unit = np.zeros((width, 1))
    unit[int(width / 2),:] = 1.0
    gausses = [unit]
    sigmas = [0.0]
    peak_values = list()

    # Create Gaussian Kernel
    k = np.sqrt(2)
    sigma = 1.0

    # Create filtered image list.
    loop_count = 10
    for i in range(loop_count):

        # Genearate Gaussian Kernel & Filter
        sigma = k * sigma
        sigmas.append(sigma)
        gauss = cv2.getGaussianKernel(width, sigma)
        gausses.append(gauss)

        # Apply Gaussian Filter
        filtered = cv2.filter2D(org_img, -1, gauss.transpose())
        filtered_imgs.append(filtered)

    # Genearete Plots
    for i in range(len(filtered_imgs)):

        # Plot Profile
        img = filtered_imgs[i]
        ax = fig.add_subplot(11, 2, 2 * i + 1)
        ax.axis('off')
        ax.set_ylim([0, 255])
        ax.set_title('sigma=' + str('{:.2f}'.format(sigmas[i])), fontsize=9)
        line_1d = img[int(height / 2.0), :]
        ax.plot(line_1d)

        # Plot Image
        peak_value = filtered_imgs[i][int(height / 2.0), int(width / 2.0)]/peak
        ax = fig.add_subplot(11, 2, 2 * i + 2)
        ax.axis('off')
        ax.set_title('peak=' + str('{:.2f}'.format(peak_value)), fontsize=9)
        ax.imshow(filtered_imgs[i], cmap='gray', vmin=0, vmax=255)
        peak_values.append(peak_value)

        ax = fig3.add_subplot(11,1,i + 1)
        ax.set_title('sigma=' + str('{:.2f}'.format(sigmas[i])), fontsize=9)
        ax.set_ylim([0.0, 1.0])
        ax.plot([k for k in range(width)], gausses[i])

    fig2 = plt.figure()
    ax = fig2.add_subplot(1,1,1)
    ax.scatter(sigmas, peak_values)

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":

    gaussian_filter2d()