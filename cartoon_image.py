#Robert Adams
#Cartoon Image
"""
Creates a 'cartoonified' image by combining the effects of a bilateral filter with outlines
created from canny edge detection.

Algorithms implemented from:
Canny, John, "A Computational Approach to Edge Detection," in Pattern
Analysis and Machine Intelligence, IEEE Transactions on , vol.PAMI-8,
no.6, pp.679-698, Nov. 1986
Tomasi, C.; Manduchi, R., "Bilateral Filtering for gray and color images,"
in Computer Vision, 1998. Sixth International Conference on , vol., no.,
pp.839-846, 4-7 Jan 1998
"""

import Queue
import numpy as np
import cv2
import sys
import time
from matplotlib import pyplot as plt


#takes an x,y pair of image coordinates,
#returns a 1d array of its neighbors
def neighbors(x, y):
    return np.array([(x-1, y), (x, y-1), (x+1, y), (x, y+1),
                     (x+1, y+1), (x-1, y+1), (x+1, y-1), (x-1, y-1)])

#takes BGR image, low threshold, high threshold
#returns single channel image of floats containing edge values (0 or 255)
def canny(img, low, high):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.getGaussianKernel(5, -1)

    sobel1 =np.matrix([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobel2 =np.matrix([[1,2,1], [0,0,0], [-1,-2,-1]])

    new_img = cv2.sepFilter2D(gray, -1, g, g)
    mag_x = cv2.filter2D(new_img, cv2.cv.CV_32F,sobel1)
    mag_y = cv2.filter2D(new_img, cv2.cv.CV_32F, sobel2)
    #mag_x = cv2.Sobel(new_img,cv2.CV_64F,1,0,ksize=5)
    #mag_y = cv2.Sobel(new_img,cv2.CV_64F,0,1,ksize=5)
    new_img = np.hypot(mag_x, mag_y)

    theta = np.arctan2(mag_y, mag_x) # save edge direction
    theta = ((theta) * 4.0 / np.pi + 0.5).astype('int') % 4 # convert edge direction into 0-3


    p1 = np.pad(new_img,((1, 0),(1,0)), mode='edge')[:-1,:-1]
    p2 = np.pad(new_img,((1, 0),(0,0)), mode='edge')[:-1,:]
    p3 = np.pad(new_img,((1, 0),(0,1)), mode='edge')[:-1,1:]
    p4 = np.pad(new_img,((0, 0),(1,0)), mode='edge')[:,:-1]
    p5 = np.pad(new_img,((0, 0),(0,1)), mode='edge')[:,1:]
    p6 = np.pad(new_img,((0, 1),(1,0)), mode='edge')[1:,:-1]
    p7 = np.pad(new_img,((0, 1),(0,0)), mode='edge')[1:,:]
    p8 = np.pad(new_img,((0, 1),(0,1)), mode='edge')[1:,1:]
    total = np.dstack((new_img,p1, p2, p3, p4, p5, p6, p7, p8, theta))


    edges = np.zeros(new_img.shape)

    for (i,j), v in np.ndenumerate(edges):
            item = total[i,j,:]
            if item[9] == 0:
                edges.itemset((i,j), item[0] > item[2] and item[0] > item[7])
            elif item[9] == 1:
                edges.itemset((i,j), item[0] > item[3] and item[0] > item[6])
            elif item[9] == 2:
                edges.itemset((i,j), item[0] > item[5] and item[0] > item[4])
            elif item[9] == 3:
                edges.itemset((i,j), item[0] > item[8] and item[0] > item[1])

    new_img = new_img * edges

    #dual thresholding
    weak = np.zeros(new_img.shape)
    strong = np.zeros(new_img.shape)

    widx = new_img > low
    sidx = new_img > high

    weak[widx] = 50
    strong[sidx] = 255

    dual = cv2.add(weak, strong)

    indices = np.argwhere(dual>=255)

    #hysteresis
    q = Queue.Queue()

    for pair in indices:
        q.put(pair)

    while not q.empty():
        current = q.get()
        dual.itemset((current[0],current[1]), 255)
        #neighbors
        for n in neighbors(current[0], current[1]):
            if n[0] < 0 or n[0] > dual.shape[0]-1 or \
                n[1] < 0 or n[1] > dual.shape[1]-1:

                continue
            else:
                #valid
                if dual.item(n[0],n[1]) == 200:
                    dual.itemset((n[0],n[1]), 255)
                    print n
                    q.put(n)

    final = np.zeros(dual.shape, np.float32)

    fidx = dual == 255
    final[fidx] = 255

    return final

#unoptimized
#takes BGR image, range sigma, domain sigma
#returns filtered image
def bilateral(img, rsigma, dsigma):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


    gaussian_weights = cv2.getGaussianKernel(5, 2*dsigma)
    gaussian_weights = cv2.mulTransposed(gaussian_weights, False)
    gaussian_weights = gaussian_weights.flatten()



    iter = np.zeros((img.shape[0],img.shape[1]))
    final = np.zeros(img.shape, dtype=img.dtype)

    for (i,j), v in np.ndenumerate(iter):
        current = img[i,j,:].astype('float')
        index = 0
        total_weight = 0.0
        sum_l = 0.0
        sum_a = 0.0
        sum_b = 0.0
        for r,c in zip(xrange(i-2, i+3), xrange(j-2, j+3)):
            if ((r >= 0) & (r < img.shape[0]) & (c >= 0) & (c < img.shape[1])):
                n = img[r,c,:].astype('float')
            else:
                n = np.array((0.0,0.0,0.0)) #out of bounds
            color_dist = (float)( (n[0] - current[0])**2 +
                   (n[1] - current[1])**2 +
                   (n[2] - current[2])**2 ) #euclidean distance

            n_weight = 1.0 / (np.sqrt(2.0 * np.pi) * rsigma) * np.exp(-1.0 * color_dist /(2 * rsigma**2))\
                       * gaussian_weights[index]
            sum_l += (n_weight * n[0])
            sum_a += (n_weight * n[1])
            sum_b += (n_weight * n[2])
            total_weight += n_weight
            index += 1
        final.itemset((i, j, 0), sum_l / total_weight)
        final.itemset((i, j, 1), sum_a / total_weight)
        final.itemset((i, j, 2), sum_b / total_weight)

    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    return final

#main function. runs canny, then bilateral, then combines
def main():

    if len(sys.argv) != 6:
        print("Invalid command line arguments (image name, low threshold, high threshold, range sigma, domain sigma)")
        return

    img = cv2.imread(sys.argv[1],-1)

    l_thresh = float(sys.argv[2])
    h_thresh = float(sys.argv[3])
    r_sig = float(sys.argv[4])
    d_sig = float(sys.argv[5])

    print "Beginning Canny Edge Detection..."
    curr = time.clock()
    canny_img = canny(img, l_thresh, h_thresh)
    print "Time Elapsed: %8.2f"% ((time.clock()-curr))
    print "Beginning Bilateral filter..."
    curr = time.clock()
    bilateral_img = bilateral(img, r_sig, d_sig)
    print "Time Elapsed: %8.2f"% ((time.clock()-curr))

    print "Creating cartoon image..."
    inverted = np.ones(canny_img.shape, np.float32)*255 - canny_img
    ret, inv_mask = cv2.threshold(inverted,128,255,cv2.THRESH_BINARY)
    inv_mask = cv2.convertScaleAbs(inv_mask)

    final_img = cv2.bitwise_and(bilateral_img, bilateral_img, mask=inv_mask)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(1)
    fig.canvas.set_window_title("Robert Adams") # set figure title to name
    plt.imshow(final_img, cmap='gray')
    plt.title("Cartoon image")
    plt.show()  #display image and wait for close


main()
