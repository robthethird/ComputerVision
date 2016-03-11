#Robert Adams
#Feature matching on transformed images
"""
Uses harris corners to detect features and histogram of gradients to
match the features between similar images of the same object.

Algorithms implemented from:
Harris, C., Stephens, M. A combined corner and edge detector. Proceedings
of the Alvey Vision Conference. 1988. pp. 147{151.
"""

import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

#takes image and integer number
#returns tuple containing list of x and y vertices
#representing the n largest indices in img
def getLargest(img, n):
    ind = ([],[])
    for i in xrange(n):
        idx = np.unravel_index(img.argmax(), img.shape)
        ind[0].append(idx[0])
        ind[1].append(idx[1])
        img[idx] = 0

    return ind

#takes image and size of window
#performs local non max suppression on image
def nonMaximalSupress(img,size):
    x_size, y_size = size
    m, n = img.shape
    for i in xrange(0,m-x_size+1):
        for j in xrange(0,n-y_size+1):
            window = img[i:i+x_size, j:j+y_size]
            if np.sum(window)==0:
                localMax=0
            else:
                localMax = np.amax(window)
            maxCoord = np.argmax(window)
            window[:] = 0 #suppress all
            window.flat[maxCoord] = localMax #except max
    return img

#takes an image, sigma for gaussian, and alpha for response function
#returns an image where each value is the result of the cornerness
#response function
def harrisCorners(img, sigI, alpha):
   #out = cv2.cornerHarris(img,2,5,0.04)

    mag_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    mag_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    gaussian = cv2.getGaussianKernel(5,sigI)
    x_2 = cv2.sepFilter2D(mag_x * mag_x, -1, gaussian, gaussian)
    y_2 = cv2.sepFilter2D(mag_y * mag_y, -1, gaussian, gaussian)
    x_y = cv2.sepFilter2D(mag_x * mag_y, -1, gaussian, gaussian)

    det = x_2*y_2 - x_y**2
    trace = x_2 + y_2

    out = det - alpha * trace**2
    out = nonMaximalSupress(out, (3,3))
    return out

#takes image and list of feature points
#computes histogram of gradients feature points
#returns list of tuples, where the first value in the tuple
#is the feature point and the second value is the feature vector
def histOfGrad(img, points):
    out = []
    g = cv2.getGaussianKernel(5, -1)
    new_img = cv2.sepFilter2D(img, -1, g, g)
    mag_x = cv2.Sobel(new_img,cv2.CV_64F,1,0,ksize=5)
    mag_y = cv2.Sobel(new_img,cv2.CV_64F,0,1,ksize=5)
    new_img = np.hypot(mag_x, mag_y)

    theta = np.arctan2(mag_y, mag_x) # save edge direction
    theta = ((theta) * 4.0 / (2*np.pi) + 8.5).astype('int') % 8 # convert edge direction into 0-7


    indices = np.nonzero(points)
    for i,j in zip(indices[0], indices[1]):
        i_new = i - 15
        j_new = j - 15
        vec = []
        for k in range(4):
            for l in range(4):
                bins = [0] * 8
                ts = theta.take(range(i+k,i+k+4),mode='wrap', axis=0).take(range(j+l,j+l+4),mode='wrap',axis=1)
                mags = new_img.take(range(i+k,i+k+4),mode='wrap', axis=0).take(range(j+l,j+l+4),mode='wrap',axis=1)

                for (m,n), o in np.ndenumerate(ts):
                    bins[ts[m,n]] += mags[m,n]
                #print bins
                for b in bins:
                    vec.append(b)

        total = sum(vec)
        if total > 0:
            vec = [float(val)/total for val in vec] #normalize

        vec = np.array(vec)
        out.append(((i,j),vec))

    return out

#takes two lists of point-vector tuples for two images
#matches two sets of feature vectors using L2 norm
#returns two lists of equal length where the value at position i
#in the first list is the point in the first image that matches the point
#at position i in the second list for the second image
def match(p1, p2):
    out = ([],[])

    while len(p1) > 0:
        current = p1.pop()
        min_point = p2[0]
        min_dist = np.linalg.norm(current[1]-min_point[1])
        for point in p2:
            dist = np.linalg.norm(current[1]-point[1])
            if dist < min_dist:
                min_point = point
                min_dist = dist
        out[0].append(current[0])
        out[1].append(min_point[0])
        p2.remove(min_point) #assuming 1-1 correspondence

    return out

#takes two image names, uses harris corner detection
#and hog features to match the two images
#returns combined image and lists of matches
def feature_match(name1, name2, sigma, alpha):
    print "Matching " + name1 + " with " + name2
    img = cv2.imread(name1,0)
    corners = harrisCorners(img, float(sigma),float(alpha))
    ind = getLargest(corners, 10)
    corners[ind] = 255
    corners[corners != 255] = 0

    img2 = cv2.imread(name2,0)
    corners2 = harrisCorners(img2, float(sigma),float(alpha))
    ind = getLargest(corners2, 10)
    corners2[ind] = 255
    corners2[corners2 != 255] = 0


    vec1 = histOfGrad(img, corners)
    vec2 = histOfGrad(img2, corners2)



    matches = match(vec1, vec2)
    print "Image 1:",
    print matches[0]
    print "Image 2:",
    print matches[1]

    combined = np.concatenate((img, img2), axis=1)
    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)

    return combined, matches

def main():
    if len(sys.argv) > 1 and len(sys.argv) != 5:
        print("Invalid command line arguments ([sigma, alpha, image1, image2])")
        return

    if len(sys.argv) == 5:
        combined, matches = feature_match(sys.argv[3], sys.argv[4], sys.argv[1], sys.argv[2])

        for i in range(len(matches[0])):
            cv2.line(combined, tuple(reversed(matches[0][i])), tuple((combined.shape[1]/2+matches[1][i][1], matches[1][i][0])),(0,0,255),1)

        fig = plt.figure(0)
        fig.canvas.set_window_title("Robert Adams") # set figure title to name
        plt.imshow(combined)
        plt.title("Unlabeled User Test")
        fig.show()

    #test 1
    combined, matches = feature_match("test1_1.jpg", "test1_2.jpg", 0, 0.04)

    for i in range(len(matches[0])):
        cv2.line(combined, tuple(reversed(matches[0][i])), tuple((combined.shape[1]/2+matches[1][i][1], matches[1][i][0])),(0,255,0),1)
    cv2.line(combined, tuple(reversed(matches[0][4])), tuple((combined.shape[1]/2+matches[1][4][1], matches[1][4][0])),(255,0,0),1)
    cv2.line(combined, tuple(reversed(matches[0][6])), tuple((combined.shape[1]/2+matches[1][6][1], matches[1][6][0])),(255,0,0),1)
    cv2.line(combined, tuple(reversed(matches[0][9])), tuple((combined.shape[1]/2+matches[1][9][1], matches[1][9][0])),(255,0,0),1)

    fig = plt.figure(1)
    fig.canvas.set_window_title("Robert Adams") # set figure title to name
    plt.imshow(combined)
    plt.title("Labeled Test 1")
    fig.show()

    #test 2
    combined, matches = feature_match("test2_1.jpg", "test2_2.jpg", 0, 0.04)

    for i in range(len(matches[0])):
        cv2.line(combined, tuple(reversed(matches[0][i])), tuple((combined.shape[1]/2+matches[1][i][1], matches[1][i][0])),(255,0,0),1)
    cv2.line(combined, tuple(reversed(matches[0][0])), tuple((combined.shape[1]/2+matches[1][0][1], matches[1][0][0])),(0,255,0),1)
    cv2.line(combined, tuple(reversed(matches[0][6])), tuple((combined.shape[1]/2+matches[1][6][1], matches[1][6][0])),(0,255,0),1)
    cv2.line(combined, tuple(reversed(matches[0][7])), tuple((combined.shape[1]/2+matches[1][7][1], matches[1][7][0])),(0,255,0),1)
    cv2.line(combined, tuple(reversed(matches[0][8])), tuple((combined.shape[1]/2+matches[1][8][1], matches[1][8][0])),(0,255,0),1)

    fig = plt.figure(2)
    fig.canvas.set_window_title("Robert Adams") # set figure title to name
    plt.imshow(combined)
    plt.title("Labeled Test 2")
    fig.show()

    #test 3
    combined, matches = feature_match("test3_1.jpg", "test3_2.jpg", 0, 0.04)

    for i in range(len(matches[0])):
        cv2.line(combined, tuple(reversed(matches[0][i])), tuple((combined.shape[1]/2+matches[1][i][1], matches[1][i][0])),(0,255,0),1)
    cv2.line(combined, tuple(reversed(matches[0][6])), tuple((combined.shape[1]/2+matches[1][6][1], matches[1][6][0])),(255,0,0),1)

    fig = plt.figure(3)
    fig.canvas.set_window_title("Robert Adams") # set figure title to name
    plt.imshow(combined)
    plt.title("Labeled Test 3")
    plt.show()

    return

main()
