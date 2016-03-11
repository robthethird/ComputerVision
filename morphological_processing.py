#Robert Adams
#Morphological processing of shapes
"""
Uses morphological processing to identify and label geometric shapes.
Calculates and displays convex hull of image.

Algorithms implemented from:
Andrew, A. M. Another Efficient Algorithm for Convex Hulls in Two Dimensions.
Info. Proc. Letters 9, 216-219. 1979.
He, L., Chao, Y., and Suzuki, K. A Run-Based Two-Scan Labeling Algorithm. in
Image Processing, IEEE Transactions on , vol.17, no.5, pp.749-756, May 2008.
"""


import numpy as np
import cv2
import sys
import time
from matplotlib import pyplot as plt

#label helper function
#takes label text, point, image
#applies text to image
def label(img, txt, point):
    t_size, b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    new_x = point[1]-t_size[0]/2
    new_y = point[0]+t_size[1]/2
    cv2.putText(img, txt, (new_x, new_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)

#clockwise/counter clockwise turn for convex hull
#takes three (x,y) pairs
#returns integer representing cross product
def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

#takes a list of equivalences
#returns a list of disjoint sets
def union(a):
    a = map(set, a)
    unions = []
    for item in a:
        temp = []
        for s in unions:
            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        unions = temp
    return unions

#takes a binary or grayscale image, structuring element, and Bool indicating operation
#returns transformed image after applying max or min using structuring element
def morph(img, struct, max=True):
    a = img.view(np.ma.MaskedArray)

    n = struct.shape[0]
    w_size = (struct.shape[0] - 1) / 2

    mask = np.logical_not(struct)

    out = np.zeros(img.shape)
    for (i,j), v in np.ndenumerate(out):
        win = a.take(range(i-w_size,i+w_size+1),mode='wrap', axis=0).take(range(j-w_size,j+w_size+1),mode='wrap',axis=1)#neighbors(img, i, j, n)
        win.mask = mask
        if max:
            out.itemset((i,j), win.max())
        else:
            out.itemset((i,j), win.min())

    return out

#takes binary (or grayscale) image and structuring element
#returns image dilated by the structuring element
def dilate(img, struct):
    return morph(img, struct, True)

#takes binary (or grayscale) image and structuring element
#returns image eroded by the structuring element
def erode(img, struct):
    return morph(img, struct, False)


#takes binary image
#returns array of labeled components
def connected_components(img):
    out = np.zeros(img.shape)

    num_comps = 0
    relations = []

    indices = np.nonzero(img)

    for i,j in zip(indices[0], indices[1]):
        win = out.take(range(i-1,i+1),mode='wrap', axis=0).take(range(j-1,j+2),mode='wrap',axis=1).flatten()
        val = 0
        for n in win:
            if n != 0:
                if val == 0:
                    val = n
                elif val < n:
                    relations.append([val,n])
                elif val > n:
                    relations.append([n,val])
                    val = n
        if val == 0:
            num_comps += 1
            out.itemset((i,j), num_comps)
            relations.append([num_comps])
        else:
            out.itemset((i,j), val)

    sets = union(relations)

    #relabeled = np.zeros(out.shape)
    #for k in range(len(sets)):
    #    relabeled[out in sets[k]] = k+1

    for i,j in zip(indices[0], indices[1]):
        for k in range(len(sets)):
            if out[i,j] in sets[k]:
                out.itemset((i,j), k+1)
                break

    return out

#takes labeled image
#return list of areas, one for each component
def areas(labels):
    n = len(np.unique(labels))-1
    out = np.zeros(n)
    indices = np.nonzero(labels)
    for i,j in zip(indices[0], indices[1]):

        out[int(labels.item(i,j)) - 1] = out[int(labels.item(i,j)) - 1] + 1

    return out

#takes labeled image
#return list of perimeters, one for each component
def perimeters(labels):
    n = len(np.unique(labels))-1
    struct = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    perims = labels - erode(labels, struct)

    out = np.zeros(n)
    indices = np.nonzero(perims)
    for i,j in zip(indices[0], indices[1]):

        out[int(perims.item(i,j)) - 1] = out[int(perims.item(i,j)) - 1] + 1

    return out

#takes labeled image
#returns list of (x,y) pairs representing width and height of the bounding box of each component
def width_height(labels):
    n = len(np.unique(labels))-1
    out = []
    for i in range(1,n+1):
        indices = np.where(labels == i)

        c = (np.max(indices[1])-np.min(indices[1]),np.max(indices[0])-np.min(indices[0]))
        out.append(c)
    return out


#takes labeled image
#return list of centroid pairs (x,y)
def centroids(labels):
    n = len(np.unique(labels))-1
    out = []
    for i in range(1,n+1):
        indices = np.where(labels == i)
        c = (int(np.mean(indices[0])), int(np.mean(indices[1])))
        out.append(c)
    return out

#takes labeled image, two labels to be compared
#return distance between two objects in labeled image
def distance(labels, o1, o2):
    index1 = o1 - 1
    index2 = o1 - 2
    c = centroids(labels)
    c1 = c[o1]
    c2 = c[o2]
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

#monotone chain convex hull
#takes array of object points in (x,y) format
#returns list of points describing the convex hull of image
def convex_hull(nonzero):
    # Build lower hull
    lower = []
    for p in nonzero:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull (heading back)
    upper = []
    for p in reversed(nonzero):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    #combine
    return lower[:-1] + upper[:-1]

def main():
    if len(sys.argv) != 3:
        print("Invalid command line arguments (image name, threshold value)")
        return

    img = cv2.imread(sys.argv[1], 0)

    print "Beginning processing..."
    curr = time.clock()


    #binary threshold
    ret,thr = cv2.threshold(img,float(sys.argv[2]),255,cv2.THRESH_BINARY_INV)
    #thr = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)

    #erode to clean up image
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    e = erode(thr, struct)
    #e = cv2.erode(e, struct)

    # labeled image
    labels = connected_components(e)
    #labels[labels != 7] = 0

    #normalize for output
    norm = cv2.normalize(labels)

    #region properties

    #perimeters + areas
    p = perimeters(labels)
    a = areas(labels)

    #bounding box width and height
    r = width_height(labels)

    #distance
    #print distance(labels,1,3)

    #centroids
    c = centroids(labels)


    for i in range(len(np.unique(labels))-1):
        perimeter = p[i]
        area = a[i]
        rect = r[i]
        center = c[i]



        if area/(float(rect[0])*float(rect[1])) >= 1:
            #rect or square
            if float(rect[0])/float(rect[1]) <= 1.04 and\
                float(rect[0])/float(rect[1]) >= 0.96:
                #square
                label(norm, "Square", center)
            else:
                #rect
                label(norm, "Rectangle", center)

        elif perimeter/(2*float(rect[0]) + 2*float(rect[1])) >= 0.96 and\
            perimeter/(2*float(rect[0]) + 2*float(rect[1])) <= 1.04:
            #cross
            label(norm, "Cross", center)

        elif area/(float(rect[1])/2 * float(rect[0])/2 * np.pi) <= 1.04 and\
            area/(float(rect[1])/2 * float(rect[0])/2 * np.pi) >= 0.96:
            #ellipse or circle
            if float(rect[0])/float(rect[1]) <= 1.04 and\
                float(rect[0])/float(rect[1]) >= 0.96:
                #circle
                label(norm, "Circle", center)
            else:
                #ellipse
                label(norm, "Ellipse", center)
        else:
            indices= np.transpose(np.where(labels == i+1))
            hull = convex_hull(indices)
            if len(hull) == 5:
                #5 point star
                label(norm, "Star", center)
            elif len(hull) > 20:
                #curves -- heart
                label(norm, "Heart", center)
            else:
                #star burst
                label(norm, "Burst", center)


    #convex hull
    #d = cv2.dilate(e, struct)
    indices= np.transpose(np.nonzero(e))
    hull = convex_hull(indices)
    for i in xrange(len(hull)-1):
        cv2.line(e, tuple(reversed(hull[i])), tuple(reversed(hull[i+1])),(255,255,255),1)
    cv2.line(e, tuple(reversed(hull[len(hull)-1])), tuple(reversed(hull[0])),(255,255,255),1)

    print "Total Time Elapsed: %8.2f"% ((time.clock()-curr))

    fig = plt.figure(0)
    fig.canvas.set_window_title("Robert Adams") # set figure title to name
    plt.imshow(norm, cmap='gray')
    plt.title("Labeled Output")
    fig.show()

    fig = plt.figure(1)
    fig.canvas.set_window_title("Robert Adams") # set figure title to name
    plt.imshow(e, cmap='gray')
    plt.title("Convex Hull")
    plt.show()

main()
