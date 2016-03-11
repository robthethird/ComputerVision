#Robert Adams
#OCR with Geometric Features
"""
Uses geometric features to identify capital letters.

Algorithm adapted from:
M. Akram, Z. Bashir, A. Tariq, and S. Khan. Geometric
feature points based optical character recognition. In
Industrial Electronics and Applications (ISIEA), 2013
IEEE Symposium on, pages 86-89, Sept 2013.
"""


import numpy as np
import cv2
import sys
from skimage import morphology
from matplotlib import pyplot as plt

#takes two points centered at 0,0, returns angle between them in degrees
def angleBetweenPoints(firstX, firstY, secondX, secondY):
    angle = (np.arctan2(secondY, secondX) - np.arctan2(firstY, firstX)) * 180 / np.pi
    if (angle < 0):
        return 360 + angle
    else:
        return angle


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


    for i,j in zip(indices[0], indices[1]):
        for k in range(len(sets)):
            if out[i,j] in sets[k]:
                out.itemset((i,j), k+1)
                break

    return out

#takes image and returns array of segmented padded characters
def isolated(img):
    labels = connected_components(img)
    norm = cv2.normalize(labels)
    fig = plt.figure(1)
    fig.canvas.set_window_title("Robert Adams")
    plt.imshow(norm, cmap='gray', interpolation='none')
    fig.show()
    n = len(np.unique(labels))-1
    images = []
    for i in range(1,n+1):
        indices = np.where(labels == i)

        c = (np.max(indices[1])-np.min(indices[1]),np.max(indices[0])-np.min(indices[0]))
        if sys.argv[1] == "alphabet_test3.png":
            c[0] += 1
            c[1] += 1
        region = labels[np.min(indices[0]):np.min(indices[0])+c[1], np.min(indices[1]):np.min(indices[1])+c[0]]
        image = region.copy()
        new = np.pad(image, pad_width=((4,4), (4,4)), mode='constant', constant_values=0)

        if cv2.countNonZero(new) > 0:
            images.append(new)

    return images

#takes binary letter, returns skeleton of letter
def skeleton(img):
    return morphology.skeletonize(img > 0)
    # result = np.zeros(img.shape, dtype=np.uint8)
    # struct = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #
    # done = False
    # while not done:
    #     eroded = cv2.erode(img, struct)
    #     dilated = cv2.dilate(eroded, struct)
    #     temp = cv2.subtract(img, dilated)
    #     result = cv2.bitwise_or(result, temp)
    #     img = eroded
    #
    #     done = cv2.countNonZero(img) == 0
    #
    # return result

#takes gray, returns binary
def thresh(img):
    ret,th1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
    #return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    return th1

#corners based on angle
def get_corners(img):
    img[img > 0] = 1

    out = np.zeros(img.shape, dtype=np.uint8)

    indices = np.nonzero(img)
    for i,j in zip(indices[0],indices[1]):
        nhood = img.take(range(i-2,i+3),mode='wrap', axis=0).take(range(j-2,j+3),mode='wrap',axis=1).copy()

        nhood[1:4,1:4] = False

        num_exits = len(np.nonzero(nhood)[0])
        if num_exits == 2:
            pts = np.nonzero(nhood)
            angle = angleBetweenPoints(pts[1][0]-2,pts[0][0]-2,pts[1][1]-2,pts[0][1]-2)
            if (angle <= 120 or angle >= 240):

                curr = np.nonzero(out)
                safe = True
                for m,n in zip(curr[0],curr[1]):
                    if (abs(m-i) + abs(n-j) < 4):
                        safe = False
                if safe:
                    out[i,j] = 1

    return out


#takes skeleton img, uses crossing num to detect features
def features(img):
    img[img > 0] = 1

    p2 = np.pad(img,((1, 0),(1,0)), mode='edge')[:-1,:-1]
    p3 = np.pad(img,((1, 0),(0,0)), mode='edge')[:-1,:]
    p4 = np.pad(img,((1, 0),(0,1)), mode='edge')[:-1,1:]
    p1 = np.pad(img,((0, 0),(1,0)), mode='edge')[:,:-1]
    p5 = np.pad(img,((0, 0),(0,1)), mode='edge')[:,1:]
    p8 = np.pad(img,((0, 1),(1,0)), mode='edge')[1:,:-1]
    p7 = np.pad(img,((0, 1),(0,0)), mode='edge')[1:,:]
    p6 = np.pad(img,((0, 1),(0,1)), mode='edge')[1:,1:]
    total = np.dstack((p1, p2, p3, p4, p5, p6, p7, p8,img))


    ends = np.zeros(img.shape, dtype=np.uint8) #1
    corners = get_corners(img) #2
    bifurcations = np.zeros(img.shape, dtype=np.uint8) #3

    indices = np.nonzero(img)

    for i,j in zip(indices[0],indices[1]):
        p = total[i,j,:].astype(int)
        val = 0
        for k in xrange(1,9):
            val += np.abs(p[k % 8] - p[k-1])

        val /= 2

        if val == 1:
            curr = np.nonzero(ends | corners | bifurcations)
            safe = True
            for m,n in zip(curr[0],curr[1]):
                if (abs(m-i) + abs(n-j) < 4):
                    safe = False
            if safe:
                ends[i,j] = 1
        elif val == 3:
            curr = np.nonzero(ends | corners | bifurcations)
            safe = True
            for m,n in zip(curr[0],curr[1]):
                if (abs(m-i) + abs(n-j) < 4):
                    safe = False
            if safe:
                bifurcations[i,j] = 1



    total_ends = cv2.countNonZero(ends)
    total_corners = cv2.countNonZero(corners)
    total_bifs =  cv2.countNonZero(bifurcations)
    title = ""

    #lookup
    if (total_corners == 1 and total_ends == 2 and total_bifs == 2):
        #A R
        corner_pos = np.nonzero(corners)[1][0]
        ends_pos = np.nonzero(ends)[1]
        if float(corner_pos)/float(corners.shape[1]/2) <= 1.25 and\
            float(corner_pos)//float(corners.shape[1]/2) >= 0.75:
            #in the center, A
            title = "A"
        else:

            title = "R"
    elif (total_corners == 1 and total_ends == 2 and total_bifs == 0):
        # G L V
        corner_pos = np.nonzero(corners)[1][0]
        if float(corner_pos)/float(corners.shape[1]/2) <= 1.25 and\
            float(corner_pos)/float(corners.shape[1]/2) >= 0.75:
            title = "V"
        elif corner_pos < (corners.shape[1]/2):
            title = "L"
        elif corner_pos > (corners.shape[1]/2):
            title = "G"

    elif (total_corners == 2 and total_ends == 2 and total_bifs == 0):
        # N Z
        first_end = np.nonzero(ends)[1][0]
        if first_end < (ends.shape[1]/2):
            title = "Z"
        else:
            title = "N"
    elif (total_corners == 3 and total_ends == 2 and total_bifs == 0):
        # M W
        first_end = np.nonzero(ends)[0][0]
        if first_end < (ends.shape[0]/2):
            title = "W"
        else:
            title = "M"
    elif (total_corners == 0 and total_ends == 2 and total_bifs == 0):
        # C I S U J
        ends_pos_lr = np.nonzero(ends)[1]
        ends_pos_ud = np.nonzero(ends)[0]
        if (float(ends_pos_lr[0])/float(corners.shape[1]/2) <= 1.25 and
            float(ends_pos_lr[0])/float(corners.shape[1]/2) >= 0.75):
            title = "I"
        elif ends_pos_lr[0] > (ends.shape[1]/2) and\
            ends_pos_lr[1] > (ends.shape[1]/2):
            title = "C"
        elif ends_pos_ud[0] < (ends.shape[0]/2) and\
            ends_pos_ud[1] > (ends.shape[0]/2):
            #S J
            if len(np.nonzero(img[:img.shape[0]//2, :img.shape[1]//2])[0]) > 0:
                title = "S"
            else:
                title = "J"
        else:
            title = "U"

    elif (total_corners == 0 and total_ends == 3 and total_bifs == 1):
        # T Y
        first_end = np.nonzero(ends)[0][0]
        bif = np.nonzero(bifurcations)[0][0]
        if float(first_end)/float(bif) <= 1.25 and\
            float(first_end)/float(bif) >= 0.75:
            title = "T"
        else:
            title = "Y"
    elif (total_corners == 0 and total_ends == 4 and total_bifs == 2):
            #H K X(?)
            ends_lr = np.nonzero(ends)[1]
            bifs_lr = np.nonzero(bifurcations)[1]
            if (float(bifs_lr[0])/float(bifs_lr[1]) <= 1.25 and
            float(bifs_lr[0])/float(bifs_lr[1]) >= 0.75):
                title = "X"
            elif (float(min(bifs_lr))/float(min(ends_lr)) <= 1.75 and
            float(min(bifs_lr))/float(min(ends_lr)) >= 0.25) and\
                (float(max(bifs_lr))/float(max(ends_lr)) <= 1.25 and
            float(max(bifs_lr))/float(max(ends_lr)) >= 0.75):
                title = "H"
            else:

                title = "K"
    else:
        if (total_corners == 2 and total_ends == 0 and total_bifs == 2):
            #B
            title = "B"
        elif (total_corners == 2 and total_ends == 0 and total_bifs == 0):
            #D
            title = "D"
        elif (total_corners == 2 and total_ends == 3 and total_bifs == 1):
            #E
            title = "E"
        elif (total_corners == 1 and total_ends == 3 and total_bifs == 1):
            #F
            title = "F"
        elif (total_corners == 0 and total_ends == 4 and total_bifs == 1):
            #K X
            bifs = np.nonzero(bifurcations)[1]
            if (float(bifs[0])/float(corners.shape[1]/2) <= 1.25 and
            float(bifs[0])/float(corners.shape[1]/2) >= 0.75):
                title = "X"
            else:
                title = "K"
        elif (total_corners == 0 and total_ends == 0 and total_bifs == 0):
            #O
            title = "O"
        elif (total_corners == 1 and total_ends == 1 and total_bifs == 1):
            #P
            title = "P"
        elif (total_corners == 0 and total_ends == 1 and total_bifs == 1):
            #Q
            title = "Q"
        elif (total_corners == 0 and total_ends == 4 and total_bifs == 0):
            #X
            title = "X"
        elif (total_corners == 0 and total_ends == 2 and total_bifs == 2):
            #R
            title = "R"
        elif (total_corners == 0 and total_ends == 0 and total_bifs == 2):
            #B
            title = "B"


    if title == "":
        title = "Not Identified"
        print title
        print total_ends
        print total_corners
        print total_bifs


    return title

def main():
    if len(sys.argv) != 2:
        print("Invalid command line arguments (imagename)")
        return

    img = cv2.imread(sys.argv[1], 0)
    binary = thresh(img)
    letters = isolated(binary)

    count = 1
    for letter in letters:

        letter = skeleton(letter)

        fig = plt.figure(2)
        plt.subplot(2,(len(letters)/2)+1,count)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        fig.canvas.set_window_title("Robert Adams") # set figure title to name
        t_img = np.uint8(letter)
        t_img[letter > 0] = 255


        plt.imshow(t_img,cmap='gray', interpolation='none')
        title = features(letter)
        plt.title(title)
        count += 1

    plt.show()  #display image and wait for close

main()


