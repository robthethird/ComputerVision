#Robert Adams
#Fourier transform visualization and hybrid image
"""
Creates a time lapse visualization of the effects of a fourier transform on an image.
Combines two images using highpass and lowpass filters to create hybrid.

Algorithms implemented from:
Cooley, J.W. and Tukey, J.W. An Algorithm for the Machine Calculation of Complex
Fourier Series. Mathematics of Computation. Vol. 19, No. 90 (Apr., 1965) , 297-301.
Oliva, A., Torralba, A., and Schyns, P. G. 2006. Hybrid images. ACM Trans. Graph.
(Proc. SIGGRAPH) 25, 527{532.
"""

import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

def addText(img,text,y=0,x=0):
    """
    A helper function that addes text on top of an image
    :param image: Image for text to be added to
    :param text: String to be placed on image
    :param y: The y offset of the anchor
    :param x: The x offset of the anchor
    """
    image2 = np.copy(img)
    cv2.putText(image2,text,(5+x,25+y),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0.0),4)
    cv2.putText(image2,text,(5+x,25+y),cv2.FONT_HERSHEY_TRIPLEX,0.7,(1.0),2)
    return image2

def logMagDisp(inFFT):
    """
    Calcuates the log magnitude of an image for display
    The image should also be shifted so the fundamental frequency
    is in the center of the image
    :param inFFT: the input complex array from the FFT
    :return: output image from 0 to 1
    """
    shift = np.fft.fftshift(inFFT)
    comp = np.log10(np.absolute(shift))
    real = np.array([[x.real for x in row] for row in comp])
    return real

def dFFT(list):
    """
    Helper recursive 1d FFT for 2d fft
    :param list: 1d np array to be transformed
    :return: 1d complex array of nums
    """
    size = list.shape[0]

    #base
    if size == 1:
        return list
    else:
        evens = dFFT(list[::2]) #0 by 2s
        odds = dFFT(list[1::2]) #1 by 2s
        #Precompute twiddles - faster than for loop (i think...)
        terms = np.exp(-2j * np.pi * np.arange(size) / size) #-2pi i k / N (is this necessary? maybe should do half)
        # combine pos and neg
        combined = np.concatenate([evens + terms[:size/2] * odds, evens + terms[size/2:] * odds])
        return combined

def dFFT2(img, use_numpy=True):
    """
    Calculated the discrete Fourier
    Transform of a 2D image
    :param img: image to be transformed
    :return: complex valued result
    """
    if use_numpy:
        return np.fft.fft2(img)
    else:
        rows = np.array([dFFT(row) for row in img])
        cols = rows.transpose()
        final = np.array([dFFT(col) for col in cols]).transpose()
        return final


def idFFT(list):
    """
    Helper 1d iFFT for 2d ifft
    uses forward fft then scales
    :param list: 1d complex np array to be transformed
    :return: 1d complex array of nums
    """

    t = np.conj(list) # conjugate
    t = dFFT(t) # forward fft
    t = np.conj(t) # conjugate again
    t /= list.shape[0] #scale by size

    return t


def idFFT2(img, use_numpy=True):
    """
    Calculated the inverse discrete Fourier
    Transform of a 2D image
    :param img: image to be transformed
    :return: complex valued result
    """
    if use_numpy:
        return np.fft.ifft2(img)
    else:
        rows = np.array([idFFT(row) for row in img])
        cols = rows.transpose()
        final = np.array([idFFT(col) for col in cols]).transpose()
        return final


def outImgArray(img,inFFT,usedFFT,deltas,scaledSin,outImg):
    """
    Creates the output image for display
    Merges the six images together into a display
    Adds textual labels to images
    Calls logMagDisp for each of the FFTs
    ALL IMAGES MUST BE THE SAME SHAPE!!!
    :param img: input image
    :param inFFT: dFFT of the input image
    :param usedFFT: components of dFFT used so far
    :param deltas: the pair of points (current component)
    :param scaledSin: normalized idFFT of the current component
    :param outImg: idFFT of the usedFFT
    :return: the output image to be displayed
    """
    imgDisp = addText(img,"Original Image")
    inFFTdisp = logMagDisp(inFFT)
    inFFTdisp = addText(inFFTdisp,"Input dFFT log(Mag)")
    usedFFTdisp = logMagDisp(usedFFT)
    usedFFTdisp = addText(usedFFTdisp,"Used dFFT log(Mag)")
    topRow = np.hstack((imgDisp,inFFTdisp,usedFFTdisp))
    deltaDisp = logMagDisp(deltas)
    deltaDisp = addText(deltaDisp,"Current Component")
    bottomRow = np.hstack((deltaDisp,scaledSin,outImg))
    bottomRow = addText(bottomRow,"Scaled Sine",0,img.shape[1])
    bottomRow = addText(bottomRow,"Inverse dFFT",0,img.shape[1]*2)
    outImg = np.vstack((topRow,bottomRow))
    return outImg

def mainLoop(img,name="Temp"):
    if len(img.shape) != 2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)/255.0
    paused = True
    lastKey = ord(' ')

    cv2.namedWindow(name+"'s FFT Display",cv2.WINDOW_AUTOSIZE)
    inFFT = dFFT2(img)
    usedFFT = np.zeros(img.shape, dtype=np.complex)


    # sort ordering by magnitudes of fourier image
    mag = np.absolute(inFFT)
    indexList = [(i,j) for i in range(mag.shape[0]) for j in range(mag.shape[1])]
    indexList.sort(key=lambda x: -mag[x])

    for i,j in indexList[::2]:
    # double for loop over sorted indices made more sense to me
    #while components < img.size//2:
        comp = np.zeros(img.shape, dtype=np.complex)

        curr = inFFT[i,j]
        usedFFT[i, j] =  comp[i, j] = curr
        usedFFT[-i, -j] = comp[-i, -j] = curr.conjugate()

        sine = idFFT2(comp)
        scaledSin = np.array([[x.real for x in row] for row in sine])
        scaledSin *= 255
        outImg = idFFT2(usedFFT)
        outImg = np.array([[x.real for x in row] for row in outImg])
        out = outImgArray(img,inFFT,usedFFT,comp,scaledSin,outImg)
        cv2.imshow(name+"'s FFT Display",out)
        if paused:
            lastKey = cv2.waitKey(0) & 0xFF
        else:
            lastKey = cv2.waitKey(5) & 0xFF
        if lastKey == ord('q') or lastKey == ord('Q') or lastKey == 27:
            break
        elif lastKey == ord('p') or lastKey == ord('P'):
                paused = not paused

    cv2.destroyAllWindows()


def highpass(img, sigma):
    """
    Performs highpass filter on fourier image
    :param img: Fourier image
    :param sigma: sigma for gaussian filter
    :return: filtered fourier image
    """
    cx = img.shape[0]/2
    cy = img.shape[1]/2

    #make filter
    gaussian = np.zeros(np.shape(img))
    for (i,j), v in np.ndenumerate(img):
        gaussian[i,j] = 1 - np.exp(-1.0 * ((i - cx)**2 + (j - cy)**2) / (2 * sigma**2))

    #shift image
    shifted = np.fft.fftshift(img)
    filtered = shifted * gaussian
    unshifted = np.fft.ifftshift(filtered)

    return unshifted

def lowpass(img, sigma):
    """
    Performs lowpass filter on fourier image
    :param img: Fourier image
    :param sigma: sigma for gaussian filter
    :return: filtered fourier image
    """
    cx = img.shape[0]/2
    cy = img.shape[1]/2

    #make filter
    gaussian = np.zeros(np.shape(img))
    for (i,j), v in np.ndenumerate(img):
        gaussian[i,j] = np.exp(-1.0 * ((i - cx)**2 + (j - cy)**2) / (2 * sigma**2))

    #shift image
    shifted = np.fft.fftshift(img)
    filtered = shifted * gaussian
    unshifted = np.fft.ifftshift(filtered)

    return unshifted

def hybrid_image(img_low, sigma_low, img_high, sigma_high):
    """
    Creates hybrid image using two images and sigmas
    :param img_low: image to use for low filter
    :param sigma_low: sigma to use for low pass
    :param img_high: image to use for high pass
    :param sigma_high: sigma to use for high pass
    :return:
    """
    dft_low = dFFT2(img_low, False)
    dft_high = dFFT2(img_high, False)
    low = lowpass(dft_low, sigma_low)
    high = highpass(dft_high, sigma_high)

    return idFFT2(low, False) + idFFT2(high, False)

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Invalid command line arguments (vis img name, img_low, sigma_low, img_high, sigma_high)")

    img = cv2.imread(sys.argv[1],0)
    name = "Robert Adams"

    #visualization (using system implementation of fft and ifft)
    mainLoop(img,name)

    # my hybrid image (using my implementations of fft and ifft)
    low_img = cv2.imread(sys.argv[2],0)
    if len(low_img.shape) != 2:
        low_img = cv2.cvtColor(low_img,cv2.COLOR_BGR2GRAY)
    low_img = low_img.astype(np.float64)/255.0

    high_img = cv2.imread(sys.argv[4],0)
    if len(high_img.shape) != 2:
        high_img = cv2.cvtColor(high_img,cv2.COLOR_BGR2GRAY)
    high_img = high_img.astype(np.float64)/255.0

    outImg = hybrid_image(low_img, float(sys.argv[3]), high_img, float(sys.argv[5]))
    outImg = [[x.real for x in row] for row in outImg]
    fig = plt.figure(1)
    fig.canvas.set_window_title("Robert Adams") # set figure title to name
    plt.imshow(outImg, cmap='gray')
    plt.title("Hybrid Image")
    plt.show()  #display image and wait for close
