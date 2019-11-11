import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage import exposure
import h5py
from scipy import fftpack
from scipy.ndimage.filters import gaussian_filter


def lowFreqCut(radius = 50, size = (512,512),smoothing = True):
    """Makes a mask for an FFT of size given by size and will
    eliminate all frequencies within a given radius"""
    mask = np.ones(size)
    for x in np.arange(-size[0],size[0]):
        for y in np.arange(-size[1],size[1]):
            r = np.sqrt(np.square(x)+np.square(y))
            if r < radius:
                mask[x,y] = 0
    if smoothing == True:
        mask = gaussian_filter(mask,2)
    return mask

def makeAperture(radius = 100, size = (512,512),smoothing = True):
    """Makes a matrix representation of an aperture with matrix size determined
    by size and aperture size given by radius."""
    aperture = np.zeros(size)
    for x in np.arange(-aperture.shape[0],aperture.shape[0]):
        for y in np.arange(-aperture.shape[1],aperture.shape[1]):
            r = np.sqrt(np.square(x)+np.square(y))
            if r < radius:
                aperture[x,y] = 1
    if smoothing == True:
        aperture = gaussian_filter(aperture,2)
    return aperture


def freqCutBoth(imagesData, highRadius,lowRadius, h5dir, smoothing = True):
    """Asks for h5 image stack as input and a sirectory to save new filtered h5
    file to. Will perform both a high and low frequency cut"""
    numImages = imagesData.shape[0]
    newh5 = input('Input name for new h5 file: ')
    newh5name = h5dir + '/' + newh5
    aperture = makeAperture(highRadius,(imagesData.shape[1],imagesData.shape[2]),smoothing)
    lowmask = lowFreqCut(lowRadius,(imagesData.shape[1],imagesData.shape[2]),smoothing)
    fout = []
    for idx in np.arange(0,numImages,1):
        img = imagesData[idx].reshape((imagesData.shape[1],imagesData.shape[2]))
        fft = fftpack.fft2(img)
        fft = fft*lowmask
        fft = fft * aperture
        newimg = np.real(fftpack.ifft2(fft))
        fout.append(newimg)
    fout = np.asanyarray(fout)
    fout = np.expand_dims(fout,axis = 3)
    i = h5py.File(newh5name, 'w')
    i.create_dataset('images', fout.shape, data=fout)
    print('Done!')
    return fout

def freqCutHigh(highRadius):
    """Asks for h5 image stack as input and a sirectory to save new filtered h5
    file to. Will perform a high and low frequency cut"""
    ogh5 = input('input directory and h5 filename containing image stack to be processed ')
    newh5 = input('input label for h5 file ')
    imagesFile = h5py.File(ogh5,'r')
    imagesData = imagesFile['images']
    numImages = imagesData.shape[0]//imagesData.shape[1]
    h5dir = ''
    for strpart in ogh5.split('/')[0:-1]:
        h5dir = h5dir + '/' + strpart
    newh5name = h5dir + '/' + newh5
    i = h5py.File(newh5name, 'w')
    i.create_dataset('images', (imagesData.shape[0],imagesData.shape[1]), maxshape = (None, None))
    aperture = makeAperture(highRadius,(imagesData.shape[1],imagesData.shape[1]))
    for idx in np.arange(0,numImages,1):
        img = imagesData[idx*imagesData.shape[1]:(idx+1)*imagesData.shape[1]]
        fft = fftpack.fft2(img)
        fft = fft * aperture
        newimg = np.real(fftpack.ifft2(fft))
        i['images'][idx*img.shape[0]:img.shape[0]*(idx+1),0:img.shape[1]] = newimg
    print('Done!')

def freqCutLow(lowRadius):
    """Asks for h5 image stack as input and a sirectory to save new filtered h5
    file to. Will perform low frequency cut"""
    ogh5 = input('input directory and h5 filename containing image stack to be processed ')
    newh5 = input('input label for h5 file ')
    imagesFile = h5py.File(ogh5,'r')
    imagesData = imagesFile['images']
    numImages = imagesData.shape[0]//imagesData.shape[1]
    h5dir = ''
    for strpart in ogh5.split('/')[0:-1]:
        h5dir = h5dir + '/' + strpart
    newh5name = h5dir + '/' + newh5
    i = h5py.File(newh5name, 'w')
    i.create_dataset('images', (imagesData.shape[0],imagesData.shape[1]), maxshape = (None, None))
    lowmask = lowFreqCut(lowRadius,(imagesData.shape[1],imagesData.shape[1]))
    for idx in np.arange(0,numImages,1):
        img = imagesData[idx*imagesData.shape[1]:(idx+1)*imagesData.shape[1]]
        fft = fftpack.fft2(img)
        fft = fft * lowmask
        newimg = np.real(fftpack.ifft2(fft))
        i['images'][idx*img.shape[0]:img.shape[0]*(idx+1),0:img.shape[1]] = newimg
    print('Done!')

def immFFT(fftpic):
    plt.figure(figsize=(10,10))
    plt.imshow(np.real(np.sqrt(np.square(fftpack.fftshift(fftpic)))).astype('uint8'))
    plt.axis('off')

def freqCutBothH5(highRadius,lowRadius, smoothing = True):
    """Asks for h5 image stack as input and a sirectory to save new filtered h5
    file to. Will perform both a high and low frequency cut"""
    ogh5 = input('input directory and h5 filename containing image stack to be processed ')
    newh5 = input('input label for h5 file ')
    imagesFile = h5py.File(ogh5,'r')
    imagesData = imagesFile['images'][:,:,:,:]
    numImages = imagesData.shape[0]
    h5dir = ''
    for strpart in ogh5.split('/')[0:-1]:
        h5dir = h5dir + '/' + strpart
    newh5name = h5dir + '/' + newh5
    aperture = makeAperture(highRadius,(imagesData.shape[1],imagesData.shape[2]),smoothing)
    lowmask = lowFreqCut(lowRadius,(imagesData.shape[1],imagesData.shape[2]),smoothing)
    fout = []
    for idx in np.arange(0,numImages,1):
        img = imagesData[idx].reshape((imagesData.shape[1],imagesData.shape[2]))
        fft = fftpack.fft2(img)
        fft = fft*lowmask
        fft = fft * aperture
        newimg = np.real(fftpack.ifft2(fft))
        fout.append(newimg)
    fout = np.asanyarray(fout)
    fout = np.expand_dims(fout,axis = 3)
    i = h5py.File(newh5name, 'w')
    i.create_dataset('images', fout.shape, data=fout)
    print('Done!')
    return fout
