import sys
from ncempy import mrc
import dm
import numpy as np
import h5py
from glob import glob
import datetime
from scipy.ndimage.filters import gaussian_filter
from skimage import io
import shutil



def filenameDigitFix(oldDir,fileend):
    """Takes directory and file ending (mrc,png,etc) and checks to make sure all
    files in diredctory have same length and ordered numbering."""
    filelist = sorted(glob(oldDir+'/*'+fileend))
    maxLen = 0
    for f in filelist:
        l = len(f)
        if l > maxLen:
            maxLen = l
    for f in filelist:
        if len(f) < maxLen:
            numZeros = maxLen - len(f)
            zeros = ''
            for z in np.arange(0,numZeros):
                zeros += '0'
            part = f.split('/')[-1].split('.')[0]
            newname = part[:-4]+zeros+part[-4:]+'.png'
            shutil.copy(f,newDir+'/'+newname)
        else:
            part = f.split('/')[-1]
            shutil.copy(f,newDir+'/'+part)
        print('Done!')

def mrc2h5stack(mrcdir, h5dir,idxs):
    """Takes in directory where mrcs containing individual images are and
    directory where h5 should go. Stacks all mrcs into one h5 file.
    Also returns stacked images and filenames"""
    mrcd = mrcdir + '/*.mrc'
    filenames = sorted(glob(mrcd))
    print(filenames)
    lenfilename = len(filenames[0])
    for f in filenames:
        if len(f) != lenfilename:
            raise RuntimeError('changing length of file names indicates that files do not have ordered, decimal numbering')
    d = datetime.datetime.today()
    d = d.strftime('%Y%m%d')
    h5name = h5dir + '/' + mrcdir.split('/')[-1]+'_'+d+'.h5'
    array = []
    size = []
    for idx in np.arange(idxs[0],idxs[-1],1):
        struct = mrc.fileMRC(filenames[idx])
        data = struct.getDataset()['data'][0]
        print(data.shape)
        data = np.expand_dims(data,axis = 3)
        array.append(data)
    array = np.asanyarray(array)
    if len(array.shape) == 5:
        print('Had to reshape array because its shape is: ', array.shape)
        array = array.reshape(array.shape[0],array.shape[2],array.shape[3],array.shape[4])
    print(array.shape)
    i = h5py.File(h5name, 'w')
    h5data = i.create_dataset('images', (array.shape[0],array.shape[1],array.shape[2],array.shape[3]), data = array)
    h5data.attrs['files'] = np.array(filenames, dtype = 'S')
    print('Done!')
    return(array,filenames)

def dm32h5stack(mrcdir, h5dir):
    """Takes in directory where dm3s containing individual images are and
    directory where h5 should go. Stacks all mrcs into one h5 file.
    Also returns stacked images and filenames"""
    mrcd = mrcdir + '/*.dm3'
    filenames = sorted(glob(mrcd))
    lenfilename = len(filenames[0])
    for f in filenames:
        if len(f) != lenfilename:
            print('changing length of file names indicates that files do not have ordered, decimal numbering')
            break
    d = datetime.datetime.today()
    d = d.strftime('%Y%m%d')
    h5name = h5dir + '/' + mrcdir.split('/')[-1]+'_'+d+'.h5'
    array = []
    for idx in np.arange(0,len(filenames),1):
        struct = dm.dmReader(filenames[idx])
        data = struct['data']
        data = np.expand_dims(data,axis = 3)
        array.append(data)
    array = np.asanyarray(array)
    if len(array.shape) == 5:
        print('Had to reshape array because its shape is: ', array.shape)
        array = array.reshape(array.shape[0],array.shape[2],array.shape[3],array.shape[4])
    i = h5py.File(h5name, 'w')
    h5data = i.create_dataset('images', (array.shape[0],array.shape[1],array.shape[2],array.shape[3]), data = array)
    h5data.attrs['files'] = np.array(filenames, dtype = 'S')
    print('Done!')
    return(array,filenames)

def xray_correct(stack, mrcdir, h5dir):
    """Uses gaussian filter to remove stray xrays. Saves cleaned stack to h5
    file and returns clean stack. Inputs are image stack,original filenames, and
    original mrc directory"""
    if len(stack.shape) != 4:
        raise TypeError('Input must be in keras format')
    cleanstack = []
    d = datetime.datetime.today()
    d = d.strftime('%Y%m%d')
    h5name = h5dir + '/' + mrcdir.split('/')[-1]+'_xrayclean_'+d+'.h5'
    for idx in np.arange(0,stack.shape[0],1):
        data = gaussian_filter(stack[idx],2)
        cleanstack.append(data)
    cleanstack = np.asanyarray(cleanstack)
    i = h5py.File(h5name, 'w')
    i.create_dataset('images', cleanstack.shape, data = cleanstack)
    print('Done!')
    return(cleanstack)

def oneHot(img,labels):
    """Takes in multiclass segmentation map and list of labels. Returns one hot encoded map."""
    newImg = np.zeros((img.shape[0],img.shape[1],len(labels)))
    mval = max(labels)
    for idx, label in enumerate(labels):
        temp = img.copy()+(mval*2)
        temp[temp == label+(mval*2)] = 1
        temp[temp != 1] = 0
        newImg[:,:,idx] = temp
    return np.asanyarray(newImg)

def maps2h5stack(mapdir,h5dir, labels):
    """Takes in directory where maps are, directory where h5 should go, and
    labels for one hot encoding. Stacks all maps into one h5 file. Also returns
    stacked images and filenames"""
    mdir = mapdir + '/*.png'
    filenames = sorted(glob(mdir))
    lenfilename = len(filenames[0])
    for f in filenames:
        if len(f) != lenfilename:
            print('changing length of file names indicates that files do not have ordered, decimal numbering')
            break
    d = datetime.datetime.today()
    d = d.strftime('%Y%m%d')
    h5name = h5dir + '/' + mapdir.split('/')[-1]+'_'+d+'.h5'
    array = []
    for idx in np.arange(0,len(filenames),1):
        data = oneHot(io.imread(filenames[idx]),labels)
        data = np.expand_dims(data,axis = 0)
        array.append(data)
    array = np.asanyarray(array)
    if len(array.shape) == 5:
        print('Had to reshape array because its shape is: ', array.shape)
        array = array.reshape(array.shape[0],array.shape[2],array.shape[3],array.shape[4])
    i = h5py.File(h5name, 'w')
    h5data = i.create_dataset('images', (array.shape[0],array.shape[1],array.shape[2],array.shape[3]), data = array)
    h5data.attrs['files'] = np.array(filenames, dtype = 'S')
    return(array,filenames)

def cutData(istack, mstack, filenames, maplist, endSize, h5dir, numlabels):
    maplist = maplist.tolist()
    for idx, f in enumerate(filenames):
        temp1 = f.split('/')[-1].split('.')[0]
        temp2 = maplist[idx].split('/')[-1].split('.')[0]
        if temp1 != temp2:
            raise RuntimeError('File list and map list do not match. Maps and images likely do not match.')
    if istack.shape[0] != mstack.shape[0]:
        raise RuntimeError('image stack and map stack do not have the same shape')
    if istack.shape[1]%endSize != 0:
        raise RuntimeError('Images not divisiable by end size')
    h5imgname = input('name for h5 image file:')
    h5mapname = input('name for h5 map file: ')
    h5imgname = h5dir + '/' + h5imgname
    h5mapname = h5dir + '/' + h5mapname
    numImages = (istack.shape[1]//endSize)*(istack.shape[2]//endSize)*istack.shape[0]
    print('Numimages: ', numImages)
    h5shapeimg = (numImages,endSize,endSize,1)
    h5shapemap = (numImages,endSize,endSize,numlabels)
    cutimages = []
    cutmaps = []
#     cutimages = np.empty(h5shapeimg)
#     cutmaps = np.empty(h5shapemap)
    for idx in np.arange(0,istack.shape[0],1):
        image2split = istack[idx]
        map2split = mstack[idx]
        for x in range(0,istack.shape[1],endSize):
            for y in range(0,istack.shape[2],endSize):
                img = image2split[x:x+endSize,y:y+endSize,:]
                mask = map2split[x:x+endSize,y:y+endSize,:]
                cutimages.append(img)
                cutmaps.append(mask)
    cutimages = np.asanyarray(cutimages)
    cutmaps = np.asanyarray(cutmaps)
    print(cutimages.shape, cutmaps.shape)
    i = h5py.File(h5imgname, 'w')
    m = h5py.File(h5mapname, 'w')
    i.create_dataset('images', h5shapeimg, data = cutimages)
    m.create_dataset('maps', h5shapemap, data = cutmaps)
    print('Done!')
    return(cutimages,cutmaps)

def containsParticles(maps):
    """Function takes in segmentation maps in Keras tensor format and returns a list containing whether
    each map contained a particle written as a boolean"""
    particlelist = []
    count = 0
    for segmap in maps:
        if 1.0 in segmap:
            particlelist.append(True)
            count += 1
        else:
            particlelist.append(False)
    print('Fraction of maps which contained particles: {0}'.format(count/maps.shape[0]))
    return particlelist

def saveasH5BalMulticlass(istack,mstack,h5dir,particleIdx):
    """Requests two directories one of images one of segmentation maps both in
    png format. It then will make two sets of h5 files named after the directory
    the images and maps were found in  respectively."""
    if istack.shape[0] != mstack.shape[0]:
        raise RuntimeError('Different number of images and maps')
    particlelist = []
    count = 0
    for segmap in mstack[:,:,:,particleIdx]:
        if 1.0 in segmap:
            particlelist.append(True)
            count += 1
        else:
            particlelist.append(False)
    print('Fraction of maps which contained particles: {0}'.format(count/mstack.shape[0]))
    h5name = input('Name for files: ')
    h5imgname = h5dir + '/' + 'Bal_' + h5name+'.h5'
    h5mapname = h5dir  + '/' + 'Bal_' + h5name +'_maps.h5'
    goodpics = []
    goodmaps = []
    for idx, part in enumerate(particlelist):
        if part == True:
            goodpics.append(istack[idx])
            goodmaps.append(mstack[idx])
    goodpics = np.asanyarray(goodpics)
    goodmaps = np.asanyarray(goodmaps)
    i = h5py.File(h5imgname, 'w')
    m = h5py.File(h5mapname, 'w')
    i.create_dataset('images', goodpics.shape, data = goodpics)
    m.create_dataset('maps', goodmaps.shape, data = goodmaps )
    print('Done!')

def save_for_labeller(directory,dm3file):
    stack = dm.dmReader(dm3file)
    warnings.filterwarnings("ignore")
    if os.path.isdir(directory) == False:
        os.mkdir(directory)
    fname_base = directory + '/' + stack['filename'].split('.')[0]
    if len(stack['data'].shape) == 2:
        fname = fname_base +'.png'
        io.imsave(fname,img)
    else:
        for idx,img in enumerate(stack['data']):
            fname = fname_base + '_' + str(idx) +'.png'
            imsave(fname,median_filter(img))
    print('done!')

def save_for_labeller_from_h5stack(stack,directory,filename, medfilt = False):
    warnings.filterwarnings("ignore")
    if os.path.isdir(directory) == False:
        os.mkdir(directory)
    fname_base = directory + '/' + filename
    data = stack.reshape((stack.shape[0],stack.shape[1],stack.shape[2]))
    for idx,img in enumerate(data):
        fname = fname_base + '_' + str(idx) +'.png'
        if medfilt == False:
            imsave(fname,img)
        else:
            imsave(fname,median_filter(img))
    print('done!')
