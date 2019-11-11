import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from ncempy.io import dm
from matplotlib.image import imsave
from matplotlib import cm
from skimage import transform
from skimage import exposure
from ast import literal_eval
import shutil
from skimage import io
import sys
from ncempy.io import dm
import shutil
from numpy.random import random_integers
import h5py
from scipy import fftpack
from matplotlib.patches import Rectangle
import ncempy.io.ser as ser
from scipy.signal import medfilt2d
from skimage.util import pad

# This code is for taking png versions of real TEM images and the associated txts
# segmenation files

#This section defines functions for processing raw dm3s
def processDM3s():
    """This function calls to the user to give a directory full of unprocessed dm3s.
    Note if these dm3s have been processed before this function will overwrite
    previously created DM3s. All DM3s will be converted to pngs and cleaned of
    spurious xray peaks"""
    fname = input('paste directory path of directory containing experiment directories or type q to quit: ',)
    new_directory = input('paste directory path where you want pngs to end up or type q to quit')
    if fname == 'q' or new_directory == 'q':
        sys.exit()
    dir_list = glob(fname+'/*')
    i = 0
    for d1 in dir_list:
        dir_list2 = glob(d1+'/*')
        for d2 in dir_list2:
            for dm3 in glob(d2+'/*.dm3'):
                img = dm.dmReader(dm3)['data']
                if img.shape[0] < 1024:
                    pass
                elif img.shape[0] == 2048 and img.shape[1] == 2048:
                    img = transform.resize(img,(1024,1024))
                    name = dm3.split('/')[-1].split('.')[0]
                    name = d2.split('/')[-2].split('_')[0] + '_' + name
                    imsave(new_directory+'/'+name+'.png', xray_correct(img), format="png", cmap=cm.gray)
                elif img.shape[0] == 1024 and img.shape[1] == 1024:
                    name = dm3.split('/')[-1].split('.')[0]
                    name = d2.split('/')[-2].split('_')[0] + '_' + name
                    imsave(new_directory+'/'+name+'.png', xray_correct(img), format="png", cmap=cm.gray)
        i += 1
        print('{} / {} directories complete'.format(i,len(dir_list)))
    print('done!')


#This section defines all the code for creating masks from a directory of label txt file_list
def transferTxts():
    """For the transfer of label txt files from one directory to another. Will
    request input both for the original directory and the new directory where
    txt files should end up"""
    og_directory = input('paste directory path of directory containing experiment directories or type q to quit: ',)
    new_directory = input('paste directory path where you want txts to end up or type q to quit')
    if og_directory == 'q' or new_directory == 'q':
        sys.exit()
    dir_list = glob(og_directory+'/*')
    i = 0
    for d1 in dir_list:
        dir_list2 = glob(d1+'/*')
        for d2 in dir_list2:
            for txt in glob(d2+'/*/*.txt'):
                name = txt.split('/')[-1]
                name = d2.split('/')[-2].split('_')[0] + '_' + name
                shutil.copy2(txt,new_directory+'/'+name)

def txt_reader(file):
    """Reads a txt label file created by the labeling GUI given a filename.
    Returns location of the center as a list of tuples, the radii of the circles
    and the image size"""
    txt_info = open(file,'r')
    txt = []
    centers = []
    radii = []
    for line in txt_info:
        if line == '\n':
            pass
        else:
            line = line.strip('\n')
            txt.append(line)
    center_start = txt.index('Particle Location:')
    center_stop = txt.index('Radius Size:')
    radius_stop = txt.index('Defect Label:')
    for loc in txt[center_start+1:center_stop]:
        centers.append(literal_eval(loc))
    for loc in txt[center_stop+1:radius_stop] :
        radii.append(int(loc))
    image_size = literal_eval(txt[-1])
    return centers, radii, image_size


def spot_maker(location, radius, label_mask):
    for x in np.arange(location[0]-radius,location[0]+radius,1):
        for y in np.arange(location[1]-radius,location[1]+radius,1):
            dx = x - location[0]
            dy = y - location[1]
            if np.sqrt((dx**2+dy**2)) <= radius \
            and int(x) < label_mask.shape[0] and int(y) < label_mask.shape[1]:
                label_mask[int(y),int(x)] = 1
    return label_mask

def mask_maker(file):
    centers, radii, image_size = txt_reader(file)
    label_mask = np.zeros(image_size)
    for idx,radius in enumerate(radii):
        label_mask = spot_maker(centers[idx],radius,label_mask)
    return label_mask

def mask_pipeline():
    """Function when called asks for directory of txt label files to create
    segmentation maps from and a directory to put created maps in. The function
    then creates a segmentation map for every txt file in the first directory."""
    directory = input('paste directory path where txt files are stored or process q to quit')
    new_directory = input('paste directory path where you want txts to end up or type q to quit')
    if directory == 'q' or new_directory == 'q':
        sys.exit()
    file_list = glob(directory+'/*.txt')
    name_list = [name.split('/')[-1].split('.')[0] for name in file_list]
    for idx, file in enumerate(file_list):
        if len(open(file,'r').readlines()) == 0:
            pass
        else:
            label_mask = mask_maker(file)
            plt.imsave(new_directory+'/'+name_list[idx]+'.png',label_mask, cmap='gray')
    print('done!')

# This section defines all the code for breaking up pngs and maps
def mapPngMatch():
    """Function asks for directory to images and maps. Returns list of images
    that match maps and count of matched items"""
    png_dir = input('paste directory path where png files are stored or process q to quit')
    map_dir = input('paste directory path where map files are stored or process q to quit')
    pngs = sorted(glob(png_dir+'/*.png'))
    maps = sorted(glob(map_dir+'/*.png'))
    count = 0
    good_pngs = []
    good_maps = []
    lone_pngs = []
    for idx, fnpic in enumerate(pngs):
        # print(fnpic.split('/')[-1].split('.')[0])
        # print(maps[idx-len(lone_pngs)].split('/')[-1].split('.')[0])
        # break
        if fnpic.split('/')[-1].split('.')[0] == maps[idx-len(lone_pngs)].split('/')[-1].split('.')[0]:
            good_pngs.append(fnpic)
            count += 1
        else:
            lone_pngs.append(fnpic)
    return good_pngs, maps, png_dir, map_dir


def sliceBoth(endSize):
    """used to break up the 1024x1024 images and maps into segments defined as
    endSize x endSize. Takes a diretory that is the source. Saves to the correct
    size directory for images and maps"""
    endSize = int(endSize)
    good_pngs, maps, png_dir, map_dir = mapPngMatch()
    image_new_directory=input('paste directory path where split image files should be stored')
    map_new_directory=input('paste directory path where split map files should be stored')
    for idx, png in enumerate(good_pngs):
        image2split = io.imread(png, as_grey=True)
        map2split = io.imread(maps[idx], as_grey=True)
        if image2split.shape != (1024,1024):
            raise RuntimeError('Image: ', png, 'is not 1024x1024')
        if map2split.shape != (1024,1024):
            raise RuntimeError('Map: ', maps[idx], 'is not 1024x1024')
        if 1024%endSize != 0:
            raise RuntimeError('1024 is not evenly divisible by endSize chosen. Please choose a different value')
        splitSize = int((1024/endSize))
        for x in range(0,splitSize*endSize,endSize):
            for y in range(0,splitSize*endSize,endSize):
                image = image2split[x:x+endSize,y:y+endSize]
                map = map2split[x:x+endSize,y:y+endSize]
                imgnamepart = png.split('/')[-1].split('.')[0]
                mapnamepart = png.split('/')[-1].split('.')[0]
                image_name = '/'+imgnamepart+ '_' + str(x)+ str(y) + '.png'
                map_name = '/'+mapnamepart+ '_' + str(x)+ str(y) + '.png'
                plt.imsave(image_new_directory+image_name,image, cmap='gray')
                plt.imsave(map_new_directory+map_name,map, cmap='gray')
    print('done!')

def makeNpyStack():
    """Requests a directory full of pngs to convert into a numpy array which
    is then saved to the second directory specified. Also creates an array of
    experiment labels so each image in the stack can be identified"""
    og_directory = input('Paste directory path where current png files are stored')
    new_directory = input('Paste directory path where npy files should be saved')
    filename = input('Please specify the file name you want for the npy file')
    stack = []
    experiment_labels = []
    og_pngs = glob(og_directory+'/*.png')
    lenpngs = len(og_pngs)
    for idx,png in enumerate(og_pngs):
        img = io.imread(png, as_grey=True)
        stack.append(np.copy(img))
        if idx < lenpngs-1:
            experiment_labels.append(png.split('/')[-1].split('.')[0]+'\n')
        else:
            experiment_labels.append(png.split('/')[-1].split('.')[0])
    stack = np.asanyarray(stack)
    print('Finished accumulating. Stack is size:',stack.shape)
    np.save(new_directory+'/'+filename+'.npy',stack)
    f = open(new_directory+'/'+filename+'_expmntLabels.txt','w')
    f.writelines(experiment_labels)

def makeExperimentLabels():
    """Requests a directory full of pngs to convert into a numpy array which
    is then saved to the second directory specified. Also creates an array of
    experiment labels so each image in the stack can be identified"""
    og_directory = input('Paste directory path where current png files are stored')
    new_directory = input('Paste directory path where text file should be saved')
    filename = input('Please specify the file name you want for the text file')
    experiment_labels = []
    og_pngs = glob(og_directory+'/*.png')
    lenpngs = len(og_pngs)
    for idx,png in enumerate(og_pngs):
        if idx < lenpngs-1:
            experiment_labels.append(png.split('/')[-1].split('.')[0]+'\n')
        else:
            experiment_labels.append(png.split('/')[-1].split('.')[0])
    f = open(new_directory+'/'+filename+'_expmntLabels.txt','w')
    f.writelines(experiment_labels)
    print('Finished')

def valSplit(fraction):
    """Requests a directory of already split pngs and maps and then creates a
    validation set. The input fraction requires a decimal value between 0 and 1.
    Fraction sets the percent of original images that should be used for
    validation"""
    ogpngdir = input('Paste directory path where current png files are stored')
    ogmapdir = input('Paste directory path where current map files are stored')
    newpngdir = input('Paste directory path where validation png files should go')
    newmapdir = input('Paste directory path where validation map files should go')
    pngs = sorted(glob(ogpngdir+'/*.png'))
    maps = sorted(glob(ogmapdir+'/*.png'))
    if len(maps) != len(pngs):
        raise RuntimeError('different number of pngs and maps. Quitting now.')
    split = int(len(pngs)*fraction)
    for count in np.arange(split):
        idx = random_integers(0,len(pngs))
        shutil.move(pngs[idx],newpngdir)
        shutil.move(maps[idx],newmapdir)
        pngs = glob(ogpngdir+'/*.png')
        maps = glob(ogmapdir+'/*.png')
    print('Done!')

def saveasH5():
    """Requests two directories one of images one of segmentation maps both in
    png format. It then will make two sets of h5 files named after the directory
    the images and maps were found in  respectively."""
    ogpngdir = input('Paste directory path where current png files are stored')
    ogmapdir = input('Paste directory path where current map files are stored')
    h5dir = input('Paste directory path where h5 files should go')
    pngs = sorted(glob(ogpngdir+'/*.png'))
    maps = sorted(glob(ogmapdir+'/*.png'))
    h5imgname = h5dir + '/' + ogpngdir.split('/')[-1]+'.h5'
    h5mapname = h5dir  + '/' + ogmapdir.split('/')[-1]+'.h5'
    img = io.imread(pngs[0], as_grey = True)
    mask = io.imread(maps[0],as_grey = True)
    i = h5py.File(h5imgname, 'w')
    m = h5py.File(h5mapname, 'w')
    i.create_dataset('images', (img.shape[0]*len(pngs),img.shape[1]), maxshape = (None, None))
    m.create_dataset('maps', (mask.shape[0]*len(maps),mask.shape[1]), maxshape = (None, None))
    i['images'][0:img.shape[0],0:img.shape[1]] = img
    m['maps'][0:mask.shape[0],0:mask.shape[1]] = mask
    if len(pngs) != len(maps):
        raise RuntimeError('Different number of images and maps')
    for idx in np.arange(1,len(pngs),1):
        img = io.imread(pngs[idx], as_grey = True)
        mask = io.imread(maps[idx],as_grey = True)
        i['images'][idx*img.shape[0]:img.shape[0]*(idx+1),0:img.shape[1]] = img
        m['maps'][idx*mask.shape[0]:mask.shape[0]*(idx+1),0:mask.shape[1]] = mask
    print('Done!')


def file_relabel(og_dir, og_subdir, new_dir):
    """Takes the old directory with directories containing microscope files
    that have already been converted to pngs and labeled. The pngs and txt files
    are moved to the unified random forest data directory. While
    doing this it renames the files to include the experiment date in the
    filename. og_dir defines the root directory containing all the experiments.
    og_subdir defines the nam of the subdirectory where are the pngs and txt
    files are stored (this assumes that the images have been processed from dm3s
    to pngs that were then labeled using the labeling gui and therefore the
    associated txt label file is stored). new_dir defines the new directory
    where you want files moved to."""
    dir_list = os.listdir(og_dir)
    skip_count = 0
    for direct in dir_list:
        if direct == '.DS_Store':
            pass
        else:
            pngs = glob(og_dir+'/'+direct+'/*/' + og_subdir +'/*.png')
            print(len(pngs))
            txts = glob(og_dir+'/'+direct+'/*/'+ og_subdir +'/*.txt')
            print(len(txts))
            for idx, txt in enumerate(txts):
                txtname = txt.split('.')[0].split('/')[-1]
                newname = direct+'_'+txtname
                if txtname == pngs[idx].split('.')[0].split('/')[-1]:
                    shutil.copy2(txt,new_dir+'/'+newname+'.txt')
                    shutil.copy2(pngs[idx],new_dir+'/'+newname+'.png')
                else:
                    if txtname == pngs[skip_count+idx+1].split('.')[0].split('/')[-1]:
                        shutil.copy2(txt,new_dir+'/'+newname+'.txt')
                        shutil.copy2(pngs[idx],new_dir+'/'+newname+'.png')
                        skip_count += 1
                    else:
                        pass
    print('done')


def imm(img_tensor, size = (512,512), colorbar = True):
    plt.figure(figsize=(10,10))
    plt.imshow(img_tensor.reshape(size),cmap='gray')
    if colorbar == True:
        plt.colorbar()

def img2tensor(img):
    img_tensor = np.expand_dims(image.img_to_array(img),axis =0)
    return img_tensor

def h5stack2tensor(stack, size = (512,512),norm = False):
    stack = stack[:,:]
    num = stack.shape[0]//size[0]
    if norm == False:
        stack = stack.reshape((num,size[0],size[1],1))
    else:
        stack = stack.reshape((num,size[0],size[1],1))/stack.max()
    return stack

def np_dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def makeFFTstack(imgstack, size = (512,512)):
    fftstack = []
    for img in imgstack:
        fft = fftpack.fft2(img.reshape(512,512))
        fftstack.append(fft)
    fftstack = np.asanyarray(fftstack)
    fftstack = fftstack.reshape((len(fftstack),size[0],size[1],1))
    return fftstack

def makeFFTImgJoint(imgstack):
    fftstack = []
    for img in imgstack:
        fft = fftpack.fftshift(fftpack.fft2(img.reshape(512,512)))
        img = np.copy(img)
        img = np.append(img,fft,axis = 0)
        fftstack.append(fft)
    fftstack = np.asanyarray(fftstack)
    fftstack = fftstack.reshape((len(fftstack),size[0],size[1],1))
    return fftstack

def combineStacks(stack1, stack2):
    """Take two stack of images in keras input format and combine them such that
    the first image from stack1 will then be followed by the first image from
    stack2 and so on."""
    if stack1.shape != stack2.shape:
        raise RuntimeError('input stacks are not the same size.')
    finalStack = []
    for idx, img1 in enumerate(stack1):
        combo = np.append(img1,stack2[idx,:,:,:],axis = 2)
        finalStack.append(combo)
    finalStack = np.asanyarray(finalStack)
    return finalStack

def immMulticlass(img_tensor, size = (512,512), colorbar = True):
    plt.figure(figsize=(10,10))
    if len(img_tensor.shape) == 3:
        if img_tensor.shape[-1] == 3:
            sizeI = (size[0],size[1],img_tensor.shape[-1])
            finalVersion = img_tensor.reshape(sizeI)
        elif img_tensor.shape[-1] == 2:
            sizeI = (size[0],size[1],img_tensor.shape[-1])
            rgbVersion = np.zeros((sizeI[0],sizeI[1],1))
            finalVersion = img_tensor.reshape(sizeI)
            finalVersion = np.concatenate((finalVersion,rgbVersion),axis = 2)
        elif img_tensor.shape[-1] == 1:
            sizeI = (size[0],size[1],img_tensor.shape[-1])
            finalVersion = img_tensor.reshape(sizeI)
        else:
            raise(RuntimeError('Unexpected shape of input image tensor'))
    elif len(img_tensor.shape) == 2:
        sizeI = size
        finalVersion = img_tensor.reshape(sizeI)
    else:
        raise(RuntimeError('Unexpected shape of input image tensor'))
    plt.imshow(finalVersion)
    if colorbar == True:
        plt.colorbar()

def immStack(stack, size = (512,512), colorbar = True):
    command = input('Enter index of image you would like to view next (enter q to quit): ')
    while command != 'q':
        imm(stack[int(command)],size,colorbar)
        command = input('Enter index of image you would like to view next (enter q to quit): ')

def immOverlay(imgStack,labelStack,idx):
    img = imgStack[idx]
    label = labelStack[idx]
    size = (img.shape[0],img.shape[1])
    plt.figure(figsize=(10,10))
    if len(label.shape) == 3:
        if label.shape[-1] == 3:
            sizeI = (size[0],size[1],label.shape[-1])
            finalVersion = label.reshape(sizeI)
        elif label.shape[-1] == 2:
            sizeI = (size[0],size[1],label.shape[-1])
            rgbVersion = np.zeros((sizeI[0],sizeI[1],1))
            finalVersion = label.reshape(sizeI)
            finalVersion = np.concatenate((finalVersion,rgbVersion),axis = 2)
        elif label.shape[-1] == 1:
            sizeI = (size[0],size[1],label.shape[-1])
            finalVersion = label.reshape(sizeI)
        else:
            raise(RuntimeError('Unexpected shape of input image tensor'))
    elif len(label.shape) == 2:
        sizeI = size
        finalVersion = label.reshape(sizeI)
    else:
        raise(RuntimeError('Unexpected shape of input image tensor'))
    plt.imshow(img.reshape(size),cmap = 'gray')
    plt.imshow(finalVersion,alpha = 0.4)

def add_scalebar(ax,size,scale,unit,loc,offx = 10,fontSize=20):
    """Add scale bar to already existing image plotted with plt.imshow"""
    if unit[0] == 'nm':
        pixels = size//scale[0]
    elif unit[0] == 'µm':
        newscale = scale[0]*1000
        pixels = size//newscale
    else:
        raise('RuntimeError')
    x = loc[0]
    y = loc[1]
    rect = Rectangle((x,y),pixels,10,linewidth=1,edgecolor='w',facecolor='w')
    ax.text(x+offx,y-5 ,str(size)+unit[0],color = 'w',fontsize=fontSize)
    ax.add_patch(rect)

def plot_ser(pics,scales,labels,index,barSize,loc):
    scales = scales.copy()*1000000000.0
    newshape = (pics[index].shape[1],pics[index].shape[2])
    rdp.imm(pics[index].reshape(newshape),size = newshape)
    ax = plt.gca()
    rdp.add_scalebar(ax,barSize,[scales[index]],['nm'],loc)
    plt.title(labels[index])
    ax.axis('off')

def plot_ser_stack(directory,barSize,loc):
    sers = glob(directory+'/*.ser')
    labels = [ser.serReader(file)['filename'].split('/')[-1] for file in sers]
    scales = np.array([ser.serReader(file)['Calibration'][0]['CalibrationDelta'] for file in sers])
    # units = [ser.serReader(file)['pixelUnit'] for file in dm3s]
    pics = [ser.serReader(file)['data'] for file in sers]
    for idx in np.arange(0,len(sers)):
        plot_ser(pics,scales,labels,idx,barSize,loc)

def plot_dm3(pics,scales,labels,units,index,barSize,loc,filt):
    if filt == False:
        imm(pics[index], size = pics[index].shape)
    else:
        imm(median_filter(pics[index]), size = pics[index].shape)
    ax = plt.gca()
    add_scalebar(ax,barSize,scales[index],units[index],loc)
    plt.title(labels[index])

def plot_dm3_fromdir(directory,barSize,loc,filt = False,setScale = False):
    dm3s = glob(directory+'/*.dm3')
    labels = [dm.dmReader(file)['filename'] for file in dm3s]
    scales = [dm.dmReader(file)['pixelSize'] for file in dm3s]
    units = [dm.dmReader(file)['pixelUnit'] for file in dm3s]
    pics = [dm.dmReader(file)['data'] for file in dm3s]
    if setScale == True:
        print('scale of first image is {} {}'.format(scales[0],units[0]))
        barSize = float(input('What should length of scale bar be: '))
        print('Size of first image is {}'.format(pics[0].shape))
        locY = float(input('Y val here should scale bar be (in pixels): '))
        locX = float(input('X val here should scale bar be (in pixels): '))
        loc = (locY,locX)
    for idx in np.arange(0,len(pics)):
        plot_dm3(pics,scales,labels,units,idx,barSize,loc,filt)
    print('Done!')

def median_filter(image):
    shape = (image.shape[0],image.shape[1])
    pad_image = image.reshape(shape)
    pad_image = pad(pad_image.astype('float32'),3,'reflect')
    pad_shape = pad_image.shape
    filt_img = medfilt2d(pad_image,3)
    thresh = 5*image.std()
    center_cut = filt_img[3:pad_shape[0]-3,3:pad_shape[1]-3]
    dif = abs(image.reshape(shape) - center_cut)
    final = image.copy().reshape(shape)
    final[dif > thresh] = center_cut[dif>thresh]
    final = final/final.std()
    final = final - final.min()
    final = final/final.max()
    return final

def median_filter_stack(stack):
    new_stack = []
    for img in stack:
        new_image = median_filter(img)
        new_stack.append(new_image)
    new_stack = np.expand_dims(np.array(new_stack),3)
    return new_stack

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
