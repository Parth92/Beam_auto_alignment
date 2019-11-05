import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio

from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.filters import gaussian


def Find_Peaks(IMG, separation=10, Sigma=1, show_ada_thresh=False, show_fig=False):
    if show_ada_thresh: image_max = ndi.maximum_filter(IMG, size=separation, mode='constant')
    Smooth_img = gaussian(IMG,  sigma=Sigma)
    coordinates = peak_local_max(Smooth_img, min_distance=separation, exclude_border=False)
    coordinates = np.array(coordinates)
    dummy = coordinates[:,0].copy()
    coordinates[:,0] = coordinates[:,1]
    coordinates[:,1] = dummy
    if show_fig:
        plt.figure()
        plt.imshow(Smooth_img, cmap=plt.cm.gray)
        plt.colorbar()
        plt.plot(coordinates[:, 0],
                 coordinates[:, 1], 'r*')
    if show_ada_thresh:
        plt.figure()
        plt.imshow(image_max, cmap=plt.cm.gray)
        plt.colorbar()
    return coordinates

def Thresh(IMG1, thresh=0.5):
    IMG = IMG1.copy()
    IMG[IMG < thresh*IMG.max()] = 0.
    return IMG

# get the basis vectors
def find_basis(P0, P1):
    Basis_vect = np.array(P1-P0, dtype='d')
    Basis_vect /= float(np.sqrt(Basis_vect[0]**2. + Basis_vect[1]**2.))
    # slope
    slope = (P1[1]-P0[1])
    slope /= (P1[0]-P0[0])
    return Basis_vect, slope

def distance_from_vect(vect, Point):
    m = vect[1] / vect[0]
    if np.isinf(m) or np.isnan(m):
        m = 1e2
    return (Point[1] - m*Point[0]) / np.sqrt(m**2+1)

def points_along_vect(P_corner, Basis, Points, width):
    # Perpendicular distance
    d = []
    for point in Points:
        # (point-P_corner) to translate P_corner to origin
        d.append(abs(distance_from_vect(Basis, point-P_corner)))
    d = np.array(d)
    return len(d[d<width])

def peaks_to_mode(Peak_corner, Peaks, Width, Basis1, slope1, Basis2=[], slope2=None):
    # ensure that the basis vects have at least pi/4 angle between them
    if not bool(len(Basis2)) and not slope2:
        m = points_along_vect(Peak_corner, Basis1, Peaks, Width)
        n = 1
    elif (abs(np.arctan(slope1) - np.arctan(slope2)) < np.pi/4.):
        m = points_along_vect(Peak_corner, Basis1, Peaks, Width)
        n = 1
    else:
        m = points_along_vect(Peak_corner, Basis1, Peaks, Width)
        n = points_along_vect(Peak_corner, Basis2, Peaks, Width)
    if abs(slope1) < 1.:
        return (m-1,n-1)
    else:
        return (n-1,m-1)

def Find_mode(img_loc, separation1=5, Sigma1=1, Width=10, thresh=0.5, show_ada_thresh=False, show_fig=True, corner=0, show_peaks=False):
    # Read image
    img = 255 - imageio.imread(img_loc)
    # thresholding and smoothing image to find peaks
    img1 = Thresh(img, thresh)
    peaks = Find_Peaks(img1, separation=separation1, Sigma=Sigma1, show_ada_thresh=show_ada_thresh, show_fig=show_fig)
    if len(peaks) == 1:
        return (0,0)
    # corner point
    if corner==0:
        i_corner = peaks[:,0].argmin()
    elif corner==1:
        i_corner = peaks[:,0].argmax()
    elif corner==2:
        i_corner = peaks[:,1].argmin()
    elif corner==3:
        i_corner = peaks[:,1].argmax()
    peak_corner = peaks[i_corner]
    # distances from corner point
    d = np.array([((peaks[i,0]-peak_corner[0])**2 + (peaks[i,1]-peak_corner[1])**2) for i in range(len(peaks[:,0]))])
    d[d==0] = d.max() + 1.
    # catch compact basis vectors
    i_near1 = d.argmin()
    peak_near1 = peaks[i_near1]
    # constructing unit vect and slope
    basis1, m1 = find_basis(peak_corner, peak_near1)
    if len(peaks) > 2:
        d[i_near1] = d.max() + 1.
        i_near2 = d.argmin()
        peak_near2 = peaks[i_near2]
        # print(peak_corner, peak_near1, peak_near2)
        # constructing unit vect and slope
        basis2, m2 = find_basis(peak_corner, peak_near2)
        # print(basis1, basis2, m1, m2)
        if show_peaks:
            kk = 4
            plt.plot([peak_corner[0], peak_corner[0]+(peak_near1[0]-peak_corner[0])*kk], 
                     [peak_corner[1], peak_corner[1]+(peak_near1[1]-peak_corner[1])*kk], 'r')
            plt.plot([peak_corner[0], peak_corner[0]+(peak_near2[0]-peak_corner[0])*kk], 
                     [peak_corner[1], peak_corner[1]+(peak_near2[1]-peak_corner[1])*kk], 'y')
    # find mode
    if len(peaks) == 2:
        mode = peaks_to_mode(peak_corner, peaks, Width, basis1, m1)
    else:
        mode = peaks_to_mode(peak_corner, peaks, Width, basis1, m1, basis2, m2)
    print(mode)
    if (mode[0]+1)*(mode[1]+1) < len(peaks):
        print('Warning: Something went wrong! \nNumber of peaks ({}) \
        inferred from mode does not match with actual number of peaks({})'.format((mode[0]+1)*(mode[1]+1), len(peaks)))
        corner += 1
        if corner < 4:
            mode = Find_mode(img_loc, separation1=separation1, Sigma1=Sigma1, Width=Width, thresh=thresh, 
                             show_ada_thresh=show_ada_thresh, show_fig=show_fig, corner=corner, show_peaks=show_peaks)
            print(mode)
        else:
            pass
    plt.show()
    return mode

def test_consistency(mode, Peaks):
    if (mode[0]+1)*(mode[1]+1) == len(Peaks) or (mode[0]+1)*(mode[1]+1) == len(Peaks)+1:
        return True
    else:
        print('Warning: Something went wrong! \nNumber of inferred peaks ({}) from mode does not match with actual number of peaks({})'.format((mode[0]+1)*(mode[1]+1), len(Peaks)))
        return False

def case1_m_0(Peaks, w=10, show_basis=False):
    # bottom & top
    i_b = Peaks[:,0].argmin()
    i_t = Peaks[:,0].argmax()
    Peak_b = Peaks[i_b]
    Peak_t = Peaks[i_t]
    # constructing unit vect and slope
    basis, m = find_basis(Peak_b, Peak_t)
    # Show basis vects
    if show_basis:
        kk = 2
        plt.plot([Peak_b[0], Peak_b[0]+(Peak_t[0]-Peak_b[0])*kk], 
                 [Peak_b[1], Peak_b[1]+(Peak_t[1]-Peak_b[1])*kk], 'b')
    if abs(m) > 1.:
        return (0,0)
    else:
        return peaks_to_mode(Peak_b, Peaks, w, basis, m)

def case2_0_n(Peaks, w=10, show_basis=False):
    # left & right
    i_l = Peaks[:,1].argmin()
    i_r = Peaks[:,1].argmax()
    Peak_l = Peaks[i_l]
    Peak_r = Peaks[i_r]
    # constructing unit vect and slope
    basis, m = find_basis(Peak_l, Peak_r)
    # Show basis vects
    if show_basis:
        kk = 2
        plt.plot([Peak_l[0], Peak_l[0]+(Peak_r[0]-Peak_l[0])*kk], 
                 [Peak_l[1], Peak_l[1]+(Peak_r[1]-Peak_l[1])*kk], 'g')
    if abs(m) < 1.:
        return (0,0)
    else:
        return peaks_to_mode(Peak_l, Peaks, w, basis, m)

def case3_m_n(Peaks, w=10, corner=0, show_basis=False):
    # catching 4 corner points
    i_b = Peaks[:,1].argmax()  # bottom point (plot is inverted in top/bottom)
    i_t = Peaks[:,1].argmin()  # top point
    i_r = Peaks[:,0].argmax()  # right point
    i_l = Peaks[:,0].argmin()  # left point
    P_b, P_t, P_r, P_l = Peaks[i_b], Peaks[i_t], Peaks[i_r], Peaks[i_l]
    # attempting each one of them as base point (P0) sequentially if prev fails
    if corner==0:
        P0, P1, P2 = P_b.copy(), P_l.copy(), P_r.copy()
    elif corner==1:
        P0, P1, P2 = P_l.copy(), P_b.copy(), P_t.copy()
    elif corner==2:
        P0, P1, P2 = P_r.copy(), P_b.copy(), P_t.copy()
    elif corner==3:
        P0, P1, P2 = P_t.copy(), P_l.copy(), P_r.copy() 
    # constructing unit vect and slope
    basis1, m1 = find_basis(P0, P1)
    basis2, m2 = find_basis(P0, P2)
    # Show basis vects
    if show_basis:
        kk = 2
        plt.plot([P0[0], P0[0]+(P1[0]-P0[0])*kk], 
                 [P0[1], P0[1]+(P1[1]-P0[1])*kk], 'r')
        plt.plot([P0[0], P0[0]+(P2[0]-P0[0])*kk], 
                 [P0[1], P0[1]+(P2[1]-P0[1])*kk], 'y')
    # find mode
    return peaks_to_mode(P0, Peaks, w, basis1, m1, basis2, m2)
    
def Find_mode2(img_loc, separation1=5, Sigma1=1, Width=10, thresh=0.5, corner=0, show_ada_thresh=False, show_fig=False, show_basis=False):
    # Read image
    # if image location
    if isinstance(img_loc, str):
        img = 255 - imageio.imread(img_loc)
    # if image itself
    elif isinstance(img_loc, np.ndarray):
        img = img_loc
    # img = imageio.imread(img_loc)[43:243, 54:320, 0]
    # thresholding and smoothing image to find peaks
    img1 = Thresh(img, thresh)
    peaks = Find_Peaks(img1, separation=separation1, Sigma=Sigma1, show_ada_thresh=show_ada_thresh, show_fig=show_fig)
    # Case 0: (0,0)
    if len(peaks) == 1:
        mode = (0,0)
        print("Case0: ", mode)
        if show_ada_thresh or show_basis: plt.show()
        return mode
    # Case 1: (m,0)
    mode = case1_m_0(peaks, w=Width, show_basis=show_basis)
    print("Case1 result: ", mode)
    if test_consistency(mode, peaks):
        if show_ada_thresh or show_basis: plt.show()
        return mode
    else:
        # Case 2: (0,n)
        mode = case2_0_n(peaks, w=Width, show_basis=show_basis)
        print("Case2 result: ", mode)
        if test_consistency(mode, peaks):
            if show_ada_thresh or show_basis: plt.show()
            return mode
        else:
            # Case 3: (m,n)
            for corner in range(4):
                mode = case3_m_n(peaks, w=Width, corner=corner, show_basis=show_basis)
                print("Case3 corner_{} result: ".format(corner), mode)
                if test_consistency(mode, peaks) or corner==3:
                    if show_ada_thresh or show_basis: plt.show()
                    return mode



################################ Beam alignment functions ################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pypylon import pylon
from pypylon import genicam
# Importing busworks
import busworks
import keras
from keras import backend as k
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle
from math import factorial
from copy import deepcopy
from tqdm import tqdm
import datetime
import time
import os
import sys


# Initialization
n_pixl = 128
RepoDir = '/home/controls/Beam_auto_alignment'
# RepoDir = '/home/user1/Dropbox/Academic/WORK/Beam_auto_alignment'
ImagesFolder = RepoDir + '/Data/Actual_cavity_Fittest_points_per_gen_' + \
str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')[:-10]
if not os.path.exists(ImagesFolder): os.mkdir(ImagesFolder)
SaveModelFolder = RepoDir + '/Data/TrainedModels/'
Model_Name = 'Trained_Model_2019-07-03_17-15'

# Read the pre-trained CNN model
cnn = keras.models.load_model(SaveModelFolder + Model_Name + '.h5')
# load the encoder
loaded_Encoder = pickle.load(open(SaveModelFolder + 'Encoder_of_' + Model_Name + '.npy', 'rb'))

#seed
np.random.seed(187)
# geometric parameters
g1 = 1
g2 = 0.268
# wavelength
Lambda = 1.064e-6
# waist size in m
waist = 140e-6
# range of movement of the waist center at the waist location in the units of waist size
Range = 3.5
# max voltage o/p of DAC
V_DAC_max = 10.
HV_op_gain = 1.
# Max angular deviation of steering mirror (on each side)
phi_SM_max = 26.2e-3 #(rad) for newport mirrors. (Thorlabs : 73e-6 radians)
SM_drive_gain = 1.
SM_ip_voltage_range = [0, 10]
# Max displacement of cavity mirror (PZT) (on one side)
phi_CM_PZT_max = 2.8e-6 #(microns). starts from zero
CM_PZT_ip_voltage_range = [0, 200]
print('Cavity Mirror scanning range is [{}, {}] micron. The used range [{}, {}] micron should not go out.'\
      .format(0, phi_CM_PZT_max, 0, phi_CM_PZT_max*V_DAC_max*HV_op_gain/CM_PZT_ip_voltage_range[1]))
print('Steering Mirror scanning range is [-{0}, {0}] rad. The used range [-{1}, {1}] rad should not go out.'\
      .format(phi_SM_max, phi_SM_max*V_DAC_max*SM_drive_gain/SM_ip_voltage_range[1]))
# cumulative distance of waist from SM1 in m
d1 = 0.35+0.0884
# cumulative distance of waist from SM2 in m
d2 = 0.0884
scale_params = np.array([waist/d1, waist/d1, waist/d2, waist/d2, Lambda/Range])   # Scanning of cavity should only happen in one lambda (Check for possible probs)
PZT_scaling = np.array([V_DAC_max/phi_SM_max, V_DAC_max/phi_SM_max, \
                        V_DAC_max/phi_SM_max, V_DAC_max/phi_SM_max, V_DAC_max/phi_CM_PZT_max])
pop_per_gen = 200
Sz = 100    # number of z_CM scan steps
num_generations = 25
num_params = len(scale_params)
num_parents_mating = pop_per_gen // 10  # 10% of new population are parents
num_offsprings_per_pair = 2 * (pop_per_gen - num_parents_mating) // num_parents_mating + 1
# after each iteration, range shrinks by
shrink_factor = 2. / pop_per_gen ** 0.2  # make sure it is < 1.
# Defining the population size.
pop_size = (pop_per_gen,num_params) # The population will have sol_per_pop chromosome \
# where each chromosome has num_weights genes.
fitness = np.empty(pop_per_gen)

# camera exposure (check if right command!)
Exposure = 300


def read_mode(Img):
    Img /= np.max(Img)
    Img = 1. - Img
    plt.imshow(Img[::-1], cmap=cm.binary_r)
    plt.colorbar()
    plt.show()
    mode = cnn.predict(Img[::-1].reshape(1,n_pixl,n_pixl,1))
    probs = cnn.predict_proba(Img[::-1].reshape(1,n_pixl,n_pixl,1))
    print(probs)
    mode = loaded_Encoder['label_enc'].inverse_transform(loaded_Encoder['one_hot_enc'].\
                                                inverse_transform(mode).astype('int'))
    return mode

def pad_and_resize_image(image):
    # padding with const value of first pixel in image
    im_new = np.ones((max(image.shape),max(image.shape)))*image[0,0]
    pad = int((max(image.shape)-min(image.shape))/2)
    im_new[pad:-pad,:] = image
    # resize 
    im = tf.image.resize_nearest_neighbor(
        im_new.reshape(1, im_new.shape[0], im_new.shape[1], 1),
        (n_pixl,n_pixl),
        align_corners=False,
        name=None
    )
    with sess.as_default():
        im_new = im.eval()
    im_new = im_new[0,:,:,0]
    return im_new

def Capture_image(exposure, Camera):
    Dat = Camera.GrabOne(exposure)
    img = Dat.Array
    return img

def sample_d(Rng, shape=pop_size):
    """
    Takes i/p range in umits of beam waist at the waist location.
    Outputs sets of deltas (in radians) to be fed to steering mirrors.
    O/P shape : pop_size
    """
    delta = np.random.uniform(low=-Rng, high=Rng, size=shape)
    # CM pzt does not take -ve values
    delta[:,shape[1]-1] = np.abs(delta[:,shape[1]-1])
    delta *= scale_params
    return delta

def scan_cavity(Beam_status, pop_deltas, Rng, Size, Camera, Bus, show_fig=False):
    """
    Takes i/p range in umits of wavelength.
    Outputs the best scan position and updated deltas along with image.
    """
    R = []
    # CM pzt does not take -ve values
    delta_z = np.zeros((Size, len(scale_params)))
    ##### better z_CM sampling #####
    if Beam_status[-1] - Rng < 0.:
        Rng_l = 0.
        Rng_h = Rng
    elif Beam_status[-1] + Rng < 0.:
        Rng_l = -Rng
        Rng_h = 0.
    else:
        Rng_l = -Rng / 2.
        Rng_h = Rng / 2.
    delta_z[:,-1] = np.random.uniform(low=Rng_l, high=Rng_h, size=Size)
    for ii in range(Size):
        # delta_z updated at each step
        Beam_status, delta_z, R_new, _ = Reward(Beam_status, delta_z, delta_z[ii], Camera, Bus)
        R.append(R_new)
    i_max = np.argmax(R)
    # pop_deltas updated at this final step
    Beam_status, pop_deltas, _, Img = Reward(Beam_status, pop_deltas, delta_z[i_max], Camera, Bus)
    if show_fig:
        plt.imshow(Img[::-1], cmap=cm.binary_r)
        plt.colorbar()
        plt.show()
    return Beam_status, pop_deltas, Img

def Set_Voltage(Beam_status, Bus):
    ip_V = Beam_status * PZT_scaling
    # making sure that the DAC voltages remain in the required range
    # if (np.abs(ip_V) > V_DAC_max).sum() > 0:
        # print('DAC I/P voltage exceeded limit! Forcefully brought down.')
    ip_V[ip_V > V_DAC_max] = V_DAC_max
    ip_V[ip_V < -V_DAC_max] = -V_DAC_max
    try:
        # print(ip_V)
        Bus.set_voltages(ip_V, 1)
    except:
        print('Error!')

def Reward_fn(Beam_status, Camera, Bus, dummy_reward=False):
    # dummy_reward = True
    if dummy_reward:
        return np.random.randint(255), np.zeros((n_pixl,n_pixl))
    else:
        Set_Voltage(Beam_status, Bus)
        # reward fn as total power in the image
        Img1 = Capture_image(Exposure, Camera)
        R_fn1 = Img1.sum()/n_pixl**2
        # R_fn1 = Img1.max()
        return R_fn1, Img1

def Reward(Beam_status, pop_deltas, step, Camera, Bus):
    # take the delta step
    Beam_status += step
    # cumulatively subtracting each delta step from all deltas
    pop_deltas -= step
    R_new, Img = Reward_fn(Beam_status, Camera, Bus)
    return Beam_status, pop_deltas, R_new, Img

def calc_pop_fitness(Current_beam_status, New_pop_deltas, fitness, Camera, Bus, only_offsprings=False):
    """
    Calculating the fitness value of each solution in the current population.
    Also returns the current beam location (after adding the steps taken so far)
    """
    # if only_offsprings:
    #     range_vals = range(num_parents_mating, pop_per_gen)
    # else:
    #     range_vals = range(pop_per_gen)
    range_vals = range(pop_per_gen)
    for ii in range_vals:
        # # take the delta step
        # Current_beam_status += New_pop_deltas[ii]
        # # cumulatively subtracting each delta step from all deltas
        # New_pop_deltas -= New_pop_deltas[ii]
        # fitness[ii], _ = Reward_fn(Current_beam_status, Camera, Bus)
        Current_beam_status, New_pop_deltas, fitness[ii], _ = Reward(Current_beam_status, New_pop_deltas, New_pop_deltas[ii], Camera, Bus)
    return Current_beam_status, New_pop_deltas, fitness

def select_mating_pool(Beam_status, pop, fitness, num_parents_mating, t0, gen, Camera, Bus, show_the_best=False, save_best=False):
    """
    Selecting the best candidates in the current generation as parents for 
    producing the offspring of the next generation.
    """
    img_is_saturated = False
    Img = []
    parents = np.empty((num_parents_mating, num_params))
    isort = np.argsort(fitness)[::-1]
    parents_fitness = fitness[isort][:num_parents_mating]
    parents = pop[isort][:num_parents_mating,:]
    if show_the_best:
        t1 = time.time() - t0
        print('Time: {}, Fittest Parent: {}, Fitness: {}'.format(t1, parents[0], parents_fitness[0]))
        Beam_status, parents, _, Img = Reward(Beam_status, parents, parents[0], Camera, Bus)
        if Img.max() == 255:
            img_is_saturated = True
        plt.imshow(Img[::-1], cmap=cm.binary_r)
        plt.colorbar()
        if save_best:
            if gen < 10:
                plt.savefig(ImagesFolder + '/Gen_0%d_time_%d_Power_%1.2f_alignments_%f_%f_%f_%f_endMirror_%f.png' \
                     %(gen, (t1), parents_fitness[0], parents[0][0], \
                       parents[0][1], parents[0][2], parents[0][3], parents[0][4]))
            else:
                plt.savefig(ImagesFolder + '/Gen_%d_time_%d_Power_%1.2f_alignments_%f_%f_%f_%f_endMirror_%f.png' \
                     %(gen, (t1), parents_fitness[0], parents[0][0], \
                       parents[0][1], parents[0][2], parents[0][3], parents[0][4]))
        plt.show()
    return Beam_status, parents, parents_fitness, img_is_saturated, Img

def get_offsprings_Uniform(pairs, parents, offspring_size):
    """create offsprings using uniform crossover"""
    offsprings = np.empty(offspring_size)
    nn = 0
    for i in range(len(pairs)):
        for j in range(num_offsprings_per_pair):
            if nn == offspring_size[0] : break
            while True:
                # To make sure not all True/False
                i_select_genes = np.random.choice([True, False], num_params)
                if (i_select_genes.sum() != num_params) & (i_select_genes.sum() != 0): break
            offsprings[nn][i_select_genes] = parents[pairs[i][0]][i_select_genes]
            offsprings[nn][np.logical_not(i_select_genes)] = \
            parents[pairs[i][1]][np.logical_not(i_select_genes)]
            nn += 1
    return offsprings

def crossover(parents, offspring_size):
    # get all possible pairs
    pairs = []
    for p1 in range(num_parents_mating):
        for p2 in range(p1+1, num_parents_mating):
            pairs.append([p1,p2])
    pairs = np.array(pairs)
    # Give preference to combinations of top performing parents
    i_sort = np.argsort(pairs.sum(axis=1))
    pairs = pairs[i_sort]
    offsprings = get_offsprings_Uniform(pairs, parents, offspring_size)
    return offsprings

def mutation(Offspring_crossover, Rng):
    # Mutation changes a single gene in each offspring randomly.
    mutations = sample_d(Rng, shape=Offspring_crossover.shape)
    # The random value to be added to the gene.
    Offspring_crossover += mutations
    return Offspring_crossover

def jump_2_fundamental(Beam_status, pop_deltas, Mode, Camera, Bus, Sign=1., show_fig=True):
    # delta z_CM
    dz = -Lambda * (Mode[0] + Mode[1]) * np.arccos(Sign*np.sqrt(g1*g2)) / 2 / np.pi
    # i/p to dac
    z_step = np.array([0., 0., 0., 0., dz])
    # taking delta z_CM jump in cavity length
    Beam_status, pop_deltas, _, img = Reward(Beam_status, pop_deltas, z_step, Camera, Bus)
    return Beam_status, pop_deltas, img
