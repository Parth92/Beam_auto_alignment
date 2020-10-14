import numpy as np
import matplotlib.pyplot as plt

import imageio
# import pickle
import datetime
import time
import os
import sys
import busworks

from math import factorial
from copy import deepcopy
from tqdm import tqdm
from pypylon import pylon
from pypylon import genicam
from matplotlib import cm
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from scipy import ndimage as ndi

# import keras
# from keras import backend as k
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import tensorflow as tf

################################ Beam alignment functions ################################

# Initialization
n_pixl = 128
imdim = (540,720)
RepoDir = '/home/controls/Beam_auto_alignment'
ImagesFolder = RepoDir + '/Data/Actual_cavity_Fittest_points_per_gen_' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')[:-10]
if not os.path.exists(ImagesFolder): os.mkdir(ImagesFolder)
print('Saving imgages in ' + ImagesFolder + '\n')
# SaveModelFolder = RepoDir + '/Data/TrainedModels/'
# Model_Name = 'Trained_Model_2019-07-03_17-15'

# # Read the pre-trained CNN model
# cnn = keras.models.load_model(SaveModelFolder + Model_Name + '.h5')
# # load the encoder
# loaded_Encoder = pickle.load(open(SaveModelFolder + 'Encoder_of_' + Model_Name + '.npy', 'rb'))

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
Range_orig = 2.
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
scale_params = np.array([waist/d1, waist/d1, waist/d2, waist/d2, Lambda/2./Range_orig])   # dist bet two peaks is lambda/2; taking a slightly higher range lambda/1.5
PZT_scaling = np.array([V_DAC_max/phi_SM_max, V_DAC_max/phi_SM_max, \
                        V_DAC_max/phi_SM_max, V_DAC_max/phi_SM_max, V_DAC_max/phi_CM_PZT_max])
pop_per_gen = 70
N_CM_STEPS = 30    # number of z_CM scan steps
num_generations = 25
num_params = len(scale_params) - 1
num_parents_mating = pop_per_gen // 10  # 10% of new population are parents
num_offsprings_per_pair = 2 * (pop_per_gen - num_parents_mating) // num_parents_mating + 1
# after each iteration, range shrinks by
shrink_factor = 1.5 / pop_per_gen ** (1./len(scale_params))  # make sure it is < 1.
print('shrink factor: ', shrink_factor)
# Defining the population size.
pop_size = (pop_per_gen,num_params) # The population will have sol_per_pop chromosome \
# where each chromosome has num_weights genes.
# timeout for image retrieval
Timeout = 500 # microseconds
# Initial exposure in microsecs
Exposure = 80.
# Exposure reduction factor at each occurance of saturation
ENSURE_UNSATURATED = False
Exposure_red_factor = 0.6

# best parent
BEST_REWARD = 0.
BEST_BEAM_STATUS = []
BEST_IMG = []
BEST_IMG_OF_GEN = []

# Mode params
SEPARATION = 5
SIGMA = 1
WIDTH = 40
THRESH = 0.3

# Digital locking
P_thresh = 0.9 # thresh wrt max power that has to be maintained
P_max = 300.
P_old = 0.
locking_loop_on = False


def sigmoid(x):
    return 10./(1.+np.exp(-x))

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

# def pad_and_resize_image(image):
#     # padding with const value of first pixel in image
#     im_new = np.ones((max(image.shape),max(image.shape)))*image[0,0]
#     pad = int((max(image.shape)-min(image.shape))/2)
#     im_new[pad:-pad,:] = image
#     # resize 
#     im = tf.image.resize_nearest_neighbor(
#         im_new.reshape(1, im_new.shape[0], im_new.shape[1], 1),
#         (n_pixl,n_pixl),
#         align_corners=False,
#         name=None
#     )
#     with sess.as_default():
#         im_new = im.eval()
#     im_new = im_new[0,:,:,0]
#     return im_new

def Capture_image(Camera, timeout):
    ## Old method
    # Dat = Camera.GrabOne(timeout)
    # img = Dat.Array
    ## New method
    # attempt 5 times to retrieve an image
    for j in range(5):
        if Camera.IsGrabbing():
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = Camera.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
        else:
            print("Camera didn't grab image..")
            continue
        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data
            img = grabResult.Array
            break
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    return img

def sample_d(Rng, shape=pop_size, first_sample=False):
    """
    Takes i/p range in umits of beam waist at the waist location.
    Outputs sets of deltas (in radians) to be fed to steering mirrors.
    O/P shape : pop_size
    """
    delta = np.random.uniform(low=-Rng, high=Rng, size=shape)
    # delta[:,shape[1]-1] *= Range_orig/Rng # always scan full FSR range
    # if first_sample:
    #     # CM pzt does not take -ve values
    #     i_n = np.where(delta[:,shape[1]-1] < 0.)
    #     delta[:,shape[1]-1][i_n] += Range_orig
    delta *= scale_params[:shape[1]]
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
        Beam_status, delta_z, R_new, _, _, _ = Reward(Beam_status, delta_z, delta_z[ii], Camera, Bus)
        R.append(R_new)
    i_max = np.argmax(R)
    # pop_deltas updated at this final step
    Beam_status, pop_deltas, _, Img, _, _ = Reward(Beam_status, pop_deltas, delta_z[i_max], Camera, Bus)
    if show_fig:
        plt.imshow(Img[::-1], cmap=cm.binary_r)
        plt.colorbar()
        plt.show()
    return Beam_status, pop_deltas, Img

def Set_Voltage(Beam_status, Bus):
    ip_V = Beam_status * PZT_scaling
    # making sure that the DAC voltages remain in the required range
    if (np.abs(ip_V) > V_DAC_max).sum() > 0:
        print('DAC I/P voltage exceeded limit! Forcefully brought down.\nExceeding points: ', np.abs(ip_V) > V_DAC_max, '\nVoltage status: ', ip_V)
    ip_V[ip_V > V_DAC_max] = V_DAC_max
    ip_V[ip_V < -V_DAC_max] = -V_DAC_max
    try:
        Bus.set_voltages(ip_V, 1)
    except:
        print('Error! Failed to set voltages: ', ip_V)

def Max_Power_image_in_FSR(Beam_status, Camera, Bus):
    global Exposure
    # scan full FSR
    dCMphi = Lambda / N_CM_STEPS
    Zsteps = np.arange(0., Lambda, dCMphi)
    im_array = np.zeros((imdim[0],imdim[1],len(Zsteps)))
    for j in range(N_CM_STEPS):
        Set_Voltage(np.append(Beam_status, Zsteps[j]), Bus)
        # reward fn as total power in the image
        Img1 = Capture_image(Camera, Timeout)
        # while saturated, keep reducing the exposure
        while Img1.max() >= 254:
            plt.imshow(Img1[::-1], cmap=cm.binary_r)
            plt.colorbar()
            plt.show()
            print('Image saturated! Max power: {}'.format(Img1.max()))
            if ENSURE_UNSATURATED:
                Exposure = Exposure*Exposure_red_factor
                Camera.ExposureTimeAbs = Exposure
                Img1 = Capture_image(Camera, Timeout)
                print('Exposure updated to {} microsec. Max power in image" {}'.format(Exposure, Img1.max()))
                plt.imshow(Img1[::-1], cmap=cm.binary_r)
                plt.colorbar()
                plt.show()
            else: break
        im_array[:,:,j] = Img1
    # pick the image with max power
    imaxpowr = np.argmax(im_array.sum(axis=(0,1)))
    Imgmax = im_array[:,:,imaxpowr]
    return Imgmax

def Reward_fn(Img1):
    # option 1
    # finding the mode
    Mode = Find_mode2(Img1, separation1=SEPARATION, Sigma1=SIGMA, Width=WIDTH, thresh=THRESH, corner=0)
    # R_fn1 = Img1.sum()/n_pixl**2./(Mode[0]+Mode[1]+1.)
    R_fn1 = 2e4*Img1.sum()/n_pixl**2./(Mode[0]+Mode[1]+1.)**4./Exposure
    # option 2
    # finding chisqr based reward function
    # R_fn1 = 100*(sigmoid(Img1.sum()/Exposure/chi_sq(Img1)/1000.) + 0.5)
    # option 3
    # R_fn1 = Img1.sum()/Exposure
    return R_fn1

def Reward(Beam_status, pop_deltas, step, Camera, Bus):
    global BEST_IMG, BEST_REWARD, BEST_BEAM_STATUS
    # take the delta step
    Beam_status += step
    # cumulatively subtracting each delta step from all deltas
    pop_deltas -= step
    # get image
    dummy_reward = False
    if dummy_reward:
        return np.random.randint(255), np.zeros((n_pixl,n_pixl))
    else:
        Img = Max_Power_image_in_FSR(Beam_status, Camera, Bus)
        # reward for the image
        R_new = Reward_fn(Img)
        if R_new > BEST_REWARD:
            BEST_IMG = Img
            BEST_REWARD = R_new
            BEST_BEAM_STATUS = Beam_status.copy()
            print('Best reward: ', BEST_REWARD)
            print('Best alignment: ', BEST_BEAM_STATUS)
    return Beam_status, pop_deltas, R_new, Img

def calc_pop_fitness(Current_beam_status, New_pop_deltas, Camera, Bus, only_offsprings=False):
    """
    Calculating the fitness value of each solution in the current population.
    Also returns the current beam location (after adding the steps taken so far)
    """
    global BEST_IMG_OF_GEN
    fitness = np.empty(pop_per_gen)
    R_best_gen = 0.
    range_vals = range(pop_per_gen)
    for ii in range_vals:
        # # take the delta step
        # Current_beam_status += New_pop_deltas[ii]
        # # cumulatively subtracting each delta step from all deltas
        # New_pop_deltas -= New_pop_deltas[ii]
        # fitness[ii], _ = Reward_fn(Current_beam_status, Camera, Bus)
        Current_beam_status, New_pop_deltas, fitness[ii], Img1 = Reward(Current_beam_status, New_pop_deltas, New_pop_deltas[ii], Camera, Bus)
        if ii == 0:
            BEST_IMG_OF_GEN = Img1
        else:
            if fitness[ii] > R_best_gen:
                R_best_gen = fitness[ii]
                BEST_IMG_OF_GEN = Img1
    return Current_beam_status, New_pop_deltas, fitness

def select_mating_pool(Beam_status, pop, fitness, num_parents_mating, t0, gen, Camera, Bus, Best_im_gen, show_the_best=False, save_best=False):
    """
    Selecting the best candidates in the current generation as parents for 
    producing the offspring of the next generation.
    """
    img_is_saturated = False
    parents = np.empty((num_parents_mating, num_params))
    isort = np.argsort(fitness)[::-1]
    parents_fitness = fitness[isort][:num_parents_mating]
    parents = pop[isort][:num_parents_mating,:]
    if show_the_best:
        t1 = time.time() - t0
        print('Time: {}, Fittest Parent: {}, Fitness: {}'.format(t1, Beam_status+parents[0], parents_fitness[0]))
        # Beam_status, parents, _, Img = Reward(Beam_status, parents, parents[0], Camera, Bus)
        if Best_im_gen.max() == 255:
            img_is_saturated = True
        plt.imshow(Best_im_gen, cmap=cm.binary_r)
        plt.colorbar()
        print(parents_fitness[0])
        if save_best:
            if gen < 10:
                plt.savefig(ImagesFolder + '/Gen_0%d_time_%1.2e_Power_%1.2f_alignments_%f_%f_%f_%f.png' \
                     %(gen, (t1), parents_fitness[0], parents[0][0], \
                       parents[0][1], parents[0][2], parents[0][3]))
            else:
                plt.savefig(ImagesFolder + '/Gen_%.2e_time_%d_Power_%1.2f_alignments_%f_%f_%f_%f.png' \
                     %(gen, (t1), parents_fitness[0], parents[0][0], \
                       parents[0][1], parents[0][2], parents[0][3]))
        plt.show()
    return Beam_status, parents, parents_fitness, img_is_saturated

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

def mutation(Beam_status, Offspring_crossover, Rng):
    # Mutation changes a single gene in each offspring randomly.
    mutations = sample_d(Rng, shape=Offspring_crossover.shape)
    # The random value to be added to the gene.
    Offspring_crossover += mutations
    # CM pzt does not take -ve values
    i_N = np.where(Beam_status[-1]+Offspring_crossover[:,Offspring_crossover.shape[1]-1] < 0.)
    Offspring_crossover[:,Offspring_crossover.shape[1]-1][i_N] += Lambda/2.
    # CM pzt does not take values beyond V_DAC_max
    i_P = np.where(Beam_status[-1]+Offspring_crossover[:,Offspring_crossover.shape[1]-1] > phi_CM_PZT_max)
    Offspring_crossover[:,Offspring_crossover.shape[1]-1][i_P] -= Lambda/2.
    return Offspring_crossover

def jump_2_fundamental(Beam_status, pop_deltas, Mode, Camera, Bus, reverse=False, show_fig=True):
    # delta z_CM
    if reverse:
        dz = -Lambda * (Mode[0] + Mode[1]) * np.arccos(-np.sqrt(g1*g2)) / 2 / np.pi + Lambda * (Mode[0] + Mode[1]) * np.arccos(np.sqrt(g1*g2)) / 2 / np.pi
    else:
        dz = -Lambda * (Mode[0] + Mode[1]) * np.arccos(np.sqrt(g1*g2)) / 2 / np.pi
    # i/p to dac
    z_step = np.array([0., 0., 0., 0., dz])
    # taking delta z_CM jump in cavity length
    Beam_status, pop_deltas, _, img = Reward(Beam_status, pop_deltas, z_step, Camera, Bus)
    return Beam_status, pop_deltas, img

# Fundamental mode position is [5.37932319e-04, 3.33051668e-04, -1.06754584e-03, -2.70645706e-03, 1.14001752e-06]
# [-4.85114666e-04, -2.79346829e-06, -1.57814286e-03, -2.89688008e-03, 3.61083367e-07]

def mean_sigma(pdf_x):
    # mean
    xx = np.arange(0.5, len(pdf_x), 1.) # centres of pixels
    m = np.sum(pdf_x*xx)/pdf_x.sum()
    # variance
    var = np.sum((xx-m)**2.)/(len(xx)-1.)
    return m, var

def get_mean_sigmax_sigmay(Img):
    # x
    px = np.sum(Img, axis=0)
    x_mean, x_var = mean_sigma(px)
    # y
    py = np.sum(Img, axis=1)
    y_mean, y_var = mean_sigma(py)
    return (x_mean, y_mean), (x_var, y_var)

def Gauss(mu, var, Img):
    # axes sampled at pixel centres
    x, y = np.meshgrid(np.arange(0.5, Img.shape[1], 1.), 
                       np.arange(0.5, Img.shape[0], 1.))
    return np.exp(-((x-mu[0])**2./2./var[0] + (y-mu[1])**2./2./var[1]))

def chi_sq(Img):
    # get mean and variance
    mean, var = get_mean_sigmax_sigmay(Img)
    # fit gaussian
    g = Gauss(mean, var, Img)
    # chi sqr
    scale_up = Img.shape[0] * Img.shape[1]
    chisqr = scale_up*(g/g.sum() - Img/Img.sum())**2.
    return chisqr.sum()


################################ Mode identification functions ################################


def one_peak_per_filter(peaks, separation):
    Updated_peaks = []
    i = 0
    while len(peaks>0):
        # collecting all the peaks in a filter: peak1
        if len(peaks)>1:
            i_pick = np.where((peaks[:,0] >= peaks[i,0]-separation) & (peaks[:,0] <= peaks[i,0]+separation) & \
                              (peaks[:,1] >= peaks[i,1]-separation) & (peaks[:,1] <= peaks[i,1]+separation))
            peak1 = peaks[i_pick]
            Len = len(peak1)
            # getting only one peak which is avg of all peaks in one filter
            if Len>1:
                peak_new = [int(np.sum(peak1[:,0])/Len), int(np.sum(peak1[:,1])/Len)]
            else:
                peak_new = peak1.ravel().tolist()
            # add this new peak in a new list of updated peaks
            Updated_peaks.append(peak_new)
            # delete the peaks from this filter from the original list
            peaks = np.delete(peaks, i_pick, 0)
        else:
            peak_new = peaks.ravel().tolist()
            Updated_peaks.append(peak_new)
            peaks = np.delete(peaks, 0, 0)
        # print(i, 'peak_new: ', peak_new, '\nPeaks: ', peaks)
        # i += 1
    return np.array(Updated_peaks)

def two_pt_line(point1, point2, xval):
    if point1[0] == point2[0]:
        return 
    else:
        Ys = (point2[1]-point1[1])
        Ys *= (xval-point1[0])
        Ys /= (point2[0]-point1[0])
        Ys += point1[1]
        return Ys

def linking_pts(point1, point2):
    if np.abs(point1[0]-point2[0]) > np.abs(point1[1]-point2[1]):
        if point1[0] > point2[0]:
            dummy = point2
            point2 = point1
            point1 = dummy
        xs = np.arange(float(point1[0]), float(point2[0])+1., 1.)
        ys = two_pt_line(point1.astype(float), point2.astype(float), xs)
    else:
        # y intersections of vertical line
        if point1[1] > point2[1]:
            dummy = point2
            point2 = point1
            point1 = dummy
        ys = np.arange(float(point1[1]), float(point2[1])+1., 1.)
        xs = two_pt_line(point1.astype(float)[::-1], point2.astype(float)[::-1], ys)
    # linking pts
    Lnk_Pts = np.array([xs, np.ceil(ys)]).T
    Lnk_Pts = np.append(Lnk_Pts, np.array([xs, np.floor(ys)]).T, axis=0)
    Lnk_Pts = Lnk_Pts.astype(int)
    return Lnk_Pts

def one_peak_per_island(peaks, img):
    Updated_peaks = []
    i = 0
    while len(peaks>0):
        i_pick = [0]
        if len(peaks)>1:
            for ii in range(1,len(peaks)):
                Lnk_pts = linking_pts(peaks[0], peaks[ii])
                if np.prod(img[Lnk_pts[:,1], Lnk_pts[:,0]]):
                    i_pick.append(ii)
            peak1 = peaks[np.array(i_pick)]
            Len = len(i_pick)
            if Len>1:
                peak_new = [int(np.sum(peak1[:,0])/Len), int(np.sum(peak1[:,1])/Len)]
            else:
                peak_new = peak1.ravel().tolist()
            Updated_peaks.append(peak_new)
            peaks = np.delete(peaks, i_pick, 0)
        else:
            peak_new = peaks.ravel().tolist()
            Updated_peaks.append(peak_new)
            peaks = np.delete(peaks, 0, 0)
    return np.array(Updated_peaks)

def Find_Peaks(IMG, separation=10, Sigma=1, thresh=0.5, show_ada_thresh=False, show_fig=False):
    Smooth_img = gaussian(IMG,  sigma=Sigma)
    Smooth_img = Thresh(Smooth_img, thresh)
    if show_ada_thresh: image_max = ndi.maximum_filter(Smooth_img, size=separation, mode='constant')
    coordinates = peak_local_max(Smooth_img, min_distance=separation, exclude_border=False)
    np.random.shuffle(coordinates)
    coordinates = np.array(coordinates[:36])
    dummy = coordinates[:,0].copy()
    coordinates[:,0] = coordinates[:,1]
    coordinates[:,1] = dummy
    coordinates = one_peak_per_filter(coordinates, separation)
    coordinates = one_peak_per_island(coordinates, Smooth_img)
    if show_fig:
        plt.figure()
        plt.title('thresholded smooth img')
        plt.imshow(Smooth_img, cmap=plt.cm.gray)
        plt.colorbar()
        plt.plot(coordinates[:, 0],
                 coordinates[:, 1], 'r*')
    if show_ada_thresh:
        plt.figure()
        plt.title('filters')
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

def test_consistency(mode, Peaks, verbose=False):
    if (mode[0]+1)*(mode[1]+1) == len(Peaks) or (mode[0]+1)*(mode[1]+1) == len(Peaks)+1:
        return True
    else:
        if verbose:
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
    
def Find_mode2(img_loc, separation1=10, Sigma1=1, Width=10, thresh=0.5, corner=0, show_ada_thresh=False, show_fig=False, show_basis=False, verbose=False):
    # Read image
    # if image location
    if isinstance(img_loc, str):
        print('reading image from ', img_loc)
        img1 = 255 - imageio.imread(img_loc)
    # if image itself
    elif isinstance(img_loc, np.ndarray):
        img1 = img_loc
    peaks = Find_Peaks(img1, separation=separation1, Sigma=Sigma1, thresh=thresh, show_ada_thresh=show_ada_thresh, show_fig=show_fig)
    # Case 0: (0,0)
    if len(peaks) == 1:
        mode = (0,0)
        if verbose:
            print("Case0: ", mode)
        if show_ada_thresh or show_basis: plt.show()
        return mode
    # Case 1: (m,0)
    mode = case1_m_0(peaks, w=Width, show_basis=show_basis)
    if verbose:
        print("Case1 result: ", mode)
    if test_consistency(mode, peaks, verbose=verbose):
        if show_ada_thresh or show_basis: plt.show()
        return mode
    else:
        # Case 2: (0,n)
        mode = case2_0_n(peaks, w=Width, show_basis=show_basis)
        if verbose:
            print("Case2 result: ", mode)
        if test_consistency(mode, peaks, verbose=verbose):
            if show_ada_thresh or show_basis: plt.show()
            return mode
        else:
            # Case 3: (m,n)
            for corner in range(4):
                mode = case3_m_n(peaks, w=Width, corner=corner, show_basis=show_basis)
                if verbose:
                    print("Case3 corner_{} result: ".format(corner), mode)
                if test_consistency(mode, peaks, verbose=verbose) or corner==3:
                    if show_ada_thresh or show_basis: plt.show()
                    return mode
