#!/usr/bin/env python

# Generate a txt file containing input args:
# import numpy as np
# np.savetxt('condor_input_arg_data.txt', np.append(np.append(np.append(np.repeat('--i-start', 200).reshape(200,1), np.arange(0, 20000, 100).reshape(200,1).astype('str'), axis=1), np.repeat('--i-end', 200).reshape(200,1), axis=1), np.arange(100, 20100, 100).reshape(200,1).astype('str'), axis=1), fmt='%s', delimiter=',')

import pykat, os, argparse
import numpy as np
import h5py as hp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage.filters import gaussian
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("--i-start", help="start index", type=int)
parser.add_argument("--i-end", help="end index", type=int)
parser.add_argument("--output-data-folder", 
    default='/home/shreejit.jadhav/WORK/Beam_auto_alignment/Data/CavityScanData', 
    help="Path to the directory where hdf file is to be stored.")

args = parser.parse_args()

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)


# Funcs

def scan_cavity(kat, phimin=-90, phimax=90, N=2000, show_scan=False, return_max=False, **kwargs):
    """
    Scans the cavity to give the location of max power.
    """
    # run a copy
    kat2 = kat.deepcopy()
    kat2.parse("""
        # photo-diode
        pd P nOut
        # scanning the cavity
        xaxis M2 phi lin {} {} {}
        # plotting the amplitude of the detector measurements
        yaxis abs
    """.format(phimin, phimax, N))
    out = kat2.run()
    # find the highest power tuning of cavity
    imax = np.argmax(out['P'].ravel())
    # peaks
    peaks, _ = find_peaks(out['P'].ravel())
    if show_scan:
        plt.plot(out.x, out['P'])
        plt.plot(out.x[peaks], out['P'][peaks], 'r*')
        plt.xlabel('Position of mirror M2 [deg]')
        plt.ylabel('Power [W]',)
        plt.title('Transmitted power vs tuning of M2')
        plt.show()

    if return_max:
        return out.x[imax], out['P'].ravel()[imax]
    else:
        return out.x[peaks], out['P'].ravel()[peaks]

def get_CCD_image(kat, phi, Sigma=1, rng=15, n_pxls=128, pixl_thresh=5e-3, show_ccd_image=False, **kwargs):
    # tune the cavity to catch highest power
    kat2 = kat.deepcopy()
    kat2.parse("""
    # Beam camera
    beam CCD nOut
    """)
    kat2.M2.phi = phi
    # read image
    # print(kat.SM1.xbeta.value, kat.SM1.ybeta.value, kat.SM2.xbeta.value, kat.SM2.ybeta.value)
    kat2.parse("""
    xaxis CCD x lin -{0} {0} {1}
    x2axis CCD y lin -{0} {0} {1}
    yaxis abs
    """.format(rng, n_pxls-1))

    out = kat2.run()
    ret_img = out['CCD'] + pixl_thresh*np.abs(np.random.normal(size=out['CCD'].shape))
    # ret_img = gaussian(ret_img, sigma=Sigma)

    if show_ccd_image:
        plt.figure()
        # float16 dtype fails with imshow
        plt.imshow(ret_img.astype(np.float32))
        plt.colorbar()
        plt.show()

    return ret_img.astype(np.float16)

def get_scan_imstack(kat, i, beam_samples, n_ims=5, n_pxls=128, pixl_thresh=5e-3, show_imstack=False, **kwargs):
    """
    Returns a stack of images corresponding to the maxima in the scan of cavity.
    Each image has to have maximum pixel above the set threshold value.
    """
    # set i/p alignment
    kat = base.deepcopy()
    # set i/p alignment
    kat.SM1.xbeta = beam_samples['SM1']['x'][i]
    kat.SM1.ybeta = beam_samples['SM1']['y'][i]
    kat.SM2.xbeta = beam_samples['SM2']['x'][i]
    kat.SM2.ybeta = beam_samples['SM2']['y'][i]
    # catch phi's for all peaks in the scan
    phis, _ = scan_cavity(kat, return_max=False, **kwargs)
    # image stack
    nn = max(n_ims, len(phis))
    imstack = np.zeros((nn,n_pxls,n_pxls), dtype=np.float16)
    phi_maxs = np.zeros(nn, dtype=np.float16)
    # get corresponding images
    for i in range(len(phis)):
        imm = get_CCD_image(kat, phis[i], n_pxls=n_pxls, **kwargs)
        # consider only if pixl power crosses threshold
        if imm.max() > 30*pixl_thresh:
            imstack[i] = imm
            phi_maxs[i] = phis[i]
    powrs = np.max(np.max(imstack, axis=-1), axis=-1)
    # sort in decreasing max power
    isort = np.argsort(powrs)[::-1]
    imstack = imstack[isort]
    phi_maxs = phi_maxs[isort]
    # keep only those above thresh
    imstack = imstack[:n_ims]
    phi_maxs = phi_maxs[:n_ims]
    # phis kept wrt max powr peak and ones corr to empty images made 0
    phi_maxs -= phi_maxs[0]
    phi_maxs[len(phis):] *= 0
    # if show image
    if show_imstack:
        plt.figure(figsize=(25,5))
        for ii in range(n_ims):
            plt.subplot(1,n_ims,ii+1)
            # float16 dtype fails with imshow
            plt.imshow(imstack[ii].astype(np.float32))
        plt.show()
    # phi_maxs[0] is always going to be 0. so omit it.
    return imstack, phi_maxs[1:]


# Base Model

base = pykat.finesse.kat()
base.verbose = False
base.parse("""

# Input laser
# -----------------------------
l laser 1e-2 0 n0            # Laser (Power = 10 mW, wavelength offset = 0)

# Gaussian Beam
gauss GB laser n0 64.5e-6 0   # define beam waist

s s00 0.21801 n0 nL1a         # Space (Length = 0.218 m)

# Lens of f=150mm
lens Lns 0.150 nL1a nL1b

s s0 0.032 nL1b nSM1a        # Space (Length = 0.032 m)

# Steering mirrors
# -----------------------------
bs SM1 1 0 0 45 nSM1a nSM1b nSM1c nSM1d  # Beam splitter (R=1, T=0, phi(tuning)=0, alpha=45)

s s1 0.35 nSM1b nSM2a

bs SM2 1 0 0 45 nSM2a nSM2b nSM2c nSM2d

s s2 0.0884 nSM2b nM1a

# Cavity
# -----------------------------
# cavity mirror1 (R=0.95, T=0.05, phi=0)
m M1 0.95 0.05 0 nM1a nM1b

# cavity length 122.7mm for waist 140um
s lCav 0.1227 nM1b nM2a

# cavity mirror2 (R=0.99, T=0.01, phi=0)
m M2 0.99 0.01 0 nM2a nM2b

# Setting RoC of M2
attr M2 Rc 0.150

# Defning the cavity for spatial mode basis computation
cav cav1 M1 nM1b M2 nM2a

# Output
# -----------------------------
s sOut 0.1 nM2b nOut

# Photo diode
# pd P nOut

# Amplitude detector
# ad AD11 1 1 1064 nOut

# Beam camera
# beam CCD nOut

# max order of modes
maxtem 10

trace 8
""")


# Initialization

# seed
np.random.seed(314)

# waist size in m
waist = 140e-6
# range of movement of the waist center at the waist location in the units of waist size
a = 3.
# cumulative distance of waist from SM1 in m
d1 = 0.35+0.0884
# cumulative distance of waist from SM2 in m
d2 = 0.0884
dist_to_w = {'SM1': d1, 'SM2': d2}
# number of alignments for generating training data
samples = int(args.i_end - args.i_start)
# no of images in cavity scan image stack
N_IMS=7
# other vars
N_PXLS=128
# amplitude of noise that is added to all the images
NOISE_AMP=1e-3
SIGMA=2
SHOW_IMSTACK=False
SHOW_SCAN=False
SHOW_CCD_IMAGE=False

if not os.path.isdir(args.output_data_folder):
    os.mkdir(args.output_data_folder)

imstack = np.zeros((samples,N_IMS,N_PXLS,N_PXLS), dtype=np.float16)
phim = np.zeros((samples,N_IMS-1), dtype=np.float16)

beam_status = {'SM1': {'x': np.random.random(samples), 'y': np.random.random(samples)}, 'SM2': {'x': np.random.random(samples), 'y': np.random.random(samples)}}

# random alignments
for sm in beam_status.keys():
    for direction in beam_status[sm].keys():
        # give angle in deg
        beam_status[sm][direction] = 2. * a * beam_status[sm][direction] - a
        beam_status[sm][direction] *= waist / dist_to_w[sm]

kat = base.deepcopy()

# randomly sample the range of SM1 and SM2 in (x,y) space
t0 = time()
for i in range(samples):
    print(i, time()-t0)
    # collect stack of images and relative spacings for peaks in the cavity scan (arranged in decreasing order of peak pixl power)
    imstack[i], phim[i] = get_scan_imstack(kat, i, beam_status, Sigma=SIGMA, n_ims=N_IMS, n_pxls=N_PXLS, pixl_thresh=NOISE_AMP)

# save the data in a hdf file
hdffile = args.output_data_folder+'/training_data_cavity_scan_{}-{}.hdf'.format(args.i_start, args.i_end)
data = hp.File(hdffile, 'w')
data.create_dataset('SM1x', data=beam_status['SM1']['x'])
data.create_dataset('SM1y', data=beam_status['SM1']['y'])
data.create_dataset('SM2x', data=beam_status['SM2']['x'])
data.create_dataset('SM2y', data=beam_status['SM2']['y'])
# images of transmitted beam for peaks in the cavity scan
# (arranged by decreasing order of max pixel power)
data.create_dataset('image_stack', data=imstack)
# relative positions of other peaks in the scan wrt the highest one (deg)
data.create_dataset('phi_stack', data=phim)
data.close()

print("data saved to {}".format(hdffile))
