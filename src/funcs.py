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
    coordinates = peak_local_max(Smooth_img, min_distance=separation)
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

def Find_mode(img_loc, separation1=5, Sigma1=1, Width=10, thresh=0.5, Show_ada_thresh=False, Show_fig=True, corner=0, show_peaks=False):
    # Read image
    img = 255 - imageio.imread(img_loc)
    # thresholding and smoothing image to find peaks
    img1 = Thresh(img, thresh)
    peaks = Find_Peaks(img1, separation=separation1, Sigma=Sigma1, show_ada_thresh=Show_ada_thresh, show_fig=Show_fig)
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
                             Show_ada_thresh=Show_ada_thresh, Show_fig=Show_fig, corner=corner, show_peaks=show_peaks)
            print(mode)
        else:
            pass
    plt.show()
    return mode

def test_consistency(mode, Peaks):
    if (mode[0]+1)*(mode[1]+1) < len(Peaks):
        print('Warning: Something went wrong! \nNumber of peaks ({}) \
        inferred from mode does not match with actual number of peaks({})'.format((mode[0]+1)*(mode[1]+1), len(Peaks)))
        return False
    else:
        return True

def case1_m_0(Peaks, w=10):
    # bottom & top
    i_b = Peaks[0,:].argmin()
    i_t = Peaks[0,:].argmax()
    Peak_b = Peaks[i_b]
    Peak_t = Peaks[i_t]
    # constructing unit vect and slope
    basis, m = find_basis(Peak_b, Peak_t)
    if abs(m) > 1.:
        return (0,0)
    else:
        return peaks_to_mode(Peak_b, Peaks, w, basis, m)

def case2_0_n(Peaks, w=10):
    # left & right
    i_l = Peaks[:,0].argmin()
    i_r = Peaks[:,0].argmax()
    Peak_l = Peaks[i_l]
    Peak_r = Peaks[i_r]
    # constructing unit vect and slope
    basis, m = find_basis(Peak_l, Peak_r)
    if abs(m) < 1.:
        return (0,0)
    else:
        return peaks_to_mode(Peak_l, Peaks, w, basis, m)

def case3_m_n(Peaks, w=10, corner=0, show_basis=False):
    # catching 4 corner points
    i_b = Peaks[:,1].argmin()  # bottom point
    i_t = Peaks[:,1].argmax()  # top point
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
    
def Find_mode2(img_loc, separation1=10, Sigma1=1, Width=10, thresh=0.5, Show_ada_thresh=False, Show_fig=True, corner=0, show_peaks=False, show_basis=False):
    # Read image
    img = 255 - imageio.imread(img_loc)
    # img = imageio.imread(img_loc)[43:243, 54:320, 0]
    # thresholding and smoothing image to find peaks
    img1 = Thresh(img, thresh)
    peaks = Find_Peaks(img1, separation=separation1, Sigma=Sigma1, show_ada_thresh=Show_ada_thresh, show_fig=Show_fig)
    # Case 0: (0,0)
    if len(peaks) == 1:
        mode = (0,0)
        print("Case0: ", mode)
        return mode
    # Case 1: (m,0)
    mode = case1_m_0(peaks, w=Width)
    print("Case1 result: ", mode)
    if test_consistency(mode, peaks):
        return mode
    else:
        # Case 2: (0,n)
        mode = case2_0_n(peaks, w=Width)
        print("Case2 result: ", mode)
        if test_consistency(mode, peaks):
            return mode
        else:
            # Case 3: (m,n)
            for corner in range(4):
                mode = case3_m_n(peaks, w=Width, corner=corner)
                print("Case3 corner_{} result: ".format(corner), mode)
                if test_consistency(mode, peaks) or corner==3:
                    # _ = case3_m_n(peaks, w=Width, corner=corner, show_basis=True)
                    return mode
