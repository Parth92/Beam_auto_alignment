#!/usr/bin/env python

"""
Script to combine the training data
"""

import numpy as np
import h5py as hp
import glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training-data-folder",
    default='/shome/shreejit.jadhav/WORK/Beam_auto_alignment/Data/CavityScanData',
    help="Path to the directory from where hdf files are to be read.")
parser.add_argument("--combined-data-file-path",
    default='/scratch/shreejit/WORK/Beam_auto_alignment/Data/CavityScanData/combined_training_data_cavity_scan.hdf',
    help="Path of the hdf file where combined data is to be stored.")

args = parser.parse_args()

# read training data files
files = glob.glob(args.training_data_folder+'/*.hdf')
files.sort()

# create combined data file
comb_file = hp.File(args.combined_data_file_path, 'w')

# var dict
data = {}
# initialize the empty data arrays
with hp.File(files[0], 'r') as f1:
    for k in f1.keys():
        shape1 = list(f1[k].shape)
        shape1[0] *= len(files)
        shape1 = tuple(shape1)
        data[k] = comb_file.create_dataset(k, shape=shape1, dtype=f1[k].dtype)

# combine data loop
for i, f1 in enumerate(files):
    print(i, f1)
    with hp.File(f1, 'r') as ff1:
        for k in comb_file.keys():
            for j in range(len(ff1[k])):
                data[k][i*len(ff1[k])+j] = ff1[k][j][()]

comb_file.close()

print('File saved as {}'.format(args.combined_data_file_path))
print('Finished..')
