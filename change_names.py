import sys
import os

dir = 'VISC_Dataset_SON/'



for file in os.listdir(dir):

    old_filepath = os.path.join(dir, file)
    new_name = str(file[0]) + '_' + str(file[3:-5]) + str(file[-4:])

    new_filepath = os.path.join(dir, new_name)
    os.rename(old_filepath, new_filepath)