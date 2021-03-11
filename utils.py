import os, glob
import numpy as np
import scipy.io as sio
from PIL import Image


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_list_files(scene_path, ext='', sort_list=True):
    working_dir = os.getcwd()

    os.chdir(scene_path)
    file_list = [] 

    for file in glob.glob("*{}".format(ext)):
        file_list.append(file)

    if sort_list:
        file_list.sort() #file_list.sort(key=lambda x: int(x[-7:-4]))

    os.chdir(working_dir)

    return file_list
