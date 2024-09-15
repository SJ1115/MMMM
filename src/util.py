import os
import random


def callpath(filename):
    return os.path.join(os.path.dirname(__file__), '..', filename)

def makedir(dir):
    # checking if the directory demo_folder2 
    # exist or not.
    if not os.path.isdir(dir):
        # if the demo_folder2 directory is 
        # not present then create it.
        os.makedirs(dir)
