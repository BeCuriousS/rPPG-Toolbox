"""The dataloader for BP4D+ datasets.

Details for the BP4D+ Dataset see https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html
If you use this dataset, please cite the following publications:
Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, Peng Liu, and Jeff Girard
“BP4D-Spontaneous: A high resolution spontaneous 3D dynamic facial expression database”
Image and Vision Computing, 32 (2014), pp. 692-706  (special issue of the Best of FG13)

AND

Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, and Peng Liu
“A high resolution spontaneous 3D dynamic facial expression database”
The 10th IEEE International Conference on Automatic Face and Gesture Recognition (FG13),  April, 2013. 
"""

import glob
import zipfile
import os
import re

import cv2
from skimage.util import img_as_float
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

from dataset.data_loader.CustomDataLoader import CustomDataLoader


class CustomBP4DPlusLoader(CustomDataLoader):
    """The data loader for custom data given in a specific preprocessed format."""

    def __init__(self, name, data_path, config_data, subject_list, num_workers, batch_sample_len, channel_types):
        """Initializes an BP4D+ dataloader.
        """
        super().__init__(
            name,
            data_path,
            config_data,
            subject_list,
            num_workers,
            batch_sample_len,
            channel_types)

    def get_raw_data(self, data_path):
        """Returns data directories under the path."""
        # the datapath points to the directory containing the datasets as the training data comprises 3 datasets
        dirs = []
        db_path = os.path.join(data_path, 'BP4D_db')
        for subj in os.listdir(db_path):
            subj_path = os.path.join(db_path, subj)
            for rec in os.listdir(subj_path):
                dirs.append({'index': 'BP4D_db' + '#' + subj + '_' + rec,
                             'path': os.path.join(db_path, subj, rec)})
        return dirs
