"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

from dataset.data_loader.CustomDataLoader import CustomDataLoader


class CustomUBFCrPPGLoader(CustomDataLoader):
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
        db_path = os.path.join(data_path, 'UBFCRPPG_db')
        for subj in os.listdir(db_path):
            subj_path = os.path.join(db_path, subj)
            dirs.append({'index': 'PURE_db' + '#' + subj,
                         'path': os.path.join(db_path, subj)})
        return dirs
