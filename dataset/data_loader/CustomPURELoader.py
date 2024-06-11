"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import glob
import glob
import json
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

from dataset.data_loader.CustomDataLoader import CustomDataLoader


class CustomPURELoader(CustomDataLoader):
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
        db_path = os.path.join(data_path, 'PURE_db')
        for subj in os.listdir(db_path):
            subj_path = os.path.join(db_path, subj)
            for rec in os.listdir(subj_path):
                dirs.append({'index': 'PURE_db' + '#' + subj + '_' + rec,
                             'path': os.path.join(db_path, subj, rec)})
        return dirs
