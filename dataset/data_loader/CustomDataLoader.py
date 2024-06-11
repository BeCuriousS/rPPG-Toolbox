"""
-------------------------------------------------------------------------------
Created: 04.06.2024, 22:36
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
-------------------------------------------------------------------------------
Purpose: To enable the loading of our custom dataset that is given in an already preprocessed form as it was used for the training and validation of DeepPerfusion. We used the existing data loaders as a starting point.
-------------------------------------------------------------------------------
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager
import pickle

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class CustomDataLoader(BaseLoader):
    """The data loader for custom data given in a specific preprocessed format."""

    def __init__(self, name, data_path, config_data, subject_list, num_workers, batch_sample_len, channel_types):
        """Initializes custom dataloader.
        """
        super().__init__(
            name,
            data_path,
            config_data,
            subject_list,
            num_workers)
        self.batch_sample_len = batch_sample_len
        self.channel_types = channel_types

    # NOTE This method is overwritten because we want to use only one sample of the chunked data during training and not the whole chunk
    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        if self.channel_types == 'Standardized':
            data = data[..., 3:]
        elif self.channel_types == 'DiffNormalized':
            data = data[..., :3]
        elif self.channel_types in ['None', 'none', None]:
            pass
        else:
            raise ValueError(
                f'The parameter for channel_types is not correct: {self.channel_types}')
        # NOTE the following was changed to enable a more diverse generated batch in case the data is processed frame by frame (e.g. Deepphys) and not as a sequence of frames (e.g. Physformer)
        #############################
        if self.batch_sample_len is not None:
            start_i = np.random.randint(len(data)-self.batch_sample_len)
            data = data[start_i:start_i+self.batch_sample_len]
            label = label[start_i:start_i+self.batch_sample_len]
        #############################
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)
        label = np.float32(label)
        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments,
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id

    def get_raw_data(self, data_path):
        """Returns data directories under the path."""
        # the datapath points to the directory containing the datasets as the training data comprises 3 datasets
        dirs = []
        for db_name in ['ColdStressStudy_db',
                        'CardioVisioIBMT320x420_db',
                        'CardioVisioIBMT640x840_db']:
            db_path = os.path.join(data_path, db_name)
            for subj in os.listdir(db_path):
                dirs.append({'index': db_name + '#' + subj,
                             'path': os.path.join(db_path, subj)})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_frames_from_segments(
                os.path.join(data_dirs[i]['path']))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], '*.npy')))
        else:
            raise ValueError(
                f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(
                frames, fs=self.config_data.FS)
        else:
            bvps, bvps_ts = self.read_wave(
                os.path.join(data_dirs[i]['path']))

        frames_clips, bvps_clips, bvps_ts_clips = self.preprocess(
            frames, bvps, bvps_ts, config_preprocess)
        input_name_list, label_name_list, label_ts_name_list = self.save_multi_process(
            frames_clips, bvps_clips, bvps_ts_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_frames_from_segments(segments_path):
        """Reads a video file, returns frames(T, H, W, 3) """
        frame_files = sorted([fn for fn in os.listdir(
            segments_path) if fn.startswith('record_frames')])
        frames = []
        for ff in frame_files:
            ff_path = os.path.join(segments_path, ff)
            with open(ff_path, 'rb') as f:
                frames_segments = pickle.load(f)['frames']
            frames.append(frames_segments)
        return np.concatenate(frames, axis=0)

    @staticmethod
    def read_wave(segments_path):
        """Reads a bvp signal file."""
        signal_files = sorted([fn for fn in os.listdir(
            segments_path) if fn.startswith('record_signals')])
        labels = []
        labels_ts = []
        for sf in signal_files:
            sf_path = os.path.join(segments_path, sf)
            with open(sf_path, 'rb') as f:
                labels_segments = pickle.load(f)
                labels_ts_segments = labels_segments['ts']
                labels_segments = labels_segments['ppg']
            labels.append(labels_segments)
            labels_ts.append(labels_ts_segments)
        return np.concatenate(labels, axis=0), np.concatenate(labels_ts, axis=0)
