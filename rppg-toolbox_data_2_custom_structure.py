"""
-------------------------------------------------------------------------------
Created: 06.06.2024, 20:33
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
-------------------------------------------------------------------------------
Purpose: To insert the predictions made with the rppg-toolbox for the test data into the custom data structure.
-------------------------------------------------------------------------------
"""
import pickle
import os
from tqdm import tqdm
import numpy as np

exp_log_dir = '/media/super_fast_storage_2/matthieu_scherpf/_tmp/rppg_toolbox/exp_logs'
datasets_dir = '/media/super_fast_storage_2/matthieu_scherpf/_tmp/rppg_toolbox/datasets'

src_dir_suffix = '_SizeW72_SizeH72_ClipLength61_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceFalse_BackendNone_Large_boxFalse_Large_sizeNone_Dyamic_DetFalse_det_lenNone_Median_face_boxFalse'

dst_dirs = {
    'BP4DPlus_custom': '/media/super_fast_storage_2/matthieu_scherpf/2019_06_26_BP4D+_v0.2/processing/phys',
    'PURE_custom': '/media/super_fast_storage_2/matthieu_scherpf/2014_01_PURE/processing/phys',
    'UBFC-rPPG_custom': '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/processing/phys',
}

db_names = [
    'BP4DPlus_custom',
    'PURE_custom',
    'UBFC-rPPG_custom'
]

models = [
    'physnet_valid_loss_Epoch29',
    'deepphys_valid_loss_Epoch29',
    'tscan_valid_loss_Epoch29'
]


def create_dict_and_save_with_ts(k, v, has_multiple_recs):
    subj = k.split('#')[1]
    rec = ''
    if has_multiple_recs:
        subj, rec = subj.split('_')
    ts = []
    preds = []
    for i in sorted(list(v.keys())):
        ts_file = k + f'_label_ts{i}.npy'
        ts_path = os.path.join(datasets_dir, dbn+src_dir_suffix, ts_file)
        ts_i = np.load(ts_path)
        preds_i = v[i].cpu().numpy().squeeze()
        # truncate timestamps; for some models, there can be one value more than there are predictions
        ts_i = ts_i[:len(preds_i)]
        ts.append(ts_i)
        preds.append(preds_i)
    ts = np.hstack(ts)
    preds = np.hstack(preds)
    to_save = {'ts': ts, 'preds': preds}
    save_path = os.path.join(
        dst_dirs[dbn], m.split('_')[0], subj, rec)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'predictions.p'), 'wb') as f:
        pickle.dump(to_save, f)


for dbn in tqdm(db_names, ncols=80):
    for m in tqdm(models, leave=False, ncols=80):
        with open(os.path.join(exp_log_dir, dbn+src_dir_suffix, 'saved_test_outputs', m+'_'+dbn+'_outputs.pickle'), 'rb') as f:
            data = pickle.load(f)
        for k, v in tqdm(data['predictions'].items(), leave=False, ncols=80):
            has_multiple_recs = False
            if dbn in ['BP4DPlus_custom', 'PURE_custom']:
                has_multiple_recs = True
            create_dict_and_save_with_ts(k, v, has_multiple_recs)
