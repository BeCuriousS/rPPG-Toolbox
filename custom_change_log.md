# This file contains the changes that we introduced to the rppg-toolbox to use it for our comparison study.
## We made the following adjustments to the BaseLoader class:
- Added support to only load specific subjects from a "subject_list"
- Added support to set the "multi_process_quota"
- Added support to process timestamp files/data (for later alignment of data from different approaches)
## We made the following adjustments to the DeepPhysTrainer, TscanTrainer and PhysnetTrainer class:
- Added support for "steps_per_epoch" (to run only part of the data in one epoch; for in-depth explanation see our publication)
- Added support for "valid_metric" which was used to enable switching either between the model loss used for optimization or the validation loss regarding heart rate estimation for the validation data (the mode loss was used in the end as it delivered the best results)
## We added the following files:
- helpers.py: Just incorporates code to extract the heart rate from a given BVP window (was used to compute the valid_metric for heart rate). This code was copied from our ippg-toolbox (https://github.com/BeCuriousS/ippg-toolbox)
- We created a CustomDataLoader class in CustomDataLoader.py with the purpose to enable processing our preprocessed private datasets used within our study.
- We created three custom classes CustomBP4DPlusLoader, CustomPURELoader and CustomUBFCrPPGLoader in the files CustomBP4DPlusLoader.py, CustomPURELoader.py and CustomUBFCrPPGLoader.py respectively. These classes are used to load the preprocessed public datasets used within our study.
- We added the "rppg-toolbox_data_2_custom_structure.py" which we used to convert the predictions for the DeepPhys, Tscan and Physnet models to a structure that is compatible with our evaluation scripts where we need the timestamps of the predictions.
## We created the custom configs in configs/custom_configs folder:
- These configs were created based on the existing configs in the configs/train_configs folder. The primary adjustment was made for the "CHUNK_LENGTH" parameter which was set to 61 to enable a comparability to our study.
- We created configs for the DeepPhys, Tscan and Physnet models.
## We created a custom_main.py file:
- To enable the easy integration of our adjustments.