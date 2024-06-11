""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time

import pickle
import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)
# custom setup
num_workers = 16
steps_per_epoch = 128


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=True,
                        type=str, help="Config file name.")
    parser.add_argument('--batch_sample_len', required=False,
                        default=None, type=str, help="batch_sample_len")
    parser.add_argument('--valid_metric', required=True,
                        type=str, help="Can be set to hr_mae is to use best model from minimal hr_mae of validation data.")
    parser.add_argument('--channel_types', required=True,
                        type=str, help="One of: [DiffNormalized, Standardized, None. Used to enable dataset reusage, e.g. for networks that only rely on DiffNormalized input instead of standardized input.")
    return parser


def train_and_test(config, data_loader_dict):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(
            config, data_loader_dict, args.valid_metric, steps_per_epoch)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(
            config, data_loader_dict, args.valid_metric, steps_per_epoch)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(
            config, data_loader_dict, args.valid_metric, steps_per_epoch)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(
            config, data_loader_dict, args.valid_metric, steps_per_epoch)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(
            config, data_loader_dict, args.valid_metric, steps_per_epoch)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(
            config, data_loader_dict, args.valid_metric, steps_per_epoch)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(
            config, data_loader_dict, args.valid_metric, steps_per_epoch)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(
            config, data_loader_dict, args.valid_metric, steps_per_epoch)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        elif unsupervised_method == "OMIT":
            unsupervised_predict(config, data_loader, "OMIT")
        else:
            raise ValueError("Not supported unsupervised method!")


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # convert custom args
    if args.batch_sample_len in ['None', 'none']:
        args.batch_sample_len = None
    else:
        args.batch_sample_len = int(args.batch_sample_len)

    # use the exact same subjects for validation as for the development of DeepPerfusion (therefore this is hard coded)
    validation_subjects_bp4d = [
        'F001',
        'F003',
        'F015',
        'F020',
        'F027',
        'F028',
        'F033',
        'F043',
        'F048',
        'F053',
        'F064',
        'F065',
        'F069',
        'F074',
        'F077',
        'F078',
        'M006',
        'M013',
        'M017',
        'M019',
        'M024',
        'M025',
        'M027',
        'M032',
        'M035',
        'M039',
        'M040',
        'M042'
    ]

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    data_loader_dict = dict()  # dictionary of data loaders
    if config.TOOLBOX_MODE == "train_and_test":
        # train_loader
        if config.TRAIN.DATA.DATASET == "Private_dataset":
            train_loader = data_loader.CustomDataLoader.CustomDataLoader
        elif config.TRAIN.DATA.DATASET == "UBFC-rPPG_custom":
            train_loader = data_loader.CustomUBFCrPPGLoader.CustomUBFCrPPGLoader
        elif config.TRAIN.DATA.DATASET == "PURE_custom":
            train_loader = data_loader.CustomPURELoader.CustomPURELoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlus_custom":
            train_loader = data_loader.CustomBP4DPlusLoader.CustomBP4DPlusLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")

        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset paths
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):
            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                subject_list=[],
                num_workers=num_workers,
                batch_sample_len=args.batch_sample_len,
                channel_types=args.channel_types)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=num_workers,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
        else:
            data_loader_dict['train'] = None

        # valid_loader
        if config.VALID.DATA.DATASET == "Private_dataset":
            valid_loader = data_loader.CustomDataLoader.CustomDataLoader
        elif config.VALID.DATA.DATASET == "UBFC-rPPG_custom":
            valid_loader = data_loader.CustomUBFCrPPGLoader.CustomUBFCrPPGLoader
        elif config.VALID.DATA.DATASET == "PURE_custom":
            valid_loader = data_loader.CustomPURELoader.CustomPURELoader
        elif config.VALID.DATA.DATASET == "BP4DPlus_custom":
            valid_loader = data_loader.CustomBP4DPlusLoader.CustomBP4DPlusLoader
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError(
                "Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP")

        # Create and initialize the valid dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA,
                subject_list=validation_subjects_bp4d,
                num_workers=num_workers,
                batch_sample_len=None,
                channel_types=args.channel_types)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=num_workers,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # test_loader
        if config.TEST.DATA.DATASET == "Private_dataset":
            test_loader = data_loader.CustomDataLoader.CustomDataLoader
        elif config.TEST.DATA.DATASET == "UBFC-rPPG_custom":
            test_loader = data_loader.CustomUBFCrPPGLoader.CustomUBFCrPPGLoader
        elif config.TEST.DATA.DATASET == "PURE_custom":
            test_loader = data_loader.CustomPURELoader.CustomPURELoader
        elif config.TEST.DATA.DATASET == "BP4DPlus_custom":
            test_loader = data_loader.CustomBP4DPlusLoader.CustomBP4DPlusLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")

        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print(
                "Testing uses last epoch, validation dataset is not required.", end='\n\n')

        # Create and initialize the test dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA,
                subject_list=[],
                num_workers=num_workers,
                batch_sample_len=None,
                channel_types=args.channel_types)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=num_workers,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        if config.UNSUPERVISED.DATA.DATASET == "UBFC-rPPG":
            unsupervised_loader = data_loader.CustomUBFCrPPGLoader.CustomUBFCrPPGLoader
        elif config.UNSUPERVISED.DATA.DATASET == "PURE":
            unsupervised_loader = data_loader.CustomPURELoader.CustomPURELoader
        elif config.UNSUPERVISED.DATA.DATASET == "SCAMPS":
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "MMPD":
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.UNSUPERVISED.DATA.DATASET == "BP4DPlus":
            unsupervised_loader = data_loader.CustomBP4DPlusLoader.CustomBP4DPlusLoader
        elif config.UNSUPERVISED.DATA.DATASET == "UBFC-PHYS":
            unsupervised_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "iBVP":
            unsupervised_loader = data_loader.iBVPLoader.iBVPLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+, UBFC-PHYS and iBVP.")

        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA,
            subject_list=[],
            num_workers=num_workers,
            batch_sample_len=None,
            channel_types=args.channel_types)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=num_workers,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    else:
        raise ValueError(
            "Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')
