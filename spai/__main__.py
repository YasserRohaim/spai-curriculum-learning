# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF 
#ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pathlib
import time
import datetime
from pathlib import Path
from typing import Optional

import csv
import numpy as np

import neptune
import cv2
import click
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import yacs
import filetype
from torch import nn
from torch.nn import TripletMarginLoss
from torch.utils.tensorboard import SummaryWriter
from timm.utils import AverageMeter
from yacs.config import CfgNode

import spai.data.data_finetune
from spai.config import get_config
from spai.models import build_cls_model
from spai.data import build_loader, build_loader_test
from spai.lr_scheduler import build_scheduler
from spai.models.sid import AttentionMask
from spai.onnx import compare_pytorch_onnx_models
from spai.optimizer import build_optimizer
from spai.logger import create_logger
from spai.utils import (
    load_pretrained,
    save_checkpoint,
    get_grad_norm,
    find_pretrained_checkpoints,
    inf_nan_to_num
)
from spai.models import losses
from spai import metrics
from spai import data_utils
from spai.data.data_finetune import CurriculumCSVDataset


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

cv2.setNumThreads(1)
logger: Optional[logging.Logger] = None


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int,
              help="Batch size for a single GPU.")
@click.option("--learning-rate", type=float)
@click.option("--data-path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="path to dataset")
@click.option("--csv-root-dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--pretrained",
              type=click.Path(exists=True, dir_okay=False),
              help="path to pre-trained model")
@click.option("--resume", is_flag=True,
              help="resume from checkpoint")
@click.option("--accumulation-steps", type=int, default=1,
              help="Gradient accumulation steps.")
@click.option("--use-checkpoint", is_flag=True,
              help="Whether to use gradient checkpointing to save memory.")
@click.option("--amp-opt-level", type=click.Choice(["O0", "O1", "O2"]), default="O1",
              help="mixed precision opt level, if O0, no amp is used")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--local_rank", type=int, default=0,
              help="local_rank for distributed training")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--data-workers", type=int,
              help="Number of worker processes to be used for data loading.")
@click.option("--disable-pin-memory", is_flag=True)
@click.option("--data-prefetch-factor", type=int)
@click.option("--save-all", is_flag=True)
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def train(
    cfg: Path,
    batch_size: Optional[int],
    learning_rate: Optional[float],
    data_path: Path,
    csv_root_dir: Optional[Path],
    lmdb_path: Optional[Path],
    pretrained: Optional[Path],
    resume: bool,
    accumulation_steps: int,
    use_checkpoint: bool,
    amp_opt_level: str,
    output: Path,
    tag: str,
    local_rank: int,
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    data_workers: Optional[int],
    disable_pin_memory: bool,
    data_prefetch_factor: Optional[int],
    save_all: bool,
    extra_options: tuple[str, str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = data_path.parent
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "data_path": str(data_path),
        "csv_root_dir": str(csv_root_dir),
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "pretrained": str(pretrained) if pretrained is not None else None,
        "resume": resume,
        "accumulation_steps": accumulation_steps,
        "use_checkpoint": use_checkpoint,
        "amp_opt_level": amp_opt_level,
        "output": str(output),
        "tag": tag,
        "local_rank": local_rank,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "data_workers": data_workers,
        "disable_pin_memory": disable_pin_memory,
        "data_prefetch_factor": data_prefetch_factor,
        "opts": extra_options
    })
    config.defrost()
    if not isinstance(config.TEST.VIEWS_REDUCTION_APPROACH, str):
        # turn function 'max'/'mean' into its name
        try:
            config.TEST.VIEWS_REDUCTION_APPROACH = config.TEST.VIEWS_REDUCTION_APPROACH.__name__
        except Exception:
            config.TEST.VIEWS_REDUCTION_APPROACH = "max"
    config.freeze()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)

    if config.AMP_OPT_LEVEL != "O0" and not config.AMP_OPT_LEVEL:
        assert amp is not None, "amp not installed!"

    # Set a fixed seed to all the random number generators.
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    if config.TRAIN.SCALE_LR:
        # Linear scale the learning rate according to total batch size - may not be optimal.
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
        # Gradient accumulation also need to scale the learning rate.
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export and display current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config, logger, is_pretrain=False, is_test=False
    )

    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=["mfm", "train", config.TRAIN.MODE, data_path.stem]
    )

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion: nn.Module = losses.build_loss(config)
    logger.info(f"Loss: \n{criterion}")

    if config.PRETRAINED:
        load_pretrained(config, model_without_ddp.get_vision_transformer(), logger)
    else:
        model_without_ddp.unfreeze_backbone()
        logger.info(f"No pretrained model. Backbone parameters are trainable.")

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger)

    train_model(
        config,
        model,
        model_without_ddp,
        data_loader_train,
        data_loader_val,
        test_loaders,
        dataset_val,
        test_datasets,
        test_datasets_names,
        criterion,
        optimizer,
        lr_scheduler,
        log_writer,
        neptune_run,
        save_all=save_all
    )


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int,
              help="Batch size for a single GPU.")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--split", type=str, default="test",
              help="The data split which will be tested. Actually, this value is expected to be "
                   "present in the `split` column of the provided csv files. Only samples "
                   "in the csv belonging to the provided split will be tested.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--update-csv", is_flag=True,
              help="When this flag is provided the predicted score for each sample is "
                   "written to the dataset csv, under a new column named as "
                   "{tag}_epoch_{epoch_num}_{crop_approach}.")
def test(
    cfg: Path,
    batch_size: Optional[int],
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    split: str,
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    resize_to: Optional[int],
    extra_options: tuple[str, str],
    update_csv: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })
    config.defrost()
    if not isinstance(config.TEST.VIEWS_REDUCTION_APPROACH, str):
        # turn function 'max'/'mean' into its name
        try:
            config.TEST.VIEWS_REDUCTION_APPROACH = config.TEST.VIEWS_REDUCTION_APPROACH.__name__
        except Exception:
            config.TEST.VIEWS_REDUCTION_APPROACH = "max"
    config.freeze()

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    neptune_tags: list[str] = ["mfm", "test"]
    neptune_tags.extend([p.stem for p in test_csv])
    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=neptune_tags
    )

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger,
                                                                         split=split)
    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    criterion = losses.build_loss(config)

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        model.cuda()
        if i == 0:
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of Params: {n_parameters}")
            if hasattr(model, "flops"):
                flops = model.flops()
                logger.info(f"Number of GFLOPs: {flops / 1e9}")

        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i==0)

        # Test the model.
        for test_data_loader, test_dataset, test_data_name in zip(test_loaders,
                                                                  test_datasets,
                                                                  test_datasets_names):
            predictions: Optional[dict[int, tuple[float, Optional[AttentionMask]]]] = None
            if update_csv:
                acc, ap, auc, loss, predictions = validate(
                    config, test_data_loader, model, criterion, neptune_run,
                    return_predictions=True
                )
            else:
                acc, ap, auc, loss = validate(config, test_data_loader,
                                              model, criterion, neptune_run)
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch} | "
                        f"Images: {len(test_dataset)} | loss: {loss:.4f}")
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
                        f"Images: {len(test_dataset)} | ACC: {acc:.3f}")
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
                        f"Images: {len(test_dataset)} | AP: {ap:.3f}")
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
                        f"Images: {len(test_dataset)} | AUC: {auc:.3f}")
            neptune_run[f"test/{test_data_name}/acc"].append(acc, step=checkpoint_epoch)
            neptune_run[f"test/{test_data_name}/ap"].append(ap, step=checkpoint_epoch)
            neptune_run[f"test/{test_data_name}/auc"].append(auc, step=checkpoint_epoch)
            neptune_run[f"test/{test_data_name}/loss"].append(loss, step=checkpoint_epoch)

            if predictions is not None:
                column_name: str = f"{tag}_epoch_{checkpoint_epoch}"
                scores: dict[int, float] = {i: t[0] for i, t in predictions.items()}
                attention_masks: dict[int, pathlib.Path] = {
                    i: t[1].mask for i, t in predictions.items() if t[1] is not None
                }
                test_dataset.update_dataset_csv(
                    column_name, scores, export_dir=Path(config.OUTPUT)
                )
                if len(attention_masks) == len(scores):
                    test_dataset.update_dataset_csv(
                        f"{column_name}_mask", attention_masks, export_dir=Path(config.OUTPUT)
                    )

        if log_writer is not None:
            log_writer.flush()
      


@cli.command()
@click.option("--cfg", default="./configs/spai.yaml", show_default=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a configuration file for SPAI.")
@click.option("--batch-size", type=int, default=1, show_default=True,
              help="Inference batch size.")
@click.option("--input", "input_paths", multiple=True, required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Can be either a directory containing the images to be analyzed or a "
                   "CSV file describing the data. This CSV file should include at least a "
                   "column named `image` with the path to the images.")
@click.option("--input-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the input csv files. "
                   "If this option is omitted, the parent directory of each input csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the input csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "input csv file. In that case, the number of provided input csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--split", type=str, default="test",
              help="The data split which will be tested. Actually, this value is expected to be "
                   "present in the `split` column of the provided csv files. Only samples "
                   "in the csv belonging to the provided split will be tested.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model", default="./weights/spai.pth",
              type=click.Path(exists=True, path_type=Path),
              help="Path to the a weight file of SPAI.")
@click.option("--output", default="./output",
              type=click.Path(file_okay=False, path_type=Path),
              help="Output directory where a CSV file containing .")
@click.option("--tag", type=str, help="Tag of experiment", default="spai")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def infer(
    cfg: Path,
    batch_size: int,
    input_paths: list[Path],
    input_csv_root_dir: list[Path],
    split: str,
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    resize_to: Optional[int],
    extra_options: tuple[str, str],
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in input_paths],
        "test_csv_root": [str(p) for p in input_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })
    config.defrost()
    if not isinstance(config.TEST.VIEWS_REDUCTION_APPROACH, str):
        # turn function 'max'/'mean' into its name
        try:
            config.TEST.VIEWS_REDUCTION_APPROACH = config.TEST.VIEWS_REDUCTION_APPROACH.__name__
        except Exception:
            config.TEST.VIEWS_REDUCTION_APPROACH = "max"
    config.freeze()

    output.mkdir(exist_ok=True, parents=True)

    # Create a console logger.
    global logger
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create dataloaders.
    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split, dummy_csv_dir=output
    )

    # Load the trained weights' checkpoint.
    model_ckpt: pathlib.Path = find_pretrained_checkpoints(config)[0]
    criterion = losses.build_loss(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    load_pretrained(config, model, logger,  checkpoint_path=model_ckpt, verbose=False)

    # Infer predictions and compute performance metrics (only on csv inputs with ground-truths).
    for test_data_loader, test_dataset, test_data_name, input_path in zip(test_loaders,
                                                                          test_datasets,
                                                                          test_datasets_names,
                                                                          input_paths):
        predictions: Optional[dict[int, tuple[float, Optional[AttentionMask]]]]
        acc, ap, auc, loss, predictions = validate(
            config, test_data_loader, model, criterion, None, return_predictions=True
        )

        if input_path.is_file():  # When input path is a dir, no ground-truth exists.
            logger.info(f"Test | {test_data_name} | Images: {len(test_dataset)} | loss: {loss:.4f}")
            logger.info(f"Test | {test_data_name} | Images: {len(test_dataset)} | ACC: {acc:.3f}")
            if test_dataset.get_classes_num() > 1:  # AUC and AP make no sense with only 1 class.
                logger.info(
                    f"Test | {test_data_name} | Images: {len(test_dataset)} | AP: {ap:.3f}")
                logger.info(
                    f"Test | {test_data_name} | Images: {len(test_dataset)} | AUC: {auc:.3f}")

        # Update the output CSV.
        if predictions is not None:
            column_name: str = f"{tag}"
            scores: dict[int, float] = {i: t[0] for i, t in predictions.items()}
            attention_masks: dict[int, pathlib.Path] = {
                i: t[1].mask for i, t in predictions.items() if t[1] is not None
            }
            test_dataset.update_dataset_csv(
                column_name, scores, export_dir=Path(output)
            )
            if len(attention_masks) == len(scores):
                test_dataset.update_dataset_csv(
                    f"{column_name}_mask", attention_masks, export_dir=Path(output)
                )


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def tsne(
    cfg: Path,
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    resize_to: Optional[int],
    extra_options: tuple[str, str],
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": 1,  # Currently, required to be 1 for correctly distinguishing embeddings.
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })
    config.defrost()
    if not isinstance(config.TEST.VIEWS_REDUCTION_APPROACH, str):
        # turn function 'max'/'mean' into its name
        try:
            config.TEST.VIEWS_REDUCTION_APPROACH = config.TEST.VIEWS_REDUCTION_APPROACH.__name__
        except Exception:
            config.TEST.VIEWS_REDUCTION_APPROACH = "max"
    config.freeze()
    from spai import tsne as tsne_utils

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    neptune_tags: list[str] = ["mfm", "tsne"]
    neptune_tags.extend([p.stem for p in test_csv])
    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=neptune_tags
    )

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger)
    model_ckpt: pathlib.Path = find_pretrained_checkpoints(config)[0]

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of Params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    checkpoint_epoch: int = load_pretrained(config, model, logger, checkpoint_path=model_ckpt)

    # Test the model.
    for test_data_loader, test_dataset, test_data_name in zip(test_loaders,
                                                              test_datasets,
                                                              test_datasets_names):
        tsne_utils.visualize_tsne(config, test_data_loader, test_data_name, model, neptune_run)

        if log_writer is not None:
            log_writer.flush()
        


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--exclude-preprocessing", is_flag=True,
              help="When this flag is provided the exported encoder does not include the spectral "
                   "filtering and normalization preprocessing operations. Instead, it accepts "
                   "three inputs, requiring these operations to be previously performed.")
def export_onnx(
    cfg: Path,
    model: Path,
    output: Path,
    tag: str,
    extra_options: tuple[str, str],
    exclude_preprocessing: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "opts": extra_options
    })

    output: Path = Path(config.OUTPUT)
    output.mkdir(exist_ok=True, parents=True)

    global logger
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    config_export_path: Path = output / "config.json"
    with config_export_path.open("w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config_export_path}")
    logger.info(config.dump())

    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    onnx_export_dir: Path = output / "onnx"
    onnx_export_dir.mkdir(exist_ok=True, parents=True)

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i == 0)

        model.to("cpu")
        model.eval()

        patch_encoder: Path = onnx_export_dir / "patch_encoder.onnx"
        patch_aggregator: Path = onnx_export_dir / "patch_aggregator.onnx"
        model.export_onnx(patch_encoder, patch_aggregator,
                          include_fft_preprocessing=not exclude_preprocessing)


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int, help="Batch size.")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--split", type=str, default="test",
              help="The data split which will be tested. Actually, this value is expected to be "
                   "present in the `split` column of the provided csv files. Only samples "
                   "in the csv belonging to the provided split will be tested.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str, help="tag of experiment")
@click.option("--device", type=str, default="cpu")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--exclude-preprocessing", is_flag=True)
def validate_onnx(
    cfg: Path,
    batch_size: Optional[int],
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    split: str,
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    device: str,
    extra_options: tuple[str, str],
    exclude_preprocessing: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "opts": extra_options
    })

    output: Path = Path(config.OUTPUT)
    output.mkdir(exist_ok=True, parents=True)

    global logger
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    config_export_path: Path = output / "config.json"
    with config_export_path.open("w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config_export_path}")
    logger.info(config.dump())

    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split
    )

    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    onnx_export_dir: Path = output / "onnx"
    if not onnx_export_dir.exists():
        raise FileNotFoundError(f"No onnx model at {onnx_export_dir}. Use the export-onnx "
                                f"command first.")

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i == 0)
        model.to(device)
        model.eval()

        patch_encoder: Path = onnx_export_dir / "patch_encoder.onnx"
        patch_aggregator: Path = onnx_export_dir / "patch_aggregator.onnx"

        compare_pytorch_onnx_models(
            model,
            patch_encoder,
            patch_aggregator,
            includes_preprocessing=not exclude_preprocessing,
            device=device
        )

## curriculum sanity check helper
def _assert_curriculum_counts(ds: CurriculumCSVDataset, logger, tol: float = 0.01) -> None:
    """Recompute composition from ds._epoch_indices and assert it matches logged mix."""
    idxs = ds._epoch_indices
    entries = ds.entries
    path_key = ds.path_column
    cls_key = ds.class_column

    c0 = c1m = c1s = 0
    for i in idxs:
        e = entries[i]
        label = int(e[cls_key])
        p = str(e[path_key])
        if label == 0:
            c0 += 1
        else:
            if "matched" in p:
                c1m += 1
            else:
                c1s += 1

    total = len(idxs)
    mix = ds.get_current_mix()
    ok = (c0 == mix["class0"] and c1m == mix["class1_matched"] and c1s == mix["class1_synth"])
    if not ok:
        logger.warning(
            "Curriculum(CHECK) mismatch | computed c0=%d c1m=%d c1s=%d vs mix %s",
            c0, c1m, c1s, mix
        )

    # Also check fraction vs schedule (after pool constraints)
    cls1 = c1m + c1s
    actual_frac = (c1m / cls1) if cls1 > 0 else 0.0
    expected_frac = float(mix["matched_fraction"])
    # When pools are limiting, fractional drift is expected; use a loose tolerance
    if cls1 > 0 and abs(actual_frac - expected_frac) > tol:
        logger.info(
            "Curriculum(FRAC) actual=%.4f expected=%.4f (tol=%.3f) [pool limited? c1=%d]",
            actual_frac, expected_frac, tol, cls1
        )

# ========================
# Training / Validation
# ========================

def train_model(
    config: yacs.config.CfgNode,
    model: nn.Module,
    model_without_ddp: nn.Module,
    data_loader_train: torch.utils.data.DataLoader,
    data_loader_val: torch.utils.data.DataLoader,
    data_loaders_test: list[torch.utils.data.DataLoader],
    dataset_val: spai.data.data_finetune.CSVDataset,
    datasets_test: list[spai.data.data_finetune.CSVDataset],
    datasets_test_names: list[str],
    criterion,
    optimizer,
    lr_scheduler,
    log_writer,
    neptune_run,
    save_all: bool = False
) -> None:
    logger.info("Start training")

    # ----- Early stopping (on by default) -----
    ES_PATIENCE: int = 6          # stop after N consecutive non-improving epochs
    ES_MIN_DELTA: float = 0.0     # required improvement in monitored metric
    best_val_loss: float = float("inf")
    epochs_no_improve: int = 0
    # -----------------------------------------

    start_time: float = time.time()
    val_accuracy_per_epoch: list[float] = []
    val_ap_per_epoch: list[float] = []
    val_auc_per_epoch: list[float] = []
    val_loss_per_epoch: list[float] = []

    # Prepare consolidated CSV (append across epochs)
    consolidated_csv = Path(config.OUTPUT) / "val_predictions_all.csv"
    write_consolidated_header = not consolidated_csv.exists()
    
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        epoch_start_time: float = time.time()
        
       # In train_model(...), right before calling train_one_epoch(...)
        ds = data_loader_train.dataset
        if hasattr(ds, "set_epoch") and hasattr(ds, "get_current_mix"):
            ds.set_epoch(epoch)
            mix = ds.get_current_mix()
            logger.info("Curriculum(MIX) | epoch=%d matched_frac=%.3f | cls0=%d | cls1_matched=%d | cls1_synth=%d",
                        epoch, mix["matched_fraction"], mix["class0"], mix["class1_matched"], mix["class1_synth"])
            _assert_curriculum_counts(ds, logger, tol=0.01)
        else:
            logger.info("Curriculum disabled (dataset: %s)", type(ds).__name__)

        # Strong check: recompute counts from indices and compare
        
        train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            lr_scheduler,
            log_writer,
            neptune_run
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        neptune_run["train/last_epoch"] = epoch + 1
        neptune_run["train/epochs_trained"] = epoch + 1 - config.TRAIN.START_EPOCH

        # ===== Validate the model and request full per-sample details =====
        acc: float
        ap: float
        auc: float
        loss: float
        acc, ap, auc, loss, predictions, details = validate(
            config, data_loader_val, model, criterion, neptune_run,
            return_predictions=True, return_details=True
        )
        logger.info(f"Val | Epoch {epoch} | Images: {len(dataset_val)} | loss: {loss:.4f}")
        logger.info(f"Val | Epoch {epoch} | Images: {len(dataset_val)} | ACC: {acc:.3f}")
        logger.info(f"Val | Epoch {epoch} | Images: {len(dataset_val)} | AP: {ap:.3f}")
        logger.info(f"Val | Epoch {epoch} | Images: {len(dataset_val)} | AUC: {auc:.3f}")
        neptune_run["val/auc"].append(auc)
        neptune_run["val/ap"].append(ap)
        neptune_run["val/accuracy"].append(acc)
        neptune_run["val/loss"].append(loss)

        # Track bests for logging
        val_accuracy_per_epoch.append(acc)
        val_ap_per_epoch.append(ap)
        val_auc_per_epoch.append(auc)
        val_loss_per_epoch.append(loss)
        logger.info(f"Val | Min loss: {min(val_loss_per_epoch):.4f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmin(val_loss_per_epoch)}")
        logger.info(f"Val | Max ACC: {max(val_accuracy_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH+np.argmax(val_accuracy_per_epoch)}")
        logger.info(f"Val | Max AP: {max(val_ap_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmax(val_ap_per_epoch)}")
        logger.info(f"Val | Max AUC: {max(val_auc_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmax(val_auc_per_epoch)}")

        # ===== Save full validation results (per-sample) =====
        # 2A) Per-epoch standalone CSV
        per_epoch_csv = Path(config.OUTPUT) / f"val_epoch_{epoch}_preds.csv"
        with per_epoch_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch", "dataset_index", "image", "target", "prob", "logit", "loss", "pred", "mask_path"])
            for di in sorted(details.keys()):
                d = details[di]
                pred_label = 1 if d["prob"] >= 0.5 else 0
                w.writerow([
                    epoch, di, d.get("image", ""), d["target"],
                    f"{d['prob']:.6f}", f"{d['logit']:.6f}", f"{d['loss']:.6f}",
                    pred_label, d.get("mask_path", "")
                ])

        # 2B) Append to consolidated CSV across epochs
        with consolidated_csv.open("a", newline="") as fh:
            w = csv.writer(fh)
            if write_consolidated_header:
                w.writerow([ "image", "class", "spai"])
                write_consolidated_header = False
            for di in sorted(details.keys()):
                d = details[di]
                pred_label = 1 if d["prob"] >= 0.5 else 0
                w.writerow([
                        d.get("image", ""), d["target"],
                    f"{d['prob']:.6f}"

                ])

        # 2C) Also write columns back to the dataset CSV (mirrors your test() flow)
        col_base = f"{config.TAG}_val_epoch_{epoch}"
        dataset_val.update_dataset_csv(
            f"{col_base}_prob",
            {i: d["prob"] for i, d in details.items()},
            export_dir=Path(config.OUTPUT)
        )
        dataset_val.update_dataset_csv(
            f"{col_base}_logit",
            {i: d["logit"] for i, d in details.items()},
            export_dir=Path(config.OUTPUT)
        )
        dataset_val.update_dataset_csv(
            f"{col_base}_loss",
            {i: d["loss"] for i, d in details.items()},
            export_dir=Path(config.OUTPUT)
        )
        dataset_val.update_dataset_csv(
            f"{col_base}_pred",
            {i: int(d["prob"] >= 0.5) for i, d in details.items()},
            export_dir=Path(config.OUTPUT)
        )
        mask_map = {i: d["mask_path"] for i, d in details.items() if d.get("mask_path")}
        if len(mask_map) == len(details):
            dataset_val.update_dataset_csv(
                f"{col_base}_mask", mask_map, export_dir=Path(config.OUTPUT)
            )

        # ===== Always save a checkpoint each epoch (for resumes) =====
        # save_checkpoint handles naming (usually includes the epoch number and best tracking).
        save_checkpoint(config, epoch, model_without_ddp, max(val_accuracy_per_epoch),
                        optimizer, lr_scheduler, logger)

        # ===== Early stopping (monitoring val loss) =====
        improved: bool = (loss < (best_val_loss - ES_MIN_DELTA))
        if improved:
            best_val_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= ES_PATIENCE:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no val loss improvement for {ES_PATIENCE} consecutive epoch(s)). "
                f"Best val loss: {best_val_loss:.4f}"
            )
            if neptune_run is not None:
                neptune_run["train/early_stop_triggered"] = True
                neptune_run["train/early_stop_epoch"] = epoch
                neptune_run["train/best_val_loss"] = float(best_val_loss)
            # Save one more checkpoint at the stopping point just in case
            save_checkpoint(config, epoch, model_without_ddp, max(val_accuracy_per_epoch),
                            optimizer, lr_scheduler, logger)
            break

        # ===== Optional: run tests each epoch =====
        for test_data_loader, test_dataset, test_data_name in zip(data_loaders_test,
                                                                    datasets_test,
                                                                    datasets_test_names):
            acc_t, ap_t, auc_t, loss_t = validate(config, test_data_loader, model,
                                                    criterion, neptune_run)
            logger.info(f"Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| loss: {loss_t:.4f}")
            logger.info(f"Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| ACC: {acc_t:.3f}")
            logger.info(f"Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| AP: {ap_t:.3f}")
            logger.info(f"Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| AUC: {auc_t:.3f}")
            neptune_run[f"test/{test_data_name}/acc"].append(acc_t)
            neptune_run[f"test/{test_data_name}/ap"].append(ap_t)
            neptune_run[f"test/{test_data_name}/auc"].append(auc_t)
            neptune_run[f"test/{test_data_name}/loss"].append(loss_t if not np.isnan(loss_t) else -100.)

        # Timing
        epoch_time: float = time.time() - epoch_start_time
        logger.info(f"Epoch training time: {epoch_time:.3f}s")
        neptune_run["train/epoch_train_time"].append(epoch_time)


    # Total training time
    total_time: float = time.time() - start_time
    total_time_str: str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Overall training time: {total_time_str}")
    neptune_run["train/total_train_time"].append(total_time_str)


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    lr_scheduler,
    log_writer,
    neptune_run
):
   

    model.train()
    criterion.train()
    optimizer.zero_grad()

    logger.info(
        "Current learning rate for different parameter groups: "
        f"{[it['lr'] for it in optimizer.param_groups]}"
    )

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if isinstance(criterion, TripletMarginLoss):
            anchor, positive, negative = batch
            batch_size: int = anchor.size(0)
            anchor = anchor.cuda(non_blocking=True)
            positive = positive.cuda(non_blocking=True)
            negative = negative.cuda(non_blocking=True)
            anchor_outputs = model(anchor)
            positive_outputs = model(positive)
            negative_outputs = model(negative)
            logits = None  # not used in triplet mode
            targets = None
        else:
            samples, targets, _ = batch
            batch_size: int = samples.size(0)
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).float()  # BCE targets must be float

            # Forward each augmented view separately to save memory.
            outputs_views: list[torch.Tensor] = [
                model(samples[:, i, :, :, :]) for i in range(samples.size(1))
            ]  # each: [B, 1]
            outputs: torch.Tensor = torch.stack(outputs_views, dim=1)  # [B, V, 1]
            logits: torch.Tensor = outputs.squeeze(-1)  # [B, V]
            if logits.dim() > 1:
                # Average predictions across augmented views to match [B] targets.
                logits = logits.mean(dim=1)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if isinstance(criterion, TripletMarginLoss):
                loss = criterion(anchor_outputs, positive_outputs, negative_outputs)
            else:
                loss = criterion(logits, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())

            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            if isinstance(criterion, TripletMarginLoss):
                loss = criterion(anchor_outputs, positive_outputs, negative_outputs)
            else:
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[-1]["lr"]
        loss_value_reduce = float(loss.detach().cpu().numpy())
        grad_norm_cpu = (float(grad_norm.detach().cpu().numpy())
                         if isinstance(grad_norm, torch.Tensor) else float(grad_norm))

        if log_writer is not None and (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            # Calibrate x-axis across batch sizes.
            epoch_1000x = int((idx / num_steps + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('grad_norm', grad_norm_cpu, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            neptune_run["train/loss"].append(
                inf_nan_to_num(loss_value_reduce, nan_value=-100., inf_value=-50.)
            )
            neptune_run["train/grad_norm"].append(
                inf_nan_to_num(grad_norm_cpu,  nan_value=-100., inf_value=-50.)
            )
            neptune_run["train/lr"].append(lr)

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
@torch.no_grad()
def validate(
    config,
    data_loader,
    model,
    criterion,
    neptune_run,
    verbose: bool = True,
    return_predictions: bool = False,
    return_details: bool = False,
):
    """Validate and (optionally) return per-sample predictions/details.

    - Computes overall AUC/AP/ACC (as before).
    - Additionally aggregates metrics per subset inferred from path:
        * negatives (label=0) -> "real"
        * positives (label=1) with "matched" in path -> "matched"
        * other positives -> "synthetic"
    - Logs per-subset + worst-group metrics to logger and Neptune.
    """
    from collections import defaultdict

    model.eval()
    criterion.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_metrics: metrics.Metrics = metrics.Metrics(metrics=("auc", "ap", "accuracy"))

    # Optional outputs
    predicted_scores: dict[int, tuple[float, Optional[AttentionMask]]] = {}
    details_by_idx: dict[int, dict] = {}

    # For per-subset aggregation (we do it at the end to avoid changing the return signature)
    subset_probs = defaultdict(list)   # name -> list[float]
    subset_tgts  = defaultdict(list)   # name -> list[int]

    ds = getattr(data_loader, "dataset", None)

    def _get_img_path(ds_obj, idx: int) -> str:
        try:
            if hasattr(ds_obj, "get_image_path"):
                return str(ds_obj.get_image_path(idx))
            if hasattr(ds_obj, "get_path"):
                return str(ds_obj.get_path(idx))
            for attr in ("df", "data_frame", "_df"):
                if hasattr(ds_obj, attr):
                    df = getattr(ds_obj, attr)
                    cols = getattr(df, "columns", [])
                    if "image" in cols:
                        try:
                            return str(df.loc[idx, "image"])
                        except Exception:
                            return str(df.iloc[idx]["image"])
        except Exception:
            pass
        return ""

    def _subset_name(img_path: str, tgt_int: int) -> str:
        # label 0 is "real" (negatives)
        if tgt_int == 0:
            return "real"
        # label 1 (positives): split by path hint
        p = (img_path or "").lower()
        if "matched" in p:
            return "matched"
        return "synthetic"

    end = time.time()
    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        if isinstance(images, list):
            # Arbitrary-resolution path returns list of tensors
            images = [img.cuda(non_blocking=True) for img in images]
            images = [img.squeeze(dim=1) for img in images]  # remove views dim (always 1 at test)
        else:
            images = images.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True).float()

        # ---- Forward (logits) ----
        if isinstance(images, list) and config.TEST.EXPORT_IMAGE_PATCHES:
            export_dirs: list[pathlib.Path] = [
                pathlib.Path(config.OUTPUT) / "images" / f"{dataset_idx.detach().cpu().tolist()[i]}"
                for i in range(len(dataset_idx))
            ]
            output, attention_masks = model(
                images, config.MODEL.FEATURE_EXTRACTION_BATCH, export_dirs
            )
        elif isinstance(images, list):
            output = model(images, config.MODEL.FEATURE_EXTRACTION_BATCH)
            attention_masks = [None] * len(images)
        else:
            if images.size(dim=1) > 1:
                preds_per_view: list[torch.Tensor] = [
                    model(images[:, i]) for i in range(images.size(dim=1))
                ]  # each [B,1]
                predictions: torch.Tensor = torch.stack(preds_per_view, dim=1)  # [B,V,1]
                if config.TEST.VIEWS_REDUCTION_APPROACH == "max":
                    output: torch.Tensor = predictions.max(dim=1).values  # [B,1]
                elif config.TEST.VIEWS_REDUCTION_APPROACH == "mean":
                    output: torch.Tensor = predictions.mean(dim=1)         # [B,1]
                else:
                    raise TypeError(f"{config.TEST.VIEWS_REDUCTION_APPROACH} is not a supported views reduction approach")
            else:
                images = images.squeeze(dim=1)  # Remove views dim.
                output = model(images)          # [B,1]
            attention_masks = [None] * output.size(0)

        logits = output.squeeze(dim=1)  # [B]

        # ---- Loss (per-sample + mean) ----
        per_sample_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = per_sample_loss.mean()

        # ---- Probs ----
        probs = torch.sigmoid(logits)

        # ---- Accumulate overall metrics ----
        loss_meter.update(loss.item(), target.size(0))
        cls_metrics.update(probs.detach().cpu(), target.cpu())

        # ---- Collect per-subset data ----
        batch_idx_list: list[int] = dataset_idx.detach().cpu().tolist()
        probs_list = probs.detach().cpu().tolist()
        logits_list = logits.detach().cpu().tolist()
        loss_list = per_sample_loss.detach().cpu().tolist()
        tgt_list = [int(t) for t in target.detach().cpu().tolist()]
        img_paths = []
        if ds is not None:
            for di in batch_idx_list:
                img_paths.append(_get_img_path(ds, di))
        else:
            img_paths = [""] * len(batch_idx_list)

        for pr, tg, path in zip(probs_list, tgt_list, img_paths):
            name = _subset_name(path, tg)
            subset_probs[name].append(float(pr))
            subset_tgts[name].append(int(tg))

        # ---- Optional: store predictions/details ----
        if return_predictions or return_details:
            if isinstance(images, list):
                attn_list = attention_masks
            else:
                attn_list = [None] * len(batch_idx_list)

            for i_row, di in enumerate(batch_idx_list):
                if return_predictions:
                    predicted_scores[di] = (float(probs_list[i_row]), attn_list[i_row])
                if return_details:
                    details_by_idx[di] = {
                        "prob": float(probs_list[i_row]),
                        "logit": float(logits_list[i_row]),
                        "loss": float(loss_list[i_row]),
                        "target": int(tgt_list[i_row]),
                        "image": img_paths[i_row],
                        "mask_path": str(attn_list[i_row].mask) if attn_list[i_row] is not None else ""
                    }

        # ---- Timing/logging ----
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 and verbose:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}] | '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
                f'Mem {memory_used:.0f}MB'
            )

    # ---- Overall metrics ----
    metric_values: dict[str, np.ndarray] = cls_metrics.compute()
    auc: float = metric_values["auc"].item()
    ap: float = metric_values["ap"].item()
    acc: float = metric_values["accuracy"].item()

    # ---- Per-subset + worst-group metrics (log only; do not change returns) ----
    results = {
        "overall": {"auc": auc, "ap": ap, "acc": acc}
    }

    # Compute per-subset metrics if available
    for name in ("real", "synthetic", "matched"):
        if len(subset_probs.get(name, [])) > 0:
            subM = metrics.Metrics(metrics=("auc", "ap", "accuracy"))
            p_t = torch.tensor(subset_probs[name])
            y_t = torch.tensor(subset_tgts[name])
            subM.update(p_t, y_t)
            mv = subM.compute()
            results[name] = {
                "auc": float(mv["auc"]),
                "ap": float(mv["ap"]),
                "acc": float(mv["accuracy"]),
            }

    # Worst group across the three (when all exist)
    if all(k in results for k in ("real", "synthetic", "matched")):
        worst_auc = min(results["real"]["auc"], results["synthetic"]["auc"], results["matched"]["auc"])
        worst_ap  = min(results["real"]["ap"],  results["synthetic"]["ap"],  results["matched"]["ap"])
        worst_acc = min(results["real"]["acc"], results["synthetic"]["acc"], results["matched"]["acc"])
        results["worst_group"] = {"auc": worst_auc, "ap": worst_ap, "acc": worst_acc}

    # Log nicely
    for k, v in results.items():
        logger.info(f"VAL[{k}] AUC={v['auc']:.4f} AP={v['ap']:.4f} ACC={v['acc']:.4f}")
        if neptune_run is not None:
            # keep short keys: auc/ap/acc
            for m in ("auc", "ap", "acc"):
                neptune_run[f"val/{k}/{m}"].append(v[m])

    # ---- Return same signature as before ----
    if return_predictions and return_details:
        return acc, ap, auc, loss_meter.avg, predicted_scores, details_by_idx
    if return_predictions:
        return acc, ap, auc, loss_meter.avg, predicted_scores
    return acc, ap, auc, loss_meter.avg


if __name__ == '__main__':
    cli()
