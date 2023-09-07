# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

# Edited by Nicholas M. Synovic

# Original code from
# https://github.com/oneapi-src/disease-prediction/blob/main/src/utils/train.py

"""
Training code for the model
"""

import logging
import pathlib
from argparse import Namespace
from contextlib import nullcontext

import torch
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger()


def train(
    tokenizer: AutoTokenizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    enable_bf16: bool,
    flags: Namespace,
    epochs: int = 5,
    max_grad_norm: float = 10,
) -> None:
    """train a model on the given dataset

    Args:
        dataloader (torch.utils.data.DataLoader): training dataset
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer to use
        enable_bf16 (bool): Enable bf16 mixed precision training
        epochs (int, optional): number of training epochs. Defaults to 5.
        max_grad_norm (float, optional): gradient clipping. Defaults to 10.
    """

    writer: SummaryWriter = SummaryWriter(log_dir="tensorboard")
    acc: float = 0


    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0
        train_preds = []
        train_labels = []
        for _, (batch, labels) in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch}",
        ):
            optimizer.zero_grad()
            # use mixed precision bf16 training only if enabled
            with torch.cpu.amp.autocast() if enable_bf16 else nullcontext():
                ids = batch["input_ids"]
                mask = batch["attention_mask"]
                token_type_ids = batch["token_type_ids"]

                out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

                loss = out.loss
                writer.add_scalar(
                    tag="Loss/train",
                    scalar_value=loss,
                    global_step=epoch,
                )

                train_preds.extend(out.logits.argmax(-1))
                train_labels.extend(labels)

                # clip gradients for stability
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=max_grad_norm
                )

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        trainAcc = accuracy_score(train_preds, train_labels)
        logger.info("Epoch Train Accuracy %.4f", acc)

        test_preds = []
        test_labels = []
        for _, (batch, labels) in enumerate(val_dataloader):
            # use mixed precision bf16 training only if enabled
            with torch.cpu.amp.autocast() if enable_bf16 else nullcontext():
                ids = batch["input_ids"]
                mask = batch["attention_mask"]
                token_type_ids = batch["token_type_ids"]

                pred = model(
                    input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids
                )
                test_preds.extend(pred.logits.argmax(-1))
                test_labels.extend(labels)

        testAcc = accuracy_score(test_preds, test_labels)

        logger.info("Test Accuracy %.4f", testAcc)

        logger.info("Epoch %d, Loss %.4f", epoch, running_loss)


        writer.add_scalar(
            tag="Accuracy/train",
            scalar_value=trainAcc,
            global_step=epoch,
        )

        writer.add_scalar(
            tag="Accuracy/test",
            scalar_value=testAcc,
            global_step=epoch,
        )

        if (testAcc > acc) and (testAcc > 0.8) and (testAcc < 0.95):
            if flags.save_model_dir:
                path = pathlib.Path(flags.save_model_dir)
                path.mkdir(parents=True, exist_ok=True)

                tokenizer.save_pretrained(path)
                model.save_pretrained(path)
                logger.info("Saved model files to %s", path)
                logger.info(f"Saved model at epoch {epoch}")

    writer.close()
