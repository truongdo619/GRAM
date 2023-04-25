#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pytorch_lightning import Trainer
from trainer.GRAM_trainer import GRAMLabeling

def evaluate(ckpt, hparams_file, gpus=[0, 1], max_length=300):
    trainer = Trainer(gpus=gpus, distributed_backend="dp")

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=1,
        max_length=max_length,
        workers=0
    )
    trainer.test(model=model)


if __name__ == '__main__':
    # example of running evaluate.py
    # GPUS="1,2,3"
    print(sys.argv)
    CHECKPOINTS = sys.argv[1]
    HPARAMS = sys.argv[2]
    try:
        GPUS = [int(gpu_item) for gpu_item in sys.argv[3].strip().split(",")]
    except:
        GPUS = [0]

    try:
        MAXLEN = int(sys.argv[4])
    except:
        MAXLEN = 512

    evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS, gpus=GPUS, max_length=MAXLEN)