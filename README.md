# Predict protein holo distance matrix from apo distance matrix

## Overview

I forked this project from [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

The files I modified may have something like My or my_ at the begining of files' name.

So please read [readme.md](readme.md) first, because it is the origin README about the project that I forked.

## Requirement

I also installed **mdtraj** and **torchinfo**.

Mdtraj is for reading md result.

Torchinfo is for model structure.

## Datasets

I can't upload the data because of the size. So you need to make the data yourself.

I used pickle to load input data, but you can modified yourself. See [CryptoSiteDataMD_creator.py](data/MyFunction/CryptoSiteDataMD_creator.py).

## Usage

1. To view training results and loss plots, run `python -m visdom.server &` .
2. Make sure you have a dir maybe named xxx(anything you like) to save the result.
3. Now run `python my_train.py --dataroot ./frames --model my_pix2pix --name xxx --num threads 1 --lambda_L1 50 --n_epochs (epochs you want to run) --n_epoches_decay (epochs you want to decay the lr) --lr (learning rate) --gan_mode wgangp` and enjoy.

## Result

### result_train(pickle) & result_val(pickle)

- result
- Have a list for [train epochs, len(data), 3].
- 3 means [apo to holo, apo to predict, holo to predict]

### train\_(epoch) (pickle) & val\_(epoch) (pickle)

- origin distance matrix and predict distance matrix.
- Have a list for [len(data), 3].
- 3 means [apo, predict, holo]