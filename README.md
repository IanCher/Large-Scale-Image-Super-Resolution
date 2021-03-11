# Super Resolution of Large TIFF Images (code from 2019)
Custom implementation of  [Fast and Accurate Single Image Super-Resolution via Information Distillation Network](https://arxiv.org/pdf/1803.09454.pdf) by Hui et al.
Adapted for large TIFF images (for instance satellites).

Project realized while working on [Learned Multi-View Texture Super-Resolution](https://arxiv.org/pdf/2001.04775.pdf)

## Code
The code was implemented using python 3.6.7 on Ubuntu 16.04, using the following packages:
* numpy 1.16.1
* opencv 4.0.0
* tensorflow-gpu 1.12.0
* matplotlib 3.0.2
* pickle

## Inference
The inference code `eval_superres_net.py` superresolves large images by tiling them, and superresolving each tiles separately before recombining them.

The network is initialized by passing the path to the pretrained weights. 
The code can be run as follows:

```shell
python eval_superres_net.py --data_path /path/to/data/ --pretrained_path /path/to/pretrained/ --mean_values_path /path/to/mean_values/  --upsampling_factor $upsampling_factor --num_tiles_w $num_tiles_w --num_tiles_h $num_tiles_h
```

The code expects the following organization for the pretrained_path and mean_values_path:

```shell
pretrained_path
   |-- x2
      |-- ckpts
   |-- x4
      |-- ckpts

mean_values_path
   |-- x2
      |-- mean_values.pickle
   |-- x4
      |-- mean_values.pickle
```
where the ckpts folders contain Tensorflow's saved checkpoints that we want to use to restore the network.

The main input parameters are the following
* --data_path: path to a folder containing the images to superresolve
* --pretrained_path: path to the weights to be used to restore the network
* --mean_values_path: path to the mean values used to center the data for the network
* --upsampling_factor: the upsampling factore for super resolution
* --num_tiles_w: number of tiles to subdivied the width
* --num_tiles_h: number of tiles to subdivied the height
* --img_ext: the extension of the images to superresolve in data_path (default .tif)

The results are saved in a folder with the name /path/to/data_x${upsampling_factor}/. 
For instance, if the data to superresolved is stored in the folder 00325_PANSHARP, then the superresolved image for an upsampling factor of 2 will be saved in the folder 00325_PANSHARP_x2.

## Training
### DataSet generation
To perform training, we first need to generate a dataset, using `create_training_dataset.py`. 
The code takes a set of large images, tiles them, and creates a dataset from these tiles.
Each tile is considered as a single image, independant from the others.
Each tile is downsampled by the factor for which we want to learn the super resolution (2 or 4 usually).
The original tiles before downsampling are considered as the groundtruth superresolved images. 
 
```shell
python create_training_dataset.py --data_path /path/to/data/ --out_path /path/to/output --tile_size_w $tile_size_w --tile_size_h $tile_size_h
```

Where the main parameters are:
* --data_path: path to a folder containing the images to superresolve
* --out_path: path where to store the datasets
* --upsampling_factor: the upsampling factor to train the super resolution network
* --tile_size_w: width of each tile
* --tile_size_h: height of each tile

For each tile we generate, we store a tensorflow feature in the TFRecord which contains 3 (key, value) pairs:
* 'gt': the tile itself that serves as groundtruth
* 'down': the downsampled version of the tile
* 'bic_up': the bicubic upsampling of the downsampled tile (used in the network, precomputed here)

The TFRecords generated are stored in the output path, with the following organization.

```shell
out_path
   |-- x${upsampling_factor}
      |-- Training
      |-- Validation
```
Where we keep one TFRecord to serve as validation set.

### Training
The training is performed using `train_superres_net.py`.

```shell
python train_superres_net.py --data_path /path/to/data/ --pretrained_path /path/to/pretrained/ --out_path /path/to/output/ --upsampling_factor ${upsampling_factor} --tile_size_w ${tile_size_w} --tile_size_h ${tile_size_h} --restore
```

The data and pretrained folder should have the following organization:

```shell
data_path
   |-- x2
      |-- Training
      |-- Validation
   |-- x4
      |-- Training
      |-- Validation

pretrained_path
   |-- x2
      |-- ckpts
   |-- x4
      |-- ckpts
```

Typically, data_path should be the same as the out_path for `create_training_dataset.py`. 
The pretrained_path can be a path to pretrained weights, or to previous checkpoints to restore.

The input parameters are the following:
* --data_path: path to a folder containing the images to superresolve
* --pretrained_path: path to weights to restore
* --out_path: path where to store the results
* --upsampling_factor: the upsampling factor to train the super resolution network
* --tile_size_w: width of each tile (same as the one used in `create_training_dataset.py`)
* --tile_size_h: height of each tile (same as the one used in `create_training_dataset.py`)
* --batch_size: batch size
* --nepochs: number of epochs
* --learning_rate: learning_rate
* --restore: flag to indicate that we want to restore weights from pretrained_path (if --restore is not given as parameters, we train from scratch)
* --ssim_loss: flag to indicat that we use the ssim as a loss function instead of the L1 loss.

The output folder is organized as follows

```shell
out_path
   |-- x${upsampling_factor}
      |-- ckpts
      |-- Log
```

where checkpoints are stored in ckpts, and log files are stored in Log.
The log files can be visualized using tensorboard.