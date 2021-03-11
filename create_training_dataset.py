import argparse 
import cv2
import numpy as np 
import os 
import pickle
import tensorflow as tf 
import utils
from random import shuffle


def _bytes_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_scenes_data(
    img_list, data_path, tile_size_w, tile_size_h,
    upsampling_factor=2, dtype=np.float32, debug=False
    ):
    
    gt_list     = [] 
    down_list   = []
    bic_up_list = []
    
    mean_val_gt = 0.0
    mean_val_down = 0.0
    mean_val_bic_up = 0.0
    
    num_tiles = 0

    for img_name in img_list:
        print("Loading img {}".format(img_name))
        img_path = os.path.join(data_path, img_name)

        # Read ground truth directly as Y-channel
        full_img = cv2.imread(img_path)
        height_full_lr, width_full_lr, _ = full_img.shape

        # Tile the image
        for tile_w_start in range(0, width_full_lr, tile_size_w):
            tile_w_end = tile_w_start + tile_size_w
            
            if tile_w_end > width_full_lr:
                continue

            for tile_h_start in range(0, height_full_lr, tile_size_h):
                tile_h_end = tile_h_start + tile_size_h
                
                if tile_h_end > height_full_lr:
                    continue
                
                # Extract tile image
                tile_img_gt = full_img[
                    tile_h_start: tile_h_end,
                    tile_w_start: tile_w_end,
                    :
                ]

                # Downsample image - BICUBIC
                height_tile_lr = tile_size_h // upsampling_factor 
                width_tile_lr  = tile_size_w  // upsampling_factor

                tile_img_down = cv2.resize(
                    tile_img_gt, (height_tile_lr, width_tile_lr), 
                    interpolation=cv2.INTER_CUBIC
                )

                # Upsample image - BICUBIC
                tile_img_bic_up = cv2.resize(
                    tile_img_down, (tile_size_h, tile_size_w), 
                    interpolation=cv2.INTER_CUBIC
                )

                # Extract y channel
                tile_img_gt_y     = cv2.cvtColor(tile_img_gt, cv2.COLOR_BGR2YCR_CB)[:,:,0]
                tile_img_down_y   = cv2.cvtColor(tile_img_down, cv2.COLOR_BGR2YCR_CB)[:,:,0]
                tile_img_bic_up_y = cv2.cvtColor(tile_img_bic_up, cv2.COLOR_BGR2YCR_CB)[:,:,0]

                # Normalize
                tile_img_gt_y = tile_img_gt_y.astype(np.float32)
                tile_img_gt_y /= 255.0
                
                tile_img_down_y = tile_img_down_y.astype(np.float32)
                tile_img_down_y /= 255.0

                tile_img_bic_up_y = tile_img_bic_up_y.astype(np.float32)
                tile_img_bic_up_y /= 255.0

                # Extract mean value
                tile_img_gt_y_mean = np.mean(tile_img_gt_y)
                tile_img_down_y_mean = np.mean(tile_img_down_y)
                tile_img_bic_up_y_mean = np.mean(tile_img_bic_up_y)

                # Add to full mean
                mean_val_gt     += tile_img_gt_y_mean
                mean_val_down   += tile_img_down_y_mean
                mean_val_bic_up += tile_img_bic_up_y_mean

                num_tiles += 1

                # Store all tiles in a list
                gt_list.append(tile_img_gt_y)
                down_list.append(tile_img_down_y)
                bic_up_list.append(tile_img_bic_up_y)

                # DEBUG
                if debug:
                    debug_gt = 255*gt_list[-1]
                    debug_gt = debug_gt.astype(np.uint8)
                    cv2.imwrite(
                        "Debug/{}_{:03d}_{:03d}_gt.png".format(img_name[:-4], tile_h_start, tile_w_start), 
                        debug_gt
                    )

                    cv2.imwrite(
                        "Debug/{}_{:03d}_{:03d}_down.png".format(img_name[:-4], tile_h_start, tile_w_start), 
                        tile_img_down
                    )

                    debug_bic_up = 255*bic_up_list[-1]
                    debug_bic_up = debug_bic_up.astype(np.uint8)
                    cv2.imwrite(
                        "Debug/{}_{:03d}_{:03d}_up_bic.png".format(img_name[:-4], tile_h_start, tile_w_start), 
                        debug_bic_up
                    )
    
    # Mean values
    mean_val_gt /= num_tiles
    mean_val_down /= num_tiles 
    mean_val_bic_up /= num_tiles   

    # Center all data
    for tile_id in range(num_tiles):
        gt_list[tile_id]     -= mean_val_gt
        down_list[tile_id]   -= mean_val_down
        bic_up_list[tile_id] -= mean_val_bic_up

    return gt_list, bic_up_list, down_list, mean_val_gt, mean_val_bic_up, mean_val_down

    



def parse_args():
    """ Argument reader for the function """

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="00325_PANSHARP/")
    parser.add_argument("--out_path", type=str, default="Training_Data/")
    parser.add_argument("--img_ext", type=str, default='.tif')
    parser.add_argument("--upsampling_factor", type=int, default=2)
    parser.add_argument("--num_data_per_file", type=int, default=250)
    parser.add_argument("--tile_size_w", type=int, default=400)
    parser.add_argument("--tile_size_h", type=int, default=400)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main():

    # Read the arguments
    args = parse_args()

    # Read the paths arguments
    data_path  = args.data_path
    upsampling_factor = args.upsampling_factor
    num_data_per_file = args.num_data_per_file

    # Create output paths
    out_path = os.path.join(
        args.out_path, "x{}".format(upsampling_factor)
    )
    utils.make_dir_if_not_exists(out_path)
    utils.make_dir_if_not_exists(
        os.path.join(out_path, "Training")
    )
    utils.make_dir_if_not_exists(
        os.path.join(out_path, "Validation")
    )

    if args.debug:
        utils.make_dir_if_not_exists("Debug")

    # Read list of all scenes
    img_list = utils.get_list_files(data_path, ext=args.img_ext)
    num_img = len(img_list)

    # And compute directly on the fly the upsampled texture init 
    gt_list, bic_up_list, down_list, mean_val_gt, mean_val_bic_up, mean_val_down = read_scenes_data(
        img_list, data_path, args.tile_size_w, args.tile_size_h, 
        upsampling_factor=upsampling_factor,
    )

    num_data = len(gt_list)
    assert len(bic_up_list) == num_data

    # Save mean value
    mean_val = {
        "gt": mean_val_gt,
        "down": mean_val_down,
        "bic_up": mean_val_bic_up
    }

    mean_val_file = os.path.join(out_path, "mean_values.pickle") 

    with open(mean_val_file, "wb") as fid:
        pickle.dump(mean_val, fid)

    print("GT     Mean = {}".format(mean_val_gt))
    print("Down   Mean = {}".format(mean_val_down))
    print("Bic up Mean = {}".format(mean_val_bic_up))

    # DEBUG
    if args.debug:
        mean_val_gt_debug = 0
        mean_val_down_debug = 0
        mean_val_bic_up_debug = 0

        for tile_img_gt_y, tile_img_bic_up_y, tile_img_down_y in zip(
            gt_list, bic_up_list, down_list
            ):
            mean_val_gt_debug += np.mean(tile_img_gt_y)
            mean_val_down_debug += np.mean(tile_img_down_y) 
            mean_val_bic_up_debug += np.mean(tile_img_bic_up_y) 

        num_data = len(gt_list)

        mean_val_gt_debug /= num_data
        mean_val_down_debug /= num_data
        mean_val_bic_up_debug /= num_data

        print(50 * "#")
        print("# DEBUG ")
        print("GT     Mean = {}".format(mean_val_gt_debug))
        print("Down   Mean = {}".format(mean_val_down_debug))
        print("Bic up Mean = {}".format(mean_val_bic_up_debug))
        print(50 * "#")

    # Save the data in a TF.Records
    tf_rec_writer = None
    tile_img_indices = list(range(num_data))

    if not args.no_shuffle:
        shuffle(tile_img_indices)

    for data_id, img_id in enumerate(tile_img_indices):

        if args.debug:
            print("Tile {}".format(img_id))

        # Open the corresponding TF Record file
        if data_id % num_data_per_file == 0:
            print("\n##### CREATING A NEW FILE #####")

            if tf_rec_writer:
                tf_rec_writer.close()
            
            file_id = data_id // num_data_per_file

            if file_id == 0:
                tf_rec_name = os.path.join(
                    out_path, "Validation/data_{0:03d}.tfrecords".format(file_id)
                )
            else:
                tf_rec_name = os.path.join(
                    out_path, "Training/data_{0:03d}.tfrecords".format(file_id)
                )

            tf_rec_writer = tf.python_io.TFRecordWriter(tf_rec_name)

        tile_img_gt = gt_list[img_id].astype(np.float32)
        tile_img_down = down_list[img_id].astype(np.float32)
        tile_img_bic_up = bic_up_list[img_id].astype(np.float32)
        
        # Convert to bytes
        tile_img_gt_raw = tile_img_gt.tostring()
        tile_img_down_raw = tile_img_down.tostring()
        tile_img_bic_up_raw = tile_img_bic_up.tostring()
        
        # Create feature
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'gt'    : _bytes_feature(tile_img_gt_raw),
                    'down'  : _bytes_feature(tile_img_down_raw),
                    'bic_up': _bytes_feature(tile_img_bic_up_raw)
                }
            )
        )

        # Write the sample
        tf_rec_writer.write(example.SerializeToString())

    tf_rec_writer.close()


if __name__ == "__main__":
    main()
