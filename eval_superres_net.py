import argparse
import cv2
import numpy as np
import os
import pickle 
import tensorflow as tf
import utils

from model import build_model


def restore_model(pretrained_path, upsampling_factor, model_saver, sess):
    model_weights_path = os.path.join(
        pretrained_path, 
        "x{}/ckpts".format(upsampling_factor)
    )

    model_saver.restore(
        sess, 
        tf.train.latest_checkpoint(model_weights_path)
    )


def add_batch_chan_dim(tile_img_y):
    tile_img_y = np.expand_dims(tile_img_y, axis=-1)
    tile_img_y = np.expand_dims(tile_img_y, axis=0)

    return tile_img_y


def normalize_center(tile_img_y, tile_img_y_mean):
    tile_img_y = tile_img_y.astype(np.float32)
    tile_img_y /= 255.0

    tile_img_y -= tile_img_y_mean

    return tile_img_y


def create_placeholders(batch_size, nchan):
    """ Create placeholders for the tensorflow model """

    # Create the placeholder
    model_placeholders = {
        "init": tf.placeholder(tf.float32, [batch_size, None, None, nchan]),
        "upsampled": tf.placeholder(tf.float32, [batch_size, None, None, nchan])
    }
    
    return model_placeholders


def create_feed_dict(tile_img_y, tile_img_bic_up_y, model_ph):
    tile_img_y        = add_batch_chan_dim(tile_img_y)
    tile_img_bic_up_y = add_batch_chan_dim(tile_img_bic_up_y)

    feed_dict = {
        model_ph["init"]: tile_img_y, 
        model_ph["upsampled"]: tile_img_bic_up_y, 
    }
    return feed_dict


def parse_args():
    """ Argument reader for the function """

    parser = argparse.ArgumentParser()

    # Common Meta-Parameters (Training parameters)
    parser.add_argument("--data_path"      , type=str, default="./00325_PANSHARP/")
    parser.add_argument("--pretrained_path", type=str, default="./Pretrained_on_DIV2K/")
    parser.add_argument("--mean_values_path", type=str, default="./Pretrained_on_DIV2K/")

    parser.add_argument("--img_ext", type=str, default='.tif')

    parser.add_argument("--upsampling_factor", type=int, default=4) 
    parser.add_argument("--batch_size"       , type=int, default=1)
    parser.add_argument("--nchan"            , type=int, default=1)

    parser.add_argument("--num_tiles_w", type=int, default=5)
    parser.add_argument("--num_tiles_h", type=int, default=5)

    # Input Parameters for the 2nd Network 
    parser.add_argument("--num_FB_layers"    , type=int       , default=2) # Number of layers in the Feature Block (FB)
    parser.add_argument("--num_Dist_blocks"  , type=int       , default=4) # Number of Distillation Blocks (DB)
    parser.add_argument("--weight_decay"     , type=np.float32, default=1.0e-4) 

    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    upsampling_factor = args.upsampling_factor
    num_tiles_w = args.num_tiles_w
    num_tiles_h = args.num_tiles_h

    data_path = args.data_path
    if data_path[-1] == '/':
        data_path = data_path[:-1]

    # Create output path
    out_path  = "{}_x{}".format(data_path, upsampling_factor)
    utils.make_dir_if_not_exists(out_path)

    # List of all tiles
    data_path = args.data_path
    img_list = utils.get_list_files(
        data_path, ext=args.img_ext
    )

    # Load the mean values
    mean_values_path = os.path.join(
        args.mean_values_path, 
        "x{}".format(upsampling_factor),
        "mean_values.pickle"
    )
    
    with open(mean_values_path, "rb") as fid:
        mean_values = pickle.load(fid)

    mean_val_gt     = mean_values["gt"]
    mean_val_down   = mean_values["down"]
    mean_val_bic_up = mean_values["bic_up"]
    
    """
    BUILD THE MODEL
    """
    # Create placeholders 
    model_ph = create_placeholders(args.batch_size, args.nchan)
        
    # Graph Model
    params_superres_net = {
        "num_FB_layers"    : args.num_FB_layers,
        "num_dist_blocks"  : args.num_Dist_blocks,
        "upsampling_factor": upsampling_factor
    }
    
    input_superres_net = {
        "tf_init"     : model_ph["init"],
        "tf_upsampled": model_ph["upsampled"], 
    }

    tf_output_text, _, _ = build_model(
        params_superres_net, input_superres_net
    )

    # Create saver for the model
    model_saver = tf.train.Saver(save_relative_paths=True)

    """
    SUPER RESOLVE ALL IMAGES
    """

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore the model
        restore_model(
            args.pretrained_path, upsampling_factor, model_saver, sess
        )

        # Loop over all images
        for img_id, img_name in enumerate(img_list):
            print("Processing image {}".format(img_name))

            # Debug
            if args.debug:
                debug_tile_path = os.path.join(
                    out_path, "{}_Tiles".format(img_name[:-4])
                )
                utils.make_dir_if_not_exists(debug_tile_path)

            # Open image
            img_path = os.path.join(data_path, img_name)
            full_img  = cv2.imread(img_path)

            height_full_lr, width_full_lr, _ = full_img.shape

             # Create tiles
            res_tiles_w = width_full_lr // num_tiles_w + 1
            res_tiles_h = height_full_lr // num_tiles_h + 1

            # Reconstructing 
            super_resolved_img = []
            bic_up_img = []

            for tile_w_id in range(num_tiles_w):
                tile_w_start = tile_w_id * res_tiles_w
                tile_w_end   = min(tile_w_start + res_tiles_w, width_full_lr)

                super_resolved_col = []
                bic_up_col = []

                for tile_h_id in range(num_tiles_h):
                    tile_h_start = tile_h_id * res_tiles_h
                    tile_h_end   = min(tile_h_start + res_tiles_h, height_full_lr)
                    
                    print("   ---> Tile ({:2d}, {:2d})".format(tile_w_id, tile_h_id))

                    # Extract tile image
                    tile_img = full_img[
                        tile_h_start: tile_h_end,
                        tile_w_start: tile_w_end,
                        :
                    ]

                    height_tile_lr, width_tile_lr, _ = tile_img.shape

                    # Upsample image - BICUBIC
                    height_tile_hr = upsampling_factor * height_tile_lr 
                    width_tile_hr  = upsampling_factor * width_tile_lr

                    tile_img_bic_up = cv2.resize(
                        tile_img, (0,0), 
                        fx=upsampling_factor, 
                        fy=upsampling_factor, 
                        interpolation = cv2.INTER_CUBIC
                    )

                    bic_up_col.append(tile_img_bic_up)
                    
                    # Convert low res to YCbCr
                    tile_img_ycbcr = cv2.cvtColor(tile_img, cv2.COLOR_BGR2YCR_CB)
                    tile_img_y = tile_img_ycbcr[:,:,0]

                    # Convert high res to YCbCr
                    tile_img_bic_up_ycbcr = cv2.cvtColor(tile_img_bic_up, cv2.COLOR_BGR2YCR_CB)
                    tile_img_bic_up_y     = tile_img_bic_up_ycbcr[:,:,0]
                    tile_img_bic_up_cbcr  = tile_img_bic_up_ycbcr[:,:,1:]

                    # Normalize and center y channel
                    tile_img_y        = normalize_center(tile_img_y, mean_val_down)
                    tile_img_bic_up_y = normalize_center(tile_img_bic_up_y, mean_val_bic_up)

                    # Create the feed dictionary
                    feed_dict = create_feed_dict(tile_img_y, tile_img_bic_up_y, model_ph)

                    # Run the model 
                    result = sess.run(
                        tf_output_text, feed_dict=feed_dict
                    )

                    # Unnoramlize the results
                    result += mean_val_gt
                    result  = np.maximum(0.0, np.minimum(result, 1.0))
                    result *= 255.0

                    # Colorize  image
                    result_colored = np.concatenate([result[0,::], tile_img_bic_up_cbcr], axis=-1)
                    super_resolved_col.append(result_colored)

                    # Save image
                    if args.debug:
                        tile_img_hr = result_colored.copy() #np.concatenate([result[0,::], tile_img_bic_up_cbcr], axis=-1)
                        tile_img_hr = cv2.cvtColor(tile_img_hr.astype(np.uint8), cv2.COLOR_YCR_CB2BGR)

                        tile_img_name = os.path.join(
                            debug_tile_path, "{}_{:02d}_{:02d}.tif".format(
                                img_name[:-4], tile_w_id, tile_h_id
                            )
                        )
                        cv2.imwrite(tile_img_name, tile_img_hr)

                        tile_img_bic_up_name = os.path.join(
                            debug_tile_path, "{}_{:02d}_{:02d}_bic_up.tif".format(
                                img_name[:-4], tile_w_id, tile_h_id
                            )
                        )
                        cv2.imwrite(tile_img_bic_up_name, tile_img_bic_up)

                super_resolved_img.append(np.concatenate(super_resolved_col, axis=0))
                bic_up_img.append(np.concatenate(bic_up_col, axis=0))

            full_img_hr = np.concatenate(super_resolved_img, axis=1)
            full_img_hr = cv2.cvtColor(full_img_hr.astype(np.uint8), cv2.COLOR_YCR_CB2BGR)
            full_img_name = os.path.join(out_path, img_name)
            cv2.imwrite(full_img_name, full_img_hr)

            full_img_bicup_hr = np.concatenate(bic_up_img, axis=1)
            full_img_bicup_name = os.path.join(
                out_path, "{}_bic_up.tif".format(img_name[:-4])
            )
            cv2.imwrite(full_img_bicup_name, full_img_bicup_hr)








if __name__ == '__main__':
    main()




