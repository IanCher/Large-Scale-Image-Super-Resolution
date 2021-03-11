import argparse
import cv2
import numpy as np
import os
import pickle

import tensorflow as tf
import utils

from model import build_model
import matplotlib.cm

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    From: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b

    !!! Converted with numpy array instead of tf.Tensor object !!!
    """

    # normalize
    vmin = np.min(value) if vmin is None else vmin
    vmax = np.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    indices = (np.rint(value[:,:,:,0] * 255)).astype(np.int32) # round to the nearest integer and convert to int32

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')

    colors = cm(np.arange(256))[:, :3] # so we replace with the two following lines. Colors is of dim [256, 3] = 256 color code of RGB
    colors = np.array(colors, dtype=np.float32)

    value_colored = np.zeros((value.shape[0], value.shape[1], value.shape[2], 3), dtype=np.float32) # 3channels since it is colored
    value_colored[:,:,:,0] = np.take(colors[:,0], indices)
    value_colored[:,:,:,1] = np.take(colors[:,1], indices)
    value_colored[:,:,:,2] = np.take(colors[:,2], indices)

    return value_colored


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


def parse_record(serialized, shape_params):
    # Data-names and types we expect find in the TFRecords files.
    features = {
        'gt'    : tf.FixedLenFeature([], tf.string),
        'down'  : tf.FixedLenFeature([], tf.string),
        'bic_up': tf.FixedLenFeature([], tf.string),
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)

    # Get the data as raw bytes.
    gt_raw     = parsed_example['gt'] 
    down_raw   = parsed_example['down']
    bic_up_raw = parsed_example['bic_up']

    # Decode the raw bytes so they become tensors with specified types.
    gt_tf     = tf.decode_raw(gt_raw    , tf.float32)
    down_tf   = tf.decode_raw(down_raw  , tf.float32)
    bic_up_tf = tf.decode_raw(bic_up_raw, tf.float32)
    
    # Reshape the data
    gt_tf     = tf.reshape(gt_tf    , shape_params["gt"])
    down_tf   = tf.reshape(down_tf  , shape_params["down"])
    bic_up_tf = tf.reshape(bic_up_tf, shape_params["bic_up"])

    # Output the data in a dictionnary
    out_dict = {
        "gt"    : gt_tf,
        "down"  : down_tf,
        "bic_up": bic_up_tf
    }

    return out_dict


def parse_args():
    """ Argument reader for the function """

    parser = argparse.ArgumentParser()

    # Common Meta-Parameters (Training parameters)
    parser.add_argument("--data_path"      , type=str, default="./Training_Data/")
    parser.add_argument("--out_path"       , type=str, default="./Training_Results/")
    parser.add_argument("--pretrained_path", type=str, default="./Pretrained_on_DIV2K/")

    parser.add_argument("--img_ext", type=str, default='.tif')

    parser.add_argument("--upsampling_factor", type=int, default=2) 
    parser.add_argument("--batch_size"       , type=int, default=1)
    parser.add_argument("--nchan"            , type=int, default=1)
    parser.add_argument("--tile_size_w"      , type=int, default=400)
    parser.add_argument("--tile_size_h"      , type=int, default=400)

    # Input Parameters for the 2nd Network 
    parser.add_argument("--num_FB_layers"  , type=int       , default=2) # Number of layers in the Feature Block (FB)
    parser.add_argument("--num_Dist_blocks", type=int       , default=4) # Number of Distillation Blocks (DB)
    parser.add_argument("--weight_decay"   , type=np.float32, default=1.0e-4) 

    parser.add_argument("--nepochs"      , type=int, default=3)
    parser.add_argument("--shuffle"      , type=int, default=50)
    parser.add_argument("--learning_rate", type=np.float32, default=1.0e-4)
    parser.add_argument("--num_img_viz"  , type=int, default=3)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--restore", action="store_true")

    parser.add_argument("--ssim_loss", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    upsampling_factor = args.upsampling_factor
    data_path = os.path.join(
        args.data_path, "x{}".format(upsampling_factor)
    )

    # Create output path
    out_path  = os.path.join(
        args.out_path, "x{}".format(upsampling_factor)
    )
    utils.make_dir_if_not_exists(out_path)

    # Shape parameters
    nchan = args.nchan

    width_hr = args.tile_size_w
    height_hr = args.tile_size_h

    width_lr  = width_hr // upsampling_factor
    height_lr = height_hr // upsampling_factor

    shape_params = {
        "gt"    : [height_hr, width_hr, nchan], 
        "down"  : [height_lr, width_lr, nchan],
        "bic_up": [height_hr, width_hr, nchan]
    }

    """
    CREATE DATASET
    """
    batch_size = args.batch_size

    # Training dataset
    train_tf_rec_path = os.path.join(data_path, "Training")
    tfrec_filename_list = utils.get_list_files(
        train_tf_rec_path, ext='.tfrecords', sort_list=True
    )
    train_tfrec_list = [
        os.path.join(train_tf_rec_path, tfrec_filename) 
        for tfrec_filename in tfrec_filename_list
    ]

    train_dataset = tf.data.TFRecordDataset(filenames=train_tfrec_list)
    train_dataset = train_dataset.shuffle(args.shuffle)
    train_dataset = train_dataset.map(
        lambda x: parse_record(x, shape_params), num_parallel_calls=4
    )
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

    # Validation dataset
    val_tf_rec_path = os.path.join(data_path, "Validation")
    val_tfrec_filename_list = utils.get_list_files(
        val_tf_rec_path, ext='.tfrecords', sort_list=True
    )
    val_tfrec_list = [
        os.path.join(val_tf_rec_path, tfrec_filename) 
        for tfrec_filename in val_tfrec_filename_list
    ]

    val_dataset = tf.data.TFRecordDataset(filenames=val_tfrec_list)
    val_dataset = val_dataset.shuffle(args.shuffle) 
    val_dataset = val_dataset.map(
        lambda x: parse_record(x, shape_params), num_parallel_calls=4
    ) 
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True) 

    # Create a reinitializable iterator of the correct shape and type
    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes
    )
    next_element = iterator.get_next()

    # Create the initialization operations
    train_data_init_op = iterator.make_initializer(train_dataset)
    val_data_init_op   = iterator.make_initializer(val_dataset)

    # Read the mean values of our dataset
    mean_val_path = os.path.join(
        data_path, "mean_values.pickle"
    )
    with open(mean_val_path, "rb") as fid:
        mean_values = pickle.load(fid)

    mean_val_gt     = mean_values["gt"]
    mean_val_down   = mean_values["down"]
    mean_val_bic_up = mean_values["bic_up"]
    
    """
    BUILD THE MODEL
    """        
    # Graph Model
    params_superres_net = {
        "num_FB_layers"    : args.num_FB_layers,
        "num_dist_blocks"  : args.num_Dist_blocks,
        "upsampling_factor": upsampling_factor
    }
    
    input_superres_net = {
        "tf_init"     : next_element["down"],
        "tf_upsampled": next_element["bic_up"], 
    }

    tf_superres, weights_conv_list, bias_list = build_model(
        params_superres_net, input_superres_net
    )

    """
    TRAINING OPERATORS
    """

    # SSIM operators
    ssim_gt_res_op = tf.image.ssim(next_element["gt"], tf_superres, max_val=1.0)
    ssim_gt_res_op = tf.reduce_mean(ssim_gt_res_op)

    # Loss operator
    if args.ssim_loss:
        loss_op = -1.0 * ssim_gt_res_op
    else:
        loss_op = tf.losses.absolute_difference(
            next_element["gt"], tf_superres
        ) 
   
    # Weight decay 
    loss_decay = [tf.nn.l2_loss(w) for w in weights_conv_list] 
    loss_decay = args.weight_decay * tf.add_n(loss_decay)
    loss_op   += loss_decay
    
    # Create the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op  = optimizer.minimize(loss_op)

    # Create summaries 
    num_img_viz = args.num_img_viz

    summary_train_loss_ph = tf.placeholder(tf.float32, name="summary_train_loss_ph")
    summary_val_loss_ph   = tf.placeholder(tf.float32, name="summary_val_loss_ph")

    summary_train_ssim_gt_res_ph = tf.placeholder(tf.float32, name="summary_train_ssim_gt_res_ph")
    summary_val_ssim_gt_res_ph = tf.placeholder(tf.float32, name="summary_val_ssim_gt_res_ph")

    summary_train_gt_ph     = tf.placeholder(tf.float32, [None, None, None, nchan], name="summary_train_gt")
    summary_train_result_ph = tf.placeholder(tf.float32, [None, None, None, nchan], name="summary_train_result")
    summary_train_bic_up_ph = tf.placeholder(tf.float32, [None, None, None, nchan], name="summary_train_bic_up")

    summary_train_residual_gt_res_ph = tf.placeholder(
        tf.float32, [None, None, None, 3], name="summary_train_residual_gt_res_ph"
    )
    summary_train_residual_bicup_res_ph = tf.placeholder(
        tf.float32, [None, None, None, 3], name="summary_train_residual_bicup_res_ph"
    )

    summary_val_gt_ph     = tf.placeholder(tf.float32, [None, None, None, nchan], name="summary_val_gt")
    summary_val_result_ph = tf.placeholder(tf.float32, [None, None, None, nchan], name="summary_val_result")
    summary_val_bic_up_ph = tf.placeholder(tf.float32, [None, None, None, nchan], name="summary_val_bic_up")

    summary_val_residual_gt_res_ph = tf.placeholder(
        tf.float32, [None, None, None, 3], name="summary_val_residual_gt_res_ph"
    )
    summary_val_residual_bicup_res_ph = tf.placeholder(
        tf.float32, [None, None, None, 3], name="summary_val_residual_bicup_res_ph"
    )

    tf.summary.scalar("01_train_loss", summary_train_loss_ph)
    tf.summary.scalar("02_train_ssim_gt_res", summary_train_ssim_gt_res_ph)
    
    tf.summary.scalar("12_val_loss", summary_val_loss_ph)
    tf.summary.scalar("12_val_ssim_gt_res", summary_val_ssim_gt_res_ph)
    
    tf.summary.image ('00_train_gt'    , summary_train_gt_ph    , num_img_viz) 
    tf.summary.image ('01_train_bic_up', summary_train_bic_up_ph, num_img_viz) 
    tf.summary.image ('02_train_result', summary_train_result_ph, num_img_viz) 

    tf.summary.image (
        '03_train_residual_gt_res', 
        summary_train_residual_gt_res_ph, num_img_viz
    ) 
    tf.summary.image (
        '04_train_residual_bicup_res', 
        summary_train_residual_bicup_res_ph, num_img_viz
    ) 

    tf.summary.image ('10_val_gt'    , summary_val_gt_ph    , num_img_viz) 
    tf.summary.image ('11_val_bic_up', summary_val_bic_up_ph, num_img_viz) 
    tf.summary.image ('12_val_result', summary_val_result_ph, num_img_viz) 

    tf.summary.image (
        '13_val_residual_gt_res', 
        summary_val_residual_gt_res_ph, num_img_viz
    ) 
    tf.summary.image (
        '14_val_residual_bicup_res', 
        summary_val_residual_bicup_res_ph, num_img_viz
    ) 

    summary_op = tf.summary.merge_all()

    # Model and Train Saver     
    with tf.name_scope("Train_Saver"):
        train_saver = tf.train.Saver(
            max_to_keep=5, keep_checkpoint_every_n_hours=2,
            save_relative_paths=True, pad_step_number=True
        )
        

    """
    TRAINING SESSION
    """

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Create Summary
        log_path   = os.path.join(out_path, "Log")
        log_writer = tf.summary.FileWriter(log_path)
        log_writer.add_graph(sess.graph)

        # Savers 
        checkpoint_path = os.path.join(out_path, "ckpts")
        utils.make_dir_if_not_exists(checkpoint_path)
        train_saver.save(
            sess, os.path.join(checkpoint_path, "initial"), 
            write_meta_graph=True
        )

        # Restore the model or if necessary
        if args.restore:
            restore_model(
                args.pretrained_path, upsampling_factor, train_saver, sess
            )

        # Start training
        for epoch in range(args.nepochs):
            print("Epoch {}".format(epoch))
            
            """ 
            TRAINING 
            """
            sess.run(train_data_init_op)

            num_batches = 0
            loss_train_mean = 0
            ssim_gt_res_train_mean = 0

            while True:
                try:
                    # Run model
                    (
                        gt_value_train,
                        bic_up_value_train,
                        result_value_train,
                        ssim_gt_res_value,
                        loss_value,
                        _
                    ) = sess.run(
                        [
                            next_element["gt"],
                            next_element["bic_up"],
                            tf_superres,
                            ssim_gt_res_op,
                            loss_op,
                            train_op,
                        ]
                    )

                    # Update values
                    num_batches += 1
                    loss_train_mean += loss_value
                    ssim_gt_res_train_mean += ssim_gt_res_value

                except tf.errors.OutOfRangeError:
                    break

            # Update mean values
            loss_train_mean /= num_batches
            ssim_gt_res_train_mean /= num_batches

            # Arrange images ot be visualized in tensorboard
            gt_value_train += mean_val_gt
            bic_up_value_train += mean_val_bic_up
            result_value_train += mean_val_gt

            result_value_train = np.maximum(0.0, np.minimum(result_value_train, 1.0))

            # Residual
            residual_gt_result_train = gt_value_train - result_value_train
            residual_bicup_result_train = bic_up_value_train - result_value_train

            min_colorscale = min(
                np.min(residual_gt_result_train), np.min(residual_bicup_result_train)
            )
            max_colorscale = max(
                np.max(residual_gt_result_train), np.max(residual_bicup_result_train)
            )

            residual_bicup_result_train = colorize(
                residual_bicup_result_train, min_colorscale, max_colorscale, cmap='jet'
            )
            residual_gt_result_train = colorize(
                residual_gt_result_train, min_colorscale, max_colorscale, cmap='jet'
            )

            """ 
            VALIDATION 
            """
            sess.run(val_data_init_op)

            num_batches = 0
            loss_val_mean = 0
            ssim_gt_res_val_mean = 0

            while True:
                try:
                    # Run model
                    (
                        gt_value_val,
                        bic_up_value_val,
                        result_value_val,
                        ssim_gt_res_value,
                        loss_value,
                    ) = sess.run(
                        [
                            next_element["gt"],
                            next_element["bic_up"],
                            tf_superres,
                            ssim_gt_res_op,
                            loss_op,
                        ]
                    )

                    # Update values
                    num_batches += 1
                    loss_val_mean += loss_value
                    ssim_gt_res_val_mean += ssim_gt_res_value

                except tf.errors.OutOfRangeError:
                    break

            # Update mean values
            loss_val_mean /= num_batches
            ssim_gt_res_val_mean /= num_batches

            # Arrange images ot be visualized in tensorboard
            gt_value_val += mean_val_gt
            bic_up_value_val += mean_val_bic_up
            result_value_val += mean_val_gt

            result_value_val = np.maximum(0.0, np.minimum(result_value_val, 1.0))

            # Residual
            residual_gt_result_val = gt_value_val - result_value_val
            residual_bicup_result_val = bic_up_value_val - result_value_val

            min_colorscale = min(
                np.min(residual_gt_result_val), np.min(residual_bicup_result_val)
            )
            max_colorscale = max(
                np.max(residual_gt_result_val), np.max(residual_bicup_result_val)
            )

            residual_bicup_result_val = colorize(
                residual_bicup_result_val, min_colorscale, max_colorscale, cmap='jet'
            )
            residual_gt_result_val = colorize(
                residual_gt_result_val, min_colorscale, max_colorscale, cmap='jet'
            )


            # Write out summaries
            summary = sess.run(
                summary_op, 
                feed_dict={
                    summary_train_loss_ph: loss_train_mean,
                    summary_train_ssim_gt_res_ph: ssim_gt_res_train_mean,

                    summary_train_gt_ph    : gt_value_train, 
                    summary_train_bic_up_ph: bic_up_value_train,
                    summary_train_result_ph: result_value_train,         

                    summary_train_residual_gt_res_ph: residual_gt_result_train,
                    summary_train_residual_bicup_res_ph: residual_bicup_result_train, 

                    summary_val_loss_ph: loss_val_mean,
                    summary_val_ssim_gt_res_ph: ssim_gt_res_val_mean,

                    summary_val_gt_ph    : gt_value_val, 
                    summary_val_bic_up_ph: bic_up_value_val,
                    summary_val_result_ph: result_value_val,         

                    summary_val_residual_gt_res_ph: residual_gt_result_val,
                    summary_val_residual_bicup_res_ph: residual_bicup_result_val,            
                }
            )

            log_writer.add_summary(summary, epoch)

            # Save a checkpoint
            train_saver.save(
                sess, os.path.join(checkpoint_path, "checkpoint"), 
                global_step=epoch, write_meta_graph=False
            )
        
        # Save the final model
        train_saver.save(
            sess, os.path.join(checkpoint_path, "final"), 
            write_meta_graph=True
        )








if __name__ == '__main__':
    main()




