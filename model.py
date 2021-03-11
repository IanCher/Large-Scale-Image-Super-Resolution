import tensorflow as tf


def build_model(params, input_tensors): 

    # Get parameters
    num_FB_layers     = params["num_FB_layers"]
    num_dist_blocks   = params["num_dist_blocks"]     
    upsampling_factor = params["upsampling_factor"]

    # Extract/Access data from input tensor
    tf_texture        = input_tensors["tf_init"]
    tf_upsampled_text = input_tensors["tf_upsampled"]

    # Build the "Feature extraction blocks" (FBlocks) 
    weights_conv_list      = []
    bias_list              = []
    tf_value_in_fb_layer_l =  tf_texture
    nfilter_fb_in          = 1
    nfilter_fb_out         = 64
    
    for l in range(num_FB_layers):

        layer_key = "feature_block_{}".format(l)

        # Create variables that we can learn for convolutions
        with tf.variable_scope("Weights_conv_fb_layer_{}".format(l)):
            tf_conv_fb_weights = tf.get_variable(
                "weights_conv2d_fb", 
                [3, 3, nfilter_fb_in, nfilter_fb_out], 
                dtype=tf.float32, 
                initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            )

        weights_conv_list.append(tf_conv_fb_weights)

        # Add a Convolution
        tf_conv_fb_text = tf.nn.conv2d(
            tf_value_in_fb_layer_l, tf_conv_fb_weights, 
            [1, 1, 1, 1], padding='SAME'
        ) 

        # Create variables that we can learn for bias 
        with tf.variable_scope("Weights_bias_fb_layer_{}".format(l)):
            tf_bias_fb_weights = tf.get_variable(
                "weights_bias_fb", [nfilter_fb_out], 
                dtype=tf.float32, initializer=tf.zeros_initializer()
            )  
        
        bias_list.append(tf_bias_fb_weights)

        # Add a bias
        tf_bias_fb_text = tf.nn.bias_add(tf_conv_fb_text, tf_bias_fb_weights) 

        # Add the leaky RELU
        tf_out_fb_text = tf.nn.leaky_relu(tf_bias_fb_text, 0.05)
        tf_value_in_fb_layer_l = tf_out_fb_text

        # Update the filter size for next block
        nfilter_fb_in = nfilter_fb_out

    # Build the "Distillation Blocks" (DBlocks)
    tf_value_in_block_b = tf_out_fb_text
    for b in range(num_dist_blocks):

        block_key = "distillation_block_{}".format(b)

        # -- Enhancement Unit--
        # LAYER 0
        with tf.variable_scope("Weights_conv_db_{}_eunit_layer_0".format(b)):
            tf_conv_db_eunit_layer_0_weights = tf.get_variable(
                "weights_conv2d_db_eunit_layer_0", [3, 3, 64, 48], 
                dtype=tf.float32, 
                initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            ) 

        weights_conv_list.append(tf_conv_db_eunit_layer_0_weights)

        tf_conv_db_eunit_layer_0_text = tf.nn.conv2d(
            tf_value_in_block_b, tf_conv_db_eunit_layer_0_weights, 
            [1, 1, 1, 1], padding='SAME'
        )

        with tf.variable_scope("Weights_bias_db_{}_eunit_layer_0".format(b)):
            tf_bias_db_layer_0_weights = tf.get_variable(
                "weights_bias_db_eunit_layer_0", [48], 
                dtype=tf.float32, initializer=tf.zeros_initializer()
            ) 

        bias_list.append(tf_bias_db_layer_0_weights)

        tf_bias_db_eunit_layer_0_text = tf.nn.bias_add(
            tf_conv_db_eunit_layer_0_text, tf_bias_db_layer_0_weights
        ) 

        # Add the leaky RELU
        tf_out_db_eunit_layer_0_text = tf.nn.leaky_relu(
            tf_bias_db_eunit_layer_0_text, 0.05
        )

        # LAYER 1 (grouped convolution)
        group = 4
        shape_conv_layer_1 = [3, 3, 48, 32] 
        split_shape_conv_layer_1 = [shape_conv_layer_1[2] // group for _ in range(group)]

        grouped_feature_layer_1 = tf.split(
            tf_out_db_eunit_layer_0_text, split_shape_conv_layer_1, axis=-1
        )
        
        idx = 0
        tf_out_layer_1_group_text = []
        for feature in grouped_feature_layer_1:
            
            with tf.variable_scope("Weights_conv_db_{}_eunit_layer_1_group_{}".format(b, idx)):
                tf_conv_db_eunit_layer_1_weights = tf.get_variable(
                    "weights_conv2d_db_eunit_layer_1_group", 
                    [3, 3, shape_conv_layer_1[2] // group, shape_conv_layer_1[3] // group], 
                    dtype=tf.float32, 
                    initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
                ) 

            weights_conv_list.append(tf_conv_db_eunit_layer_1_weights)

            tf_conv_db_eunit_layer_1_text = tf.nn.conv2d(
                feature, tf_conv_db_eunit_layer_1_weights, 
                [1, 1, 1, 1], padding='SAME'
            )

            with tf.variable_scope("Weights_bias_db_{}_eunit_layer_1_group_{}".format(b, idx)):
                tf_bias_db_eunit_layer_1_weights = tf.get_variable(
                    "weights_bias_db_eunit_layer_1_group", 
                    [shape_conv_layer_1[3] // group], dtype=tf.float32, 
                    initializer=tf.zeros_initializer()
                ) 

            bias_list.append(tf_bias_db_eunit_layer_1_weights)

            tf_bias_db_eunit_layer_1_text = tf.nn.bias_add(
                tf_conv_db_eunit_layer_1_text, tf_bias_db_eunit_layer_1_weights
            ) 

            tf_out_layer_1_group_text.append(tf_bias_db_eunit_layer_1_text)

            # Update index of the group 
            idx += 1 

        # Concatenate the output of the groups as output of the layer 
        tf_concat_db_eunit_layer_1_text = tf.concat(tf_out_layer_1_group_text, axis=-1)

        # Add the leaky RELU
        tf_out_db_eunit_layer_1_text = tf.nn.leaky_relu(tf_concat_db_eunit_layer_1_text, 0.05)


        # LAYER 2
        with tf.variable_scope("Weights_conv_db_{}_eunit_layer_2".format(b)):
            tf_conv_db_eunit_layer_2_weights = tf.get_variable(
                "weights_conv2d_db_eunit_layer_2", [3, 3, 32, 64], 
                dtype=tf.float32, 
                initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            ) 

        weights_conv_list.append(tf_conv_db_eunit_layer_2_weights)

        tf_conv_db_eunit_layer_2_text = tf.nn.conv2d(
            tf_out_db_eunit_layer_1_text, 
            tf_conv_db_eunit_layer_2_weights, 
            [1, 1, 1, 1], padding='SAME'
        )

        with tf.variable_scope("Weights_bias_db_{}_eunit_layer_2".format(b)):
            tf_bias_db_eunit_layer_2_weights = tf.get_variable(
                "weights_bias_db_eunit_layer_2", [64], 
                dtype=tf.float32, initializer=tf.zeros_initializer()
            ) 

        bias_list.append(tf_bias_db_eunit_layer_2_weights)

        tf_bias_db_eunit_layer_2_text = tf.nn.bias_add(
            tf_conv_db_eunit_layer_2_text, tf_bias_db_eunit_layer_2_weights
        ) 

        # Add the leaky RELU
        tf_out_db_eunit_layer_2_text = tf.nn.leaky_relu(tf_bias_db_eunit_layer_2_text, 0.05)
        
        # SLICING : Output of 3rd layer "layer_2" is sliced into two segments "segment_1" and "segment_2"
        segment_1, segment_2 = tf.split(tf_out_db_eunit_layer_2_text, [16, 48], axis=-1)
        concatenated = tf.concat([tf_value_in_block_b, segment_1], axis=-1)

        # LAYER 3
        with tf.variable_scope("Weights_conv_db_{}_eunit_layer_3".format(b)):
            tf_conv_db_eunit_layer_3_weights = tf.get_variable(
                "weights_conv2d_db_eunit_layer_3", [3, 3, 48, 64], 
                dtype=tf.float32, 
                initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            ) 

        weights_conv_list.append(tf_conv_db_eunit_layer_3_weights)

        tf_conv_db_eunit_layer_3_text = tf.nn.conv2d(
            segment_2, tf_conv_db_eunit_layer_3_weights, 
            [1, 1, 1, 1], padding='SAME'
        )   

        with tf.variable_scope("Weights_bias_db_{}_eunit_layer_3".format(b)):
            tf_bias_db_eunit_layer_3_weights = tf.get_variable(
                "weights_bias_db_eunit_layer_3", [64], 
                dtype=tf.float32, initializer=tf.zeros_initializer()
            ) 

        bias_list.append(tf_bias_db_eunit_layer_3_weights)

        tf_bias_db_eunit_layer_3_text = tf.nn.bias_add(
            tf_conv_db_eunit_layer_3_text, tf_bias_db_eunit_layer_3_weights
        ) 

        # Add the leaky RELU
        tf_out_db_eunit_layer_3_text = tf.nn.leaky_relu(
            tf_bias_db_eunit_layer_3_text, 0.05
        )


        # LAYER 4 (grouped convolution)
        group = 4
        shape_conv_layer_4 = [3, 3, 64, 48] 
        split_shape_conv_layer_4 = [shape_conv_layer_4[2] // group for _ in range(group)]
        grouped_feature_layer_4 = tf.split(
            tf_out_db_eunit_layer_3_text, split_shape_conv_layer_4, axis=-1
        )
        
        idx = 0
        tf_out_layer_4_group_text = []
        for feature in grouped_feature_layer_4:
            
            with tf.variable_scope("Weights_conv_db_{}_eunit_layer_4_group_{}".format(b, idx)):
                tf_conv_db_eunit_layer_4_weights = tf.get_variable(
                    "weights_conv2d_db_eunit_layer_4_group", 
                    [3, 3, shape_conv_layer_4[2] // group, shape_conv_layer_4[3] // group], 
                    dtype=tf.float32, 
                    initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
                ) 

            weights_conv_list.append(tf_conv_db_eunit_layer_4_weights)

            tf_conv_db_eunit_layer_4_text = tf.nn.conv2d(
                feature, tf_conv_db_eunit_layer_4_weights, [1, 1, 1, 1], padding='SAME'
            )

            with tf.variable_scope("Weights_bias_db_{}_eunit_layer_4_group_{}".format(b, idx)):
                tf_bias_db_eunit_layer_4_weights = tf.get_variable(
                    "weights_bias_db_eunit_layer_4_group", [shape_conv_layer_4[3] // group], 
                    dtype=tf.float32, initializer=tf.zeros_initializer()
                ) 

            bias_list.append(tf_bias_db_eunit_layer_4_weights)

            tf_bias_db_eunit_layer_4_text = tf.nn.bias_add(
                tf_conv_db_eunit_layer_4_text, tf_bias_db_eunit_layer_4_weights
            ) 

            tf_out_layer_4_group_text.append(tf_bias_db_eunit_layer_4_text)

            # Update index of the group 
            idx += 1 

        # Concatenate the output of the groups as output of the layer 
        tf_concat_db_eunit_layer_4_text = tf.concat(tf_out_layer_4_group_text, axis=-1)

        # Add the leaky RELU
        tf_out_db_eunit_layer_4_text = tf.nn.leaky_relu(tf_concat_db_eunit_layer_4_text, 0.05)


        # LAYER 5
        with tf.variable_scope("Weights_conv_db_{}_eunit_layer_5".format(b)):
            tf_conv_db_eunit_layer_5_weights = tf.get_variable(
                "weights_conv2d_db_eunit_layer_5", [3, 3, 48, 80], 
                dtype=tf.float32, 
                initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            ) 

        weights_conv_list.append(tf_conv_db_eunit_layer_5_weights)

        tf_conv_db_eunit_layer_5_text = tf.nn.conv2d(
            tf_out_db_eunit_layer_4_text, 
            tf_conv_db_eunit_layer_5_weights, 
            [1, 1, 1, 1], padding='SAME'
        )   

        with tf.variable_scope("Weights_bias_db_{}_eunit_layer_5".format(b)):
            tf_bias_db_eunit_layer_5_weights = tf.get_variable(
                "weights_bias_db_eunit_layer_5", [80], 
                dtype=tf.float32, 
                initializer=tf.zeros_initializer()
            ) 

        bias_list.append(tf_bias_db_eunit_layer_5_weights)

        tf_bias_db_eunit_layer_5_text = tf.nn.bias_add(
            tf_conv_db_eunit_layer_5_text, tf_bias_db_eunit_layer_5_weights
        ) 

        # Add the leaky RELU
        tf_out_db_eunit_layer_5_text = tf.nn.leaky_relu(
            tf_bias_db_eunit_layer_5_text, 0.05
        )

        # Output of the Distillation Block 
        tf_out_db_eunit = tf.add(tf_out_db_eunit_layer_5_text, concatenated) # input of next block

        # -- Compression --
        # Dimensionality reduction or distilling relevant information for the later network
        with tf.variable_scope("Weights_conv_db_{}_cunit".format(b)):
            tf_conv_db_cunit_weights = tf.get_variable(
                "weights_conv2d_db_cunit", [1, 1, 80, 64], 
                dtype=tf.float32, 
                initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            ) 

        weights_conv_list.append(tf_conv_db_cunit_weights)

        tf_conv_db_cunit_text = tf.nn.conv2d(
            tf_out_db_eunit, tf_conv_db_cunit_weights, 
            [1, 1, 1, 1], padding='SAME'
        )  

        # Update input of next DBlocks
        tf_value_in_block_b = tf_conv_db_cunit_text
        

    # Build the "Reconstruction Blocks" (RBlock) 
    # It is a transposed convolution without activation function 
    shape = tf.shape(tf_conv_db_cunit_text)
    output_shape = [shape[0], upsampling_factor * shape[1], upsampling_factor * shape[2], 1]
    stride = [1, upsampling_factor, upsampling_factor, 1]
    
    with tf.variable_scope("Weights_conv_rb"):
        tf_conv_rb_weights = tf.get_variable(
            "weights_conv2d_rb", [17, 17, 1, 64], 
            dtype=tf.float32, 
            initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        ) 

    weights_conv_list.append(tf_conv_rb_weights)

    tf_conv_reconstructed_text = tf.nn.conv2d_transpose(
        tf_conv_db_cunit_text, tf_conv_rb_weights, 
        output_shape, stride, padding='SAME'
    )   

    with tf.variable_scope("Weights_bias_rb"):
        tf_bias_rb_weights = tf.get_variable(
            "weights_bias_rb", output_shape[-1], 
            dtype=tf.float32, 
            initializer=tf.zeros_initializer()
        ) 

    bias_list.append(tf_bias_rb_weights)

    tf_bias_reconstructed_text = tf.nn.bias_add(tf_conv_reconstructed_text, tf_bias_rb_weights) 

    # Add the upsampled input image (Bicubic image)
    tf_output = tf.add(tf_bias_reconstructed_text, tf_upsampled_text) 

        
    return tf_output, weights_conv_list, bias_list
