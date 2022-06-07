# Proton Interaction Vertex Recognization using Convolutional Neural Network


Training Procedure:
[1] Prepare sample for the training\
--> run the code as: **python prepare_data_cnn_vtx-id.py -c config.json**\
(by default, config.json will be read)\

It will take a while to convert into two npy files, db_view_2_x.npy & db_view_2_y.npy ...
p.s. 
(_2: collection view)
According to the user's manual, need to change the view individually in config.json to produce the relative npy files
"selected_view_idx": 2, --> change 2 to 1 or 0

[2] Start CNN Training
**python train_cnn.py -c config.json**
p.s.
Important!! Put your db_view_2_x.npy & db_view_2_y.npy in training and testing folder under the input folder that you created
Output: modelsgd_lorate_architecture.json & modelsgd_lorate_weights.h5

[3] Combined .json and .h5 to .pb (will be used for reco later)
**python save_tf_proto.py**
