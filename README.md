# Proton Interaction Vertex Recognization using Convolutional Neural Network

Training Procedure for recognizatiing inelastic-/elastic scattering vertex:\
(codes embedded in LArSoft)

 - Setup virtual environment (skip this bullet point if env. has been set)\
--> Source bashrc as: source ~/.bashrc\
    source /home/[usr_name]/ENV/bin/activate\
    source /data/[usr_name]/root/bin/thisroot.sh

 - Prepare sample for the training\
--> Run the code as: **python prepare_data_cnn_vtx-id.py -c config.json**\
    (by default, config.json will be read)

    It will take a while to produce two npy files, db_view_2_x.npy & db_view_2_y.npy ...\
    p.s. 
    - _2: collection view
    - Need to change the view individually in config.json to produce the relative npy files\
      "selected_view_idx": 2, --> change 2 to 1 or 0

 - Start CNN Training\
--> Run as: **python train_cnn.py -c config.json**\
p.s.\
Outputs: modelsgd_lorate_architecture.json & modelsgd_lorate_weights.h5\
**Important!! Put your db_view_2_x.npy & db_view_2_y.npy in the training and the testing folder under the input folder that you created**
 - Combined .json and .h5 to .pb (will be used for reco later)\
**python save_tf_proto.py**
