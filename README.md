# NIFCyl

## 1. Introduction
This repository contains the official implementation of **NIFCyl**, a framework designed for quantifying underground mine tunnel deformation using SLAM LiDAR point clouds. It learns a Neural Implicit Function to approximate tunnel surface geometry and generate scale-invariant normal fields from point clouds, coupled with a cylinder-based module for robust spatial averaging. The two major innovations and mechanisms that make NIFCyl superior to other methods are: first, it leverages the gradients of a learned signed distance function (SDF) to generate mathematically consistent, scale-invariant normal fields. This eliminates the need for manually selecting the scale-related parameters or ground-truth labels required by existing supervised or PCA-based methods. Second, the integration of a cylinder-based distance measurement allows for robust spatial averaging along the normal fields derived from SDF, effectively decoupling true deformation signals from high-frequency surface roughness. NeuralGF (Li et al., 2023) was adopted to construct the neural network to train the SDF for deriving the normal fields from the tunnel surface point cloud data in this work. 

## 2. Environmental settings
The code is implemented in the following environmental settings:
- Ubuntu==24.04
- CUDA==11.7
- Python==3.9 
- numpy==1.24.4
- matplotlib==3.9.2
- pytorch3d==0.7.5
- torch==2.0.1
- torchvision==0.15.2

The tests are conducted on a desktop with an Intel Core i9 14900K CPU, NVIDIA RTX4080 Super GPU.


## 3. Dataset 
Users can download the dataset from https://drive.google.com/file/d/1qIlNrDqRbB83wvPal7Q3G73w63hbJheW/view?usp=sharing to reproduce the results. This dataset can be used as input to the whole NIFCyl structure from the beginning. Users can also download the data of the deformed regions introduced in Section 4.3 in the manuscript to quickly reproduce the results from https://drive.google.com/file/d/16McVLhwqvuAwQDdV7z7e3ZwFOEQYcgH8/view?usp=sharing. Users can download the data from https://drive.google.com/drive/folders/1gIjmi3HmG0DtM3mvGcod58b4wsNCkLSM?usp=sharing for the systemic uncertainty analysis of NIFCyl. Demo data related to experimental tests and scaling work can be downloaded from https://drive.google.com/drive/folders/1MohVzVi80GPG6n3YhXapCKkracC4XJLa?usp=sharing and https://drive.google.com/drive/folders/10atHa-Tb-0URsLxkUA1Gg2_vPc3Ooq-e?usp=sharing, separately.

## 4. Implementation steps
### 4.1 Prepare the dataset 
The dataset of the tunnel point cloud can be downloaded from https://drive.google.com/file/d/1qIlNrDqRbB83wvPal7Q3G73w63hbJheW/view?usp=sharing. The point cloud is stored in a .txt file. It contains seven columns. The first three columns are the original coordinates with five mathematically reconstructed regions as described in Section 3.1.1 of the manuscript. Columns 4 to 6 are coordinates of the deformed point cloud and the last column is the analytically derived deformation values (unit=m) as described in Section 3.1.2. 

### 4.2 Train the SDF function 
Before inputting the data into the neural network, the format should be changed. The .txt file of the point cloud should be changed to .xyz format. This can be done by using the open-source software CloudCompare or by the provided Python script named txt2xyz.py. The first three columns will be used to train the SDF. Users should create a new folder /Tunnel/ and put the processed .xyz point cloud into the folder and indicate the file path (your own directory of ".../Tunnel/") in the train_test.py (find '--dataset_root') Python script. Create a new folder named "list" inside /folder/ as /folder/list/, create a new .txt file named "testset_Tunnel.txt" in it and write the file name of the .xyz point cloud (without .xyz) into it. Use the following command to train the model:
- python train_test.py --mode=train --gpu=0 --data_set=Tunnel --max_iter=20000
After training, the result will be stored in .../dataset/log/, in the name format of yymmdd_hhmmss_Tunnel.

### 4.3 Derive normal fields 
Use the following command to derive the normal fields.
- python train_test.py --mode=test --gpu=0 --data_set=Tunnel --ckpt_dir=yymmdd_hhmmss_Tunnel --ckpt_iter=20000 --save_normal_xyz=True
- Please remember to use your own ckpt_dir from the previous step to replace yymmdd_hhmmss_Tunnel. 

### 4.4 Merge data columns
The normal fields result is stored in .../nifcyl/yymmdd_hhmmss_Tunnel/log/test_20000/pred_normal/. The last three columns are the normal vector for each point. However, the first three columns are normalized coordinates of the reference point cloud. There are several options to restore the original coordinates. One of the solutions is to concatenate the normal columns to the original input point cloud data (the txt file) as the last three columns. This can be done using CloudCompare or the provided Python script named merge.py. 
### 4.5 Calculate deformation and evaluation
Indicate path of the merged file derived from the previous step in the python script named deform_nifcyl.py and calculate the deformation values. For validation purposes, users can cut out the five deformed regions and merge together using CloudCompare. The point cloud of the deformed regions used in this study can be downloaded from https://drive.google.com/file/d/16McVLhwqvuAwQDdV7z7e3ZwFOEQYcgH8/view?usp=sharing. Users can quickly reproduce our NIFCyl results and the scatter plots.

## 5. Acknowledgements 
We acknowledge the work of the NeuralGF framework and the reference of NeuralGF is as the following:
@inproceedings{li2023neuralgf,
  title={{NeuralGF}: Unsupervised Point Normal Estimation by Learning Neural Gradient Function},
  author={Li, Qing and Feng, Huifang and Shi, Kanle and Gao, Yue and Fang, Yi and Liu, Yu-Shen and Han, Zhizhong},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023}
} 

The related license of NeuralGF is as follows:

MIT License

Copyright (c) 2023 Leo Q. Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
