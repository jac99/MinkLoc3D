# MinkLoc3D: Point Cloud Based Large-Scale Place Recognition

[MinkLoc3D: Point Cloud Based Large-Scale Place Recognition](http://arxiv.org/abs/2011.04530) WACV 2021

[Jacek Komorowski](mailto:jacek.komorowski@pw.edu.pl)

Warsaw University of Technology

![Overview](media/overview.jpg)

### Introduction
The paper presents a learning-based method for computing a discriminative 3D point cloud descriptor for place recognition purposes. 
Existing methods, such as PointNetVLAD, are based on unordered point cloud representation. They use PointNet as the first processing step to extract local features, which are later aggregated into a global descriptor. 
The PointNet architecture is not well suited to capture local geometric structures. Thus, state-of-the-art methods enhance vanilla PointNet architecture by adding different mechanism to capture local contextual information, such as graph convolutional networks or using hand-crafted features. 
We present an alternative approach, dubbed **MinkLoc3D**, to compute a discriminative 3D point cloud descriptor, based on a sparse voxelized point cloud representation and sparse 3D convolutions.
The proposed method has a simple and efficient architecture. Evaluation on standard benchmarks proves that MinkLoc3D outperforms current state-of-the-art.  

### Citation
If you find this work useful, please consider citing:

    @inproceedings{kom21ml,
    title={MinkLoc3D: Point Cloud Based Large-Scale Place Recognition},
    author={Jacek Komorowski},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision},
    year={2021}
    }

### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch 1.6 and MinkowskiEngine 0.4.3 on Ubuntu 18.04 with CUDA 10.2.
Other dependencies include:
* Python 1.6
* MinkowskiEngine 0.4.3

To install requirements:

```setup
pip install -r requirements.txt
```

### Datasets

**MinkLoc3D** is trained using a subset of Oxford RobotCar and In-house (U.S., R.A., B.D.) datasets introduced in
*PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition* paper ([link](https://arxiv.org/pdf/1804.03492)).
For dataset description see PointNetVLAD paper or github repository ([link](https://github.com/mikacuy/pointnetvlad)).

You can download training and evaluation datasets from 
[here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q) 
([alternative link](https://drive.google.com/file/d/1-1HA9Etw2PpZ8zHd3cjrfiZa8xzbp41J/view?usp=sharing)). 
Extract the folder on the same directory as the project code. Thus, on that directory you must have two folders: 1) benchmark_datasets and 2) MinkLoc3D

Before the network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor. 
 
```generate pickles
cd generating_queries/ 

# Generate training tuples for 'baseline' scenario
python generate_training_tuples_baseline.py

# Generate training tuples for 'refined' scenario
python generate_training_tuples_refine.py

# Generate evaluation tuples
python generate_test_sets.py
```

### Training
To train **MinkLoc3D** detector, download and decompress the dataset, edit `config.txt` and set paths to downloaded datasets.
Then, run:

```train baseline
python train_detector.py --config config.txt
```

```train refined
python train_detector.py --config config.txt
```

### Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 


### Evaluation
The pre-trained model `model_20201019_1416_final.pth` is saved in `models/` folder.
The model was trained with ISSIA-CNR dataset (cameras 1,2,3,4) and SoccerPlayerDetection dataset (set 1).
To run the trained model use the following command:

```eval baseline
python eval.py --model-file mymodel.pth --benchmark imagenet
```

```eval refined
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Results

**MinkLoc3D** performance (measured by Average Precision@1\%) compared to state-of-the-art:

### Trained on Baseline Dataset

| Method         | Oxford  | U.S. | R.A. | B.D |
| ------------------ |---------------- | -------------- |---|---|
| PointNetVLAD  |     80.3     |   72.6 | 60.3 | 65.3 |
| PCAN  |     83.8     |   79.1 | 71.2 | 66.8 |
| DAGC  |     87.5     |   83.5 | 75.7 | 71.2 |
| LPD-Net  |     94.9   |   **96.0** | 90.5 | **89.1** |
| **MinkLoc3D (our)**  |     **97.9**     |   95.0 | **91.2** | 88.5 |


### Trained on Refined Dataset

| Method         | Oxford  | U.S. | R.A. | B.D |
| ------------------ |---------------- | -------------- |---|---|
| PointNetVLAD  |     80.1     |   94.5 | 93.1 | 86.5 |
| PCAN  |     86.4     |   94.1 | 92.3 | 87.0 |
| DAGC  |     87.8     |   94.3 | 93.4 | 88.5 |
| LPD-Net  |     94.9     |   98.9 | 96.4 | 94.4 |
| **MinkLoc3D (our)**  |     **98.5**     |   **99.7** | **99.3** | **96.7** |




### License
Our code is released under the MIT License (see LICENSE file for details).
