# An unofficial PoinetNet
This is an unofficial PointNet with Tensorflow2.

## Performance on ModelNet40 classification dataset

| set      | Accuracy      |
| :---:    | :---:    |
| test    |88+         | 

## Performance on ShapeNet part segmentation dataset

| mIOU | Airplane | Bag| Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 83.5 |  82.0 | 82.7 | 82.3| 75.5 |89.7| 69.3| 91.3| 86.0| 80.6| 94.7| 66.0| 91.7| 82.2| 54.2| 72.3| 80.6| 

## Performance on Saturn multi-label classification dataset

| set      | Accuracy      |
| :---:    | :---:    |
| test    |100         | 

## Performance on Saturn part segmentation dataset

| set      | Accuracy      |
| :---:    | :---:    |
| test    |100         | 

## My Environment
- Operating System:
  - Archlinux
- CUDA:
  - CUDA V11.5.50 
- Nvidia driver:
  - 495.44
- Python:
  - python 3.6.5
- Python package:
  - h5py...
- Tensorflow:
  - tensorflow-2.6.0

## Downloading the ModelNet40 Classification Dataset
[ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)
## Downloading the Pretrain Model
[Model](https://drive.google.com/drive/u/2/folders/1n_sgQsFyKiMMZ-0MZ1XMP5mSmS_et9CU)


## For training
```
cd ./Classification
```
```
python main.py
```

## For testing
```
cd ./Classification
```
```
python main.py --phase='test'
```
## For visualization
```
cd ./Classification
```
```
python main.py --phase='test' --vis=1
```

## Downloading the ShapeNet Part Segmentation Dataset
[ShapeNet-Org](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip)
[ShapeNet-h5](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip)

## Downloading the Pretrain Model
[Model](https://drive.google.com/drive/u/2/folders/1pK490CRc3kbJ01US-xt_dLNUVpkjJYCp)

## For training
```
cd ./PartSeg
```
```
python main.py
```

## For testing (sample)
```
cd ./PartSeg
```
```
python main.py --phase='test'
```
## For testing (original)
```
cd ./PartSeg
```
```
python test_all.py
```
## For visualization
```
cd ./PartSeg
```
```
python main.py --phase='test' --vis=1
```
```
python test_all.py --vis=1
```

## Downloading the Saturn Multi-label Classification Dataset
[Saturn](https://drive.google.com/drive/u/2/folders/1gVkqTfjYX34Ul6iGxQ0i3SZZd338yPNP)

## Downloading the Pretrain Model
[Model](https://drive.google.com/drive/u/2/folders/1gVkqTfjYX34Ul6iGxQ0i3SZZd338yPNP)

## For training
```
cd ./Toy/Classification
```
```
python main.py
```

## For testing
```
cd ./Toy/Classification
```
```
python main.py --phase='test'
```
## For visualization
```
cd ./Toy/Classification
```
```
python main.py --phase='test' --vis=1
```

## Downloading the Saturn Part Segmentation Dataset
[Saturn](https://drive.google.com/drive/u/2/folders/1cKCUERmJrSbexCgpucXJs5S7EbUBe0A2)

## Downloading the Pretrain Model
[Model](https://drive.google.com/drive/u/2/folders/1cKCUERmJrSbexCgpucXJs5S7EbUBe0A2)

## For training
```
cd ./Toy/PartSeg
```
```
python main.py
```

## For testing 
```
cd ./Toy/PartSeg
```
```
python main.py --phase='test'
```
## For visualization
```
cd ./Toy/PartSeg
```
```
python main.py --phase='test' --vis=1
```


## Part Segmentation Sample Results from Saturn and ShapeNet
![alt text](https://github.com/johnnylu305/Pointnet-Tensorflow-2/blob/main/Figures/Airplane_35.png?raw=true)
![alt text](https://github.com/johnnylu305/Pointnet-Tensorflow-2/blob/main/Figures/Chair_68.png?raw=true)
![alt text](https://github.com/johnnylu305/Pointnet-Tensorflow-2/blob/main/Figures/Motorbike_1.png?raw=true)
![alt text](https://github.com/johnnylu305/Pointnet-Tensorflow-2/blob/main/Figures/saturn_0.png?raw=true)
![alt text](https://github.com/johnnylu305/Pointnet-Tensorflow-2/blob/main/Figures/saturn_8.png?raw=true)
![alt text](https://github.com/johnnylu305/Pointnet-Tensorflow-2/blob/main/Figures/saturn_9.png?raw=true)


## References

1. [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

2. [Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. 2017b. PointNet: Deep
Learning on Point Sets for 3D Classification and Segmentation. In Proc. CVPR.](https://arxiv.org/abs/1612.00593)

