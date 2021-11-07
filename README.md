# An unofficial PoinetNet
This is an unofficial PointNet with Tensorflow2.

## Performance on ModelNet40 classification dataset

| set      | Accuracy      |
| :---:    | :---:    |
| test    |88.0~89.0         | 

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

## References

1. [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

2. [Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. 2017b. PointNet: Deep
Learning on Point Sets for 3D Classification and Segmentation. In Proc. CVPR.](https://arxiv.org/abs/1612.00593)

