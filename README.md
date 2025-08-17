# SNN-DVS

Files for data preprocessing (need to be evaluated before training):
* NCars_corrected.ipynb
* NCars_corrected_trilinear.ipynb
* nCaltech101_corrected.ipynb
* nCaltech101_corrected_trilinear.ipynb
* MINST-DVS_dataset_corrected.ipynb
* MINST-DVS_dataset_corrected_trilinear.ipynb
* CIFAR10-DVS_corrected.ipynb
* CIFAR10-DVS_corrected_trilinear.ipynb
> [!WARNING]
> preprocessing makes .npy files that can be huge. Please be aware of the amount of free memory space.
Files for training ResNets:
* CIFAR10-DVS_resnet...
* NCarsReLU_resnet...
* nCaltech_resnet...
* MNIST-DVS_resnet...

there are multiple files starting with those names rest of the filename describes how model was trained for example "ReLU1_andReLUmaxpool" means there were used ReLU1 activation function besides layers strictly before maxpool. Trilinear/exp are types of kernels used in ESP trilinear and alpha.

Files for TTFS representation evaluation:
* CIFAR10-DVS_TTFS...
* NCarsReLU_TTFS...
* nCaltech_TTFS...
* MNIST-DVS_TTFS...


Folder structure of datasets:
```
Datasety/
│
├── CIFAR10-DVS/
│   ├── airplane/
│   ├── automobile/
│   ├── ...
│   └── truck/
│
├── NMINST/          <---- tutaj pomyłka w nazewnictwie (to jest folder na MINST-DVS)
│   ├── grabbed_data0/
│   ├── ...
│   └── grabbed_data9/
│
├── NCaltech/
│   ├── Caltech101/
│   │   ├── accordion/
│   │   ├── ...
│   │   └── yin_yang/
│   └── Caltech101_annotations/
├── prophesee_automotive_dataset_toolbox_master/
└── Prophesee_Dataset_n_cars/
    ├── n-cars_test/
    ├── ...
    └── n-cars_train/
```

