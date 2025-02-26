# SNN-DVS
SNN-DVS

Pliki stricte do TTFS:

* SNN_test.ipnb - TTFS standardowe
* INN.ipynb - TTFS ale użyte zostały interwały do obliczenia zakresów wyjsć
* Hardtanh.ipynb - TTFS ale z Hard Tanh (w tym kodowanie zarówno dodatniich jak i ujemnych wartości)

Modele ReLU:

MINST-DVS:
* MINST-DVS_dataset.ipynb - preprocesing zamiana na ramki
* MINST-DVS_resnet.ipynb - resnet
* MINST-DVS_resnet_Hardtanh.ipynb - resnet z Hard tanh funkcjami aktywacji
* MINST-DVS_resnet_Hardtanh_and_ReLUmaxpool.ipynb - resnet z Hard tanh funkcjami aktywacji ale ReLU przed warstwami maxpool (model wykorzystywany w pliku Hardtanh.ipynb)
* MINST-DVS_resnet_withmaxpool_same.ipynb - wytrenowana sieć wykorzystywana do pomiaru wartości $t_{max}$ w INN.ipynb

NCars:
* NCars.ipynb - preprocessing
* nCarsReLU_resnet.ipynb - podstawowy resnet bez EST (tylko rzutowanie na ramkę)
* nCarsReLU_resnet_2FC_EST.ipynb - Resnet z EST + dodatkowa warstwa fully connected

NCaltech:
* nCaltech101.ipynb - preprocessing
* nCaltech_resnet_aug.ipynb - Resnet z augmentacją danych
* nCaltech_resnet_aug_cross_val.ipynb - Resnet z augmentacją i cross validacją
* NCaltech_GoogLeNet_aug.ipynb - GoogLeNet z augmentacją

CIFAR10-DVS:
* nCIFAR10.ipynb - preprocessing
