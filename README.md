# Noise-Robust Continual Test-Time Domain Adaptation

> code for [Noise-Robust Continual Test-Time Domain Adaptation](https://dl.acm.org/doi/abs/10.1145/3581783.3612071), published in0the 31st ACM International Conference on Multimedia (ACM MM), 2023 Continual Test-Time Adaptation, About Mindspore migration of Continual Test-Time Domain Adaptation, published in CVPR 2022.

## migration
mainly use the msadapter and raw mindspore apis to transplant, the details about migration: cotta-transplant-note.md.

## Prerequisite 
 Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment. 
 ```bash 
 # It may take several minutes for conda to solve the environment 
 conda update conda 
 conda env create -f environment.yml 
 conda activate cotta  
 ``` 
  
 ## Classification Experiments 
 ### CIFAR10-to-CIFAR10C-standard task 
 ```bash 
 cd cifar 
 # This includes the comparison of all three methods as well as baseline 
 bash run_cifar10.sh  
 ``` 
 ### CIFAR10-to-CIFAR10C-gradual task 
 ```bash 
 bash run_cifar10_gradual.sh 
 ``` 
 ### CIFAR100-to-CIFAR100C task 
 ```bash 
 bash run_cifar100.sh 
 ``` 
  
 ### ImageNet-to-ImageNetC task  
 ```bash 
 cd imagenet 
 bash run.sh 
 ```
