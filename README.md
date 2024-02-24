# Noise-Robust Continual Test-Time Domain Adaptation

> code for [Noise-Robust Continual Test-Time Domain Adaptation](https://dl.acm.org/doi/abs/10.1145/3581783.3612071), published in the 31st ACM International Conference on Multimedia (ACM MM), 2023

## abstract
Abstract: Continual test-time domain adaptation (TTA) is a challenging topic in the field of source-free domain adaptation, which focuses on addressing cross-domain multimedia information during inference with a continuously changing data distribution. Previous methods have been found to lack noise robustness, leading to a significant increase in errors under strong noise. In this paper, we address the noise-robustness problem in continual TTA by offering three effective recipes to mitigate it. At the category level, we employ the Taylor cross-entropy loss to alleviate the low confidence category bias commonly associated with cross-entropy. At the sample level, we reweight the target samples based on uncertainty to prevent the model from overfitting on noisy samples. Finally, to reduce pseudo-label noise, we propose a soft ensemble negative learning mechanism to guide the model optimization using ensemble complementary pseudo labels. Our method achieves state-of-the-art performance on three widely used continual TTA datasets, particularly in the strong noise setting that we introduced.

## migration
mainly use the msadapter and raw mindspore apis to transplant, the details about migration: cotta-transplant-note.md.

after the migration, the experiment results are the same as the resluts in the paper.

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
