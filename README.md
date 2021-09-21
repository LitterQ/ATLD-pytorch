# Improving Model Robustness with Latent Distribution Locally and Globally

## Introduction
This is the implementation of the
["Improving Model Robustness with Latent Distribution Locally and Globally"].

The codes are implemented based on the released codes from ["Feature-Scattering Adversarial Training"](https://papers.nips.cc/paper/8459-defense-against-adversarial-attacks-using-feature-scattering-based-adversarial-training.pdf)

## Usage
### Installation
The training environment (PyTorch and dependencies) can be installed as follows:
```
python setup.py install
```
Tested under Python 3.6.2 and PyTorch 1.2.0.

### Train
Enter the folder named by the dataset you want to train. Specify the path for saving the trained models in ```fs_train.sh```, and then run
```
sh ./fs_train_cfiar10.sh  # for CIFAR-10 dataset
sh ./fs_train_svhn.sh     # for SVHN dataset
sh ./fs_train_cfiar100.sh # for CIFAR-100 dataset

```

### Evaluate
Specify the path to the trained models to be evaluated in ```fs_eval.sh``` and then run, using CIFAR-10 as a example. 
``` param: --init_model_pass: The load number of checkpoint, Possible values: `latest` for the last checkpoint, `199` for checkpoint-199 ``` 
```param: --attack_method_list: The attack list for evaluation, Possible values: `natural` for natural data, `fgsm`, `pgd`, `cw` ```
```
sh ./fs_eval_cifar10.sh      # for standard evaluation (ATLD-)
sh ./fs_eval_imt_cifar10.sh  # for evaluation with proposed IMT (ATLD+)

```
(For ATLD test, for example on CIFAR-10, simply comment the line 168-171 in attack_methods_new_imt_cifar10.py, then sh ./fs_eval_imt_cifar10.sh the output accuracy is changed from ATLD+ to ATLD)

### Reference Model
A reference model trained on CIFAR-10 is [here](https://drive.google.com/file/d/18NOtz_z29iMKdv92xTkXhZLVeCvg0N_o/view?usp=sharing).

A reference model trained on CIFAR-100 is [here](https://drive.google.com/file/d/1ZlNAqWvajZfNEQifTWqbBJjPdcXIjKhY/view?usp=sharing).

A reference model trained on SVHN is [here](https://drive.google.com/file/d/1T-fejNJFxNJzAiPtvmWIIOfYa9JLP7ea/view?usp=sharing).



## Reference
Haichao Zhang and JianyuWang. Defense against adversarial attacks using feature scattering-based adversarial training. In Advances in Neural Information Processing Systems, pp. 1829–1839, 2019.
