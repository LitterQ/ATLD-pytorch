export PYTHONPATH=./:$PYTHONPATH
#model_dir=./models/feature_scatter_cifar10/
model_dir=./models/test_v7_4/
CUDA_VISIBLE_DEVICES=4 python fs_eval_imt_cifar100.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=fgsm-pgd-cw \
    --dataset=cifar100\
    --batch_size_test=80 \
    --resume
