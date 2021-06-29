export PYTHONPATH=./:$PYTHONPATH
#model_dir=./models/feature_scatter_cifar10/
model_dir=./models/test_v7_svhn_re/
#model_dir=./models/test_v5_svhn_re/
CUDA_VISIBLE_DEVICES=7 python fs_eval_svhn.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=natural-fgsm-pgd-cw \
    --dataset=svhn\
    --batch_size_test=80 \
    --resume
