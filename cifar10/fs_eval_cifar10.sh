export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/test_v7_cifar10_re/
CUDA_VISIBLE_DEVICES=7 python fs_eval_cifar10.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=pgd-cw \
    --dataset=cifar10\
    --batch_size_test=80 \
    --resume
