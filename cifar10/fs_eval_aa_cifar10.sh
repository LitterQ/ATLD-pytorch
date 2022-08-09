export PYTHONPATH=./:$PYTHONPATH
#model_dir=./models/feature_scatter_cifar10/
model_dir=./models/backup_test_v7_cifar10_re/
#model_dir=./models/test9/
CUDA_VISIBLE_DEVICES=3 python fs_eval_aa_cifar10_test.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=pgd-cw \
    --dataset=cifar10\
    --batch_size_test=80 \
    --resume
