export PYTHONPATH=./:$PYTHONPATH
#model_dir=./models/feature_scatter_cifar10/
#model_dir=./models/test_v5_svhn_re/
model_dir=./models/test_v7_svhn_re/
#model_dir=./models/test9/
CUDA_VISIBLE_DEVICES=3 python fs_eval_imt_svhn.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=cw\
    --dataset=svhn\
    --batch_size_test=80 \
    --resume
