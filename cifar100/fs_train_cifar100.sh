export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/test_v7_6/
mkdir -p $model_dir
CUDA_VISIBLE_DEVICES=7 python fs_main_cifar100.py \
    --resume \
    --adv_mode='feature_scatter' \
    --lr=0.1 \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --max_epoch=400 \
    --save_epochs=10 \
    --decay_epoch1=60 \
    --decay_epoch2=90 \
    --batch_size_train=50 \
    --dataset=cifar100

