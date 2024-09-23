# batch size
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 64 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 128 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 256 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 512 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix
# pre-training epochs
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 200 --save_point 150 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 300 --save_point 200 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 500 --save_point 300 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix
# batch shuffle
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
# mask_rot
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mix_maskrot
# minimum crop ratio of FullRot
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.2 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.4 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.8 --mix_method wrmix &&
# minimum crop ratio of WRMix
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mixup_mincrop_ratio 0.1 &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mixup_mincrop_ratio 0.2 &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mixup_mincrop_ratio 0.4 &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mixup_mincrop_ratio 0.8
# base bias of WRMix
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mix_base_bias 0.1 &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mix_base_bias 0.3 &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mix_base_bias 0.4 &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mix_base_bias 0.5 &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mix_base_bias 0.7 &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix --mix_base_bias 0.9
# model
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnext50_32x4d --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name regnety_3200mf --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name regnetx_3200mf --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name efficientnet_b0 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name shufflenet_v2_x2_0 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name mobilenetv3_large --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name mobilenetv3_small --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name vgg11 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix
# crop mode
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --crop_mode 3 --mix_method wrmix &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --batch_shuffle --num_classes 12 --views 2 --mincrop_ratio 0.6 --crop_mode 0 --mix_method wrmix