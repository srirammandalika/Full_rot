# CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 200 --save_point 150 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6 &&
# CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 200 --save_point 150 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix


# Remove any CUDA references and make sure the script uses MPS or CPU.

python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 2 --save_point 150 --batch_size 16 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6 &&
python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 2 --save_point 150 --batch_size 16 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix

# # Use the below code for doing 20% training and 80% testing  
# python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 2 --save_point 150 --batch_size 16 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6
# python main.py --task classification --gpus 0 --setname STL10_unlabelled --num_workers 8 --epochs 2 --save_point 150 --batch_size 16 --lr 2e-4 --model_name resnet50 --optim adam --sup_method rotation --num_classes 12 --views 2 --mincrop_ratio 0.6 --mix_method wrmix
