# 28页 CIFAR-100数据集实验

# iCaRL + FixMatch + w/o drift
nohup python train_baseline.py --ckp_prefix cifar100_5%_baseline_wd --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar100 --num_classes 100 --nb_cl_fg 10 --nb_cl 10 --image_size 32 --k_shot 25 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --p_cutoff 0.95 --u_iter 100 --buffer_size 5120 2>&1 > cifar100_5%_baseline_wd.log &

# iCaRL + FixMatch
nohup python train_baseline.py --ckp_prefix cifar100_5%_baseline --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar100 --num_classes 100 --nb_cl_fg 10 --nb_cl 10 --image_size 32 --k_shot 25 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --p_cutoff 0.95 --u_iter 100 --buffer_size 5120 --add_drift 2>&1 > cifar100_5%_baseline.log &

# this work
nohup python train.py --ckp_prefix cifar100_5% --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar100 --num_classes 100 --nb_cl_fg 10 --nb_cl 10 --image_size 32 --k_shot 25 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift 2>&1 > cifar100_5%.log &

# 28页 BloodMNIST数据集实验

# iCaRL + FixMatch
nohup python train_baseline.py --ckp_prefix bloodmnist_5%_baseline --gpu 0 --epochs 100 --epochs_new 100 --u_ratio 7 --dim 512 --dataset bloodmnist --num_classes 8 --nb_cl_fg 2 --nb_cl 2 --image_size 28 --k_shot_rate 0.05 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --p_cutoff 0.95 --u_iter 100 --buffer_size 5120 2>&1 > bloodmnist_5%_baseline.log &

# this work
nohup python train.py --ckp_prefix bloodmnist_5% --gpu 0 --epochs 100 --epochs_new 100 --u_ratio 7 --dim 512 --dataset bloodmnist --num_classes 8 --nb_cl_fg 2 --nb_cl 2 --image_size 28 --k_shot_rate 0.05 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 2>&1 > bloodmnist_5%.log &

# 29页 PathMNIST数据集实验

# iCaRL + FixMatch
nohup python train_baseline.py --ckp_prefix pathmnist_5%_baseline --gpu 0 --epochs 100 --epochs_new 100 --u_ratio 7 --dim 512 --dataset pathmnist --num_classes 9 --nb_cl_fg 3 --nb_cl 3 --image_size 28 --k_shot_rate 0.05 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --p_cutoff 0.95 --u_iter 100 --buffer_size 5120 2>&1 > pathmnist_5%_baseline.log &

# this work
nohup python train.py --ckp_prefix pathmnist_5% --gpu 0 --epochs 100 --epochs_new 100 --u_ratio 7 --dim 512 --dataset pathmnist --num_classes 9 --nb_cl_fg 3 --nb_cl 3 --image_size 28 --k_shot_rate 0.05 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 2>&1 > pathmnist_5%.log &

# 29页 核心模块消融实验

# M0
nohup python train_baseline.py --ckp_prefix m0 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --p_cutoff 0.95 --u_iter 100 --buffer_size 5120 --add_drift 2>&1 > m0.log &

# M1
nohup python train_m1.py --ckp_prefix m1 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift 2>&1 > m1.log &

# M2
nohup python train_m2.py --ckp_prefix m2 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift 2>&1 > m2.log &

# M3
nohup python train.py --ckp_prefix m3 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift 2>&1 > m3.log &


# 30页 损失权重超参数分析实验

# lambda_2

nohup python train.py --ckp_prefix lambda_cons_0.1 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_cons 0.1 2>&1 > lambda_cons_0.1.log &

nohup python train.py --ckp_prefix lambda_cons_0.5 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_cons 0.5 2>&1 > lambda_cons_0.5.log &

nohup python train.py --ckp_prefix lambda_cons_1.5 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_cons 1.5 2>&1 > lambda_cons_1.5.log &

nohup python train.py --ckp_prefix lambda_cons_2.0 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_cons 1.5 2>&1 > lambda_cons_2.0.log &

# lambda_3

nohup python train.py --ckp_prefix lambda_kd_0.1 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_kd 0.1 2>&1 > lambda_kd_0.1.log &

nohup python train.py --ckp_prefix lambda_kd_0.5 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_kd 0.5 2>&1 > lambda_kd_0.5.log &

nohup python train.py --ckp_prefix lambda_kd_1.5 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_kd 1.5 2>&1 > lambda_kd_1.5.log &

nohup python train.py --ckp_prefix lambda_kd_2.0 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_kd 1.5 2>&1 > lambda_kd_2.0.log &

# lambda_4

nohup python train.py --ckp_prefix lambda_sep_0.1 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_sep 0.1 2>&1 > lambda_sep_0.1.log &

nohup python train.py --ckp_prefix lambda_sep_0.5 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_sep 0.5 2>&1 > lambda_sep_0.5.log &

nohup python train.py --ckp_prefix lambda_sep_1.5 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_sep 1.5 2>&1 > lambda_sep_1.5.log &

nohup python train.py --ckp_prefix lambda_sep_2.0 --gpu 0 --epochs 200 --epochs_new 200 --u_ratio 7 --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --u_iter 100 --buffer_size 5120 --add_drift --lambda_sep 1.5 2>&1 > lambda_sep_2.0.log &