CUDA_VISIBLE_DEVICES=1 nohup python3 main.py --config 'configs/ve/sr_ve.py' --mode 'train' --workdir VESDE > 1-ipsrsde.txt &

## GENERATE TFRECORDS FILES
python dataset_tool.py create_from_images /home/msantos/fasr/sample_imgs/tfrecords /home/msantos/fasr/sample_imgs/probe_HR --shuffle 0

nohup sh 2-lr-script_exec.sh >> 2-train-ip-lr.txt &

nohup sh 3-mixed-script_exec.sh >> 3-train-ip-mixed.txt &

nohup bash 04-ir101_script_quis_campi.sh >> 4-eval2_ir101.txt &
---------------------------------------------------------------
nohup bash 01-IHR_script_quis_campi.sh >> 01-eval-ckpt16.txt &

nohup bash 01-IHR_snr_script_celeba.sh >> 01-eval-celeba-snr-txt &

nohup bash 01-script_quis_campi.sh >> 01-eval_ip_hr.txt &

nohup bash 02-ILR_script_quis_campi.sh >> 02-eval_ip_lr-txt &

nohup bash 03-ICOMBINED_script_quis_campi.sh >> 03-eval_ip_mixed.txt &

nohup python3 main.py --config 'configs/ve/sr_ve.py' --mode 'train' --workdir 4-ir_101 >> 4-train_101_web.txt &

CUDA_VISIBLE_DEVICES=1 nohup python3 main.py --config 'configs/ve/sr_ve.py' --mode 'sr' --workdir 1-IP-input-hr --eval_folder 6-s-0001-celeba >> 1-eval-celeba.txt &

CUDA_VISIBLE_DEVICES=1 nohup python3 main.py --config 'configs/ve/sr_ve.py' --mode 'sr' --workdir 4-ir_101 --eval_folder 8-s-0001-snr016-celeba >> 4-eval-celeba-snr016.txt &
-------------------------------------------------------------------------------------------------------
1 - VESDE

1.1 - TRAIN
CUDA_VISIBLE_DEVICES=1 python3 main.py --config 'configs/ve/sr_ve.py' --mode 'train' --workdir VESDE

1.2 - SR GENERATION
CUDA_VISIBLE_DEVICES=1 python3 main.py --config 'configs/ve/sr_ve.py' --mode 'sr' --workdir VESDE

2 - VPSDE

1.1 - TRAIN
CUDA_VISIBLE_DEVICES=1 python3 main.py --config 'configs/vp/sr_subvp.py' --mode 'train' --workdir VPSDE

1.2 - SR GENERATION
CUDA_VISIBLE_DEVICES=0 python3 main.py --config 'configs/vp/sr_vp.py' --mode 'sr' --workdir VPSDE

3 - SUBVPSDE

1.1 - TRAIN
CUDA_VISIBLE_DEVICES=1 python3 main.py --config 'configs/subvp/sr_subvp.py' --mode 'train' --workdir SUBVPSDE

1.2 - SR GENERATION
CUDA_VISIBLE_DEVICES=0 python3 main.py --config 'configs/subvp/sr_subvp.py' --mode 'sr' --workdir SUBVPSDE

