set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1

python main.py --env QbertNoFrameskip-v4 --case atari --opr train --force \
  --num_gpus 2 --num_cpus 20 --cpu_actor 8 --gpu_actor 8 \
  --seed 0 \
  --p_mcts_num 4 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'Test' \
  --delay 15 \
  --use_forward \
  --steps_transitions 130000 \
  --stochastic_delay \
  --load_model \
  --model_path 'model.p'
