set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0

delay_values=(0 5 15 25)

for value in "${delay_values[@]}"
do
  python main.py --env RoadRunnerNoFrameskip-v4 --case atari --opr test --seed 0 --num_gpus 1 --num_cpus 20 --force \
  --test_episodes 32 \
  --load_model \
  --amp_type 'torch_amp' \
  --model_path 'model.p' \
  --info 'Test' \
  --delay $value \
  --use_forward
done
