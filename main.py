import argparse
import logging.config
import os
import wandb

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir, set_seed
if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='EfficientZero')
    parser.add_argument('--env', default='StarGunnerNoFrameskip-v4', help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', default='atari', choices=['atari'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', default='train', choices=['train', 'test'])
    parser.add_argument('--amp_type', default='torch_amp', choices=['torch_amp', 'none'],
                        help='choose automated mixed precision type')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If enabled, logs additional values  '
                             '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--save_video', action='store_true', default=False, help='save video in test.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--cpu_actor', type=int, default=8, help='batch cpu actor')
    parser.add_argument('--gpu_actor', type=int, default=3, help='batch bpu actor')
    parser.add_argument('--p_mcts_num', type=int, default=4, help='number of parallel mcts')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--num_gpus', type=int, default=1, help='gpus available')
    parser.add_argument('--num_cpus', type=int, default=96, help='cpus available')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=0.99,
                        help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--use_root_value', action='store_true', default=False,
                        help='choose to use root value in reanalyzing')
    parser.add_argument('--use_priority', action='store_true', default=True,
                        help='Uses priority for data sampling in replay buffer. '
                             'Also, priority for new data is calculated based on loss (default: False)')
    parser.add_argument('--use_max_priority', action='store_true', default=True, help='max priority')
    parser.add_argument('--test_episodes', type=int, default=10, help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='use augmentation')
    parser.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+',
                        choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity'],
                        help='Style of augmentation')
    parser.add_argument('--info', type=str, default='EfficientZero-V1', help='debug string')
    parser.add_argument('--load_model', action='store_true', default=False, help='choose to load model')
    parser.add_argument('--model_path', type=str, default='./results/test_model.p', help='load model path')
    parser.add_argument('--object_store_memory', type=int, default=150 * 1024 * 1024 * 512, help='object store memory')
    parser.add_argument('--delay', type=int, default=15, help='execution delay value')
    parser.add_argument('--stochastic_delay', action='store_true', default=False, help='stochastic delay values between 0 and delay')
    parser.add_argument('--p_prob', type=float, default=0.2, help='execution delay value')
    parser.add_argument('--use_forward', action='store_true', default=False, help='use forward in operations if delay>0')
    parser.add_argument('--steps_transitions', type=int, default=130 * 1000, help='number of training steps and total transitions')

    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
    assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
        ' Revisit policy search rate should be in [0,1]'

    if args.opr == 'train':
        ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus,
                 object_store_memory=args.object_store_memory)
    else:
        ray.init()

    # seeding random iterators
    set_seed(args.seed)

    stochastic_delays = '_stoch' if args.stochastic_delay else "_const"
    env_name_short = f"{(args.env.split('NoFrameskip'))[0]}_{args.delay}{stochastic_delays}"
    if args.opr == 'train':
        dic = vars(args)
        wandb.init(config=dic)
        wandb.run.name = env_name_short + ' use forward=' + str(args.use_forward)

    # import corresponding configuration , neural networks and envs
    if args.case == 'atari':
        from config.atari import game_config
    else:
        raise Exception('Invalid --case option')

    # set config as per arguments
    exp_path = game_config.set_config(args, env_name_short)
    print('game_config.training_steps:', game_config.training_steps)

    exp_path, log_base_path = make_results_dir(exp_path, args)

    # set-up logger
    init_logger(log_base_path)
    logging.getLogger('train').info('Path: {}'.format(exp_path))
    logging.getLogger('train').info('Param: {}'.format(game_config.get_hparams()))

    device = game_config.device
    try:
        if args.opr == 'train':
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            if args.load_model and os.path.exists(args.model_path):
                model_path = args.model_path
            else:
                model_path = None
            model, weights = train(game_config, summary_writer, model_path)
            model.set_weights(weights)
            total_steps = game_config.training_steps + game_config.last_steps
            test_score, _, test_path = test(game_config, model.to(device), total_steps, game_config.test_episodes, device,
                                            render=False, save_video=args.save_video, final_test=True, use_pb=True,
                                            delay=True, forward=game_config.test_use_forward)
            mean_score = test_score.mean()
            std_score = test_score.std()

            test_log = {
                'mean_score': mean_score,
                'std_score': std_score,
            }
            for key, val in test_log.items():
                summary_writer.add_scalar('train/{}'.format(key), np.mean(val), total_steps)

            print('test dict is not None', mean_score)
            wandb.log({"(Delay) Mean score": mean_score})

            test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
                       ''.format(total_steps, env_name_short, mean_score, test_score.max(), test_score.min(), std_score)
            logging.getLogger('train_test').info(test_msg)
            if args.save_video:
                logging.getLogger('train_test').info('Saving video in path: {}'.format(test_path))
        elif args.opr == 'test':
            assert args.load_model
            if args.model_path is None:
                model_path = game_config.model_path
            else:
                model_path = args.model_path
            assert os.path.exists(model_path), 'model not found at {}'.format(model_path)

            model = game_config.get_uniform_network().to(device)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

            '''
            test_score, _, test_path = test(game_config, model, 0, args.test_episodes, device=device, render=args.render, save_video=args.save_video, final_test=True, use_pb=True)
            mean_score = test_score.mean()
            std_score = test_score.std()
            print(f'test_score non delayed = {test_score}')
            logging.getLogger('test').info('Test Mean Score: {} (max: {}, min: {})'.format(mean_score, test_score.max(), test_score.min()))
            logging.getLogger('test').info('Test Std Score: {}'.format(std_score))
            if args.save_video:
                logging.getLogger('test').info('Saving video in path: {}'.format(test_path))
            '''
            print(f'Testing delay: {game_config.delay} and forward: {args.use_forward}')
            # Testing on delayed environment
            test_score, _, test_path = test(game_config, model, 0, args.test_episodes, device=device,
                                            render=args.render,
                                            save_video = args.save_video, final_test = True, use_pb = True,
                                            delay=True, forward=args.use_forward)
            mean_score = test_score.mean()
            median_score = np.median(test_score)
            std_score = test_score.std()
            print(f'Delayed Mean Score = {mean_score} , Std Score = {std_score} , Median Score = {median_score}')

            logging.getLogger('test').info(
                'Delay={} Test Mean Score: {} (max: {}, min: {}). Test Median Score: {}'.format(args.delay, mean_score,
                                                                                            test_score.max(),
                                                                                            test_score.min(),
                                                                                            median_score))
            logging.getLogger('test').info('Delay Test Std Score: {}'.format(std_score))

        else:
            raise Exception('Please select a valid operation(--opr) to be performed')
        ray.shutdown()
    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)
