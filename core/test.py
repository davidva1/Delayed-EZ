import os
import ray
import time
import torch
import wandb

import numpy as np
import core.ctree.cytree as cytree

from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst, update_envs_current_delay, sample_delay_queue, get_env_action_from_samples


@ray.remote(num_gpus=0.125)
def _test(config, shared_storage):
    test_model = config.get_uniform_network()
    best_test_score = float('-inf')
    episodes = 0
    while True:
        counter = ray.get(shared_storage.get_counter.remote())
        if counter >= config.training_steps + config.last_steps:
            time.sleep(30)
            break
        if counter >= config.test_interval * episodes:
            episodes += 1
            test_model.set_weights(ray.get(shared_storage.get_weights.remote()))
            test_model.eval()

            print(f'Testing on delayed environments with forward: {config.test_use_forward}')
            test_score, eval_steps, _ = test(config, test_model, counter, config.test_episodes, config.device, False,
                                             save_video=False, delay=True, forward=config.test_use_forward)

            mean_score = test_score.mean()
            median_score = np.median(test_score)
            std_score = test_score.std()
            print('Start evaluation at step {}.'.format(counter))
            if mean_score >= best_test_score:
                print('Saving new best model')
                best_test_score = mean_score
                torch.save(test_model.state_dict(), config.model_path)

            test_log = {
                'mean_score': mean_score,
                'median_score': median_score,
                'std_score': std_score,
                'max_score': test_score.max(),
                'min_score': test_score.min(),
            }
            shared_storage.add_test_log.remote(counter, test_log)
            print('Training step {}, delayed test scores: \n{} of mean {} and median {}.'.format(counter, test_score,
                                                                                                 mean_score, median_score))

            '''
            if config.delay > 0:
                # if delay > 0, test on few episodes also with delay = 0 with no forward
                print(f'Testing on non delayed environments')
                test_score, eval_steps, _ = test(config, test_model, counter, config.test_episodes//2,
                                                         config.device, False, save_video=False, delay=False, forward=False)
                mean_score = test_score.mean()
                median_score = np.median(test_score)
                print('Training step {}, non-delayed test scores: \n{} of mean {} and median {}.'.format(counter,test_score,
                                                                                                         mean_score, median_score))
            '''
        time.sleep(30)


def test(config, model, counter, test_episodes, device, render, save_video=False, final_test=False, use_pb=False,
         delay=False, forward=False):
    """evaluation test
    Parameters
    ----------
    model: any
        models for evaluation
    counter: int
        current training step counter
    test_episodes: int
        number of test episodes
    device: str
        'cuda' or 'cpu'
    render: bool
        True -> render the image during evaluation
    save_video: bool
        True -> save the videos during evaluation
    final_test: bool
        True -> this test is the final test, and the max moves would be 108k/skip
    use_pb: bool
        True -> use tqdm bars
    delay: bool
        True -> use a delayed environments
    """
    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))
    current_delay_value = config.delay
    min_delay = current_delay_value

    with torch.no_grad():
        # new games
        envs = [config.new_game(seed=i, save_video=save_video, save_path=save_path, test=True, final_test=final_test,
                              video_callable=lambda episode_id: True, uid=i, delay_enable=delay) for i in range(test_episodes)]
        print(f'Testing with delay: {delay}, forward: {forward}')
        max_episode_steps = envs[0].get_max_episode_steps()
        if use_pb:
            pb = tqdm(np.arange(max_episode_steps), leave=True)
        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [GameHistory(envs[_].env.action_space, max_length=max_episode_steps, config=config) for _ in range(test_episodes)]
        for i in range(test_episodes):
            game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])

        step = 0
        ep_ori_rewards = np.zeros(test_episodes)
        ep_clip_rewards = np.zeros(test_episodes)
        # loop
        while not dones.all():
            if render:
                for i in range(test_episodes):
                    envs[i].render()

            if config.image_based:
                stack_obs = []
                for game_history in game_histories:
                    stack_obs.append(game_history.step_obs())
                stack_obs = prepare_observation_lst(stack_obs)
                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = [game_history.step_obs() for game_history in game_histories]
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)

            with autocast():
                network_output = model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state
            reward_hidden_roots = network_output.reward_hidden
            value_prefix_pool = network_output.value_prefix
            policy_logits_pool = network_output.policy_logits.tolist()

            if config.test_use_forward:
                [envs[i].initialize_pending_actions(model, stack_obs[i], config) for i in range(test_episodes)]
            else:
                [envs[i].initialize_pending_actions() for i in range(test_episodes)]
            current_delay_value = update_envs_current_delay(envs, current_delay_value,
                                                            config.min_delay,
                                                            config.delay, p=0.2)

            # Useful when several actions queues are possible and we want to predict action for each one.
            action_queues = [[envs[i].get_pending_actions_for_agent() for i in range(test_episodes)]]

            policy_actions_per_env = [[] for _ in range(test_episodes)]
            for actions in action_queues:
                if delay and forward:
                    #actions = [env.get_pending_actions_for_agent() for env in envs]
                    for i in range(current_delay_value):
                        action_i = np.asarray([actions[env_ix][i] for env_ix in range(test_episodes)])
                        action_i = torch.from_numpy(action_i).to(device).unsqueeze(1).long()

                        hidden_states = torch.from_numpy(np.asarray(hidden_state_roots)).to(device).squeeze(0).float()
                        hidden_states_c_reward = torch.from_numpy(np.asarray(reward_hidden_roots[0])).to(device)
                        hidden_states_h_reward = torch.from_numpy(np.asarray(reward_hidden_roots[1])).to(device)

                        network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), action_i)
                        hidden_state_roots = network_output.hidden_state
                        reward_hidden_roots = network_output.reward_hidden
                        #value_prefix_pool = network_output.value_prefix
                        policy_logits_pool = network_output.policy_logits.tolist()
                    value_prefix_pool = [0. for _ in range(test_episodes)]

                roots = cytree.Roots(test_episodes, config.action_space_size, config.num_simulations)
                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
                # do MCTS for a policy (argmax in testing)
                MCTS(config).search(roots, model, hidden_state_roots, reward_hidden_roots)

                roots_distributions = roots.get_distributions()
                roots_values = roots.get_values()

                #current_delay_value = update_envs_current_delay(envs, current_delay_value, config.min_delay, config.delay, p=0.2)
                min_delay = min(min_delay, current_delay_value)
                for i in range(test_episodes):
                    if dones[i]:
                        continue

                    distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                    # select the argmax, not sampling
                    action, _ = select_action(distributions, temperature=1, deterministic=True)
                    policy_actions_per_env[i].append(action)

                    #TODO: CHECK THIS ADDITION:
                    game_histories[i].store_search_stats(distributions, value)


            for i in range(test_episodes):
                if dones[i]:
                    continue
                env = envs[i]
                action = get_env_action_from_samples(policy_actions_per_env[i])
                #print('Found sampled action :', action)
                obs, ori_reward, done, info = env.step(action)
                executed_action = info['executed_action']

                if config.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward

                game_histories[i].append(action, executed_action, obs, clip_reward, config.delay)

                dones[i] = done
                ep_ori_rewards[i] += ori_reward
                ep_clip_rewards[i] += clip_reward

            if step % 500 == 0:
                print(step, dones)
            step += 1

            if use_pb:
                pb.set_description('{} In step {}, scores: {}(max: {}, min: {}) currently.'
                                   ''.format(config.env_name_short, counter,
                                             ep_ori_rewards.mean(), ep_ori_rewards.max(), ep_ori_rewards.min()))
                pb.update(1)
        for env in envs:
            env.close()
    print(f"Minimum delay value during the test episodes: {min_delay}")
    return ep_ori_rewards, step, save_path
