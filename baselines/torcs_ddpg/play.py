import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from baselines.ddpg_torqs.models import Actor, Critic
from baselines.ddpg_torqs.memory import Memory
from baselines.ddpg_torqs.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U
from collections import deque

# dosssman
from baselines.gym_torcs import TorcsEnv
import csv # CHecking trace for GIAL

def run( seed, noise_type, layer_norm, nb_epochs, nb_epoch_cycles, reward_scale,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size,
    tau=0.01, param_noise_adaption_interval=5, checkpoint, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()

    # Gym Torcs Parameter envs
    vision = False
    throttle = True
    gear_change = False
    rendering = True
    lap_limiter = 4

    # Agent 10 Fixed First Track Second Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \


    # env = gym.make(env_id)
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
		race_config_path=race_config_path, rendering=rendering,
		lap_limiter = lap_limiter, randomisation=False, profile_reuse_ep=50)

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    # TODO: Is memeory still needed ?
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    # TODO: Is critic still needed ? probably not ...
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    # logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    # if eval_env is not None:
    #     eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()

    ### Importing Training code
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    # logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    # logger.info('Using agent with the following configuration:')
    # logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None


    if checkpoint == None:
        raise "Checkpoint path must not be empty"
    save_filename = checkpoint
    
    # "/home/z3r0/random/rl/openai_logs/openai-ddpgtorcs-2018-12-05-13-20-33-056875/model_data/epoch_1368.ckpt" # Poor pef on Track2

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()

        # Restore to trained state ?
        saver.restore( sess, save_filename)

        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()
        obss = []
        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                while not done:
                    # Predict next action.
                    # TODO: Noise on or off ?
                    action, _ = agent.pi(obs, apply_noise=False, compute_Q=False)

                    assert action.shape == env.action_space.shape

                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1

                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    # epoch_actions.append(action)
                    # epoch_qs.append(q)
                    # agent.store_transition(obs, action, r, new_obs, done)
                    obss.append( obs)
                    obs = new_obs

                    lapsed = (time.time() - start_time)

                    # if  lapsed >= 30.0:
                    #     done = True

                    if done:
                        print( "Timesteps: %d" % len( obss))
                        print( "Score: %.3f" % episode_reward)
                        # np.save( "/home/z3r0/torcs_data/ddpg_obs", np.asarray( obss))
                        print( "Time %.6f\n" % (time.time() - start_time))
                        # Episode done.
                        # epoch_episode_rewards.append(episode_reward)
                        # episode_rewards_history.append(episode_reward)
                        # epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()

                        # Restore to trained state ?
                        saver.restore( sess, save_filename)
                        # Custom: Need to hard reset Torcs 'cause of mem leak
                        # if np.mod(episodes, 5) == 0:
                        #     obs = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
                        # else:
                        #     obs = env.reset()
                        obs = env.reset()
                        done = False

                        # obs = env.reset()

    ### ENd training code

    env.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--env-id', type=str, default='Pendulum-v0')
    # boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=100)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--checkpoint', type=str, default=None)  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args

if __name__ == '__main__':
    args = parse_args()

    run(**args)
