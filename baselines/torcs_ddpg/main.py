import argparse
import time, datetime
import os
import os.path as osp
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.torcs_ddpg.training as training
from baselines.torcs_ddpg.models import Actor, Critic
from baselines.torcs_ddpg.memory import Memory
from baselines.torcs_ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI

# from baselines.gym_torcs import TorcsEnv
import gym_torcs

def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()

    if rank != 0:
        logger.set_level(logger.DISABLED)

    # GymTorcs parameters
    vision = False
    throttle = True
    gear_change = False
    rendering = False
    lap_limiter = 4

    # 10 Fixed Sparsed Config 2m not too much bots in corners
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        "/raceconfig/agent_10fixed_sparsed_4.xml"

    # The default pre process for dosssman Hybrid IL RL Experiment
    def obs_preprocess_fn(dict_obs):
        return np.hstack((dict_obs['angle'],
            dict_obs['track'],
            dict_obs['trackPos'],
            dict_obs['speedX'],
            dict_obs['speedY'],
            dict_obs['speedZ'],
            dict_obs['wheelSpinVel'],
            dict_obs['rpm'],
            dict_obs['opponents']))

    # Required to change properly build the observation space
    custom_obs_vars = [
        'angle', 'track', 'trackPos', 'speedX', 'speedY', 'speedZ',
        'wheelSpinVel',
        'rpm',
        'opponents'
    ]

    # Creating the environment
    env = gym.make( 'Torcs-v0', vision=vision, throttle=throttle, gear_change=gear_change,
        lap_limiter = lap_limiter, obs_preprocess_fn=obs_preprocess_fn, obs_vars=custom_obs_vars,
        hard_reset_interval=20, rank=rank)

    if evaluation and rank==0:
        eval_env = env
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

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
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm, hidden_sizes=[200,100])
    actor = Actor(nb_actions, layer_norm=layer_norm, hidden_sizes=[200,100])

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)

    env.close()

    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Pendulum-v0')
    boolean_flag(parser, 'render-eval', default=False)
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
    parser.add_argument('--nb-epochs', type=int, default=1500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=720)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)
    boolean_flag(parser, 'evaluation', default=True)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']

    # Fix for Torcs snakeoil parameter overide gatekeeping
    import sys
    sys.argv = [sys.argv[0]]

    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        #Custom logging method to separate DDPG and GAIL for instance
        dir = os.path.dirname(os.path.abspath(__file__)) +  "/result"

        dir = os.path.join( dir,
            datetime.datetime.now().strftime("torcs-ddpg-%Y-%m-%d-%H-%M-%S-%f"))

        assert isinstance(dir, str)
        os.makedirs(dir, exist_ok=True)
        # args.log_dir = dir

        logger.configure( dir=dir, format_strs="stdout,log,csv,tensorboard")
        print( "# DEBUG: Logging to %s" % logger.get_dir())

    # Run actual script.
    run(**args)
