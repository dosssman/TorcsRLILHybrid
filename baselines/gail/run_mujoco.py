'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier
import os
import os.path as osp
import datetime

from baselines.gym_torcs import TorcsEnv

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Torcs')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    # TODO: Adapt for desired result boy !
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=True, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(args):
    task_name = "torcs_gail"
    return task_name


def main(args):
    U.make_session(num_cpu=4).__enter__()
    set_global_seeds(args.seed)

    # Gym torcs parameters
    vision = False
    throttle = True
    gear_change = False
    rendering = False
    noisy = False

    # Agent10Fixed_Sparse
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        "/raceconfig/agent_10fixed_sparsed_4.xml"

    # Lap limitation, better left as is
    lap_limiter = 3
    timestep_limit = 320

    # env = gym.make(args.env_id)
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
		race_config_path=race_config_path, rendering=rendering,
		lap_limiter = lap_limiter, noisy=noisy, timestep_limit=timestep_limit)


    args.task = "train"
    dir = os.path.dirname(os.path.abspath(__file__)) +  "/result"

    dir = os.path.join( dir,
        datetime.datetime.now().strftime("torcs-gail-%Y-%m-%d-%H-%M-%S-%f"))

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)
    args.log_dir = dir

    logger.configure( dir=args.log_dir, format_strs="stdout,log,csv,tensorboard")
    print( "# DEBUG: Logging to %s" % logger.get_dir())

    # DamDossDamFix_35eps
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath( os.path.join( data_dir, os.pardir))
    data_dir = os.path.abspath( os.path.join( data_dir, os.pardir))
    data_dir = os.path.join( data_dir, "data")

    # Path to Human Expert data
    args.expert_path = os.path.join(data_dir, "Doss10FixedAnal_220eps/expert_data.npz")

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)

    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    # args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)

    args.checkpoint_dir = os.path.join( args.log_dir, "checkpoint")
    assert isinstance(args.checkpoint_dir, str)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    args.log_dir = osp.join(args.log_dir, task_name)

    args.num_timesteps = 10000000
    args.save_per_iter = 1

    # Uncomment if you already have a pretrained model you want to train further
    # then sent the path
    args.pretrained = False
    # args.pretrained_weight = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5/checkpoint/torcs_gail/torcs_gail_932"
    # args.pretrained_weight = None

    if args.task == 'train':
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name,
              load_model_path=args.load_model_path
              )
    elif args.task == 'evaluate':
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError

    env.close()

def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None,
          load_model_path=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    # if args.pretrained_weight is not None:
    #     pretrained_weight = args.pretrained_weight

    if algo == 'trpo':
        from baselines.gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=512,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name, load_model_path=load_model_path)
    else:
        raise NotImplementedError

    env.close()

def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs, stochastic_policy, save=False, reuse=False):
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)
    # U.load_variables(load_model_path, variables=pi.get_variables(),
    #     sess=tf.get_default_session())

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    print("Max return:", np.max( ret_list))
    print("Min return:", np.min( ret_list))

    return avg_len, avg_ret, np.max( ret_list), np.min( ret_list)


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        # print( "Timestep %d - Steer: %.3f - Accel: %.3f" % (t, ac[0], ac[1]))
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
