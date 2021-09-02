'''
This code is used to evalaute the imitators trained with different number of trajectories
and plot the results in the same figure for easy comparison.
'''

import argparse
import os
import os.path as osp
import glob
import gym
import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from baselines.remi import run_mujoco as run_torcs
from baselines.remi import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.remi.dataset.mujoco_dset import Mujoco_Dset

from baselines.gym_torcs import TorcsEnv

plt.style.use('ggplot')
CONFIG = {
    'traj_limitation': [1, 5, 10, 50, 100, 200],
    # 'traj_limitation': [ 200],
}


def load_dataset(expert_path):
    dataset = Mujoco_Dset(expert_path=expert_path)
    return dataset


def argsparser():
    parser = argparse.ArgumentParser('Do evaluation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--env', type=str, choices=['Hopper', 'Walker2d', 'HalfCheetah',
                                                    'Humanoid', 'HumanoidStandup'])
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    return parser.parse_args()


def evaluate_env(env_name, seed, policy_hidden_size, stochastic, reuse, prefix):

    def get_checkpoint_dir(checkpoint_list, limit, prefix):
        for checkpoint in checkpoint_list:
            if ('limitation_'+str(limit) in checkpoint) and (prefix in checkpoint):
                return checkpoint
        return None

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=policy_hidden_size, num_hid_layers=2)
    # XXX Readapt expert path
    dir = os.getenv('OPENAI_GEN_LOGDIR')
    # if dir is None:
    #     dir = osp.join(tempfile.gettempdir(),
    #         datetime.datetime.now().strftime("openai-remi"))
    # else:
    #     dir = osp.join( dir, datetime.datetime.now().strftime("openai-remi"))

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    log_dir = dir
    # data_path = os.path.join( log_dir,
    #     "openai-remi/best20180907damned200ep720tstpInterpolated/expert_data.npz")

    # args.stochastic_policy = True
    # data_path = os.path.join( log_dir,
    #     "openai-remi/data/Doss10FixedAnal_70eps_Sliced/expert_data.npz")

    data_path = os.path.join( log_dir,
        "openai-remi/data/Doss10FixedAnal_200eps_Sliced/expert_data.npz")

    # data_path = os.path.join('data', 'deterministic.trpo.' + env_name + '.0.00.npz')
    dataset = load_dataset(data_path)
    # checkpoint_list = glob.glob(os.path.join('checkpoint', '*' + env_name + ".*"))
    log = {
        'traj_limitation': [],
        'upper_bound': [],
        'avg_ret': [],
        'avg_len': [],
        'normalized_ret': []
    }
    for i, limit in enumerate(CONFIG['traj_limitation']):
        # Do one evaluation
        upper_bound = sum(dataset.rets[:limit])/limit
        # checkpoint_dir = get_checkpoint_dir(checkpoint_list, limit, prefix=prefix)
        # checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

        # XXX Checkpoint path
        # checkpoint_path = os.path.join( log_dir, "openai-remi/Doss10Fixed_130eps_GAILed_MildlyStrict_MaxKL_0.01/checkpoint/torcs_gail/torcs_gail_950")
        # Damned Imitated
        # checkpoint_path = os.path.join( log_dir, "openai-remi/best20180907damned200ep720tstpInterpolatedTrainLogs/checkpoint/torcs_gail/torcs_gail_460")

        # Doss 130 Episodes
        # checkpoint_path = os.path.join( log_dir, "openai-remi/Doss10Fixed_130eps_GAILed_MildlyStrict_MaxKL_0.01/checkpoint/torcs_gail/torcs_gail_816")

        # Doss Ctrl 100 Episode most promising so far 2018-10-24
        # checkpoint_path = os.path.join( log_dir, "openai-remi/DossCtrl10Fixed_170eps_BC_GAILed/checkpoint/torcs_gail/torcs_gail_1040")
        # checkpoint_path = os.path.join( log_dir, "openai-remi/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_286")
        # checkpoint_path = os.path.join( log_dir, "openai-remi/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_530")
        # checkpoint_path = os.path.join( log_dir, "openai-remi/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_715")
        #
        # # Round 2 with NoSlice an traj sample at 3600
        # checkpoint_path = os.path.join( log_dir, "openai-remi/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd/checkpoint/torcs_gail/torcs_gail_292")
        # checkpoint_path = os.path.join( log_dir, "openai-remi/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd/checkpoint/torcs_gail/torcs_gail_345")
        #
        # # ROund 3 ...
        # checkpoint_path = os.path.join( log_dir, "openai-remi/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd2/checkpoint/torcs_gail/torcs_gail_205")

        # 200eps over 5m timnesteps First effective run
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd/checkpoint/torcs_gail/torcs_gail_345" # Actually ok on second track

        # Doss 10 Fixed Analogs
        # 1 eps + DETERMINSITRC POLICY MF
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_1ep/checkpoint/torcs_gail/torcs_gail_500"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_1ep/checkpoint/torcs_gail/torcs_gail_900"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_1ep/checkpoint/torcs_gail/torcs_gail_1050"

        # 70 eps + STOCH POLICY
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_70eps/checkpoint/torcs_gail/torcs_gail_900"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_70eps/checkpoint/torcs_gail/torcs_gail_1110"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_70eps/checkpoint/torcs_gail/torcs_gail_1100" # Max wrt MazRew
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_70eps/checkpoint/torcs_gail/torcs_gail_1041" # Max wrt TrueRewMean

        # 130 eps + STOCH POLICY
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_130eps/checkpoint/torcs_gail/torcs_gail_500"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_130eps/checkpoint/torcs_gail/torcs_gail_1160"

        # 200 eps
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_200eps/checkpoint/torcs_gail/torcs_gail_750"

        # REMI Testing Because I can
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_200eps/checkpoint/torcs_gail/torcs_gail_750"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/DossCtrl_DDPGCkpt560_NoSlice_alpha_4_Run5/checkpoint/torcs_remi/torcs_remi_131" # Passes the 3rd corner + Almost goes to the end

        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_1900" # Boy this guy good, can go to the second corner but crash
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_2700" # Boy this guy good, can go to the second corner but crash
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_2732"

        # Doss10FixedAnal_DDPG_Chkp560_200eps_Run2
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_200eps_5mTsteps_Contd1/checkpoint/torcs_gail/torcs_gail_806"

        # Alpha Search
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.1_Run2/checkpoint/torcs_remi/torcs_remi_2828"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.2_Run0/checkpoint/torcs_remi/torcs_remi_859"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.3_Run0/checkpoint/torcs_remi/torcs_remi_5899"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.4_Run0/checkpoint/torcs_remi/torcs_remi_3996"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.6_Run0/checkpoint/torcs_remi/torcs_remi_1188"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.7_Run2/checkpoint/torcs_remi/torcs_remi_2747"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.8_Run2/checkpoint/torcs_remi/torcs_remi_3144"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.9_Run1/checkpoint/torcs_remi/torcs_remi_4593"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.2_Run2/checkpoint/torcs_remi/torcs_remi_5463"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.3_Run0/checkpoint/torcs_remi/torcs_remi_1238"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.4_Run1/checkpoint/torcs_remi/torcs_remi_1256"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_20181215_ckpt1368_220eps_Alpha_0.5_OnlineRL_Run0/checkint/torcs_remi/torcs_remi_1367"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.0_Run0/checkpoint/torcs_remi/torcs_remi_1367"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_1.0_Run0/checkpoint/torcs_remi/torcs_remi_4339"
        # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.3_Run2/checkpoint/torcs_remi/torcs_remi_3248"
        args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.9_Run2/checkpoint/torcs_remi/torcs_remi_3981"

        # Torcs GAIL
        args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run7/checkpoint/torcs_gail/torcs_gail_1337"
        args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run7/checkpoint/torcs_gail/torcs_gail_2656"

        # Torcs Remi
        args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run3/checkpoint/torcs_remi/torcs_remi_2014"
        args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run4/checkpoint/torcs_remi/torcs_remi_2336"

        print( "# DEBUG: Model path: ", args.load_model_path)
        # Not pretty but will do for now
        # assert( os.path.isfile( checkpoint_path + ".index"))

        # env = gym.make(env_name + '-v1')
        # XXX: Custom env declaration
        vision = False
        throttle = True
        gear_change = False
        # Agent alone
        # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        #     "/raceconfig/agent_practice.xml"

        # DamDamAgentFix
        race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
            "/raceconfig/agent_10fixed_sparsed_4.xml"


        # Agent10Fixed_Sparse Used for record Nomoto san
        race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
            "/raceconfig/agent_10fixed_sparsed_2_humanrec.xml"

        rendering = True
        lap_limiter = 2
        timestep_limit = 320

        # env = gym.make(env_id)
        env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
    		race_config_path=race_config_path, rendering=rendering,
    		lap_limiter = lap_limiter)

        env.seed(seed)
        print('Trajectory limitation: {}, Load checkpoint: {}, '.format(limit,
            args.load_model_path))
        # TODO: RUn Mujoco not meant to be used here
        avg_len, avg_ret, max_ret, min_ret = run_torcs.runner(env,
                                             policy_fn,
                                             args.load_model_path,
                                             timesteps_per_batch=3600,
                                             number_trajs=10,
                                             stochastic_policy=stochastic,
                                             reuse=((i != 0) or reuse))
        normalized_ret = avg_ret/upper_bound
        print('Upper bound: {}, evaluation returns: {}, normalized scores: {}'.format(
            upper_bound, avg_ret, normalized_ret))
        log['traj_limitation'].append(limit)
        log['upper_bound'].append(upper_bound)
        log['avg_ret'].append(avg_ret)
        log['avg_len'].append(avg_len)
        log['max_ret'] = max_ret
        log['min_ret'] = min_ret
        log['avg_avg_ret'] = np.mean( log['avg_ret'])
        log['std_avg_ret'] = np.std( log['avg_ret'])
        log['normalized_ret'].append(normalized_ret)
        env.close()
    return log


def plot(env_name, bc_log, gail_log, stochastic):
    upper_bound = bc_log['upper_bound']
    bc_avg_ret = bc_log['avg_ret']
    gail_avg_ret = gail_log['avg_ret']
    plt.plot(CONFIG['traj_limitation'], upper_bound)
    plt.plot(CONFIG['traj_limitation'], bc_avg_ret)
    plt.plot(CONFIG['traj_limitation'], gail_avg_ret)
    plt.xlabel('Number of expert trajectories')
    plt.ylabel('Accumulated reward')
    plt.title('{} unnormalized scores'.format(env_name))
    plt.legend(['expert', 'bc-imitator', 'gail-imitator'], loc='lower right')
    plt.grid(b=True, which='major', color='gray', linestyle='--')
    if stochastic:
        title_name = '{}-unnormalized-stochastic-scores.png'.format(env_name)
    else:
        title_name = '{}-unnormalized-deterministic-scores.png'.format(env_name)

    # XXX Better logging policy ?
    dir = os.getenv('OPENAI_GEN_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-remi/result"))
    else:
        dir = osp.join( dir, datetime.datetime.now().strftime("openai-remi/result"))

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    log_dir = dir
    title_name = osp.join( log_dir, title_name)
    plt.savefig(title_name)
    plt.close()

    bc_normalized_ret = bc_log['normalized_ret']
    gail_normalized_ret = gail_log['normalized_ret']
    plt.plot(CONFIG['traj_limitation'], np.ones(len(CONFIG['traj_limitation'])))
    plt.plot(CONFIG['traj_limitation'], bc_normalized_ret)
    plt.plot(CONFIG['traj_limitation'], gail_normalized_ret)
    plt.xlabel('Number of expert trajectories')
    plt.ylabel('Normalized performance')
    plt.title('{} normalized scores'.format(env_name))
    plt.legend(['expert', 'bc-imitator', 'gail-imitator'], loc='lower right')
    plt.grid(b=True, which='major', color='gray', linestyle='--')
    if stochastic:
        title_name = '{}-normalized-stochastic-scores.png'.format(env_name)
    else:
        title_name = '{}-normalized-deterministic-scores.png'.format(env_name)
    plt.ylim(0, 1.6)
    # XXX Better logging policy ?
    title_name = osp.join( log_dir, title_name)
    plt.savefig(title_name)
    plt.close()


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    args.env = "Torcs GAIL"
    print('Evaluating {}'.format(args.env))
    bc_log = evaluate_env(args.env, args.seed, args.policy_hidden_size,
                          args.stochastic_policy, False, 'BC')
    print('Evaluation for {}'.format(args.env))
    print(bc_log)
    gail_log = evaluate_env(args.env, args.seed, args.policy_hidden_size,
                            args.stochastic_policy, True, 'gail')
    print('Evaluation for {}'.format(args.env))
    print(gail_log)
    plot(args.env, bc_log, gail_log, args.stochastic_policy)


if __name__ == '__main__':
    args = argsparser()
    main(args)
