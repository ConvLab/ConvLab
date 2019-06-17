# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import shutil

import numpy as np
import pandas as pd
import pydash as ps
import torch

from convlab.lib import logger, util, viz

NUM_EVAL = 4
METRICS_COLS = [
    'strength', 'max_strength', 'final_strength',
    'sample_efficiency', 'training_efficiency',
    'stability', 'consistency',
]

logger = logger.get_logger(__name__)


# methods to generate returns (total rewards)

def gen_return(agent, env):
    '''Generate return for an agent and an env in eval mode'''
    obs = env.reset()
    agent.reset(obs)
    done = False
    total_reward = 0
    env.clock.tick('epi')
    env.clock.tick('t')
    while not done:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        env.clock.tick('t')
    return total_reward


def gen_avg_return(agent, env, num_eval=NUM_EVAL):
    '''Generate average return for agent and an env'''
    with util.ctx_lab_mode('eval'):  # enter eval context
        agent.algorithm.update()  # set explore_var etc. to end_val under ctx
        with torch.no_grad():
            returns = [gen_return(agent, env) for i in range(num_eval)]
    # exit eval context, restore variables simply by updating
    agent.algorithm.update()
    return np.mean(returns)


def gen_result(agent, env):
    '''Generate average return for agent and an env'''
    with util.ctx_lab_mode('eval'):  # enter eval context
        agent.algorithm.update()  # set explore_var etc. to end_val under ctx
        with torch.no_grad():
            _return = gen_return(agent, env)
    # exit eval context, restore variables simply by updating
    agent.algorithm.update()
    return _return 


def gen_avg_result(agent, env, num_eval=NUM_EVAL):
    returns, lens, successes, precs, recs, f1s, book_rates = [], [], [], [], [], [], []
    for _ in range(num_eval):
        returns.append(gen_result(agent, env))
        lens.append(env.clock.t)
        if env.evaluator: 
            successes.append(env.evaluator.task_success())
            _p, _r, _f1 = env.evaluator.inform_F1() 
            if _f1 is not None:
                precs.append(_p)
                recs.append(_r)
                f1s.append(_f1)
            _book = env.evaluator.book_rate()
            if _book is not None:
                book_rates.append(_book)
        elif hasattr(env, 'get_task_success'):
            successes.append(env.get_task_success())
        logger.nl(f'---A dialog session is done---')
    mean_success = None if len(successes) == 0 else np.mean(successes)
    mean_p = None if len(precs) == 0 else np.mean(precs)
    mean_r = None if len(recs) == 0 else np.mean(recs)
    mean_f1 = None if len(f1s) == 0 else np.mean(f1s)
    mean_book_rate = None if len(book_rates) == 0 else np.mean(book_rates)
    return np.mean(returns), np.mean(lens), mean_success, mean_p, mean_r, mean_f1, mean_book_rate


def calc_session_metrics(session_df, env_name, info_prepath=None, df_mode=None):
    '''
    Calculate the session metrics: strength, efficiency, stability
    @param DataFrame:session_df Dataframe containing reward, frame, opt_step
    @param str:env_name Name of the environment to get its random baseline
    @param str:info_prepath Optional info_prepath to auto-save the output to
    @param str:df_mode Optional df_mode to save with info_prepath
    @returns dict:metrics Consists of scalar metrics and series local metrics
    '''
    mean_return = session_df['avg_return'] if df_mode == 'eval' else session_df['avg_return']
    mean_length = session_df['avg_len'] if df_mode == 'eval' else None 
    mean_success = session_df['avg_success'] if df_mode == 'eval' else None 
    frames = session_df['frame']
    opt_steps = session_df['opt_step']

    # all the session local metrics
    local = {
        'mean_return': mean_return,
        'mean_length': mean_length,
        'mean_success': mean_success,
        'frames': frames,
        'opt_steps': opt_steps,
    }
    metrics = {
        'local': local,
    }
    if info_prepath is not None:  # auto-save if info_prepath is given
        util.write(metrics, f'{info_prepath}_session_metrics_{df_mode}.pkl')
    return metrics


def calc_trial_metrics(session_metrics_list, info_prepath=None):
    '''
    Calculate the trial metrics: mean(strength), mean(efficiency), mean(stability), consistency
    @param list:session_metrics_list The metrics collected from each session; format: {session_index: {'scalar': {...}, 'local': {...}}}
    @param str:info_prepath Optional info_prepath to auto-save the output to
    @returns dict:metrics Consists of scalar metrics and series local metrics
    '''
    # calculate mean of session metrics
    mean_return_list = [sm['local']['mean_return'] for sm in session_metrics_list]
    mean_length_list = [sm['local']['mean_length'] for sm in session_metrics_list]
    mean_success_list = [sm['local']['mean_success'] for sm in session_metrics_list]
    frames = session_metrics_list[0]['local']['frames']
    opt_steps = session_metrics_list[0]['local']['opt_steps']

    # for plotting: gather all local series of sessions
    local = {
        'mean_return': mean_return_list,
        'mean_length': mean_length_list,
        'mean_success': mean_success_list,
        'frames': frames,
        'opt_steps': opt_steps,
    }
    metrics = {
        'local': local,
    }
    if info_prepath is not None:  # auto-save if info_prepath is given
        util.write(metrics, f'{info_prepath}_trial_metrics.pkl')
    return metrics


def calc_experiment_df(trial_data_dict, info_prepath=None):
    '''Collect all trial data (metrics and config) from trials into a dataframe'''
    experiment_df = pd.DataFrame(trial_data_dict).transpose()
    cols = METRICS_COLS
    config_cols = sorted(ps.difference(experiment_df.columns.tolist(), cols))
    sorted_cols = config_cols + cols
    experiment_df = experiment_df.reindex(sorted_cols, axis=1)
    experiment_df.sort_values(by=['strength'], ascending=False, inplace=True)
    if info_prepath is not None:
        util.write(experiment_df, f'{info_prepath}_experiment_df.csv')
        # save important metrics in info_prepath directly
        util.write(experiment_df, f'{info_prepath.replace("info/", "")}_experiment_df.csv')
    return experiment_df


# interface analyze methods

def analyze_session(session_spec, session_df, df_mode):
    '''Analyze session and save data, then return metrics. Note there are 2 types of session_df: body.eval_df and body.train_df'''
    info_prepath = session_spec['meta']['info_prepath']
    session_df = session_df.copy()
    assert len(session_df) > 1, f'Need more than 1 datapoint to calculate metrics'
    util.write(session_df, f'{info_prepath}_session_df_{df_mode}.csv')
    # calculate metrics
    session_metrics = calc_session_metrics(session_df, ps.get(session_spec, 'env.0.name'), info_prepath, df_mode)
    # plot graph
    viz.plot_session(session_spec, session_metrics, session_df, df_mode)
    return session_metrics


def analyze_trial(trial_spec, session_metrics_list):
    '''Analyze trial and save data, then return metrics'''
    info_prepath = trial_spec['meta']['info_prepath']
    # calculate metrics
    trial_metrics = calc_trial_metrics(session_metrics_list, info_prepath)
    # plot graphs
    viz.plot_trial(trial_spec, trial_metrics)
    # zip files
    if util.get_lab_mode() == 'train':
        predir, _, _, _, _, _ = util.prepath_split(info_prepath)
        shutil.make_archive(predir, 'zip', predir)
        logger.info(f'All trial data zipped to {predir}.zip')
    return trial_metrics


def analyze_experiment(spec, trial_data_dict):
    '''Analyze experiment and save data'''
    info_prepath = spec['meta']['info_prepath']
    util.write(trial_data_dict, f'{info_prepath}_trial_data_dict.json')
    # calculate experiment df
    experiment_df = calc_experiment_df(trial_data_dict, info_prepath)
    # plot graph
    viz.plot_experiment(spec, experiment_df, METRICS_COLS)
    # zip files
    predir, _, _, _, _, _ = util.prepath_split(info_prepath)
    shutil.make_archive(predir, 'zip', predir)
    logger.info(f'All experiment data zipped to {predir}.zip')
    return experiment_df
