# Modified by Microsoft Corporation.
# Licensed under the MIT license.

# The control module
# Creates and runs control loops at levels: Experiment, Trial, Session
from copy import deepcopy

import pydash as ps
import torch.multiprocessing as mp

from convlab import agent as agent_module
from convlab.agent import Body
from convlab.agent.net import net_util
from convlab.env import make_env
from convlab.experiment import analysis, search
from convlab.lib import logger, util
from convlab.spec import spec_util


def make_agent_env(spec, global_nets=None):
    '''Helper to create agent and env given spec'''
    env = make_env(spec)
    body = Body(env, spec['agent'])
    AgentClass = getattr(agent_module, ps.get(spec['agent'][0], 'name'))
    agent = AgentClass(spec, body=body, global_nets=global_nets)
    return agent, env


def mp_run_session(spec, global_nets, mp_dict):
    '''Wrap for multiprocessing with shared variable'''
    session = Session(spec, global_nets)
    metrics = session.run()
    mp_dict[session.index] = metrics


class Session:
    '''
    The base lab unit to run a RL session for a spec.
    Given a spec, it creates the agent and env, runs the RL loop,
    then gather data and analyze it to produce session data.
    '''

    def __init__(self, spec, global_nets=None):
        self.spec = spec
        self.index = self.spec['meta']['session']
        util.set_random_seed(self.spec)
        util.set_cuda_id(self.spec)
        util.set_logger(self.spec, logger, 'session')
        spec_util.save(spec, unit='session')

        self.agent, self.env = make_agent_env(self.spec, global_nets)
        with util.ctx_lab_mode('eval'):  # env for eval
            self.eval_env = make_env(self.spec)
            self.agent.body.eval_env = self.eval_env 
        self.num_eval = ps.get(self.agent.spec, 'meta.num_eval')
        self.warmup_epi = ps.get(self.agent.agent_spec, 'algorithm.warmup_epi') or -1 
        logger.info(util.self_desc(self))

    def to_ckpt(self, env, mode='eval'):
        '''Check with clock whether to run log/eval ckpt: at the start, save_freq, and the end'''
        if mode == 'eval' and util.in_eval_lab_modes():  # avoid double-eval: eval-ckpt in eval mode
            return False
        clock = env.clock
        frame = clock.get()
        frequency = env.eval_frequency if mode == 'eval' else env.log_frequency
        if frame == 0 or clock.get('opt_step') == 0:  # avoid ckpt at init
            to_ckpt = False
        elif frequency is None:  # default episodic
            to_ckpt = env.done
        else:  # normal ckpt condition by mod remainder (general for venv)
            to_ckpt = util.frame_mod(frame, frequency, env.num_envs) or frame == clock.max_frame
        return to_ckpt

    def try_ckpt(self, agent, env):
        '''Check then run checkpoint log/eval'''
        body = agent.body
        if self.to_ckpt(env, 'log') and self.env.clock.get('epi') > self.warmup_epi:
            body.train_ckpt()
            body.log_summary('train')

        if self.to_ckpt(env, 'eval'):
            avg_return, avg_len, avg_success, avg_p, avg_r, avg_f1, avg_book_rate = analysis.gen_avg_result(agent, self.eval_env, self.num_eval) 
            body.eval_ckpt(self.eval_env, avg_return, avg_len, avg_success)
            body.log_summary('eval')
            if body.eval_reward_ma >= body.best_reward_ma:
                body.best_reward_ma = body.eval_reward_ma
                agent.save(ckpt='best')
            if self.env.clock.get('epi') > self.warmup_epi:
                if len(body.train_df) > 1:  # need > 1 row to calculate stability
                    metrics = analysis.analyze_session(self.spec, body.train_df, 'train')
                if len(body.eval_df) > 1:  # need > 1 row to calculate stability
                    metrics = analysis.analyze_session(self.spec, body.eval_df, 'eval')

    def run_eval(self):
        avg_return, avg_len, avg_success, avg_p, avg_r, avg_f1, avg_book_rate = analysis.gen_avg_result(self.agent, self.eval_env, self.num_eval) 
        result = f'{self.num_eval} episodes, {avg_return:.2f} return'
        if not avg_success is None:
            result += f', {avg_success*100:.2f}% success rate'
        if avg_len:
            result += f', {avg_len:.2f} turns'
        if avg_p:
            result += f', {avg_p:.2f} P, {avg_r:.2f} R, {avg_f1:.2f} F1'
        if avg_book_rate:
            result += f', {avg_book_rate*100:.2f}% book rate'
        logger.info(result)

    def run_rl(self):
        '''Run the main RL loop until clock.max_frame'''
        logger.info(f'Running RL loop for trial {self.spec["meta"]["trial"]} session {self.index}')
        clock = self.env.clock
        obs = self.env.reset()
        clock.tick('t')
        self.agent.reset(obs)
        done = False
        while True:
            if util.epi_done(done):  # before starting another episode
                logger.nl(f'A dialog session is done')
                self.try_ckpt(self.agent, self.env)
                if clock.get() < clock.max_frame:  # reset and continue
                    clock.tick('epi')
                    obs = self.env.reset()
                    self.agent.reset(obs)
                    done = False
            self.try_ckpt(self.agent, self.env)
            if clock.get() >= clock.max_frame:  # finish
                break
            clock.tick('t')
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            self.agent.update(obs, action, reward, next_obs, done)
            obs = next_obs

    def close(self):
        '''Close session and clean up. Save agent, close env.'''
        self.agent.close()
        self.env.close()
        self.eval_env.close()
        logger.info(f'Session {self.index} done')

    def run(self):
        if util.in_eval_lab_modes():
            self.run_eval()
            metrics = None
        else:
            self.run_rl()
            metrics = analysis.analyze_session(self.spec, self.agent.body.eval_df, 'eval')
        self.close()
        return metrics


class Trial:
    '''
    The lab unit which runs repeated sessions for a same spec, i.e. a trial
    Given a spec and number s, trial creates and runs s sessions,
    then gathers session data and analyze it to produce trial data.
    '''

    def __init__(self, spec):
        self.spec = spec
        self.index = self.spec['meta']['trial']
        util.set_logger(self.spec, logger, 'trial')
        spec_util.save(spec, unit='trial')

    def parallelize_sessions(self, global_nets=None):
        mp_dict = mp.Manager().dict()
        # spec_util.tick(self.spec, 'session')
        # mp_run_session(deepcopy(self.spec), global_nets, mp_dict)
        workers = []
        for _s in range(self.spec['meta']['max_session']):
            spec_util.tick(self.spec, 'session')
            w = mp.Process(target=mp_run_session, args=(deepcopy(self.spec), global_nets, mp_dict))
            w.start()
            workers.append(w)
        for w in workers:
            w.join()
        session_metrics_list = [mp_dict[idx] for idx in sorted(mp_dict.keys())]
        return session_metrics_list

    def run_sessions(self):
        logger.info('Running sessions')
        session_metrics_list = self.parallelize_sessions()
        return session_metrics_list

    def init_global_nets(self):
        session = Session(deepcopy(self.spec))
        session.env.close()  # safety
        global_nets = net_util.init_global_nets(session.agent.algorithm)
        return global_nets

    def run_distributed_sessions(self):
        logger.info('Running distributed sessions')
        global_nets = self.init_global_nets()
        session_metrics_list = self.parallelize_sessions(global_nets)
        return session_metrics_list

    def close(self):
        logger.info(f'Trial {self.index} done')

    def run(self):
        if self.spec['meta'].get('distributed') == False:
            session_metrics_list = self.run_sessions()
        else:
            session_metrics_list = self.run_distributed_sessions()
        metrics = analysis.analyze_trial(self.spec, session_metrics_list)
        self.close()
        # return metrics['scalar']
        return metrics


class Experiment:
    '''
    The lab unit to run experiments.
    It generates a list of specs to search over, then run each as a trial with s repeated session,
    then gathers trial data and analyze it to produce experiment data.
    '''

    def __init__(self, spec):
        self.spec = spec
        self.index = self.spec['meta']['experiment']
        util.set_logger(self.spec, logger, 'trial')
        spec_util.save(spec, unit='experiment')

    def close(self):
        logger.info('Experiment done')

    def run(self):
        trial_data_dict = search.run_ray_search(self.spec)
        experiment_df = analysis.analyze_experiment(self.spec, trial_data_dict)
        self.close()
        return experiment_df
