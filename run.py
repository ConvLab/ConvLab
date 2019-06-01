# Modified by Microsoft Corporation.
# Licensed under the MIT license.

'''
Specify what to run in `config/experiments.json`
Then run `python run_lab.py` or `yarn start`
'''
# import os
# # NOTE increase if needed. Pytorch thread overusage https://github.com/pytorch/pytorch/issues/975
# os.environ['OMP_NUM_THREADS'] = '1'
from convlab import EVAL_MODES, TRAIN_MODES
from convlab.experiment.control import Session, Trial, Experiment
from convlab.lib import logger, util
from convlab.spec import spec_util
# from convlab.experiment import analysis, retro_analysis
# from convlab.experiment.monitor import InfoSpace
from xvfbwrapper import Xvfb
import os
import pydash as ps
import sys
import torch
import torch.multiprocessing as mp


debug_modules = [
    # 'algorithm',
]
debug_level = 'DEBUG'
logger.toggle_debug(debug_modules, debug_level)
logger = logger.get_logger(__name__)


def run_spec(spec, lab_mode):
    '''Run a spec in lab_mode'''
    os.environ['lab_mode'] = lab_mode
    if lab_mode in TRAIN_MODES:
        spec_util.save(spec)  # first save the new spec
        if lab_mode == 'dev':
            spec = spec_util.override_dev_spec(spec)
        if lab_mode == 'search':
            spec_util.tick(spec, 'experiment')
            Experiment(spec).run()
        else:
            spec_util.tick(spec, 'trial')
            Trial(spec).run()
    elif lab_mode in EVAL_MODES:
        spec_util.tick(spec, 'session')
        spec = spec_util.override_enjoy_spec(spec)
        Session(spec).run()
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}')


def read_spec_and_run(spec_file, spec_name, lab_mode):
    '''Read a spec and run it in lab mode'''
    logger.info(f'Running lab spec_file:{spec_file} spec_name:{spec_name} in mode:{lab_mode}')
    if lab_mode in TRAIN_MODES:
        spec = spec_util.get(spec_file, spec_name)
    else:  # eval mode
        if '@' in lab_mode:
            lab_mode, prename = lab_mode.split('@')
            spec = spec_util.get_eval_spec(spec_file, prename)
        else:
            # spec = spec_util.get_eval_spec(spec_file)
            spec = spec_util.get(spec_file, spec_name)

    if 'spec_params' not in spec:
        run_spec(spec, lab_mode)
    else:  # spec is parametrized; run them in parallel
        param_specs = spec_util.get_param_specs(spec)
        num_pro = spec['meta']['param_spec_process']
        # can't use Pool since it cannot spawn nested Process, which is needed for VecEnv and parallel sessions. So these will run and wait by chunks
        workers = [mp.Process(target=run_spec, args=(spec, lab_mode)) for spec in param_specs]
        for chunk_w in ps.chunk(workers, num_pro):
            for w in chunk_w:
                w.start()
            for w in chunk_w:
                w.join()


def main():
    '''Main method to run jobs from scheduler or from a spec directly'''
    args = sys.argv[1:]
    if len(args) <= 1:  # use scheduler
        job_file = args[0] if len(args) == 1 else 'job/experiments.json'
        for spec_file, spec_and_mode in util.read(job_file).items():
            for spec_name, lab_mode in spec_and_mode.items():
                read_spec_and_run(spec_file, spec_name, lab_mode)
    else:  # run single spec
        assert len(args) == 3, f'To use sys args, specify spec_file, spec_name, lab_mode'
        read_spec_and_run(*args)


if __name__ == '__main__':
    torch.set_num_threads(1)  # prevent multithread slowdown
    mp.set_start_method('spawn', force=True)  # for distributed pytorch to work
    if sys.platform == 'darwin':
        # avoid xvfb on MacOS: https://github.com/nipy/nipype/issues/1400
        main()
    else:
        with Xvfb() as xvfb:  # safety context for headless machines
            main()


# debug_modules = [
#     # 'algorithm',
# ]
# debug_level = 'DEBUG'
# logger.toggle_debug(debug_modules, debug_level)


# def run_new_mode(spec_file, spec_name, lab_mode):
#     '''Run to generate new data with `search, train, dev`'''
#     spec = spec_util.get(spec_file, spec_name)
#     info_space = InfoSpace()
#     analysis.save_spec(spec, info_space, unit='experiment')  # first save the new spec
#     if lab_mode == 'search':
#         info_space.tick('experiment')
#         Experiment(spec, info_space).run()
#     elif lab_mode.startswith('train'):
#         info_space.tick('trial')
#         Trial(spec, info_space).run()
#     elif lab_mode == 'dev':
#         spec = spec_util.override_dev_spec(spec)
#         info_space.tick('trial')
#         Trial(spec, info_space).run()
#     else:
#         raise ValueError(f'Unrecognizable lab_mode not of {TRAIN_MODES}')


# def run_old_mode(spec_file, spec_name, lab_mode):
#     '''Run using existing data with `enjoy, eval`. The eval mode is also what train mode's online eval runs in a subprocess via bash command'''
#     # reconstruct spec and info_space from existing data
#     if '@' in lab_mode:
#         lab_mode, prename = lab_mode.split('@')
#         predir, _, _, _, _, _ = util.prepath_split(spec_file)
#         prepath = f'{predir}/{prename}'
#         spec, info_space = util.prepath_to_spec_info_space(prepath)
#     else:
#         prepath = f'output/{spec_file}/{spec_name}'
#         spec = spec_util.get(spec_file, spec_name)
#         info_space = InfoSpace()

#     # see InfoSpace def for more on these
#     info_space.ckpt = 'eval'
#     info_space.eval_model_prepath = prepath

#     # no info_space.tick() as they are reconstructed
#     if lab_mode == 'enjoy':
#         spec = spec_util.override_enjoy_spec(spec)
#         Session(spec, info_space).run()
#     elif lab_mode == 'eval':
#         # example eval command:
#         # python run_lab.py data/dqn_cartpole_2018_12_19_224811/dqn_cartpole_t0_spec.json dqn_cartpole eval@dqn_cartpole_t0_s1_ckpt-epi10-totalt1000
#         spec = spec_util.override_eval_spec(spec)
#         Session(spec, info_space).run()
#         util.clear_periodic_ckpt(prepath)  # cleanup after itself
#         retro_analysis.analyze_eval_trial(spec, info_space, predir)
#     else:
#         raise ValueError(f'Unrecognizable lab_mode not of {EVAL_MODES}')


# def run_by_mode(spec_file, spec_name, lab_mode):
#     '''The main run lab function for all lab_modes'''
#     logger.info(f'Running lab in mode: {lab_mode}')
#     # '@' is reserved for 'enjoy@{prename}'
#     os.environ['lab_mode'] = lab_mode.split('@')[0]
#     if lab_mode in TRAIN_MODES:
#         run_new_mode(spec_file, spec_name, lab_mode)
#     else:
#         run_old_mode(spec_file, spec_name, lab_mode)


# def main():
#     if len(sys.argv) > 1:
#         args = sys.argv[1:]
#         assert len(args) == 3, f'To use sys args, specify spec_file, spec_name, lab_mode'
#         run_by_mode(*args)
#         return

#     # experiments = util.read('config/experiment1.json')
#     # run_by_mode('experiment1.json', 'word_dst', 'train')
#     '''
#     for spec_file in experiments:
#         for spec_name, lab_mode in experiments[spec_file].items():
#             run_by_mode(spec_file, spec_name, lab_mode)
#     '''


# if __name__ == '__main__':
#     try:
#         mp.set_start_method('spawn')  # for distributed pytorch to work
#     except RuntimeError:
#         pass
#     if sys.platform == 'darwin':
#         # avoid xvfb for MacOS: https://github.com/nipy/nipype/issues/1400
#         main()
#     else:
#         with Xvfb() as xvfb:  # safety context for headless machines
#             main()
