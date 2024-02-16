import argparse

import nncore
from nncore.engine import Engine, comm, set_random_seed
from nncore.engine.utils import save_checkpoint
from nncore.nn import build_model
from tqdm import trange
from read.model import add_read
from read.utils import get_read_state_dict, get_read_params
from nncore.nn import move_to_device
from nncore.optim import build_optimizer
import torch


class READ_PVLA_Engine(Engine):
    def train_epoch(self):
        self._mode = 'train'
        self.model.train()
        self.data_loader = self.data_loaders[self._mode]

        if callable(getattr(self.data_loader.dataset, 'set_state', None)):
            self.data_loader.dataset.set_state(self._mode)

        self._call_hook('before_train_epoch')

        for data in self.data_loader:
            self.train_iter(data)

        self._call_hook('after_train_epoch')
        self._epoch += 1

    def after_train_epoch(self, hook):
        if (not hook.last_epoch(self) and not hook.every_n_epochs(self, hook._interval)):
            return

        filename = "epoch_{}.pth".format(self.epoch + 1)
        filepath = nncore.join(hook._out, filename)
        optimizer = self.optimizer if hook._save_optimizer else None

        meta = dict(
            epoch = self.epoch + 1,
            iter = self.iter,
            stages = [
                stage.to_dict() if isinstance(stage, nncore.CfgNode) else stage
                for stage in self.stages
            ]
        )

        self.logger.info("Saving checkpoint to {}...".format(filepath))
        save_checkpoint(self.model, filepath, optimizer, meta=meta)

        if hook._create_symlink:
            nncore.symlink(filename, nncore.join(hook._out, "latest.pth"))


    def _call_hook(self, name):
        for hook in self.hooks.values():
            if 'CheckpointHook' in str(type(hook)) and name == 'after_train_epoch':
                self.after_train_epoch(hook)
            else:
                getattr(hook, name)(self)

    def run_stage(self):
        if isinstance(self.cur_stage['optimizer'], dict):
            optim = self.cur_stage['optimizer'].copy()
            optim_type = optim.pop('type')
            optim_args = ['{}: {}'.format(k, v) for k, v in optim.items()]
            optim = '{}({})'.format(optim_type, ', '.join(optim_args))
        else:
            optim = '{}()'.format(
                self.cur_stage['optimizer'].__class__.__name__)

        self.logger.info('Stage: {}, epochs: {}, optimizer: {}'.format(
            self._stage + 1, self.cur_stage['epochs'], optim))

        if self.epoch_in_stage == 0:
            self.optimizer = build_optimizer(
                self.cur_stage['optimizer'], params=get_read_params(self.model))

        self._call_hook('before_stage')

        for _ in range(self.cur_stage['epochs'] - self.epoch_in_stage):
            self.train_epoch()
            cfg = self.cur_stage.get('validation')
            if (cfg is not None and 'val' in self.data_loaders
                    and cfg.get('interval', 0) > 0
                    and self.epoch_in_stage > cfg.get('offset', 0)
                    and self.epoch_in_stage % cfg.get('interval', 0) == 0):
                self.val_epoch()

        self._call_hook('after_stage')
        self._stage += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--checkpoint', help='load a checkpoint')
    parser.add_argument('--resume', help='resume from a checkpoint')
    parser.add_argument('--seed', help='random seed', type=int, default=37439364)
    parser.add_argument('--eval', help='evaluation mode', action='store_true')
    parser.add_argument('--launcher', help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    launcher = comm.init_dist(launcher=args.launcher)

    if comm.is_main_process() and not args.eval:
        timestamp = nncore.get_timestamp()
        work_dir = nncore.join('work_dirs', nncore.pure_name(args.config))
        work_dir = nncore.mkdir(work_dir, modify_path=True)
        log_file = nncore.join(work_dir, '{}.log'.format(timestamp))
    else:
        log_file = work_dir = None

    logger = nncore.get_logger(log_file=log_file)
    logger.info(f'Environment info:\n{nncore.collect_env_info()}')
    logger.info(f'Elastic launcher: {launcher}')
    logger.info(f'Config: {cfg.text}')

    seed = args.seed if args.seed is not None else cfg.get('seed')
    seed = set_random_seed(seed, deterministic=True)
    logger.info(f'Using random seed: {seed}')

    model = build_model(cfg.model, dist=bool(launcher)).module
    model = model.to('cpu')
    add_read(model)
    model = model.cuda()

    engine = READ_PVLA_Engine(
        model,
        cfg.data,
        stages=cfg.stages,
        hooks=cfg.hooks,
        work_dir=work_dir,
        seed=seed)

    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)
    elif args.resume:
        engine.resume(args.checkpoint)
    
    engine.launch(eval=args.eval)

if __name__ == '__main__':
    main()
