"""Train one FrOGS model over its whole (dmu, T) box.

    python scripts/train.py --config configs/cuau_128.yml
    python scripts/train.py --config configs/cuau_128.yml --resume results/cuau_ce/cuau128-1/ckpts/last.ckpt

Runs are written to results/<system>/<run_tag>-<n>/ (override the root with FROGS_RESULTS_DIR).
"""
import argparse
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from frogs.config import load_config, run_name
from frogs.paths import RESULTS_DIR, ensure_dir
from frogs.trainer import FrOGSModule

torch.serialization.add_safe_globals([argparse.Namespace])


def new_run_dir(config):
    base = ensure_dir(os.path.join(RESULTS_DIR, config.system))
    name = run_name(config)
    taken = [int(d.rsplit("-", 1)[1]) for d in os.listdir(base)
             if d.rsplit("-", 1)[0] == name and d.rsplit("-", 1)[1].isdigit()]
    return ensure_dir(os.path.join(base, f"{name}-{max(taken, default=0) + 1}"))


def make_logger(config, run_dir, offline):
    if not offline:
        try:
            from lightning.pytorch.loggers import WandbLogger
            import wandb  # noqa: F401
            return WandbLogger(project=config.wandb_project, name=os.path.basename(run_dir),
                               save_dir=run_dir, config=vars(config))
        except ImportError:
            pass
    return L.pytorch.loggers.CSVLogger(save_dir=run_dir, name="", version="")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", metavar="CKPT", help="continue a run from its last.ckpt")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config entries, e.g. --set seed=1 run_tag=cuau128-s1")
    p.add_argument("--offline", action="store_true", help="log to metrics.csv instead of W&B")
    args = p.parse_args()

    torch.set_default_dtype(torch.float32)
    config = load_config(args.config, args.set)
    if config.seed is not None:
        L.seed_everything(int(config.seed), workers=True)
    if args.resume:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.resume)))
    else:
        run_dir = new_run_dir(config)
    print(f"run directory: {run_dir}")

    ckpt_dir = os.path.join(run_dir, "ckpts")
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if config.last_ckpt_every == config.ckpt_every:
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1, save_last=True,
                                         every_n_train_steps=config.ckpt_every))
    else:
        callbacks += [
            ModelCheckpoint(dirpath=ckpt_dir, save_top_k=0, save_last=True,
                            every_n_train_steps=config.last_ckpt_every),
            ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1, every_n_train_steps=config.ckpt_every),
        ]
    devices = config.num_gpus * config.num_nodes
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.num_gpus if torch.cuda.is_available() else 1,
        num_nodes=config.num_nodes,
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        precision="32-true",
        logger=make_logger(config, run_dir, args.offline),
        log_every_n_steps=config.log_every,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.grad_clip,
        callbacks=callbacks,
        default_root_dir=run_dir,
        max_epochs=-1,
        max_steps=config.max_steps,
    )
    model = FrOGSModule.load_from_checkpoint(args.resume) if args.resume else FrOGSModule(config)
    trainer.fit(model, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
