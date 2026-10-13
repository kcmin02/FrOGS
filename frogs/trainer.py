"""Joint training of the prior, the transport and F_phi under the PINN loss of Eq. (18).

Each step draws conditions c uniformly from the training box, births walkers from rho_0(. | c),
rolls them out to t = k * dt, and minimizes (-d_t U_t - corrector + d_t F)^2 on (sigma, t) pairs
from that rollout or from a replay buffer of earlier rollouts. The number of CTMC steps k
ramps up quadratically to `n_steps` over `n_anneal` optimizer steps.
"""
from __future__ import annotations

import random
import sys

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from frogs.config import normalize
from frogs.energy import make_energy
from frogs.nets import FreeEnergyNet, build_prior, build_transport
from frogs.sampler import CTMCSampler, ess

_LEGACY_PREFIXES = {"net.": "transport.", "F_t_net.": "free_energy.",
                    "_coupled_ar_direct.": "prior."}


def anneal_schedule(k_min, k_max, total_steps):
    """Step counts k_min..k_max, the i-th repeated ~(1 + i d) times (quadratic ramp)."""
    n = k_max - k_min + 1
    if total_steps < n:
        raise ValueError("n_anneal must be at least the number of distinct step counts")
    if n == 1:
        return [k_min] * total_steps
    d = 2 * (total_steps - n) / (n * (n - 1))
    cum = [0.0]
    for i in range(n):
        cum.append(cum[-1] + 1 + i * d)
    scale = total_steps / cum[-1]
    bounds = [int(round(x * scale)) for x in cum]
    bounds[0], bounds[-1] = 0, total_steps
    ks = []
    for i in range(n):
        ks.extend([k_min + i] * (bounds[i + 1] - bounds[i]))
    ks.extend([k_max] * (total_steps - len(ks)))
    return ks[:total_steps]


class _Steps(Dataset):
    def __len__(self):
        return 1_000_000

    def __getitem__(self, idx):
        return torch.tensor(0.0)


class FrOGSModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        config = normalize(config)
        self.save_hyperparameters({"config": config})
        self.config = config

        # Construction order fixes the initial weights for a given seed: prior, transport, F.
        self.prior = build_prior(config) if config.prior == "autoregressive" else None
        self.energy = make_energy(config, self.prior)
        self.transport = build_transport(config) if config.transport else None
        self.free_energy = FreeEnergyNet(config.free_energy_hidden, config.cond_low, config.cond_high)

        self.dt = 1.0 / config.n_steps
        self.ks = anneal_schedule(config.starting_k, config.n_steps, config.n_anneal)
        self.buffer = []
        self._cond_low = torch.tensor(config.cond_low, dtype=torch.get_default_dtype())
        self._cond_high = torch.tensor(config.cond_high, dtype=torch.get_default_dtype())
        self._skips = self._skip_streak = self._guard_steps = 0

    def on_load_checkpoint(self, checkpoint):
        sd = checkpoint["state_dict"]
        for key in list(sd):
            for old, new in _LEGACY_PREFIXES.items():
                if key.startswith(old):
                    sd[new + key[len(old):]] = sd.pop(key)

    def setup(self, stage=None):
        self.energy.to(self.device)

    def on_fit_start(self):
        if self.prior is not None and self.config.compile_prior:
            self.prior.compile()

    def sample_conditions(self, n):
        lo, hi = self._cond_low.to(self.device), self._cond_high.to(self.device)
        c = lo + (hi - lo) * torch.rand(n, len(self.config.cond_low), device=self.device)
        return c.to(torch.get_default_dtype())

    def _rollout(self, k, ts):
        c = self.sample_conditions(self.config.batch_size)
        sampler = CTMCSampler(self.energy, ts, self.transport, n_mcmc=self.config.n_mcmc,
                              n_save=k)
        with torch.no_grad():
            sigma0 = self.energy.sample_prior(c).to(self.device)
            sigmas, As = sampler.rollout(sigma0, c)
        return sigmas.detach(), As.detach(), c

    def training_step(self, batch, batch_idx):
        cfg = self.config
        k = self.ks[min(self.global_step, len(self.ks) - 1)]
        t_final = k * self.dt
        t_set = torch.cat((torch.tensor([0.0]), t_final * torch.rand(k - 1), torch.tensor([t_final])))
        ts = torch.sort(t_set.requires_grad_(True)).values.to(self.device)

        fresh = (cfg.buffer_size == 0 or self.global_step % cfg.buffer_cycle == cfg.buffer_cycle - 1
                 or len(self.buffer) < cfg.buffer_size)
        if fresh:
            sigmas, As, c = self._rollout(k, ts)
            if cfg.buffer_size > 0:
                if len(self.buffer) >= cfg.buffer_size:
                    self.buffer.pop(0)
                self.buffer.append((sigmas.cpu(), As.cpu(), k, ts.detach().cpu(), c.cpu()))
        else:
            sigmas, As, k, ts, c = random.choice(self.buffer)
            ts, c = ts.to(self.device), c.to(self.device)

        walkers = torch.randint(0, cfg.batch_size, size=(cfg.loss_batch_size,))
        times = torch.randint(0, k + 1, size=(cfg.loss_batch_size,))
        sigma = sigmas[times, walkers].to(self.device)
        t, cond = ts[times], c[walkers]

        if self.transport is None:
            corrector = 0.0
        else:
            # full_grad also passes the prior's gradient through the Metropolis rates of the corrector.
            with torch.set_grad_enabled(bool(cfg.full_grad)):
                _, dU = self.energy.Ut(sigma, t, cond)
            corrector = self.transport.corrector(sigma, t, dU, cond)
        residual = -self.energy.dtUt(sigma, t, cond) - corrector + self.free_energy.gradt(t, cond)
        loss = (residual ** 2).mean()

        if (self.global_step + 1) % cfg.log_every == 0:
            self._log_metrics(loss, As, c, ts)
        return loss

    def _log_metrics(self, loss, As, c, ts):
        self.log("train_loss", loss, prog_bar=True)
        self.log("nonfinite_skip_rate", self._skips / max(self._guard_steps, 1))
        A1 = As[-1].to(self.device).double()
        with torch.no_grad():
            n = c.shape[0]
            F1 = self.free_energy(c.new_full((n,), float(ts[-1].item())), c).double()
            F0 = self.free_energy(c.new_zeros(n), c).double()
        # Walkers have different c, so compare A against the learned log Z_t(c) - log Z_0(c).
        A_corr = A1 + F1 - F0
        self.log("ess", ess(A_corr))
        self.log("A_corrected_std", A_corr.std())
        self.log("k", float(ts.numel() - 1))

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Skip the update when the loss or any gradient is non-finite."""
        loss = optimizer_closure() if optimizer_closure is not None else None
        bad = torch.is_tensor(loss) and not bool(torch.isfinite(loss).all())
        opt = getattr(optimizer, "optimizer", optimizer)
        grads = [p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None]
        if grads:
            bad = bad or not bool(torch.isfinite(torch.stack(torch._foreach_norm(grads)).sum()))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flag = torch.tensor([float(bad)], device=self.device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            bad = bool(flag.item() > 0)
        self._guard_steps += 1
        if not bad:
            self._skip_streak = 0
            optimizer.step()
            return
        self._skips += 1
        self._skip_streak += 1
        print(f"skipped non-finite step {self.global_step} (streak {self._skip_streak})",
              file=sys.stderr)
        if self._skip_streak >= 200:
            raise RuntimeError("200 consecutive non-finite steps; resume from last.ckpt "
                               "with a changed setup")

    def configure_optimizers(self):
        params = list(self.free_energy.parameters())
        if self.transport is not None:
            params += list(self.transport.parameters())
        groups = [{"params": params, "lr": self.config.lr}]
        if self.prior is not None:
            groups.append({"params": list(self.prior.parameters()), "lr": self.config.prior_lr})
        opt = torch.optim.Adam(groups, lr=self.config.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.config.lr_step,
                                                gamma=self.config.lr_gamma)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def train_dataloader(self):
        return DataLoader(_Steps(), batch_size=1)
