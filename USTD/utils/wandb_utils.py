# -*- coding: utf-8 -*-
# Shared W&B helper for cross-model comparison across DiffSTG, USTD, AGCRN, CLCRN
# Runs are logged independently (not simultaneously) but grouped so they can
# be compared on the same dashboard/plots after the fact.

import os
import wandb


def init_run(project, group, job_type, name, config, entity=None):
    """
    Initialize a wandb run for a single training job.

    Args:
        project: wandb project name, e.g. "stgnn-weather"
        group: dataset name, e.g. "SAWS" or "ERA5" — lets you filter/compare
               runs trained on the same data regardless of model or when they ran
        job_type: model name, e.g. "DiffSTG", "USTD", "AGCRN", "CLCRN" — lets you
                  filter/compare runs of the same model across datasets or seeds
        name: unique run name, e.g. config.trial_name (includes hyperparams/seed)
        config: dict of hyperparameters to log (params dict from get_params(), etc.)
        entity: wandb entity/team. Defaults to WANDB_ENTITY env var if set.

    Returns:
        wandb Run object. Use run.log(...) or the global wandb.log(...) —
        both work once init() has been called; this returns the object in
        case you want to be explicit or manage multiple runs in one process.
    """
    entity = entity or os.environ.get("WANDB_ENTITY", "chkkar002-university-of-cape-town")

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        job_type=job_type,
        name=name,
        config=config,
        reinit=True,  # safe if you ever loop over multiple trials in one process (e.g. NNI)
    )
    return run


def log_metrics(metrics: dict, step=None, prefix=None):
    """
    Log a dict of metrics, optionally prefixed (e.g. 'val/', 'test/').

    Args:
        metrics: dict of {metric_name: value}
        step: optional explicit step (e.g. epoch number)
        prefix: optional string prepended to every key, e.g. 'val' -> 'val/mae'
    """
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    if step is not None:
        metrics["epoch"] = step
    wandb.log(metrics)


def log_summary(summary: dict):
    """
    Write final/best values to the run summary (shown in the wandb table view,
    separate from the time-series charts). Use once at the end of a run.
    """
    for k, v in summary.items():
        wandb.run.summary[k] = v


def finish():
    """Cleanly close the run. Always call this at the end of main(),
    including in except/finally blocks, so crashed trials don't hang
    as 'running' on the dashboard."""
    wandb.finish()
