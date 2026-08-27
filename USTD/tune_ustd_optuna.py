"""
Optuna HPO driver for USTD's fine-tune stage (stdiffusionfore).

Mirrors the resume/logging/reporting structure of AGCRN's Run.py, but --
unlike AGCRN, where run_once() is an importable function Optuna can call
directly and prune mid-epoch -- USTD's train.py is a subprocess-driven
script (launched via train.sh) with its own internal early-stop logic
already keyed on val MAE plateaus (early_stopping_threshold=10 in
train.py). So each Optuna trial here launches train.py as a subprocess
for ONE seed, waits for it to finish (or self-terminate via its own
early stopping), and reads back best_val_MAE from the JSON file train.py
now writes on completion. There is no true Optuna-level pruning
(trial.report()/should_prune() are never called) -- train.py's own
early stopping is what keeps bad configs from burning the full epoch
budget. If that turns out to be too coarse, we can revisit.

IMPORTANT -- BLOCKED ON ONE THING:
This assumes --config <name> resolves by looking up a top-level key
called <name> inside stdiffusionfore_config.yaml. If the actual loader
resolves --config to a *file path* instead (or the file location isn't
CONFIG_YAML_PATH below), fix write_trial_config() accordingly.
"""

import os
import sys
import json
import copy
import subprocess
import argparse

import yaml
import optuna


# ---- fixed paths ----
# Confirmed from options/base_options.py:
#   yaml_path = os.path.join('model_configurations', opt.model + '_config.yaml')
# i.e. relative to wherever train.py is invoked from.
CONFIG_YAML_PATH = './model_configurations/stdiffusionfore_config.yaml'
BASE_CONFIG_KEY = 'config_SAWS'
PRETRAIN_TAG = 'gwavenet_NA_20260819T022245' # from tune_ustd.sh
TRAIN_PY = 'train.py'

# NOTE: write_trial_config() below reads-then-writes this same shared YAML
# file on every trial. That's safe ONLY because subprocess.run() blocks
# until train.py exits, so trials run strictly sequentially (matches the
# single-GPU SLURM job this is meant for). If this ever gets parallelized
# (multiple concurrent SLURM jobs against the same study, or n_jobs>1 in
# study.optimize), this read/write pattern will race and corrupt the file
# -- switch to per-trial temp files + a file lock before doing that.


def load_base_config():
    with open(CONFIG_YAML_PATH, 'r') as f:
        full = yaml.safe_load(f)
    return full, copy.deepcopy(full[BASE_CONFIG_KEY])


def write_trial_config(full_yaml, base_section, trial, trial_key):
    """Override ONLY top-level stdiffusionfore keys."""
    section = copy.deepcopy(base_section)

    num_heads = trial.suggest_categorical('num_heads', [4, 8])

    # FIX: 64, 96, 128, and 192 are all divisible by both 4 and 8.
    # Use a static list to avoid Optuna categorical space distribution errors.
    static_valid_dims = [64, 96, 128, 192]

    section['num_heads'] = num_heads
    section['embed_dim'] = trial.suggest_categorical('embed_dim', static_valid_dims)
    section['pos_dim'] = trial.suggest_categorical('pos_dim', static_valid_dims)

    section['encoder_depth'] = trial.suggest_int('encoder_depth', 1, 4)
    section['mlp_ratio'] = trial.suggest_categorical('mlp_ratio', [1, 2, 4])
    section['dropout'] = trial.suggest_float('dropout', 0.0, 0.4)

    section['num_steps'] = trial.suggest_categorical('num_steps', [50, 100, 200])
    section['objective'] = trial.suggest_categorical('objective', ['input', 'noise'])

    assert section['wavenet'] == base_section['wavenet'], \
        'wavenet block drifted from pretrain config -- do not tune this.'

    full_yaml = copy.deepcopy(full_yaml)
    full_yaml[trial_key] = section
    with open(CONFIG_YAML_PATH, 'w') as f:
        yaml.safe_dump(full_yaml, f, sort_keys=False)


def run_trial_subprocess(trial_key, batch_size, seed, gpu_id):
    cmd = [
        'python', TRAIN_PY,
        '--model', 'stdiffusionfore',
        '--dataset_mode', 'SAWS',
        '--pred_attr', 'NA',
        '--enable_val',
        '--gpu_ids', str(gpu_id),
        '--config', trial_key,
        '--pretrain', PRETRAIN_TAG,
        '--save_best',
        '--t_len', '48',
        '--seed', str(seed),
        '--eval_epoch_freq', '10',
        '--num_train_target', '3',
        '--num_threads', '4',
        '--batch_size', str(batch_size),
    ]
    env = dict(os.environ)
    env['USTD_SKIP_AUTO_TEST'] = '1'  # don't spawn test.py after every trial

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

    save_dir = None
    for line in proc.stdout.splitlines():
        if line.startswith('USTD_SAVE_DIR::'):
            save_dir = line.split('USTD_SAVE_DIR::', 1)[1].strip()

    if proc.returncode != 0 or save_dir is None:
        print('---- trial subprocess FAILED ----')
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError('train.py subprocess failed or did not report save_dir')

    metrics_path = os.path.join(save_dir, 'best_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics['best_val_MAE']


def objective(trial, full_yaml, base_section, batch_size, seed, gpu_id):
    trial_key = '{}_trial{}'.format(BASE_CONFIG_KEY, trial.number)
    write_trial_config(full_yaml, base_section, trial, trial_key)
    val_mae = run_trial_subprocess(trial_key, batch_size, seed, gpu_id)
    return val_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--study_name', type=str, default='ustd_stdiffusionfore_tuning')
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64, help='fixed, one seed per trial per your call -- not tuned here')
    parser.add_argument('--seed', type=int, default=2030)
    parser.add_argument('--gpu_ids', type=int, default=0)
    args = parser.parse_args()

    full_yaml, base_section = load_base_config()

    storage = args.storage or 'sqlite:///./optuna_studies/{}.db'.format(args.study_name)
    os.makedirs('./optuna_studies', exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=27),
        load_if_exists=True,
    )

    finished_states = (optuna.trial.TrialState.COMPLETE,)
    n_finished = len(study.get_trials(deepcopy=False, states=finished_states))
    n_remaining = max(0, args.n_trials - n_finished)

    if n_remaining == 0:
        print(f'{n_finished} trials already finished, target is {args.n_trials} -- nothing to run.')
    else:
        print(f'{n_finished} trials already finished; running {n_remaining} more '
              f'to reach target of {args.n_trials}.')
        study.optimize(
            lambda trial: objective(trial, full_yaml, base_section,
                                     args.batch_size, args.seed, args.gpu_ids),
            n_trials=n_remaining,
        )

        csv_path = os.path.join('./optuna_studies', f'{args.study_name}_results.csv')
        study.trials_dataframe().to_csv(csv_path, index=False)
        print(f'\nAll-trials results written to {csv_path}')

        print('\nBEST TRIAL')
        best = study.best_trial
        print(f'val_MAE: {best.value:.4f}')
        for k, v in best.params.items():
            print(f'  {k:15s}: {v}')

        print('\nTOP 5 (by val_MAE, ascending)')
        completed = [t for t in study.trials if t.value is not None]
        for t in sorted(completed, key=lambda t: t.value)[:5]:
            print(f'  #{t.number:<3} val_MAE={t.value:.4f}  {t.params}')

        print('\nPARAM IMPORTANCES')
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                print(f'  {param:15s}: {importance:.3f}')
        except Exception as e:
            print(f'(could not compute: {e})')


if __name__ == '__main__':
    main()
