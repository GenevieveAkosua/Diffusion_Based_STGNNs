
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.AGCRN import AGCRN as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
import wandb_utils
import optuna

#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'SAWS'      #PEMSD4 or PEMSD8
DEVICE = 'cuda'
MODEL = 'AGCRN'

#get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)
print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--n_trials', default=0, type=int, help='0 = single run, >0 = run Optuna HPO for this many trials')
args.add_argument('--study_name', default='agcrn_full_tuning', type=str)
args.add_argument('--storage', default=None, type=str)
args = args.parse_args()
#init_seed(args.seed)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#args.device = device
#if torch.cuda.is_available():
#    torch.cuda.set_device(int(args.device[5]))
#else:
#    args.device = 'cpu'

def run_once(args, trial=None):
    """One full train+val+test cycle. Returns best val loss. If `trial` is
    given, the tuned hyperparameters below are overridden by Optuna's
    suggestions for this trial and pruning/reporting is wired in via the
    Trainer."""
    init_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
 
    if trial is not None:
        args.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        args.lr_init = trial.suggest_float('lr_init', 1e-4, 5e-2, log=True)
        args.lag = trial.suggest_categorical('lag', [12, 24, 48])       # input window (hours)
        args.num_layers = trial.suggest_int('num_layers', 1, 4)         # stacked AGCRN layers
        args.rnn_units = trial.suggest_categorical('rnn_units', [16, 32, 64])
        args.embed_dim = trial.suggest_categorical('embed_dim', [1, 2, 3, 5, 8, 10, 15, 20, 30])
 
    #init model
    model = Network(args)
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)
 
    #load dataset
    train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                                   normalizer=args.normalizer,
                                                                   tod=args.tod, dow=False,
                                                                   weather=False, single=False)
 
    print(args.loss_func)
    #init loss function, optimizer
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError
 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                 weight_decay=0, amsgrad=False)
    #learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)
 
    #config log path
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    run_tag = 'trial{}_{}'.format(trial.number, current_time) if trial is not None else current_time
    log_dir = os.path.join(current_dir,'experiments', args.dataset, run_tag)
    args.log_dir = log_dir
 
    run = wandb_utils.init_run(
        project="stgnn-weather",
        group="SAWS",
        job_type="AGCRN",
        name='AGCRN_Hyperparam_{}'.format(trial.number) if trial is not None else 'AGCRN_realval',
        config=vars(args),
    )
 
    #start training
    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                      args, lr_scheduler=lr_scheduler, trial=trial)
    if args.mode == 'train':
        return trainer.train()
    elif args.mode == 'test':
        model.load_state_dict(torch.load('../pre-trained/{}.pth'.format(args.dataset)))
        print("Load saved model")
        trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
        return None
    else:
        raise ValueError

if __name__ == '__main__':
    if args.n_trials <= 0:
        # ---- normal single run, exactly like before ----
        run_once(args)
    else:
        # ---- Optuna HPO run ----
        if optuna is None:
            raise ImportError("optuna is not installed in this environment "
                               "(pip install optuna) but --n_trials > 0 was passed.")

        storage = args.storage or 'sqlite:///./optuna_studies/{}.db'.format(args.study_name)
        os.makedirs('./optuna_studies', exist_ok=True)

        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction='minimize',
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=27),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            load_if_exists=True,
        )

        finished_states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        n_finished = len(study.get_trials(deepcopy=False, states=finished_states))
        n_remaining = max(0, args.n_trials - n_finished)
        n_running = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)))
        if n_running:
            print(f"Note: {n_running} trial(s) stuck in RUNNING state from a previous killed job "
                  f"-- harmless, but they won't count toward n_trials and won't be resumed.")

        if n_remaining == 0:
            print(f"{n_finished} trials already finished, target is {args.n_trials} -- nothing to run.")
        else:
            print(f"{n_finished} trials already finished; running {n_remaining} more "
                  f"to reach target of {args.n_trials}.")
            study.optimize(lambda trial: run_once(args, trial=trial), n_trials=n_remaining)

            csv_path = os.path.join('./optuna_studies', f"{args.study_name}_results.csv")
            study.trials_dataframe().to_csv(csv_path, index=False)
            print(f"\nAll-trials results written to {csv_path}")

            print('\nBEST TRIAL')
            best = study.best_trial
            print(f"val_loss: {best.value:.4f}")
            for k, v in best.params.items():
                print(f"  {k:15s}: {v}")

            print('\nTOP 5 (by val_loss, ascending)')
            completed = [t for t in study.trials if t.value is not None]
            for t in sorted(completed, key=lambda t: t.value)[:5]:
                print(f"  #{t.number:<3} val_loss={t.value:.4f}  {t.params}")

            print('\nPARAM IMPORTANCES')
            try:
                importances = optuna.importance.get_param_importances(study)
                for param, importance in importances.items():
                    print(f"  {param:15s}: {importance:.3f}")
            except Exception as e:
                print(f"(could not compute: {e})")

