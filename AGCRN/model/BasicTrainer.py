import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics, vpt_batch, vpt_from_nrmse_curve
import wandb_utils
import optuna

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None, trial=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.trial = trial
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0.)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        wandb_utils.log_metrics({'loss': val_loss}, step=epoch, prefix='val')
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            loss = self.loss(output.cuda(), label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))

        wandb_utils.log_metrics({'loss': train_epoch_loss, 'tf_ratio': teacher_forcing_ratio}, step=epoch, prefix='train')
        
        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if self.trial is not None:
                self.trial.report(val_epoch_loss, epoch)
                if self.trial.should_prune():
                    wandb_utils.finish()
                    raise optuna.exceptions.TrialPruned()

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        try:
            #save the best model to file
            if not self.args.debug:
                checkpoint = {'state_dict': best_model, 'optimizer': self.optimizer.state_dict(), 'config': self.args}
                torch.save(checkpoint, self.best_path)
                self.logger.info("Saving current best model to " + self.best_path)
 
            #test
            self.model.load_state_dict(best_model)
            #self.val_epoch(self.args.epochs, self.test_loader)
            self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
            wandb_utils.log_summary({'best_val_loss': best_loss, 'training_time_min': training_time / 60})
        finally:
            wandb_utils.finish()
        return best_loss

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        if y_true_np.shape[-1] == 1:
            y_true_np = y_true_np.squeeze(-1)
            y_pred_np = y_pred_np.squeeze(-1)
        vpt_threshold = getattr(args, 'vpt_threshold', 0.5)
        vpt_results = vpt_batch(y_true_np, y_pred_np, threshold=vpt_threshold)
        logger.info(
            "VPT (threshold={:.2f}): mean={:.2f} steps, median={:.2f}, "
            "std={:.2f}, range=[{:.0f}, {:.0f}]".format(
                vpt_threshold, vpt_results['vpt_mean'], vpt_results['vpt_median'],
                vpt_results['vpt_std'], vpt_results['vpt_min'], vpt_results['vpt_max']))
        wandb_utils.log_summary({f'test_{k}': v for k, v in vpt_results.items()})

    @staticmethod
    def rollout_evaluate(model, args, raw_test_series, scaler, logger, rollout_threshold=0.5):
        """Closed-loop autoregressive rollout evaluation, ported from
        ChaosNetBench's `autoregressive_rollout_np` / `_evaluate_autoregressive`.

        Does NOT touch the model architecture -- it just calls the
        already-trained model repeatedly, feeding each pred_len-step
        output back in as the next input window, to build a continuous
        NRMSE(t) curve that extends past a single forward pass's horizon.

        Assumes univariate input (args.input_dim == 1, args.tod == False)
        so the predicted channel can be fed straight back in as the next
        window without reconstructing covariates. If you're running with
        tod=True this needs an extra covariate-reconstruction step before
        each model call.

        Args:
            model: trained model (already loaded with best weights)
            args: same args used for training (needs .lag, .horizon,
                  .input_dim, .output_dim, .device)
            raw_test_series: [T_test, N] or [T_test, N, 1] contiguous,
                  SCALED (normalized) test-split series -- NOT the
                  windowed dataloader. Pull this from wherever
                  get_dataloader keeps its chronological test split
                  before windowing (or reload flow.npy and re-slice
                  using the same val_ratio/test_ratio you trained with).
            scaler: same scaler used elsewhere (only needed if you want
                  to report rollout error in real units too)
            rollout_threshold: NRMSE threshold for VPT

        Returns:
            Dict with nrmse_t curve, vpt_steps, and the raw rollout
            predictions.
        """
        model.eval()
        device = args.device
        seq_len = args.lag
        step_size = args.horizon

        raw = raw_test_series
        if raw.ndim == 2:
            raw = raw[..., np.newaxis]  # [T, N, 1]
        T_total, N, D = raw.shape
        assert D == args.input_dim == args.output_dim, (
            "rollout_evaluate assumes input_dim == output_dim (univariate, "
            "no extra covariates). Extend the window-shift logic below if "
            "you're running with tod=True or multivariate input."
        )

        x_init = raw[:seq_len]           # [seq_len, N, D]
        y_true_full = raw[seq_len:]      # [T_total - seq_len, N, D]
        T_future = y_true_full.shape[0]

        y_pred_full = np.zeros_like(y_true_full)
        window = x_init.copy()

        # dummy target: only used for shape by the decoder when
        # teacher_forcing_ratio=0 -- VERIFY this against your AGCRN
        # model.py before trusting the rollout numbers.
        dummy_target = torch.zeros(1, step_size, N, args.output_dim, device=device)

        n_done = 0
        with torch.no_grad():
            while n_done < T_future:
                x_t = torch.FloatTensor(window[np.newaxis]).to(device)  # [1, seq_len, N, D]
                pred = model(x_t, dummy_target, teacher_forcing_ratio=0.)
                pred_np = pred.cpu().numpy()[0]  # [step_size, N, output_dim]

                end = min(n_done + step_size, T_future)
                n_to_store = end - n_done
                y_pred_full[n_done:end] = pred_np[:n_to_store]

                if n_to_store < seq_len:
                    window = np.concatenate([window[n_to_store:], pred_np[:n_to_store]], axis=0)
                else:
                    window = pred_np[-seq_len:]

                n_done = end

        mse_t = np.mean((y_true_full - y_pred_full) ** 2, axis=(1, 2))  # [T_future]
        std_signal = np.std(y_true_full) + 1e-8
        nrmse_t = np.sqrt(mse_t) / std_signal

        vpt_info = vpt_from_nrmse_curve(nrmse_t, threshold=rollout_threshold)
        logger.info(
            "Rollout VPT (threshold={:.2f}): {} steps out of {} rolled "
            "(NRMSE curve final={:.4f})".format(
                rollout_threshold, vpt_info['vpt_steps'], T_future, nrmse_t[-1]))
        wandb_utils.log_summary({
            'rollout_vpt_steps': vpt_info['vpt_steps'],
            'rollout_nrmse_final': float(nrmse_t[-1]),
        })
        return {
            'nrmse_t': nrmse_t,
            'y_pred_full': y_pred_full,
            'y_true_full': y_true_full,
            **vpt_info,
        }


    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
