import torch
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse

class DGCRN_Engine(BaseEngine):
    def __init__(self, step_size, horizon, **args):
        super(DGCRN_Engine, self).__init__(**args)
        self._step_size = step_size
        self._horizon = horizon
        self._task_level = 0


    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()

            if self._iter_cnt % self._step_size == 0 and self._task_level < self._horizon:
                self._task_level += 1

            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label, self._iter_cnt, self._task_level)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

            pred = pred[:, :self._task_level, :, :]
            label = label[:, :self._task_level, :, :]

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)