import torch
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import compute_all_metrics

class HL_Engine(BaseEngine):
    def __init__(self, **args):
        super(HL_Engine, self).__init__(**args)


    def evaluate(self, mode):
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        test_mae = []
        test_mape = []
        test_rmse = []
        print('check mask value', mask_value)
        for i in range(self.model.horizon):
            res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
            log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
            test_mae.append(res[0])
            test_mape.append(res[1])
            test_rmse.append(res[2])

        log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))