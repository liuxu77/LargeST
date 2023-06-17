import os
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.stid import STID
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--node_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--num_layer', type=int, default=4)
    parser.add_argument('--temp_dim_tid', type=int, default=32)
    parser.add_argument('--temp_dim_diw', type=int, default=32)
    parser.add_argument('--time_of_day_size', type=int, default=96)
    parser.add_argument('--day_of_week_size', type=int, default=7)
    parser.add_argument('--if_time_in_day', type=bool, default=True)
    parser.add_argument('--if_day_in_week', type=bool, default=True)
    parser.add_argument('--if_spatial', type=bool, default=True)

    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    folder_name = '{}'.format(args.dataset)
    log_dir = './experiments/{}/{}/'.format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, _, node_num = get_dataset_info(args.dataset)

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = STID(node_num=node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 model_args={
                    "node_dim": args.node_dim,
                    "input_len": args.seq_len,
                    "embed_dim": args.embed_dim,
                    "output_len": args.horizon,
                    "num_layer": args.num_layer,
                    "temp_dim_tid": args.temp_dim_tid,
                    "temp_dim_diw": args.temp_dim_diw,
                    "time_of_day_size": args.time_of_day_size,
                    "day_of_week_size": args.day_of_week_size,
                    "if_T_i_D": args.if_time_in_day,
                    "if_D_i_W": args.if_day_in_week,
                    "if_node": args.if_spatial
                    }
                  )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * 10 for i in range(1, 10)], gamma=0.5)

    engine = BaseEngine(device=device,
                        model=model,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.lrate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        seed=args.seed
                        )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()