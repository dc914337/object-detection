from argparse import ArgumentParser
import gc
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
from architectures.RAFT_SlotAttention import RAFT_SlotAttention
from architectures.SlotAttention_AutoEncoder import SlotAttentionAutoEncoder
from movi_dataset import FlowPairMoviDetectron
# from unet import get_unet
import wandb

torch.multiprocessing.set_sharing_strategy('file_system')


def main():

    #MODEL_CLASS = RAFT_SlotAttention
    MODEL_CLASS = SlotAttentionAutoEncoder

    """ARGUMENTS"""
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--visualize_freq', type=int, default=1000)
    parser.add_argument('--eval_freq', type=int, default=2500)
    parser.add_argument('--dataset_path', type=str, default="data/movi_e")
    parser.add_argument('--DEBUG', action='store_true', default=False)
    MODEL_CLASS.add_argparse_args(parser)
    cfg = parser.parse_args()

    """LOGGING"""
    wandb.init(project="Motion-conditioned-object-detection", entity="aalto", config=cfg.__dict__)

    """MODEL"""
    device = torch.device("cuda")
    model = MODEL_CLASS(cfg, device=device).to(device)
    # optimizer = model.get_optimizer(cfg)
    # scheduler = None  # torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)

    # calculate total parameters of all models
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {pytorch_total_params}")

    train_loader = torch.utils.data.DataLoader(
        FlowPairMoviDetectron("train", None, (128, 128),
                              prefix=cfg.dataset_path,
                              gt_flow=True,
                              num_frames=2),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        collate_fn=lambda x: x)

    val_loader = torch.utils.data.DataLoader(
        FlowPairMoviDetectron("validation",
                              None, (128, 128),
                              prefix=cfg.dataset_path,
                              gt_flow=True,
                              num_frames=2),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        collate_fn=lambda x: x)

    wandb.watch(model)

    """TRAINING LOOP"""
    iteration = 0
    total_iter = 200000
    with torch.autograd.set_detect_anomaly(cfg.DEBUG) and tqdm(initial=iteration, total=total_iter,
                                                               disable=utils.is_slurm()) as pbar:
        while iteration < total_iter:
            for sample in train_loader:
                iteration += 1

                visualize = (iteration % cfg.visualize_freq == 0) or (iteration in [3, 50, 100, 300, 500])

                train_log_dict = model.train_step(sample, iteration, total_iter, visualize=visualize)
                # todo: delete train_log_dict values to free memory

                wandb.log(train_log_dict, step=iteration)

                pbar.set_postfix(loss=train_log_dict["loss"])
                pbar.update()

                if (iteration) % 1000 == 0:
                    gc.collect()

                if cfg.DEBUG or (iteration) % 100 == 0:
                    wandb.log(train_log_dict, step=iteration)

                if iteration >= total_iter:
                    print("All iterations done.")


if __name__ == "__main__":
    main()
