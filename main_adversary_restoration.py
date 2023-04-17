from argparse import ArgumentParser
import gc
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
from tqdm import tqdm
import utils
from architectures.SlotAttention_AutoEncoder import SlotAttentionAutoEncoder
import wandb
from architectures.VIT_MADVERSARY_2_cluster import VIT_MADVERSARY_2_cluster
from architectures.VIT_MADVERSARY_2 import VIT_MADVERSARY_2
from dataloaders.h5_loader import TetrominoesDataset

torch.multiprocessing.set_sharing_strategy('file_system')


def main():

    MODEL_CLASS = VIT_MADVERSARY_2
    #MODEL_CLASS = VIT_MADVERSARY_2_slots

    """ARGUMENTS"""
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--visualize_freq', type=int, default=300)
    parser.add_argument('--eval_freq', type=int, default=2500)
    parser.add_argument('--dataset_path', type=str, default="data/EMORL/tetrominoes.h5")
    parser.add_argument('--DEBUG', action='store_true', default=False)
    MODEL_CLASS.add_argparse_args(parser)
    cfg = parser.parse_args()

    """LOGGING"""
    wandb.init(project="Motion-conditioned-object-detection", entity="aalto", config=cfg.__dict__)

    """MODEL"""
    device = torch.device("cuda")
    model = MODEL_CLASS(cfg, device=device)

    # calculate total parameters of all models
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {pytorch_total_params}")

    train_loader = torch.utils.data.DataLoader(
        TetrominoesDataset(cfg.dataset_path, recursive = False, load_data = True, transform=Resize((36, 36))),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=7,
        pin_memory=False)

    wandb.watch(model)

    """TRAINING LOOP"""
    iteration = 0
    epoch = 1
    total_iter = 300000
    with torch.autograd.set_detect_anomaly(cfg.DEBUG) and tqdm(initial=iteration, total=total_iter,
                                                               disable=utils.is_slurm()) as pbar:
        while iteration < total_iter:
            for batch in train_loader:
                iteration += 1

                visualize = (iteration % cfg.visualize_freq == 0) or (iteration in [3, 50, 100, 300, 500])

                train_log_dict = model.train_step(batch, iteration, total_iter, visualize=visualize)
                # todo: delete train_log_dict values to free memory

                train_log_dict["epoch"] = epoch
                wandb.log(train_log_dict, step=iteration)

                pbar.set_postfix(loss=train_log_dict["loss"])
                pbar.update()

                if (iteration) % 1000 == 0:
                    gc.collect()

                if iteration >= total_iter:
                    print("All iterations done.")
                    break
            epoch += 1


if __name__ == "__main__":
    main()
