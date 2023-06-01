from argparse import ArgumentParser
import gc
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, Pad
from tqdm import tqdm
import utils
from architectures.BEST_PATCHED_BASELINE import BEST_PATCHED_BASELINE
from architectures.SlotAttention_AutoEncoder import SlotAttentionAutoEncoder
from architectures.VIT_MADVERSARY_slotattention import VIT_MADVERSARY_slotattention
import wandb
from architectures.VIT_MADVERSARY_2_cluster import VIT_MADVERSARY_2_cluster
from architectures.VIT_MADVERSARY_2 import VIT_MADVERSARY_2
from configs.uconfig import YamlConfig
from dataloaders.h5_loader import TetrominoesDataset

torch.multiprocessing.set_sharing_strategy('file_system')


def main():


    #MODEL_CLASS = VIT_MADVERSARY_slotattention
    #MODEL_CLASS = VIT_MADVERSARY_2_slots

    """ARGUMENTS"""
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--config_path', type=str, default="configs/vit_madversary_clustering.yaml")

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--visualize_freq', type=int)
    parser.add_argument('--eval_epoch_freq', type=int)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--DEBUG', type=bool, default=False)
    #MODEL_CLASS.add_argparse_args(parser)
    cfg_args = parser.parse_args()

    cfg = YamlConfig(cfg_args.config_path)
    # merge config with arguments
    cfg.output_dir = cfg_args.output_dir or cfg.output_dir
    cfg.batch_size = cfg_args.batch_size or cfg.batch_size
    cfg.visualize_freq = cfg_args.visualize_freq or cfg.visualize_freq
    cfg.eval_freq = cfg_args.eval_epoch_freq or cfg.eval_epoch_freq
    cfg.dataset_path = cfg_args.dataset_path or cfg.dataset_path
    cfg.DEBUG = cfg_args.DEBUG or cfg.DEBUG


    if cfg.model_class == "VIT_MADVERSARY_2":
        MODEL_CLASS = VIT_MADVERSARY_2
    elif cfg.model_class == "BEST_PATCHED_BASELINE":
        MODEL_CLASS = BEST_PATCHED_BASELINE
    else:
        raise Exception("Model class not found")


    """LOGGING"""
    wandb.init(project="Motion-conditioned-object-detection", entity="aalto", notes=cfg.notes, config=cfg.dict())
    wandb.run.log_code(".")

    """MODEL"""
    device = torch.device("cuda")

    model = MODEL_CLASS(cfg.model, device=device)

    # calculate total parameters of all models
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {pytorch_total_params}")

    train_loader = torch.utils.data.DataLoader(
        TetrominoesDataset(cfg.dataset_path, recursive = False, load_data = True, transform=Pad((1, 1, 0, 0), fill=0)),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=7,
        pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        TetrominoesDataset(cfg.test_dataset_path, recursive = False, load_data = True, transform=Pad((1, 1, 0, 0), fill=0)),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=7,
        pin_memory=False,
        drop_last=True)

    wandb.watch(model)

    """TRAINING LOOP"""
    iteration = 0
    epoch = 1
    total_iter = cfg.total_iter
    train_recon = cfg.train_recon
    with torch.autograd.set_detect_anomaly(cfg.DEBUG) and tqdm(initial=iteration, total=total_iter,
                                                               disable=utils.is_slurm()) as pbar:
        while iteration < total_iter:
            for batch in train_loader:
                iteration += 1

                visualize = (iteration % cfg.visualize_freq == 0) or (iteration in [3, 50, 100, 300, 500])

                if iteration % cfg.switch_training_recon_masker_steps == 0:
                    train_recon = not train_recon

                log_dict = {}

                model.train_step(batch, iteration, total_iter, log_dict, visualize=visualize, train_recon=train_recon)
                # todo: delete train_log_dict values to free memory

                log_dict["epoch"] = epoch
                wandb.log(log_dict, step=iteration)

                #pbar.set_postfix(loss=log_dict["loss"])
                pbar.update()


                if (iteration) % 1000 == 0:
                    gc.collect()

                if iteration >= total_iter:
                    print("All iterations done.")
                    break
            epoch += 1

            if epoch % cfg.eval_epoch_freq == 0:

                # eval model
                eval_dicts = []
                for batch in test_loader:
                    eval_dict = model.eval_step(batch)
                    eval_dicts.append(eval_dict)

                # merge dicts
                eval_dict = {}
                for k in eval_dicts[0].keys():
                    if k not in {"eval.images", "eval.mask", "eval.mask_real", "eval.logits_masked_img"}:
                        #if float convert to tensor if not already and take mean
                        if isinstance(eval_dicts[0][k], torch.Tensor):
                            eval_dict[k] = torch.stack([d[k] for d in eval_dicts]).mean()
                        else:
                            eval_dict[k] = torch.stack([torch.as_tensor(float(d[k])) for d in eval_dicts]).mean()


                eval_dict["eval.images"] = eval_dicts[0]["eval.images"]
                eval_dict["epoch"] = epoch
                wandb.log(eval_dict, step=iteration)


if __name__ == "__main__":
    main()
