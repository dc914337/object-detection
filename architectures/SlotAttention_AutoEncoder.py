import torch
from torch import nn as nn
from torch.nn import functional as F

from architectures.slot_attention.slot_attention import Encoder, Decoder, SlotAttention
from utils import normalize


class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--hid_dim', type=int, default=64)
        parser.add_argument('--resolution', type=int, default=128)
        parser.add_argument('--num_slots', type=int, default=7)
        parser.add_argument('--num_iterations', type=int, default=3)
        parser.add_argument('--lr', type=float, default=0.0004)
        parser.add_argument('--warmup_steps', default=10000, type=int,
                            help='Number of warmup steps for the learning rate.')
        parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
        parser.add_argument('--decay_steps', default=100000, type=int,
                            help='Number of steps for the learning rate decay.')

        return parser

    def __init__(self, cfg, device):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        cfg.model = self.__class__.__name__
        self.hid_dim = cfg.hid_dim
        self.resolution = (cfg.resolution, cfg.resolution)
        self.num_slots = cfg.num_slots
        self.num_iterations = cfg.num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim, device=device)

        self.fc1 = nn.Linear(cfg.hid_dim, cfg.hid_dim)
        self.fc2 = nn.Linear(cfg.hid_dim, cfg.hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=cfg.hid_dim,
            iters=self.num_iterations,
            eps=1e-8,
            hidden_dim=128)

        self.decoder_cnn = Decoder(self.hid_dim, self.resolution, device=device)

        self.optimizer = self.get_optimizer(cfg)
        self.device = device
        self.to(device)
        self.cfg = cfg

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(self.device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3, 1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots

    def get_optimizer(self, cfg):
        return torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def criterion(self, y, y_pred):
        """Computes the loss function."""
        return F.mse_loss(y_pred, y)

    def train_step(self, sample, iteration, total_iter, visualize=False):
        self.train()
        log_dict = {}

        # scheduling
        if iteration < self.cfg.warmup_steps:
            learning_rate = self.cfg.lr * (iteration / self.cfg.warmup_steps)
        else:
            learning_rate = self.cfg.lr

        learning_rate = learning_rate * (self.cfg.decay_rate ** (
                iteration / self.cfg.decay_steps))

        self.optimizer.param_groups[0]['lr'] = learning_rate
        log_dict["lr"] = learning_rate

        frames = torch.stack([a['rgb'][0] for a in sample]).to(self.device) / 255.  # B, F, C, H, W

        recon_combined, recons, masks, slots = self.forward(frames)
        loss = self.criterion(recon_combined, frames)
        log_dict["loss"] = loss.item()
        if visualize:
            log_dict["images"] = self.visualize(frames, recon_combined, recons, masks, slots)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return log_dict

    def validation_step(self, sample, iteration):
        self.eval()
        log_dict = {}

        frames = torch.stack([a['rgb'] for a in sample]).to(self.device)  # B, F, C, H, W
        flows = torch.stack([a['flow'][0] for a in sample]).to(self.device)  # B, C, H, W

        with torch.no_grad():
            recon_combined, recons, masks, slots = self.forward(frames[:, 0, ...])

        loss = self.criterion(recon_combined, frames[:, 0, ...])
        log_dict["loss"] = loss.item()
        del recons, masks, slots

        return log_dict

    def visualize(self, image, recon_combined, recons, masks, slots, max_rows=6):
        """Visualizes the training progress."""
        """input, gt_flow, gt_seg, pred_seg, gt_flow_rec, gt_frec_dif, flow_rec, frec_diff, slot_{}"""
        import matplotlib.pyplot as plt
        from PIL import Image as Image, ImageEnhance
        batch_size = min(image.shape[0], max_rows)
        num_slots = self.num_slots

        fig, ax = plt.subplots(batch_size, num_slots + 2, figsize=(20, 9))

        image = normalize(image.permute(0, 2, 3, 1).cpu().detach().numpy())  # B, H, W, C
        recons = recons.squeeze(0).cpu().detach().numpy()
        recon_combined = normalize(recon_combined.permute(0, 2, 3, 1).cpu().detach().numpy())  # B, H, W, C

        masks = masks.squeeze(0).cpu().detach().numpy()
        picture = recons * masks + (1 - masks)

        for i, title in enumerate(["image", "recon"] + [f"slot {i + 1}" for i in range(num_slots)]):
            ax[0, i].set_title(title)

        for batch_id in range(batch_size):
            ax[batch_id, 0].imshow(image[batch_id])
            ax[batch_id, 0].grid(False)
            ax[batch_id, 0].axis('off')

            ax[batch_id, 1].imshow(recon_combined[batch_id])
            ax[batch_id, 1].grid(False)
            ax[batch_id, 1].axis('off')

            for i in range(num_slots):
                ax[batch_id, i + 2].imshow(picture[batch_id, i])
                ax[batch_id, i + 2].grid(False)
                ax[batch_id, i + 2].axis('off')
        plt.tight_layout()
        return fig
