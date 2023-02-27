from argparse import Namespace

import torch
from torch import nn as nn
from torch.nn import functional as F

import utils
from architectures.slot_attention.slot_attention import Encoder, Decoder, SlotAttention
from utils import normalize
import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance
from functools import partial
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, RmsNorm

from architectures.vit.vit_mae import ViTMAEModel, ViTMAEConfig, ViTMAEDecoder, ViTMAEForPreTraining, ViTMAEEmbeddings, ViTMAEEncoder, ViTMAEPreTrainedModel
from transformers import AutoImageProcessor
import math


class Config(Namespace):
    @classmethod
    def from_cfg(cls, cfg: Namespace):
        n = Config()
        for k, v in cfg.__dict__.items():
            n.__setattr__(k, v)
        return n

    def __setattr__(self, name, value):
        if '.' in name:
            name = name.split('.')
            name, rest = name[0], '.'.join(name[1:])
            if not hasattr(self, name):
                setattr(self, name, type(self)())
            setattr(getattr(self, name), rest, value)
        else:
            super().__setattr__(name, value)

    def dict(self):
        dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                dict[k] = v.dict()
            else:
                dict[k] = v
        return dict








class VIT_MADVERSARY(ViTMAEPreTrainedModel):
    """Slot Attention-based auto-encoder for object discovery."""

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)

        ## training
        parser.add_argument('--lr', type=float, default=3e-5)
        parser.add_argument('--warmup_steps', default=1000, type=int,
                            help='Number of warmup steps for the learning rate.')
        parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
        parser.add_argument('--decay_steps', default=20000, type=int,
                            help='Number of steps for the learning rate decay.')

        ## MAE parameters
        parser.add_argument('--resolution', dest = "mae.image_size", type=int, default=35)
        parser.add_argument('--patch_size', dest = "mae.patch_size", type=int, default=5)

        # encoder
        parser.add_argument('--enc_num_layers',dest = "mae.num_hidden_layers", type=int, default=5)
        parser.add_argument('--enc_num_heads',dest = "mae.num_attention_heads", type=int, default=12)
        parser.add_argument('--enc_hidden_size', dest = 'mae.hidden_size', type=int, default=768)

        parser.add_argument('--enc_intermediate_size',dest = 'mae.intermediate_size', type=int, default=3072)
        parser.add_argument('--dec_intermediate_size',dest = 'mae.decoder_intermediate_size', type=int, default=2048)

        # decoder
        parser.add_argument('--dec_num_layers',dest = "mae.decoder_num_hidden_layers", type=int, default=4)
        parser.add_argument('--dec_num_heads',dest = "mae.decoder_num_attention_heads", type=int, default=16)
        parser.add_argument('--dec_hidden_size', dest = 'mae.decoder_hidden_size', type=int, default=512)

        parser.add_argument('--mask_ratio', dest = "mae.mask_ratio", type=float, default=0.75)

        return parser

    def __init__(self, cfg, device):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        cfg.model = self.__class__.__name__
        cfg = Config.from_cfg(cfg)
        cfg_ae = ViTMAEConfig(**cfg.mae.dict())
        super().__init__(cfg_ae)


        cfg_mask = ViTMAEConfig(**cfg.mae.dict())
        cfg_mask.num_channels = 1

        # encoder
        self.embeddings = ViTMAEEmbeddings(cfg_ae)
        self.layernorm = nn.LayerNorm(cfg_ae.hidden_size, eps=cfg_ae.layer_norm_eps)
        self.vit_encoder = ViTMAEEncoder(cfg_ae)

        # decoders
        self.mask_decoder = ViTMAEDecoder(cfg_mask, num_patches=self.embeddings.num_patches)
        self.rec_decoder = ViTMAEDecoder(cfg_ae, num_patches=self.embeddings.num_patches)

        encoder_params = list(self.embeddings.parameters()) + list(self.layernorm.parameters()) + list(self.vit_encoder.parameters())
        self.optimizer_rec = torch.optim.Adam(encoder_params + list(self.rec_decoder.parameters()), lr=cfg.lr)
        self.optimizer_mask = torch.optim.Adam(encoder_params + list(self.mask_decoder.parameters()), lr=cfg.lr)
        self.to(device)
        self.cfg = cfg



    def reconstruct_image_from_logits(self, logits):
        """Reconstructs an image from logits.
        Args:
        logits: Tensor of shape [B, H, W, C] containing logits.
        Returns:
        Tensor of shape [B, H, W, C] containing the reconstructed image.
        """
        patched_pixels = torch.sigmoid(logits)
        batch_size = logits.shape[0]
        num_patches = int(math.sqrt(logits.shape[1]))
        patches = patched_pixels.reshape(batch_size,
                                    num_patches,
                                    num_patches,
                                    self.cfg.mae.patch_size,
                                    self.cfg.mae.patch_size,
                                    3)
        image = torch.zeros(batch_size, self.cfg.mae.image_size, self.cfg.mae.image_size, 3).to(self.device)
        for i in range(num_patches):
            for j in range(num_patches):
                image[:,
                i*self.cfg.mae.patch_size:(i+1)*self.cfg.mae.patch_size,
                j*self.cfg.mae.patch_size:(j+1)*self.cfg.mae.patch_size, :] = patches[:, i, j, :, :, :]
        return image


    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values


    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        #loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss.mean() # loss on reconstruction



    def forward(self, image, mask_ratio=0.75, output_attentions = False):
        B = image.shape[0]
        patches_num = self.embeddings.num_patches

        ### MASKING
        # encoder for mask
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        embedding_output, mask, ids_restore = self.embeddings(image, mask_ratio = 0)

        encoder_outputs = self.vit_encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        # decoder for mask
        decoder_outputs = self.mask_decoder(sequence_output, ids_restore)
        logits_mask = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        mask_prob = torch.sigmoid(logits_mask)
        patch_mask_probs = utils.normalize(mask_prob.sum(dim=2).to(self.device))


        ### RECONSTRUCTION
        # encoder for reconstruction
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        embedding_output, mask, ids_restore = self.embeddings(image, mask_ratio= mask_ratio, noise = patch_mask_probs)

        encoder_outputs = self.vit_encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        # decoder for reconstruction
        decoder_outputs = self.rec_decoder(sequence_output, ids_restore)
        logits_img = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)


        loss_recon = self.forward_loss(image, logits_img, mask)
        return loss_recon, logits_img, mask, ids_restore, None



    def train_step(self, sample, iteration, total_iter, visualize=False):
        self.train()
        log_dict = {}

        # scheduling
        if iteration < self.cfg.warmup_steps:
            learning_rate = self.cfg.lr * (iteration / self.cfg.warmup_steps)
        else:
            learning_rate = self.cfg.lr
        mask_ratio = 0.3 # max(iteration / total_iter, 0.2)

        learning_rate = learning_rate * (self.cfg.decay_rate ** (
                iteration / self.cfg.decay_steps))

        self.optimizer_rec.param_groups[0]['lr'] = learning_rate
        self.optimizer_mask.param_groups[0]['lr'] = learning_rate
        log_dict["lr"] = learning_rate

        image = sample['image'].permute(0, 3, 1, 2).to(self.device) / 255.  # B, C, W, H



        loss, logits, mask, ids_restore, attentions = self.forward(image, mask_ratio = mask_ratio, output_attentions = True)

        recons = self.reconstruct_image_from_logits(logits) # B, H, W, C
        image = image.permute(0, 2, 3, 1)

        log_dict["mask_ratio"] = mask_ratio
        log_dict["loss"] = loss.item()
        if visualize:
            log_dict["images"] = self.visualize(image, recons, attentions, mask)

        loss_mask = -loss
        self.optimizer_mask.zero_grad()
        loss_mask.backward(retain_graph=True)
        self.optimizer_mask.step()


        self.optimizer_rec.zero_grad()
        loss.backward()
        self.optimizer_rec.step()

        return log_dict


    def visualize(self, image, recons, attentions, mask, max_rows=6):
        """Visualizes the training progress."""
        """
        Args:
        image: Tensor of shape [B, H, W, C] containing the input image.
        recons: Tensor of shape [B, H, W, C] containing the reconstructed image.
        attentions: List of Tensors of shape [B, H, W, H, W] containing the
        mask: Tensor of shape [B, H, W, C] containing the mask.
        max_rows: Maximum number of rows in the visualization."""
        """input, reconstruction, attention_layer {}"""

        batch_size = min(image.shape[0], max_rows)
        image = image[:batch_size]
        recons = recons[:batch_size]
        mask = mask[:batch_size]
        # todo: cut attention

        titles = ["image", "mask"] + ["masked"] + ["recon"] # + [f"attention {i + 1}" for i in range(self.cfg.mae.decoder_num_hidden_layers + self.cfg.mae.num_hidden_layers)]
        fig, ax = plt.subplots(batch_size, len(titles), figsize=(20, 9))

        patch_size = self.cfg.mae.patch_size
        img_size = image.shape[1]
        patches_num = img_size//patch_size

        image = normalize(image.cpu().detach().numpy())  # B, H, W, C
        recons = normalize(recons.cpu().detach().numpy())  # B, H, W, C

        mask = F.interpolate(mask.detach().cpu().expand(3,-1,-1).reshape(-1, batch_size, patches_num, patches_num).permute(1,0,2,3), (img_size,img_size)).permute(0,2,3,1)  # B, H, W, C
        masked_input = (1-mask) * image

        for i, title in enumerate(titles):
            ax[0, i].set_title(title)

        for batch_id in range(batch_size):
            ax[batch_id, 0].imshow(image[batch_id])
            ax[batch_id, 0].grid(False)
            ax[batch_id, 0].axis('off')

            ax[batch_id, 1].imshow(mask[batch_id])
            ax[batch_id, 1].grid(False)
            ax[batch_id, 1].axis('off')

            ax[batch_id, 2].imshow(masked_input[batch_id])
            ax[batch_id, 2].grid(False)
            ax[batch_id, 2].axis('off')

            ax[batch_id, 3].imshow(recons[batch_id])
            ax[batch_id, 3].grid(False)
            ax[batch_id, 3].axis('off')

            #for i in range(num_slots):
            #    ax[batch_id, i + 2].imshow(picture[batch_id, i])
            #    ax[batch_id, i + 2].grid(False)
            #    ax[batch_id, i + 2].axis('off')
        plt.tight_layout()
        plt.close()
        return fig
