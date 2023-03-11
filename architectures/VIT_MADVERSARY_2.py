import random
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

from architectures.vit.vit_mae import ViTMAEModel, ViTMAEDecoderOutput, get_2d_sincos_pos_embed, ViTMAEConfig, ViTMAEDecoder,ViTMAELayer, ViTMAEForPreTraining, ViTMAEEmbeddings, ViTMAEEncoder, ViTMAEPreTrainedModel
from transformers import AutoImageProcessor
import math
from copy import deepcopy


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




class ViTMaskDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    None,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



class Recon(ViTMAEPreTrainedModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_vit = ViTMAEConfig(**cfg.mae.dict())
        super(Recon, self).__init__(self.cfg_vit)

        self.embeddings_recon = ViTMAEEmbeddings(self.cfg_vit)
        self.layernorm_recon = nn.LayerNorm(self.cfg_vit.hidden_size, eps=self.cfg_vit.layer_norm_eps)
        self.vit_encoder_recon = ViTMAEEncoder(self.cfg_vit)
        self.decoder_recon = ViTMAEDecoder(self.cfg_vit, num_patches=self.embeddings_recon.num_patches)
        self.params_recon = list(list(self.embeddings_recon.parameters()) +
                            list(self.layernorm_recon.parameters()) +
                            list(self.vit_encoder_recon.parameters()) +
                            list(self.decoder_recon.parameters()))


    def forward(self, image, mask_ratio=0.75, mask_noise=None, output_attentions = False):
        B = image.shape[0]
        num_patches = self.embeddings_recon.num_patches


        head_mask = self.get_head_mask(None, self.cfg_vit.num_hidden_layers)
        embedding_output, mask_learned, ids_restore = self.embeddings_recon(image, mask_ratio=mask_ratio,
                                                                      noise=mask_noise)

        encoder_outputs = self.vit_encoder_recon(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm_recon(sequence_output)

        # decoder for reconstruction
        decoder_outputs = self.decoder_recon(sequence_output, ids_restore)
        logits = decoder_outputs.logits.reshape(B, num_patches, self.cfg.mae.patch_size, self.cfg.mae.patch_size, 3)  # shape (batch_size, num_patches, patch_size, patch_size) - mask

        return logits.reshape(B, num_patches, -1), mask_learned
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        missing_keys, unexpected_keys = self.load_state_dict(torch.load(path), strict=False)
        print("missing keys: ", missing_keys)
        print("unexpected keys: ", unexpected_keys)


class Masker(Recon):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_vit_masker = ViTMAEConfig(**cfg.mae.dict())
        self.cfg_vit_masker.num_channels = 2
        super(Masker, self).__init__(cfg)
        self.mask_decoder = ViTMaskDecoder(self.cfg_vit_masker, num_patches=self.embeddings_recon.num_patches)
        self.params_masker = self.mask_decoder.parameters()

    def forward(self, image, mask_ratio=0.75, temperature=1, output_attentions=False):
        B = image.shape[0]
        num_patches = self.embeddings_recon.num_patches
        patch_size = self.cfg.mae.patch_size

        # encoder for mask
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        embedding_output, mask_random, ids_restore = self.embeddings_recon(image,
                                                                            mask_ratio=0)  # inputting unmasked image

        encoder_outputs = self.vit_encoder_recon(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm_recon(sequence_output)

        # decoder for mask
        decoder_outputs = self.mask_decoder(sequence_output,
                                              ids_restore)  # shape (batch_size, num_patches, patch_size**2 * channels)
        patch_logits = decoder_outputs.logits.reshape(B, num_patches,
                                                      2)  # shape (batch_size, num_patches, patch_size^2, 2)
        patch_mask_probs = F.gumbel_softmax(patch_logits, tau=temperature, hard=True,
                                            dim=-1)  # shape (batch_size, num_patches, patch_size, patch_size)
        return patch_mask_probs[:, :, 1], patch_logits[:, :, 1]


class VIT_MADVERSARY_2(ViTMAEPreTrainedModel):
    """Slot Attention-based auto-encoder for object discovery."""

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)

        ## training
        parser.add_argument('--lr', type=float, default=3e-5)
        parser.add_argument('--warmup_steps', default=1000, type=int,
                            help='Number of warmup steps for the learning rate.')
        parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
        parser.add_argument('--decay_steps', default=50000, type=int,
                            help='Number of steps for the learning rate decay.')

        ## MAE parameters
        parser.add_argument('--resolution', dest = "mae.image_size", type=int, default=36)
        parser.add_argument('--patch_size', dest = "mae.patch_size", type=int, default=3)

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

        # load params
        parser.add_argument('--recon_path', dest="recon_path", type=str, default=None)#"./checkpoints/recon_madversary_2_best_random.pt")
        parser.add_argument('--masker_path', dest="masker_path", type=str, default=None)#"./checkpoints/recon_madversary_2_best_random.pt")

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
        cfg_masker = ViTMAEConfig(**cfg.mae.dict())
        cfg_masker_decoder = ViTMAEConfig(**cfg.mae.dict())
        cfg_masker_decoder.num_channels = 2 # 0 - no mask, 1 - mask


        cfg_recon = ViTMAEConfig(**cfg.mae.dict())
        super().__init__(cfg_recon)


        # masker
        if False:
            self.embeddings_masker = ViTMAEEmbeddings(cfg_masker)
            self.layernorm_masker = nn.LayerNorm(cfg_masker.hidden_size, eps=cfg_masker.layer_norm_eps)
            self.vit_encoder_masker = ViTMAEEncoder(cfg_masker)

            self.decoder_masker = ViTMaskDecoder(cfg_masker_decoder, num_patches=self.embeddings_masker.num_patches)
            params_masker = list(list(self.embeddings_masker.parameters()) +
                          list(self.layernorm_masker.parameters()) +
                          list(self.vit_encoder_masker.parameters()) +
                          list(self.decoder_masker.parameters()))
            self.optimizer_masker = torch.optim.Adam(params_masker, lr=cfg.lr)

        # masker 2
        self.masker = Masker(cfg)
        self.optimizer_masker = torch.optim.Adam(self.masker.params_masker, lr=cfg.lr)

        # reconstructor
        self.reconstructor = Recon(cfg)
        self.optimizer_recon = torch.optim.Adam(self.reconstructor.params_recon, lr=cfg.lr)

        self.to(device)
        self.cfg = cfg
        self.train_mask = False
        if cfg.recon_path is not None:
            self.reconstructor.load(cfg.recon_path)
        if cfg.masker_path is not None:
            self.masker.load(cfg.recon_path)



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

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss.mean() # loss on reconstruction


    def forward_masker(self, image, mask_ratio=0.75, temperature=1, output_attentions = False):
        B = image.shape[0]
        num_patches = self.embeddings_masker.num_patches
        patch_size = self.cfg.mae.patch_size

        # encoder for mask
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        embedding_output, mask_random, ids_restore = self.embeddings_masker(image, mask_ratio=0) # inputting unmasked image

        encoder_outputs = self.vit_encoder_masker(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm_masker(sequence_output)

        # decoder for mask
        decoder_outputs = self.decoder_masker(sequence_output, ids_restore) # shape (batch_size, num_patches, patch_size**2 * channels)
        patch_logits = decoder_outputs.logits.reshape(B, num_patches, 2) # shape (batch_size, num_patches, patch_size^2, 2)
        patch_mask_probs = F.gumbel_softmax(patch_logits, tau=temperature, hard=False, dim=-1) # shape (batch_size, num_patches, patch_size, patch_size)
        return patch_mask_probs[:, :, 1], patch_logits[:, :, 1]



    def forward(self, image, mask_ratio=0.75, temperature=1., train_recon = False):

        if train_recon:
            image_logits, mask_learned = self.reconstructor(image, mask_ratio=mask_ratio, mask_noise=None)
            loss_masked_recon = self.forward_loss(image, image_logits, mask_learned)
            mask_prob = mask_learned
            mask_logits = mask_prob

        else:
            mask_prob, mask_logits = self.masker(image, mask_ratio=mask_ratio, temperature=temperature)
            image_logits, mask_learned = self.reconstructor(image, mask_ratio=mask_ratio, mask_noise = mask_prob)
            loss_masked_recon = self.forward_loss(image, image_logits, mask_prob)


        patch_means = mask_prob.mean(dim=1)
        loss_mask = ((patch_means - mask_ratio)**2).mean()

        return {
                "loss_masked_recon": loss_masked_recon,
                "loss_mask": loss_mask,
                "logits_masked_img": image_logits,
                "mask_learned": mask_learned,
                "mask_logits": mask_logits,
                "mask_prob": mask_prob}



    def train_step(self, sample, iteration, total_iter, train_recon=False, visualize=False):

        if iteration % 1000 == 0 and train_recon:
            self.reconstructor.save(f"./checkpoints/recon_madversary_2_{iteration}.pt")


        self.train()
        log_dict = {}

        # scheduling
        if iteration < self.cfg.warmup_steps:
            learning_rate = self.cfg.lr * (iteration / self.cfg.warmup_steps)
        else:
            learning_rate = self.cfg.lr
        #mask_ratio = min((iteration / 15000)+0.15, 0.75)
        #mask_ratio = max(1-(iteration / 15000)-0.1, 0.18)
        #mask_ratio = random.uniform(0.15, 0.85)
        mask_ratio = 0.3 # one object
        #temperature = max(0.1-(iteration/10000*0.1), 0.0001)
        temperature = 0.7 #max(0.1-(iteration/10000*0.1), 0.0001)

        learning_rate = learning_rate * (self.cfg.decay_rate ** (
                iteration / self.cfg.decay_steps))

        self.optimizer_recon.param_groups[0]['lr'] = learning_rate
        self.optimizer_masker.param_groups[0]['lr'] = learning_rate
        log_dict["lr"] = learning_rate

        image = sample['image'].permute(0, 3, 1, 2).to(self.device) / 255.  # B, C, W, H



        res = self.forward(image, mask_ratio = mask_ratio, temperature = temperature, train_recon = train_recon)
        recons_masked = self.reconstruct_image_from_logits(res["logits_masked_img"]) # B, H, W, C

        loss_masked_recon = res["loss_masked_recon"]
        loss_mask = res["loss_mask"]

        #scale = 1/loss_masked_recon.item()*loss_random_recon.item()
        #log_dict["scale"] = scale


        image = image.permute(0, 2, 3, 1)
        log_dict["mask_ratio"] = mask_ratio
        log_dict["loss_mask"] = loss_mask
        log_dict["loss_masked_recon"] = loss_masked_recon


        log_dict["train_mask"] = not train_recon
        log_dict["temperature"] = temperature



        if visualize:
            log_dict["images"] = self.visualize(image, recons_masked, res["mask_logits"], res["mask_prob"], res["mask_learned"])

        if train_recon:
            loss = loss_masked_recon
            self.optimizer_recon.zero_grad()
            loss.backward()
            self.optimizer_recon.step()

        else:
            loss = -loss_masked_recon + loss_mask
            self.optimizer_masker.zero_grad()
            loss.backward()
            self.optimizer_masker.step()

        log_dict["loss"] = loss.item()

        return log_dict


    def visualize(self, image, recons_masked, mask_logits, mask_prob, mask_learned, max_rows=6):
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
        recons_masked= recons_masked[:batch_size]
        mask_learned = mask_learned[:batch_size]


        titles = ["image", "mask logits", "mask prob", "mask learned", "masked input", "recon masked"]
        fig, ax = plt.subplots(batch_size, len(titles), figsize=(20, 9))

        patch_size = self.cfg.mae.patch_size
        img_size = image.shape[1]
        patches_num = img_size//patch_size

        image = normalize(image.cpu().detach().numpy())  # B, H, W, C
        recons_masked = normalize(recons_masked.cpu().detach().numpy())  # B, H, W, C

        mask_prob = normalize(mask_prob.cpu().detach().numpy())  # B, H, W, C
        mask_logits = normalize(mask_logits.cpu().detach().numpy())  # B, H, W, C

        mask_learned = F.interpolate(mask_learned.detach().cpu().expand(3,-1,-1).reshape(-1, batch_size, patches_num, patches_num).permute(1,0,2,3), (img_size,img_size)).permute(0,2,3,1)  # B, H, W, C
        masked_input_learned = (1-mask_learned) * image

        for i, title in enumerate(titles):
            ax[0, i].set_title(title)

        for batch_id in range(batch_size):
            ax[batch_id, 0].imshow(image[batch_id])
            ax[batch_id, 0].grid(False)
            ax[batch_id, 0].axis('off')

            # learned mask
            ax[batch_id, 1].imshow(mask_logits[batch_id].reshape(patches_num, patches_num))
            ax[batch_id, 1].grid(False)
            ax[batch_id, 1].axis('off')

            ax[batch_id, 2].imshow(mask_prob[batch_id].reshape(patches_num, patches_num))
            ax[batch_id, 2].grid(False)
            ax[batch_id, 2].axis('off')


            ax[batch_id, 3].imshow(mask_learned[batch_id])
            ax[batch_id, 3].grid(False)
            ax[batch_id, 3].axis('off')

            ax[batch_id, 4].imshow(masked_input_learned[batch_id])
            ax[batch_id, 4].grid(False)
            ax[batch_id, 4].axis('off')

            ax[batch_id, 5].imshow(masked_input_learned[batch_id] + mask_learned[batch_id]*recons_masked[batch_id])
            ax[batch_id, 5].grid(False)
            ax[batch_id, 5].axis('off')

            #for i in range(num_slots):
            #    ax[batch_id, i + 2].imshow(picture[batch_id, i])
            #    ax[batch_id, i + 2].grid(False)
            #    ax[batch_id, i + 2].axis('off')
        plt.tight_layout()
        plt.close()
        return fig
