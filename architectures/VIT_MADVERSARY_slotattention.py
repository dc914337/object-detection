import random
from argparse import Namespace

import torch
from torch import nn as nn
from torch.nn import functional as F
import collections.abc

import utils
from architectures.slot_attention.slot_attention import Encoder, Decoder, SlotAttention, SoftPositionEmbed
from utils import normalize
import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance
from functools import partial
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, RmsNorm

from architectures.vit.vit_mae import ViTMAEDecoderOutput, get_2d_sincos_pos_embed, ViTMAEConfig, ViTMAEDecoder,ViTMAELayer, ViTMAEEncoder, ViTMAEPreTrainedModel
from transformers import AutoImageProcessor
import math
from copy import deepcopy

import numpy as np



class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pad_token = nn.Parameter(torch.zeros(1, config.hidden_size))


        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)
        torch.nn.init.normal_(self.pad_token, std=self.config.initializer_range)


    def padded_masking(self, sequence, mask=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        noise_token = mask.reshape(batch_size, seq_length, 1)
        # apply mask
        sequence_masked = sequence * (1.-noise_token) + noise_token * self.pad_token.reshape(1, 1, dim)

        return sequence_masked

    def forward(self, pixel_values, mask=None):
        """
        Either mask_ratio or mask must be specified but not both.


        """

        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * mask_ratio
        embeddings = self.padded_masking(embeddings, mask=mask)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings


class ViTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x




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
        self.cfg = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.cfg.initializer_range)

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

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
        self.cfg_vit = ViTMAEConfig(**cfg.dict())
        super(Recon, self).__init__(self.cfg_vit)

        self.embeddings_recon = ViTMAEEmbeddings(self.cfg_vit)
        self.layernorm_recon = nn.LayerNorm(self.cfg_vit.hidden_size, eps=self.cfg_vit.layer_norm_eps)
        self.vit_encoder_recon = ViTMAEEncoder(self.cfg_vit)
        self.decoder_recon = ViTMAEDecoder(self.cfg_vit, num_patches=self.embeddings_recon.num_patches)
        self.params = list(list(self.embeddings_recon.parameters()) +
                            list(self.layernorm_recon.parameters()) +
                            list(self.vit_encoder_recon.parameters()) +
                            list(self.decoder_recon.parameters()))

    def generate_random_mask(self, batch_size, random_mask_ratio):
        num_patches = self.embeddings_recon.num_patches
        mask = (torch.rand(batch_size, num_patches, device=self.device) <= random_mask_ratio).to(torch.float) # noise in [0, 1]
        return mask

    def forward(self, image, mask=None, output_attentions = False):
        B = image.shape[0]
        num_patches = self.embeddings_recon.num_patches


        head_mask = self.get_head_mask(None, self.cfg_vit.num_hidden_layers)

        embedding_output = self.embeddings_recon(image, mask=mask)

        encoder_outputs = self.vit_encoder_recon(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm_recon(sequence_output)

        # decoder for reconstruction
        decoder_outputs = self.decoder_recon(sequence_output)
        logits = decoder_outputs.logits.reshape(B, num_patches, self.cfg.patch_size, self.cfg.patch_size, 3)  # shape (batch_size, num_patches, patch_size, patch_size) - mask

        return logits.reshape(B, num_patches, -1)
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        missing_keys, unexpected_keys = self.load_state_dict(torch.load(path), strict=False)
        print("missing keys: ", missing_keys)
        print("unexpected keys: ", unexpected_keys)


class Masker(nn.Module):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        super(Masker, self).__init__()

        self.encoder_cnn = Encoder((cfg.image_size, cfg.image_size), cfg.hidden_size)

        self.fc1 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)

        self.slot_attention = SlotAttention(
            num_slots=cfg.decoder_num_classes,
            dim=cfg.hidden_size,
            iters=3,
            eps=1e-8,
            hidden_dim=cfg.hidden_size)

        self.decoder_cnn = Decoder(cfg.hidden_size, (cfg.image_size, cfg.image_size))



        self.params = list(list(self.encoder_cnn.parameters()) +
                           list(self.fc1.parameters()) +
                           list(self.fc2.parameters()) +
                           list(self.slot_attention.parameters()) +
                           list(self.decoder_cnn.parameters()))


    def forward(self, image, temperature=1, output_attentions=False):
        B = image.shape[0]
        patch_size = self.cfg.patch_size
        num_patches = 144
        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        #x = nn.LayerNorm(x.shape[1:]).to(self.device)(x)
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
        #masks = nn.Softmax(dim=1)(masks).squeeze()

        masks = masks.squeeze()
        patch_logits = F.gumbel_softmax(masks, tau=temperature, hard=True, dim=1)
        patch_mask_probs = F.max_pool2d(patch_logits, kernel_size=3, stride=3).permute(0,2,3,1).reshape(B, num_patches, self.cfg.decoder_num_classes)  # convert to patches

        #recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        #recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].


        return patch_mask_probs[:, :, :], patch_mask_probs[:, :, :]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        # update all the keys in the state dict from recon to mask
        state_dict = torch.load(path)
        for key in list(state_dict.keys()):
            if 'decoder_pred' in key:
                state_dict.pop(key)
                continue # don't need these keys because

            if 'recon' in key:
                state_dict[key.replace('recon', 'mask')] = state_dict.pop(key)



        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print("missing keys: ", missing_keys)
        print("unexpected keys: ", unexpected_keys)



def reconstruct_image_from_logits(logits, patch_size, img_size, device):
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
                                     patch_size,
                                     patch_size,
                                     3)
    image = torch.zeros(batch_size, img_size, img_size, 3).to(device)
    for i in range(num_patches):
        for j in range(num_patches):
            image[:,
            i * patch_size:(i + 1) * patch_size,
            j * patch_size:(j + 1) * patch_size, :] = patches[:, i, j, :, :, :]
    return image

def patchify(pixel_values, patch_size, num_channels):
    """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
            Patchified pixel values.
    """
    patch_size, num_channels = patch_size, num_channels
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
        batch_size, num_patches_one_direction * num_patches_one_direction, patch_size ** 2 * num_channels
    )
    return patchified_pixel_values




class VIT_MADVERSARY_slotattention(nn.Module):
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
        super().__init__()
        cfg.model = self.__class__.__name__
        self.cfg = cfg
        self.device = device

        # masker 2
        self.masker = Masker(cfg.masker, device)
        self.optimizer_masker = torch.optim.Adam(self.masker.params, lr=cfg.lr)

        # reconstructor
        self.reconstructor = Recon(cfg.reconstructor)
        self.optimizer_recon = torch.optim.Adam(self.reconstructor.params, lr=cfg.lr)

        self.to(device)
        self.train_mask = False
        if cfg.reconstructor.pretrained_path is not None:
            self.reconstructor.load(cfg.reconstructor.pretrained_path)
        if cfg.masker.pretrained_path is not None:
            self.masker.load(cfg.masker.pretrained_path)


        # freeze masker and recon
        #for param in list(self.reconstructor.parameters()) + \
        #             list(self.masker.vit_encoder_mask.parameters()) + \
        #             list(self.masker.embeddings_mask.parameters()) + \
        #             list(self.masker.layernorm_mask.parameters()):
        #    param.requires_grad = False



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
        target = patchify(pixel_values, self.cfg.patch_size, num_channels=3)
        if self.cfg.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=1)  # [N, L], mean loss per patch

        #nonzero_mask = (mask.sum(dim=1) != 0.) # some of the items may be zero
        #if not nonzero_mask.any():
        #    return 0.  # if all masks are empty - we can't calculate the loss
        #loss = (loss * mask)[nonzero_mask].mean()   #/ mask[nonzero_mask].sum(dim=1)  # mean loss on removed patches

        #assert loss[mask==0].max() < 0.1

        return loss.mean() # loss on reconstruction

    def forward(self, image, mask_ratio=0.75, temperature=1., train_recon=False):
        if train_recon:
            return self.forward_recon(image, mask_ratio=mask_ratio)
        else:
            return self.forward_masker(image, mask_ratio=mask_ratio, temperature=temperature)

    def forward_recon(self, image, mask_ratio=0.75):
        B, C, H, W = image.shape

        mask = self.reconstructor.generate_random_mask(B, mask_ratio)
        image_logits = self.reconstructor(image, mask=mask)
        loss_masked_recon = self.forward_loss(image, image_logits, mask)

        # hack
        #image_logits = image_logits.reshape(B, 144, 27, 1).expand(-1, -1, -1, 3)
        mask_learned = mask.reshape(B, 144, 1).expand(-1, -1, 4)
        #mask_logits = mask_logits.reshape(B, 144, 1).expand(-1, -1, 4)
        mask_prob = mask

        return {
            "image_logits": image_logits,
            "mask": mask,
            "loss_recon": loss_masked_recon,
        }

    def forward_masker(self, image, mask_ratio=0.75, temperature=1.):
        mask_prob, mask_logits = self.masker(image, temperature=temperature)
        num_classes = mask_prob.shape[-1]

        image_logits_list = []
        mask_learned_list = []
        loss_masks = []
        loss_recons = []
        loss_masked_recon = 0.
        losses_mask = 0.

        res_dict = {}

        for i in range(1, num_classes):
            learned_mask = mask_prob[:, :, i]

            image_logits = self.reconstructor(image, mask = learned_mask)

            loss_masked_recon_ = self.forward_loss(image, image_logits, learned_mask)
            loss_masked_recon += loss_masked_recon_
            image_logits_list.append(image_logits.detach())
            mask_learned_list.append(learned_mask.detach())

            patch_means = mask_prob[:, :, i].mean(dim=1)
            mask_loss = ((patch_means - mask_ratio) ** 2).mean()

            losses_mask += mask_loss
            loss_masks.append(mask_loss.detach().item())
            loss_recons.append(loss_masked_recon_.detach().item())

        res_dict.update(
            {
                "mask_logits": mask_logits,
                "mask": mask_prob,
                "mask_real": mask_learned_list,

                "logits_masked_img": image_logits_list,
                "loss_masked_recon": loss_masked_recon,
                "loss_mask": losses_mask,

                "loss_recons": loss_recons,
                "loss_masks": loss_masks,
            }
        )
        return res_dict



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
        if train_recon:
            mask_ratio = random.uniform(0.15, 0.85)
        else:
            mask_ratio = self.cfg.object_size # one object

        #temperature = max(0.1-(iteration/10000*0.1), 0.0001)
        temperature = self.cfg.temperature #max(0.1-(iteration/10000*0.1), 0.03)

        learning_rate = learning_rate * (self.cfg.decay_rate ** (
                iteration / self.cfg.decay_steps))

        self.optimizer_recon.param_groups[0]['lr'] = learning_rate
        self.optimizer_masker.param_groups[0]['lr'] = learning_rate
        log_dict["lr"] = learning_rate

        image = sample['image'].permute(0, 3, 1, 2).to(self.device) / 255.  # B, C, W, H



        res = self.forward(image, mask_ratio = mask_ratio, temperature = temperature, train_recon = train_recon)

        loss_masked_recon = res["loss_masked_recon"]
        loss_mask = res["loss_mask"]


        image = image.permute(0, 2, 3, 1)
        log_dict["mask_ratio"] = mask_ratio
        log_dict["loss_mask"] = loss_mask.item()
        log_dict["loss_masked_recon"] = loss_masked_recon
        for i in range(len(res["loss_recons"])):
            log_dict[f"loss_recon_{i}"] = res["loss_recons"][i]
            log_dict[f"loss_mask_{i}"] = res["loss_masks"][i]

        log_dict["train_mask"] = not train_recon
        log_dict["temperature"] = temperature


        if visualize:
            log_dict["images"] = self.visualize(image, res)

        if train_recon:
            loss = loss_masked_recon
            self.optimizer_recon.zero_grad()
            loss.backward()
            self.optimizer_recon.step()

        else:
            loss = -loss_masked_recon + loss_mask
            self.optimizer_masker.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.masker.params, 1e-4)
            self.optimizer_masker.step()

        log_dict["loss"] = loss.item()

        return log_dict


    def visualize(self, image, res_dict, max_rows=6):
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
        recons_masked_imgs = []
        num_recons = len(res_dict["logits_masked_img"])

        for i in range(num_recons):
            logits_masked_img = res_dict["logits_masked_img"][i][:batch_size, :, :]
            recons_masked = reconstruct_image_from_logits(logits_masked_img,
                                                          patch_size = self.cfg.patch_size,
                                                          img_size = self.cfg.image_size,
                                                          device=self.device) # B, H, W, C
            recons_masked = normalize(recons_masked.cpu().detach().numpy())  # B, H, W, C
            recons_masked_imgs.append(recons_masked)

        masks_learned = res_dict["mask"][:batch_size]


        titles = ["image"]
        for i in range(num_recons):
            titles.append(f"mask logits {i+1}")
            titles.append(f"masked {i+1}")
            titles.append(f"recon {i+1}")

        fig, ax = plt.subplots(batch_size, len(titles), figsize=(20, 9))

        patch_size = self.cfg.patch_size
        img_size = image.shape[1]
        patches_num = img_size//patch_size

        image = normalize(image.cpu().detach().numpy())  # B, H, W, C

        mask_prob = normalize(res_dict["mask"].cpu().detach().numpy())  # B, H, W, C
        mask_logits = normalize(res_dict["mask_logits"].cpu().detach().numpy())  # B, H, W, C

        #masked_input_learned = (1-mask_learned) * image

        for i, title in enumerate(titles):
            ax[0, i].set_title(title)

        for batch_id in range(batch_size):
            ax[batch_id, 0].imshow(image[batch_id])
            ax[batch_id, 0].grid(False)
            ax[batch_id, 0].axis('off')


            for i in range(len(recons_masked_imgs)):
                # mask logits
                ax[batch_id, i*num_recons+1].imshow(mask_logits[batch_id, :, i+1].reshape(patches_num, patches_num), vmin=0, vmax=1)
                ax[batch_id, i*num_recons+1].grid(False)
                ax[batch_id, i*num_recons+1].axis('off')

                RGB_CHANNELS = 3
                mask_learned = F.interpolate(
                    masks_learned[:,:,i+1].detach().cpu().expand(RGB_CHANNELS, -1, -1).reshape(-1, batch_size, patches_num,
                                                                          patches_num).permute(1, 0, 2, 3),
                    (img_size, img_size)).permute(0, 2, 3, 1)  # B, H, W, C

                masked_input_learned = (1 - mask_learned) * image


                # masked
                ax[batch_id, i*num_recons+2].imshow(masked_input_learned[batch_id] + mask_learned[batch_id])
                ax[batch_id, i*num_recons+2].grid(False)
                ax[batch_id, i*num_recons+2].axis('off')

                # recon
                ax[batch_id, i*num_recons+3].imshow(recons_masked_imgs[i][batch_id])
                ax[batch_id, i*num_recons+3].grid(False)
                ax[batch_id, i*num_recons+3].axis('off')






            #ax[batch_id, 2].imshow(mask_prob[batch_id].reshape(patches_num, patches_num))
            #ax[batch_id, 2].grid(False)
            #ax[batch_id, 2].axis('off')


            #ax[batch_id, 3].imshow(mask_learned[batch_id])
            #ax[batch_id, 3].grid(False)
            #ax[batch_id, 3].axis('off')

            #ax[batch_id, 4].imshow(masked_input_learned[batch_id])
            #ax[batch_id, 4].grid(False)
            #ax[batch_id, 4].axis('off')

            #ax[batch_id, 5].imshow(masked_input_learned[batch_id] + mask_learned[batch_id]*recons_masked[batch_id])
            #ax[batch_id, 5].grid(False)
            #ax[batch_id, 5].axis('off')

            #for i in range(num_slots):
            #    ax[batch_id, i + 2].imshow(picture[batch_id, i])
            #    ax[batch_id, i + 2].grid(False)
            #    ax[batch_id, i + 2].axis('off')
        plt.tight_layout()
        plt.close()
        return fig
