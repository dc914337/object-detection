import os


def init_weights(module, prefix, state_dict, collected):
    for name, child in module.named_children():
        full_name = prefix + "." + name if prefix != "" else name
        if full_name in state_dict:
            child.load_state_dict(state_dict[full_name])
        else:
            for p_name, param in child.named_parameters(recurse=False):
                full_p_name = full_name + "." + p_name
                if full_p_name in state_dict:
                    param.data.copy_(state_dict[full_p_name])
                    collected.append(full_p_name)
            init_weights(child, full_name, state_dict, collected)



def is_slurm():
    return 'SLURM_JOB_ID' in os.environ and os.environ['SLURM_JOB_NAME'] != 'zsh' and os.environ['SLURM_JOB_NAME'] != 'bash'



def visualize_results(sample, epreds, cfg):
    import matplotlib.pyplot as plt
    from PIL import Image as Image, ImageEnhance
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
    image = image.squeeze(0)
    recon_combined = recon_combined.squeeze(0)
    recons = recons.squeeze(0)
    masks = masks.squeeze(0)
    image = image.permute(1, 2, 0).cpu().numpy()
    recon_combined = recon_combined.permute(1, 2, 0)
    recon_combined = recon_combined.cpu().detach().numpy()
    recons = recons.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Recon.')
    for i in range(7):
        picture = recons[i] * masks[i] + (1 - masks[i])
        ax[i + 2].imshow(picture)
        ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
        ax[i].grid(False)
        ax[i].axis('off')
    plt.show()


    """epreds_dict = utils.convert.list_of_dicts_2_dict_of_tensors(epreds, device=model.device)
    sample_dict = utils.convert.list_of_dicts_2_dict_of_tensors(sample, device=model.device)

    masks_softmaxed, pred_masks, true_masks = val.get_masks(cfg, epreds_dict, sample_dict)
    vis = utils.visualisation.Visualiser(cfg)
    vis.add_all(sample_dict, epreds_dict, masks_softmaxed, pred_masks, true_masks)
    imgs = [vis.img_vis()]"""


def normalize(x):
    x = x - x.min()
    x = x / x.max()
    return x