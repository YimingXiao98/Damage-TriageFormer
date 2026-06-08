import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.config import CATEGORIES, COLOR_TO_CAT, IGNORE_INDEX

# Class color definitions (RGB) matching the dataset
CLASS_COLORS = {
    0: (255, 255, 255),   # Undamaged (White)
    1: (0, 255, 83),      # Partial Roof (Green)
    2: (246, 255, 11),    # Total Roof (Yellow)
    3: (255, 138, 18),    # Partial Structural (Orange)
    4: (255, 0, 0),       # Total Structural (Red)
}

# Inverse mapping for ground truth decoding
CAT_TO_COLOR = {v: k for k, v in COLOR_TO_CAT.items() if v >= 0}

# Legacy palette for old visualize function
PALETTE = [
    (128, 128, 128),  # 0: background (black)
    (0, 255, 0),     # 1: green
    (255, 255, 0),   # 2: yellow
    (255, 0, 0),     # 3: red
    (255, 165, 0),   # 4: orange
]


def colorize_mask(mask):
    """
    Converts a 2D label mask into an RGB color mask using the PALETTE.
    mask: (H, W) with class indices.
    """
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        mask_rgb[mask == i] = color
    return mask_rgb


def denormalize_image(img_tensor):
    """Convert normalized tensor to displayable numpy array."""
    if hasattr(img_tensor, 'detach'):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = img_tensor

    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

    return img


def create_damage_overlay(inst_masks, severities, height, width):
    """
    Create RGBA overlay from instance masks and their severities.
    Returns normalized float array with alpha channel.
    """
    overlay = np.zeros((height, width, 4), dtype=np.float32)

    if inst_masks is None or len(inst_masks) == 0:
        return overlay

    for i in range(len(inst_masks)):
        if hasattr(inst_masks[i], 'cpu'):
            m = inst_masks[i].cpu().numpy() > 0.5
        else:
            m = inst_masks[i] > 0.5

        if hasattr(severities[i], 'item'):
            sev = int(severities[i].item())
        else:
            sev = int(severities[i])

        if sev == IGNORE_INDEX or sev < 0 or sev > 4:
            continue

        color = CLASS_COLORS.get(sev, (128, 128, 128))
        overlay[m, 0] = color[0] / 255.0
        overlay[m, 1] = color[1] / 255.0
        overlay[m, 2] = color[2] / 255.0
        overlay[m, 3] = 0.6

    return overlay


def visualize_with_gt(batch, model, device, output_dir, num_samples=20,
                      class_balanced=True, use_hierarchical=False,
                      use_gated_head=False):
    """
    Create side-by-side visualizations of predictions vs ground truth.

    Args:
        batch: Dict with 'image', 'inst_masks', 'inst_sev', 'image_path'
        model: Trained model (set to eval mode)
        device: torch device
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        class_balanced: If True, try to include samples from all classes
        use_hierarchical: If True, model outputs hierarchical predictions
    """
    import torch
    os.makedirs(output_dir, exist_ok=True)

    images = batch["image"]
    inst_masks_list = batch["inst_masks"]
    inst_sev_list = batch["inst_sev"]
    image_paths = batch["image_path"]

    n = min(len(images), num_samples)

    model.eval()
    with torch.no_grad():
        for idx in range(n):
            img_tensor = images[idx:idx+1].to(device)
            inst_masks = inst_masks_list[idx] if idx < len(
                inst_masks_list) else None
            inst_sev = inst_sev_list[idx] if idx < len(inst_sev_list) else None
            img_path = image_paths[idx] if idx < len(
                image_paths) else f"sample_{idx}"

            # Get prediction
            if inst_masks is not None:
                inst_input = [inst_masks.to(device)]
            else:
                inst_input = None

            seg_logits, damage_output = model(
                img_tensor, inst_masks=inst_input)

            # Process predictions based on model type
            if use_gated_head and damage_output is not None:
                # damage_output = (gate_logits_list, leaf_logits_list)
                from src.model.gated_head import predict_5class_from_gate
                if len(damage_output) == 3:
                    gate_logits_list, leaf_logits_list, _ = damage_output
                else:
                    gate_logits_list, leaf_logits_list = damage_output
                if len(gate_logits_list) > 0 and gate_logits_list[0].numel() > 0:
                    pred_sev = predict_5class_from_gate(
                        gate_logits_list[0], leaf_logits_list[0])
                else:
                    pred_sev = None
            elif use_hierarchical and damage_output is not None:
                type_logits_list, extent_logits_list = damage_output
                if len(type_logits_list) > 0 and type_logits_list[0].numel() > 0:
                    type_pred = type_logits_list[0].argmax(dim=1)
                    extent_pred = extent_logits_list[0].argmax(dim=1)

                    # Decode 3-type, 2-extent logic
                    # 0 -> 0;
                    # 1 -> 1+E; 2 -> 3+E
                    pred_sev = torch.zeros_like(type_pred)
                    mask = (type_pred > 0)
                    pred_sev[mask] = (type_pred[mask] - 1) * \
                        2 + 1 + extent_pred[mask]
                else:
                    pred_sev = None
            elif damage_output is not None:
                # Non-hierarchical: damage_output is [[tensor, tensor, ...]]
                sev_logits_list = damage_output[0] if isinstance(
                    damage_output, (list, tuple)) else damage_output
                # Flatten nested list if needed
                if isinstance(sev_logits_list, list) and len(sev_logits_list) > 0:
                    if isinstance(sev_logits_list[0], list):
                        # Unwrap outer list
                        sev_logits_list = sev_logits_list[0]
                    # Concatenate valid tensors
                    valid_tensors = [s for s in sev_logits_list if hasattr(
                        s, 'numel') and s.numel() > 0]
                    if valid_tensors:
                        pred_sev = torch.cat(
                            valid_tensors, dim=0).argmax(dim=1)
                    else:
                        pred_sev = None
                else:
                    pred_sev = None
            else:
                pred_sev = None

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 1. Original Image
            img_np = denormalize_image(images[idx])
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            # 2. Ground Truth
            axes[1].imshow(img_np)
            if inst_masks is not None and inst_sev is not None:
                gt_overlay = create_damage_overlay(
                    inst_masks, inst_sev, img_np.shape[0], img_np.shape[1])
                axes[1].imshow(gt_overlay)
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')

            # 3. Prediction
            axes[2].imshow(img_np)
            if pred_sev is not None and inst_masks is not None:
                pred_overlay = create_damage_overlay(
                    inst_masks, pred_sev, img_np.shape[0], img_np.shape[1])
                axes[2].imshow(pred_overlay)
            axes[2].set_title("Prediction")
            axes[2].axis('off')

            # Legend
            patches = [mpatches.Patch(
                color=[c/255 for c in CLASS_COLORS[i]], label=CATEGORIES[i]) for i in range(5)]
            fig.legend(handles=patches, loc='lower center',
                       ncol=4, bbox_to_anchor=(0.5, 0.02))
            plt.subplots_adjust(bottom=0.12)

            # Save
            basename = os.path.basename(img_path).replace(
                '.png', '').replace('.tif', '')
            save_path = os.path.join(output_dir, f"{idx:02d}_{basename}.png")
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

    print(f"[Visualization] Saved {n} samples to {output_dir}")


def visualize(images, preds, output_dir="out/vis"):
    """Legacy visualization function (binary segmentation only)."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (im, p) in enumerate(zip(images, preds)):
        if hasattr(im, 'detach'):
            im = im.detach().cpu().numpy()
        if hasattr(p, 'detach'):
            p = p.detach().cpu().numpy()

        # Convert image to uint8
        if im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))  # CHW to HWC

        if im.dtype == np.float32 or im.dtype == np.float64:
            im = (im - im.min()) / (im.max() - im.min())
            im = (im * 255).astype(np.uint8)

        # Resize prediction mask
        p_img = Image.fromarray(p.astype(np.uint8)).resize(
            (im.shape[1], im.shape[0]), resample=Image.NEAREST)

        # Colorize mask
        p_colored = colorize_mask(np.array(p_img))

        # Blend image and mask
        blended = Image.blend(Image.fromarray(
            im), Image.fromarray(p_colored), alpha=0.5)

        blended.save(os.path.join(output_dir, f"{i}.png"))
