
import matplotlib.pyplot as plt
import numpy as np
import torch

def iou_and_dice(pred_mask, true_mask, eps=1e-7):
    """
    pred_mask, true_mask: numpy array veya torch tensor
    binary olması beklenir
    """
    pred_mask = torch.as_tensor(pred_mask).detach().cpu()
    true_mask = torch.as_tensor(true_mask).detach().cpu()

    pred_mask = (pred_mask > 0).float()
    true_mask = (true_mask > 0).float()

    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_mask.sum() + true_mask.sum() + eps)

    return float(iou), float(dice)


def plot_model_results(
    images,
    prediction,
    prediction_dino,
    prediction_supervised,
    labels,
    index=0,
    threshold=0.5,
    show_probability=False
):
    """
    images            : [B,3,H,W]
    pred_topodistill  : [B,1,H,W] veya [B,H,W]
    pred_dino         : [B,1,H,W] veya [B,H,W]
    pred_supervised   : [B,1,H,W] veya [B,H,W]
    real_mask         : [B,1,H,W] veya [B,H,W]
    index             : batch içinden hangi örnek çizilsin
    """

    def prepare_img(x, index):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        return x[index]

    def prepare_mask(x, index, threshold=0.5, show_probability=False):
        if x is None:
            return None

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()

        x = x[index]

        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)

        if not show_probability:
            x = (x > threshold).float()

        return x

    img_tensor = prepare_img(images, index)

    pred_topodistill_vis = prepare_mask(prediction, index, threshold, show_probability)
    pred_dino_vis = prepare_mask(prediction_dino, index, threshold, show_probability)
    pred_supervised_vis = prepare_mask(prediction_supervised, index, threshold, show_probability)
    real_mask_vis = prepare_mask(labels, index, threshold=0.5, show_probability=False)

    scores = {}
    if real_mask_vis is not None:
        scores["TopoDistill"] = iou_and_dice(
            (pred_topodistill_vis > 0.5).float() if show_probability else pred_topodistill_vis,
            real_mask_vis
        ) if pred_topodistill_vis is not None else (0.0, 0.0)

        scores["DINO"] = iou_and_dice(
            (pred_dino_vis > 0.5).float() if show_probability else pred_dino_vis,
            real_mask_vis
        ) if pred_dino_vis is not None else (0.0, 0.0)

        scores["Supervised"] = iou_and_dice(
            (pred_supervised_vis > 0.5).float() if show_probability else pred_supervised_vis,
            real_mask_vis
        ) if pred_supervised_vis is not None else (0.0, 0.0)
    else:
        scores["TopoDistill"] = (0.0, 0.0)
        scores["DINO"] = (0.0, 0.0)
        scores["Supervised"] = (0.0, 0.0)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
    axes = axes.ravel()

    def safe_imshow(ax, img, title="", score=None, is_tensor=False, cmap='gray'):
        if img is not None:
            if is_tensor:
                ax.imshow(img.permute(1, 2, 0).numpy())
            else:
                ax.imshow(img, cmap=cmap)

        ax.set_title(title, fontsize=12)

        if score is not None:
            iou, dice = score
            text = f"IoU: {iou:.2f}\nDice: {dice:.2f}"
            ax.text(
                0.04, 0.96,
                text,
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                verticalalignment='top',
                color='white',
                bbox=dict(facecolor='black', alpha=0.6, pad=4)
            )

        ax.axis("off")

    safe_imshow(axes[0], img_tensor, is_tensor=True, cmap=None)
    safe_imshow(axes[1], pred_topodistill_vis, score=scores["TopoDistill"])
    safe_imshow(axes[2], pred_dino_vis, score=scores["DINO"])
    safe_imshow(axes[3], pred_supervised_vis, score=scores["Supervised"])
    safe_imshow(axes[4], real_mask_vis)

    plt.tight_layout()
    plt.show()
    return fig


def plot_topology_results(
    name,
    gray_image,
    random_walker_mask,
    morphological_mask,
    topology_mask,
    real_mask,
    image_tensor,
    clean_image,
    best_feature,
    otsu_mask,
    hole_mask,
    h1_mask,
    h0_mask,
    persistence,
    threshold,
):
    """Visualize pseudo-mask candidates and their scores against a reference mask."""
    masks = [
        ("Random walker", random_walker_mask),
        ("Morphological Chan-Vese", morphological_mask),
        ("Otsu", otsu_mask),
        ("TopoDistill mask", topology_mask),
        ("H1 mask", h1_mask),
        ("H0 mask", h0_mask),
        ("Hole mask", hole_mask),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    axes = axes.ravel()
    fig.suptitle(f"Pseudo-mask analysis: {name}", fontsize=15)

    original = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(np.clip(original, 0, 1))
    axes[0].set_title("Original")
    axes[1].imshow(np.clip(clean_image, 0, 1))
    axes[1].set_title("DullRazor")
    axes[2].imshow(gray_image, cmap="gray")
    axes[2].set_title("Combined channel")

    for axis, (mask_name, mask) in zip(axes[3:10], masks):
        axis.imshow(mask, cmap="gray")
        if real_mask is not None:
            iou, dice = iou_and_dice(torch.as_tensor(mask), torch.as_tensor(real_mask))
            mask_name += f"\nIoU {iou:.3f} | Dice {dice:.3f}"
        axis.set_title(mask_name)

    if real_mask is None:
        axes[10].text(0.5, 0.5, "Reference mask unavailable", ha="center")
    else:
        axes[10].imshow(real_mask, cmap="gray")
    axes[10].set_title("Reference mask")

    diagram_axis = axes[11]
    try:
        for dimension, color in ((0, "tab:blue"), (1, "tab:red")):
            points = persistence[dimension][1].detach().cpu().numpy()
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            if len(points):
                diagram_axis.scatter(points[:, 0], points[:, 1], s=12, color=color, label=f"H{dimension}")
        diagram_axis.legend(loc="best")
    except (IndexError, TypeError, AttributeError):
        diagram_axis.text(0.5, 0.5, "Persistence diagram unavailable", ha="center")
    feature = np.asarray(best_feature).round(2) if best_feature is not None else "N/A"
    diagram_axis.set_title(f"Persistence\nfeature={feature}, threshold={float(threshold):.2f}")
    diagram_axis.set_xlabel("Birth")
    diagram_axis.set_ylabel("Death")

    for axis in axes:
        axis.axis("off")
    diagram_axis.axis("on")
    plt.tight_layout()
    plt.show()
    return fig
