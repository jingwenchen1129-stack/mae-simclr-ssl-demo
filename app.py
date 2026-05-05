import io
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


IMAGE_SIZE = 32
PATCH_SIZE = 4
GRID = IMAGE_SIZE // PATCH_SIZE
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3


class ImageTransformSSL:
    """Small self-supervised image transformations used by the demo."""

    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
        self.image_size = image_size
        self.patch_size = patch_size

    def rotate(self, image, k):
        return np.rot90(image, k=k)

    def mask_patches(self, image, mask_ratio, rng):
        patches = image_to_patches(image, self.patch_size)
        n_patches = patches.shape[0]
        n_mask = max(1, int(round(n_patches * mask_ratio)))
        mask_idx = rng.choice(n_patches, size=n_mask, replace=False)
        masked = patches.copy()
        masked[mask_idx] = 0.5
        return patches_to_image(masked, self.image_size, self.patch_size), mask_idx

    def jigsaw(self, image, rng):
        patches = image_to_patches(image, self.patch_size)
        order = rng.permutation(len(patches))
        return patches_to_image(patches[order], self.image_size, self.patch_size), order


def make_shape_image(seed, image_size=IMAGE_SIZE):
    rng = np.random.default_rng(seed)
    bg = tuple(rng.integers(18, 70, size=3).tolist())
    fg = tuple(rng.integers(120, 245, size=3).tolist())
    accent = tuple(rng.integers(80, 230, size=3).tolist())
    img = Image.new("RGB", (image_size, image_size), bg)
    draw = ImageDraw.Draw(img)

    shape = seed % 4
    margin = int(rng.integers(4, 8))
    box = [margin, margin, image_size - margin, image_size - margin]
    if shape == 0:
        draw.polygon(
            [(image_size // 2, 3), (image_size - 5, image_size - 6), (5, image_size - 6)],
            fill=fg,
        )
    elif shape == 1:
        draw.rectangle(box, fill=fg)
        draw.rectangle([box[0] + 5, box[1] + 5, box[2] - 5, box[3] - 5], fill=bg)
    elif shape == 2:
        draw.ellipse(box, fill=fg)
    else:
        draw.polygon(
            [(4, image_size // 2), (image_size // 2, 4), (image_size - 4, image_size // 2), (image_size // 2, image_size - 4)],
            fill=fg,
        )

    # Add an orientation cue so rotation prediction is learnable.
    draw.rectangle([image_size - 10, 3, image_size - 4, 9], fill=accent)
    draw.line([4, image_size - 5, image_size - 4, 5], fill=accent, width=2)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_dataset(n=160, seed=7):
    return np.stack([make_shape_image(seed * 1000 + i) for i in range(n)])


def image_to_patches(image, patch_size=PATCH_SIZE):
    h, w, c = image.shape
    return (
        image.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
        .swapaxes(1, 2)
        .reshape(-1, patch_size, patch_size, c)
    )


def patches_to_image(patches, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    grid = image_size // patch_size
    return (
        patches.reshape(grid, grid, patch_size, patch_size, 3)
        .swapaxes(1, 2)
        .reshape(image_size, image_size, 3)
    )


def patch_features_from_masked(masked_patches, mask_idx):
    means = masked_patches.mean(axis=(1, 2))
    visible = np.ones((means.shape[0], 1), dtype=np.float32)
    visible[mask_idx] = 0.0
    coords = []
    for y in range(GRID):
        for x in range(GRID):
            coords.append([x / (GRID - 1), y / (GRID - 1)])
    coords = np.asarray(coords, dtype=np.float32)
    return np.concatenate([means, visible, coords], axis=1).reshape(-1)


@dataclass
class MAEResult:
    losses: list
    recon_image: np.ndarray
    masked_image: np.ndarray
    original_image: np.ndarray
    mask_ratio: float


class SimplePatchMAE:
    """A tiny masked-image reconstruction model trained with NumPy.

    It predicts the RGB mean of each hidden patch from all visible patch means,
    a visibility bit, and patch coordinates. The decoder expands each predicted
    mean back to a 4x4 patch, which makes the reconstruction easy to inspect.
    """

    def __init__(self, lr=0.25, seed=0):
        self.rng = np.random.default_rng(seed)
        in_dim = GRID * GRID * 6
        out_dim = GRID * GRID * 3
        self.w = self.rng.normal(0, 0.04, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr

    def train(self, images, mask_ratio, steps=160, batch_size=24, seed=0):
        rng = np.random.default_rng(seed)
        transform = ImageTransformSSL()
        losses = []
        last = None
        for step in range(steps):
            batch_ids = rng.choice(len(images), size=batch_size, replace=True)
            grad_w = np.zeros_like(self.w)
            grad_b = np.zeros_like(self.b)
            total_loss = 0.0
            for image_id in batch_ids:
                image = images[image_id]
                original_patches = image_to_patches(image)
                masked_image, mask_idx = transform.mask_patches(image, mask_ratio, rng)
                masked_patches = image_to_patches(masked_image)
                x = patch_features_from_masked(masked_patches, mask_idx)
                target = original_patches.mean(axis=(1, 2)).reshape(-1)
                pred = x @ self.w + self.b
                mask = np.zeros(GRID * GRID, dtype=np.float32)
                mask[mask_idx] = 1.0
                mask = np.repeat(mask, 3)
                diff = (pred - target) * mask
                denom = max(mask.sum(), 1.0)
                total_loss += float(np.sum(diff * diff) / denom)
                grad = 2.0 * diff / denom
                grad_w += np.outer(x, grad)
                grad_b += grad
                last = (image, masked_image, mask_idx)
            self.w -= self.lr * grad_w / batch_size
            self.b -= self.lr * grad_b / batch_size
            losses.append(total_loss / batch_size)

        image, masked_image, mask_idx = last
        recon = self.reconstruct(image, mask_ratio, seed + 99, mask_idx=mask_idx, masked_image=masked_image)
        return MAEResult(losses, recon, masked_image, image, mask_ratio)

    def reconstruct(self, image, mask_ratio, seed=0, mask_idx=None, masked_image=None):
        rng = np.random.default_rng(seed)
        transform = ImageTransformSSL()
        if mask_idx is None or masked_image is None:
            masked_image, mask_idx = transform.mask_patches(image, mask_ratio, rng)
        masked_patches = image_to_patches(masked_image)
        x = patch_features_from_masked(masked_patches, mask_idx)
        pred_means = (x @ self.w + self.b).reshape(GRID * GRID, 3)
        pred_means = np.clip(pred_means, 0, 1)
        recon_patches = masked_patches.copy()
        for idx in mask_idx:
            recon_patches[idx] = pred_means[idx]
        return patches_to_image(recon_patches)


def rotation_features(images):
    feats = []
    for image in images:
        left = image[:, : IMAGE_SIZE // 2].mean(axis=(0, 1))
        right = image[:, IMAGE_SIZE // 2 :].mean(axis=(0, 1))
        top = image[: IMAGE_SIZE // 2, :].mean(axis=(0, 1))
        bottom = image[IMAGE_SIZE // 2 :, :].mean(axis=(0, 1))
        center = image[8:24, 8:24].mean(axis=(0, 1))
        feats.append(np.concatenate([left, right, top, bottom, center, right - left, top - bottom]))
    return np.asarray(feats, dtype=np.float32)


def train_rotation_predictor(images, steps=140, lr=0.8, seed=0):
    rng = np.random.default_rng(seed)
    transform = ImageTransformSSL()
    x_list, y_list = [], []
    for image in images:
        for label in range(4):
            x_list.append(transform.rotate(image, label))
            y_list.append(label)
    x = rotation_features(np.stack(x_list))
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    y = np.asarray(y_list)
    w = rng.normal(0, 0.02, size=(x.shape[1], 4)).astype(np.float32)
    history = []
    acc_history = []
    for _ in range(steps):
        logits = x @ w
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-8).mean()
        pred = probs.argmax(axis=1)
        acc = (pred == y).mean()
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(y)), y] = 1.0
        grad = x.T @ (probs - onehot) / len(y)
        w -= lr * grad
        history.append(float(loss))
        acc_history.append(float(acc))
    return history, acc_history


def to_pil(image):
    return Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))


def plot_history(values_a, label_a, values_b=None, label_b=None, ylabel="loss"):
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=130)
    ax.plot(values_a, label=label_a, linewidth=2)
    if values_b is not None:
        ax.plot(values_b, label=label_b, linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def image_triplet(original, masked, recon):
    cols = st.columns(3)
    for col, title, image in zip(cols, ["Input", "Masked", "Output"], [original, masked, recon]):
        col.caption(title)
        col.image(to_pil(image), use_column_width=True)


@st.cache_data(show_spinner=False)
def cached_dataset(n, seed):
    return make_dataset(n, seed)


def run_mae(mask_ratio, steps, seed):
    images = cached_dataset(180, 11)
    model = SimplePatchMAE(lr=0.3, seed=seed)
    return model.train(images, mask_ratio=mask_ratio, steps=steps, seed=seed + 1)


@st.cache_data(show_spinner=False)
def run_rotation(steps, seed):
    images = cached_dataset(90, 21)
    return train_rotation_predictor(images, steps=steps, seed=seed)


def main():
    st.set_page_config(page_title="Self-Supervised Vision Demo", layout="wide")
    st.title("MAE + Rotation Self-Supervised Vision Demo")
    st.caption("A compact NumPy demo for masked image reconstruction and rotation prediction.")

    with st.sidebar:
        st.header("Controls")
        task = st.radio(
            "Task",
            ["Masked Autoencoder", "Rotation Prediction"],
            horizontal=False,
            key="ssl_task_selector_v2",
        )
        seed = st.slider("Random seed", 0, 99, 7, key="ssl_seed_v2")
        steps = st.slider("Training steps", 20, 320, 160, step=20, key="ssl_steps_v2")
        mask_ratio = st.slider("Mask ratio", 0.15, 0.75, 0.45, step=0.05, key="ssl_mask_ratio_v2")
        compare_ratio = st.slider(
            "Compare mask ratio",
            0.15,
            0.75,
            0.65,
            step=0.05,
            key="ssl_compare_ratio_v2",
        )
        run = st.button("Run experiment", type="primary", key="ssl_run_v2")

    if "ran" not in st.session_state:
        st.session_state.ran = True
    if run:
        st.session_state.ran = True

    if task == "Masked Autoencoder":
        st.subheader("Masked image reconstruction")
        with st.spinner("Training tiny patch decoder..."):
            result = run_mae(mask_ratio, steps, seed)
            comparison = run_mae(compare_ratio, steps, seed + 13)

        left, right = st.columns([1.2, 1.0])
        with left:
            image_triplet(result.original_image, result.masked_image, result.recon_image)
        with right:
            st.pyplot(plot_history(result.losses, f"mask={mask_ratio:.2f}", comparison.losses, f"mask={compare_ratio:.2f}"))
            st.metric("Final loss", f"{result.losses[-1]:.4f}", f"{result.losses[-1] - result.losses[0]:.4f}")

        st.subheader("Setting comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"Mask ratio {mask_ratio:.2f}")
            image_triplet(result.original_image, result.masked_image, result.recon_image)
        with col_b:
            st.caption(f"Mask ratio {compare_ratio:.2f}")
            image_triplet(comparison.original_image, comparison.masked_image, comparison.recon_image)

    else:
        st.subheader("Rotation prediction")
        images = cached_dataset(8, 42)
        transform = ImageTransformSSL()
        preview = images[seed % len(images)]
        cols = st.columns(4)
        for k, col in enumerate(cols):
            col.caption(f"{k * 90} deg")
            col.image(to_pil(transform.rotate(preview, k)), use_column_width=True)
        with st.spinner("Training rotation classifier..."):
            loss, acc = run_rotation(steps, seed)
        c1, c2 = st.columns(2)
        c1.pyplot(plot_history(loss, "cross entropy", ylabel="loss"))
        c2.pyplot(plot_history(acc, "accuracy", ylabel="accuracy"))
        st.metric("Final accuracy", f"{acc[-1] * 100:.1f}%", f"{(acc[-1] - acc[0]) * 100:.1f} pp")


if __name__ == "__main__":
    main()
