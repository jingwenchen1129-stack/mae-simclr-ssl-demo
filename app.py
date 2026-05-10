from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from experiment_core import (
    build_state,
    decode_latent,
    diffusion_like_generate,
    gan_generate,
    image_grid,
    interpolate_latent,
    reconstruction_payload,
)


st.set_page_config(page_title="AE / VAE / GAN / Diffusion Lab", layout="wide")


@st.cache_resource(show_spinner="Training lightweight models...")
def load_state():
    return build_state(seed=7)


def show_gray(image, caption=None, width=None):
    kwargs = {"caption": caption, "clamp": True}
    if width is not None:
        kwargs["width"] = width
    st.image(np.clip(image, 0, 1), **kwargs)


state = load_state()

st.title("Autoencoder, VAE, GAN and Prompt Parameter Lab")
st.caption(f"Dataset: {state.source_name}. Offline-friendly teaching implementation.")

tabs = st.tabs(["AE vs VAE", "VAE Latent Space", "DCGAN-style Samples", "Text Prompt Parameters", "Report Notes"])

with tabs[0]:
    left, right = st.columns([0.22, 0.78])
    with left:
        sample_idx = st.slider("Sample index", 0, len(state.x) - 1, 12)
        payload = reconstruction_payload(state, sample_idx)
        st.metric("Label", payload["label"])
        st.metric("AE MSE", f"{payload['ae_mse']:.4f}")
        st.metric("VAE MSE", f"{payload['vae_mse']:.4f}")
    with right:
        c1, c2, c3 = st.columns(3)
        with c1:
            show_gray(payload["input"], "Input")
        with c2:
            show_gray(payload["ae"], "Autoencoder reconstruction")
        with c3:
            show_gray(payload["vae"], "VAE reconstruction")

        h1, h2 = st.columns(2)
        with h1:
            fig = px.imshow(payload["ae_error"], color_continuous_scale="magma", title="AE absolute error heatmap")
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with h2:
            fig = px.imshow(payload["vae_error"], color_continuous_scale="magma", title="VAE absolute error heatmap")
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)

        loss_df = pd.DataFrame(
            {
                "epoch": list(range(1, max(len(state.ae_loss), len(state.vae_loss)) + 1)),
                "Autoencoder": pd.Series(state.ae_loss),
                "Simplified VAE": pd.Series(state.vae_loss),
            }
        )
        st.line_chart(loss_df, x="epoch", y=["Autoencoder", "Simplified VAE"], height=260)

with tabs[1]:
    color_mode = st.radio("Scatter color", ["class label", "sample index"], horizontal=True)
    df = pd.DataFrame(
        {
            "z1": state.latent[:, 0],
            "z2": state.latent[:, 1],
            "label": state.y.astype(str),
            "index": np.arange(len(state.x)),
        }
    )
    color = "label" if color_mode == "class label" else "index"
    fig = px.scatter(df, x="z1", y="z2", color=color, hover_data=["index", "label"], height=520)
    fig.update_traces(marker=dict(size=6, opacity=0.78))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([0.36, 0.64])
    with c1:
        st.subheader("Decode latent coordinate")
        z1 = st.slider("z1", float(state.latent[:, 0].min()), float(state.latent[:, 0].max()), 0.0)
        z2 = st.slider("z2", float(state.latent[:, 1].min()), float(state.latent[:, 1].max()), 0.0)
        show_gray(decode_latent(state, z1, z2), "Generated from selected latent position")
    with c2:
        st.subheader("Latent interpolation")
        a, b, steps = st.columns(3)
        with a:
            idx_a = st.number_input("Sample A", 0, len(state.x) - 1, 3)
        with b:
            idx_b = st.number_input("Sample B", 0, len(state.x) - 1, 444)
        with steps:
            n_steps = st.slider("Steps", 4, 12, 8)
        frames = interpolate_latent(state, idx_a, idx_b, n_steps)
        st.image([np.clip(f, 0, 1) for f in frames], caption=[f"{i + 1}" for i in range(n_steps)], width=88)

with tabs[2]:
    c1, c2 = st.columns([0.25, 0.75])
    with c1:
        gan_seed = st.number_input("Seed", 0, 100000, 21)
        noise_dim = st.slider("Generator noise dim", 2, 16, 10)
        truncation = st.slider("Noise truncation", 0.2, 2.0, 0.9, 0.1)
    generated = gan_generate(state, gan_seed, noise_dim, truncation, n=16)
    with c2:
        show_gray(image_grid(generated["images"], scale=18, cols=4), "Generated sample grid")
        score_df = pd.DataFrame({"sample": np.arange(16), "discriminator score": generated["scores"]})
        st.bar_chart(score_df, x="sample", y="discriminator score", height=220)
        st.dataframe(pd.DataFrame(generated["noise"][:6]).round(3), use_container_width=True)

with tabs[3]:
    c1, c2 = st.columns([0.28, 0.72])
    with c1:
        prompt = st.text_input("Prompt", "round handwritten 0 and 8")
        negative = st.text_input("Negative prompt", "7")
        steps = st.slider("Sampling steps", 1, 50, 24)
        seed = st.number_input("Seed", 0, 100000, 2026)
        guidance = st.slider("Guidance scale", 0.0, 15.0, 7.5, 0.5)
    samples = diffusion_like_generate(state, prompt, negative, steps, seed, guidance, n=8)
    with c2:
        show_gray(image_grid(samples, scale=22, cols=4), "Prompt-conditioned generated comparison")
        st.info("This panel is an offline lightweight prompt-to-image surrogate. With diffusers installed, the same controls map directly to prompt, negative_prompt, num_inference_steps, seed and guidance_scale.")

with tabs[4]:
    st.markdown(
        """
        **Submission checklist**

        - PDF report: `report/experiment_report.pdf`
        - Core source: `experiment_core.py`, `app.py`
        - Optional URL: run locally with `streamlit run app.py`
        - Agent/LLM used: Codex GPT-5

        **Notes for the report**

        The implementation uses an offline MNIST-like handwritten digits dataset from scikit-learn.
        The AE is trained as a regular MLP autoencoder. The VAE, DCGAN and diffusion sections are
        simplified interactive surrogates so the project runs without GPU downloads.
        """
    )
