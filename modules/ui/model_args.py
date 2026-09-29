"""sd.cpp-webui - UI component for model arguments"""
import gradio as gr
from modules.shared_instance import config
from modules.ui.constants import QUANTS


def create_model_args_ui():
    """Create the model arguments UI"""
    with gr.Accordion(label="Chroma", open=False):
        use_dit_mask = gr.Checkbox(
            label="Enable DiT mask for Chroma",
            value=config.get('def_chroma_use_dit_mask')
        )
        use_t5_mask = gr.Checkbox(
            label="Enable T5 mask for Chroma",
            value=config.get('def_chroma_use_t5_mask')
        )
        t5_mask_pad = gr.Number(
            label="T5 mask pad size for Chroma",
            minimum=0,
            precision=0,
            step=1,
            value=config.get('def_chroma_t5_mask_pad')
        )

    with gr.Accordion(label="Qwen Image", open=False):
        zero_cond_t = gr.Checkbox(
            label="Enable zero_cond_t for Qwen Image",
            value=config.get('def_qwen_image_zero_cond_t')
        )
        prefix_cache = gr.Checkbox(
            label="Enable prefix cache for Qwen Image 2.1",
            value=config.get('def_qwen_image_2_1_prefix_cache')
        )
        # Skip the "Default" entry, this arg uses "auto"
        prefix_cache_type = gr.Dropdown(
            label="Prefix cache type for Qwen Image 2.1",
            choices=["auto"] + QUANTS[1:],
            value=config.get('def_qwen_image_2_1_prefix_cache_type')
        )

    with gr.Accordion(label="PixArt", open=False):
        pos_embed_base_size = gr.Number(
            label="pixart_pos_embed_base_size",
            minimum=0,
            precision=0,
            value=config.get('def_pixart_pos_embed_base_size')
        )
        interpolation_scale = gr.Number(
            label="pixart_interpolation_scale",
            minimum=0,
            value=config.get('def_pixart_interpolation_scale')
        )
        vae_scale_factor = gr.Number(
            label="pixart_vae_scale_factor",
            minimum=0,
            value=config.get('def_pixart_vae_scale_factor')
        )

    return {
        'in_chroma_use_dit_mask': use_dit_mask,
        'in_chroma_use_t5_mask': use_t5_mask,
        'in_chroma_t5_mask_pad': t5_mask_pad,
        'in_qwen_image_zero_cond_t': zero_cond_t,
        'in_qwen_image_2_1_prefix_cache': prefix_cache,
        'in_qwen_image_2_1_prefix_cache_type': prefix_cache_type,
        'in_pixart_pos_embed_base_size': pos_embed_base_size,
        'in_pixart_interpolation_scale': interpolation_scale,
        'in_pixart_vae_scale_factor': vae_scale_factor
    }
