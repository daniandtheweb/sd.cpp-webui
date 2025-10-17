"""sd.cpp-webui - UI component to set environment variables"""

import gradio as gr

def create_env_ui():
    """Create env UI"""
    with gr.Accordion(
        label="Environment variables", open=False
    ):
        with gr.Row():
            vk_visible_override = gr.Checkbox(
                label="Enable Vulkan visible devices override",
                value=False
            )
            vk_visible_dev = gr.Number(
                label="Select Vulkan GPU identifier",
                value=None,
                minimum=0
            )
        with gr.Row():
            cuda_visible_override = gr.Checkbox(
                label="Enable CUDA/ROCm visible devices override",
                value=False
            )
            cuda_visible_dev = gr.Number(
                label="Select CUDA/ROCm GPU identifier",
                value=None,
                minimum=0
            )
        with gr.Row():
            disable_vk_coopmat = gr.Checkbox(
                label="Disable Vulkan cooperative matrix",
                value=False
            )
            disable_vk_int_dot = gr.Checkbox(
                label="Disable Vulkan integer dot product",
                value=False
            )
    return {
        'env_vk_visible_override': vk_visible_override,
        'env_GGML_VK_VISIBLE_DEVICES': vk_visible_dev,
        'env_cuda_visible_override': cuda_visible_override,
        'env_CUDA_VISIBLE_DEVICES': cuda_visible_dev,
        'env_GGML_VK_DISABLE_COOPMAT': disable_vk_coopmat,
        'env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT': disable_vk_int_dot
    }


