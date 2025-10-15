"""sd.cpp-webui - Reusable UI module"""

import os
from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)

from modules.utility import (
    switch_sizes, update_interactivity, SDOptionsCache
)

# initiate cache once
sd_options = SDOptionsCache()

QUANTS = [
    "Default", "f32", "f16", "q8_0", "q6_K", "q5_K", "q5_1",
    "q5_0", "q4_K", "q4_1", "q4_0", "q3_K", "q2_K"
]
SAMPLERS = sd_options.get_opt("samplers")
SCHEDULERS = sd_options.get_opt("schedulers")
MODELS = [
    "Checkpoint", "UNET", "VAE", "clip_g", "clip_l", "t5xxl", "qwen2vl",
    "TAESD", "Lora", "Embeddings", "Upscaler", "ControlNet"
]
PREVIEW = ['none'] + sd_options.get_opt("previews")
PREDICTION = ['Default'] + sd_options.get_opt("prediction")
RELOAD_SYMBOL = '\U0001F504'
RANDOM_SYMBOL = '\U0001F3B2'
SWITCH_V_SYMBOL = '\u2195'


def create_img_model_sel_ui():
    """Create the image model selection UI"""
    ckpt_dir_txt = gr.Textbox(value=config.get('ckpt_dir'), visible=False)
    vae_dir_txt = gr.Textbox(value=config.get('vae_dir'), visible=False)
    unet_dir_txt = gr.Textbox(value=config.get('unet_dir'), visible=False)
    clip_dir_txt = gr.Textbox(value=config.get('clip_dir'), visible=False)

    diffusion_mode = gr.Number(value=0, visible=False)

    # Model & VAE Selection
    with gr.Tab("Checkpoint") as ckpt_tab:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        ckpt_model = gr.Dropdown(
                            label="Checkpoint Model",
                            choices=get_models(config.get('ckpt_dir')),
                            scale=7,
                            value=config.get('def_ckpt'),
                            interactive=True
                        )
                    with gr.Row():
                        reload_ckpt_btn = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        reload_ckpt_btn.click(
                            reload_models,
                            inputs=[ckpt_dir_txt],
                            outputs=[ckpt_model]
                        )

                        gr.ClearButton(
                            ckpt_model,
                            scale=1
                        )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        ckpt_vae = gr.Dropdown(
                            label="Checkpoint VAE",
                            choices=get_models(config.get('vae_dir')),
                            scale=7,
                            value=config.get('def_ckpt_vae'),
                            interactive=True
                        )
                    with gr.Row():
                        reload_vae_btn = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        reload_vae_btn.click(
                            reload_models,
                            inputs=[vae_dir_txt],
                            outputs=[ckpt_vae]
                        )

                        gr.ClearButton(
                            ckpt_vae,
                            scale=1
                        )

    with gr.Tab("UNET") as unet_tab:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        unet_model = gr.Dropdown(
                            label="UNET Model",
                            choices=get_models(config.get('unet_dir')),
                            scale=7,
                            value=config.get('def_unet'),
                            interactive=True
                        )
                    with gr.Row():
                        reload_unet_btn = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        reload_unet_btn.click(
                            reload_models,
                            inputs=[unet_dir_txt],
                            outputs=[unet_model]
                        )

                        gr.ClearButton(
                            unet_model,
                            scale=1
                        )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        unet_vae = gr.Dropdown(
                            label="UNET VAE",
                            choices=get_models(config.get('vae_dir')),
                            scale=7,
                            value=config.get('def_unet_vae'),
                            interactive=True
                        )
                    with gr.Row():
                        reload_unet_vae_btn = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        reload_unet_vae_btn.click(
                            reload_models,
                            inputs=[vae_dir_txt],
                            outputs=[unet_vae]
                        )

                        gr.ClearButton(
                            unet_vae,
                            scale=1
                        )

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        clip_g = gr.Dropdown(
                            label="clip_g",
                            choices=get_models(config.get('clip_dir')),
                            scale=7,
                            value=config.get('def_clip_g'),
                            interactive=True
                        )
                    with gr.Row():
                        reload_clip_g_btn = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        reload_clip_g_btn.click(
                            reload_models,
                            inputs=[clip_dir_txt],
                            outputs=[clip_g]
                        )

                        gr.ClearButton(
                            clip_g,
                            scale=1
                        )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        clip_l = gr.Dropdown(
                            label="clip_l",
                            choices=get_models(config.get('clip_dir')),
                            scale=7,
                            value=config.get('def_clip_l'),
                            interactive=True
                        )
                    with gr.Row():
                        reload_clip_l_btn = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        reload_clip_l_btn.click(
                            reload_models,
                            inputs=[clip_dir_txt],
                            outputs=[clip_l]
                        )

                        gr.ClearButton(
                            clip_l,
                            scale=1
                        )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        t5xxl = gr.Dropdown(
                            label="t5xxl",
                            choices=get_models(config.get('clip_dir')),
                            scale=7, value=config.get('def_t5xxl'),
                            interactive=True
                        )
                    with gr.Row():
                        reload_t5xxl_btn = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        reload_t5xxl_btn.click(
                            reload_models,
                            inputs=[clip_dir_txt],
                            outputs=[t5xxl]
                        )

                        gr.ClearButton(
                            t5xxl,
                            scale=1
                        )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        qwen2vl = gr.Dropdown(
                            label="qwen2vl",
                            choices=get_models(config.get('clip_dir')),
                            scale=7, value=config.get('def_qwen2vl'),
                            interactive=True
                        )
                    with gr.Row():
                        reload_qwen2vl_btn = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        reload_qwen2vl_btn.click(
                            reload_models,
                            inputs=[clip_dir_txt],
                            outputs=[qwen2vl]
                        )
                        
                        gr.ClearButton(
                            qwen2vl,
                            scale=1
                        )

    # Return the dictionary with all UI components
    return {
        'inputs': {
            'in_diffusion_mode': diffusion_mode,
            'in_ckpt_model': ckpt_model,
            'in_ckpt_vae': ckpt_vae,
            'in_unet_model': unet_model,
            'in_unet_vae': unet_vae,
            'in_clip_g': clip_g,
            'in_clip_l': clip_l,
            'in_t5xxl': t5xxl,
            'in_qwen2vl': qwen2vl,
        },
        'components': {
            'ckpt_tab': ckpt_tab,
            'unet_tab': unet_tab,
        }
    }


def create_quant_ui():
    """Create the model type selection UI"""
    with gr.Row():
        model_type = gr.Dropdown(
            label="Quantization",
            choices=QUANTS,
            value=config.get('def_type'),
            interactive=True
        )
    return {
        'in_model_type': model_type
    }


def create_video_model_sel_ui():
    """Create the video model selection UI"""
    unet_dir_txt = gr.Textbox(value=config.get('unet_dir'), visible=False)
    vae_dir_txt = gr.Textbox(value=config.get('vae_dir'), visible=False)
    clip_dir_txt = gr.Textbox(value=config.get('clip_dir'), visible=False)

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    unet_model = gr.Dropdown(
                        label="UNET Model",
                        choices=get_models(config.get('unet_dir')),
                        scale=7,
                        value=config.get('def_unet'),
                        interactive=True
                    )
                with gr.Row():
                    reload_unet_btn = gr.Button(
                        value=RELOAD_SYMBOL,
                        scale=1
                    )
                    reload_unet_btn.click(
                        reload_models,
                        inputs=[unet_dir_txt],
                        outputs=[unet_model]
                    )

                    gr.ClearButton(
                        unet_model,
                        scale=1
                    )
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    unet_vae = gr.Dropdown(
                        label="UNET VAE",
                        choices=get_models(config.get('vae_dir')),
                        scale=7,
                        value=config.get('def_unet_vae'),
                        interactive=True
                    )
                with gr.Row():
                    reload_unet_vae_btn = gr.Button(
                        value=RELOAD_SYMBOL,
                        scale=1
                    )
                    reload_unet_vae_btn.click(
                        reload_models,
                        inputs=[vae_dir_txt],
                        outputs=[unet_vae]
                    )

                    gr.ClearButton(
                        unet_vae,
                        scale=1
                    )

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    clip_vision_h = gr.Dropdown(
                        label="clip_vision_h",
                        choices=get_models(config.get('clip_dir')),
                        scale=7,
                        value=config.get('def_clip_vision_h'),
                        interactive=True
                    )
                with gr.Row():
                    reload_clip_vision_h_btn = gr.Button(
                        value=RELOAD_SYMBOL,
                        scale=1
                    )
                    reload_clip_vision_h_btn.click(
                        reload_models,
                        inputs=[clip_dir_txt],
                        outputs=[clip_vision_h]
                    )

                    gr.ClearButton(
                        clip_vision_h,
                        scale=1
                    )
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    umt5_xxl = gr.Dropdown(
                        label="umt5_xxl",
                        choices=get_models(config.get('clip_dir')),
                        scale=7,
                        value=config.get('def_umt5_xxl'),
                        interactive=True
                    )
                with gr.Row():
                    reload_umt5_xxl_btn = gr.Button(
                        value=RELOAD_SYMBOL,
                        scale=1
                    )
                    reload_umt5_xxl_btn.click(
                        reload_models,
                        inputs=[clip_dir_txt],
                        outputs=[umt5_xxl]
                    )

                    gr.ClearButton(
                        umt5_xxl,
                        scale=1
                    )
    with gr.Row():
        with gr.Accordion(
            label="High Noise", open=False
        ):
            with gr.Group():
                with gr.Row():
                    high_noise_model = gr.Dropdown(
                        label="High Noise Diffusion model",
                        choices=get_models(config.get('unet_dir')),
                        value=None,
                        interactive=True
                    )

                with gr.Row():
                    reload_high_noise_model = gr.Button(
                        value=RELOAD_SYMBOL,
                        scale=1
                    )
                    reload_high_noise_model.click(
                        reload_models,
                        inputs=[unet_dir_txt],
                        outputs=[high_noise_model]
                    )
                    gr.ClearButton(
                        high_noise_model,
                        scale=1
                    )

    # Return the dictionary with all UI components
    return {
        'in_unet_model': unet_model,
        'in_unet_vae': unet_vae,
        'in_clip_vision_h': clip_vision_h,
        'in_umt5_xxl': umt5_xxl,
        'in_high_noise_model': high_noise_model
    }


def create_prompts_ui():
    """Create the prompts UI"""

    def save_and_refresh_prompts(name, p_prompt, n_prompt):
        config.add_prompt(name, p_prompt, n_prompt)
        return gr.update(choices=config.get_prompts(), value=name)

    def delete_and_refresh_prompts(name):
        config.delete_prompt(name)
        return gr.update(choices=config.get_prompts())

    def refresh_prompt_list():
        return gr.update(choices=config.get_prompts())

    with gr.Row():
        with gr.Accordion(
            label="Saved prompts", open=False
        ):
            with gr.Column():
                saved_prompts = gr.Dropdown(
                    label="Prompts",
                    choices=config.get_prompts(),
                    interactive=True,
                    allow_custom_value=True
                )
            with gr.Column():
                with gr.Row():
                    load_prompt_btn = gr.Button(
                        value="Load prompt", size="lg"
                    )
                    reload_prompts_btn = gr.Button(
                        value=RELOAD_SYMBOL
                    )
                with gr.Row():
                    save_prompt_btn = gr.Button(
                        value="Save prompt", size="lg"
                    )
                    del_prompt_btn = gr.Button(
                        value="Delete prompt", size="lg"
                    )
    with gr.Row():
        pprompt = gr.Textbox(
            placeholder="Positive prompt",
            label="Positive Prompt",
            lines=3,
            show_copy_button=True
        )
    with gr.Row():
        nprompt = gr.Textbox(
            placeholder="Negative prompt",
            label="Negative Prompt",
            lines=3,
            show_copy_button=True
        )

    save_prompt_btn.click(
        save_and_refresh_prompts,
        inputs=[saved_prompts, pprompt, nprompt],
        outputs=[saved_prompts]
    )
    del_prompt_btn.click(
        delete_and_refresh_prompts,
        inputs=[saved_prompts],
        outputs=[saved_prompts]
    )
    reload_prompts_btn.click(
        refresh_prompt_list,
        inputs=[],
        outputs=[saved_prompts]
    )
    load_prompt_btn.click(
        config.get_prompt,
        inputs=[saved_prompts],
        outputs=[pprompt, nprompt]
    )

    # Return the dictionary with all UI components
    return {
        'saved_prompts': saved_prompts,
        'in_pprompt': pprompt,
        'in_nprompt': nprompt
    }


def create_settings_ui():
    """Create settings UI"""
    with gr.Row():
        with gr.Column(scale=1):
            sampling = gr.Dropdown(
                label="Sampling method",
                choices=SAMPLERS,
                value=config.get('def_sampling'),
                interactive=True
            )
        with gr.Column(scale=1):
            steps = gr.Slider(
                label="Steps",
                minimum=1,
                maximum=100,
                value=config.get('def_steps'),
                step=1
            )
    with gr.Row():
        scheduler = gr.Dropdown(
            label="Scheduler",
            choices=SCHEDULERS,
            value=config.get('def_scheduler'),
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            width = gr.Slider(
                label="Width",
                minimum=64,
                maximum=4096,
                value=config.get('def_width'),
                step=64
            )
            height = gr.Slider(
                label="Height",
                minimum=64,
                maximum=4096,
                value=config.get('def_height'),
                step=64
            )
            switch_size = gr.Button(
                value=SWITCH_V_SYMBOL, scale=1
            )
            switch_size.click(
                switch_sizes,
                inputs=[height,
                        width],
                outputs=[height,
                         width]
            )
        with gr.Column():
            with gr.Row():
                batch_count = gr.Slider(
                    label="Batch count",
                    minimum=1,
                    maximum=99,
                    value=1,
                    step=1
                )
            with gr.Row():
                cfg = gr.Slider(
                    label="CFG Scale",
                    minimum=1,
                    maximum=30,
                    value=config.get('def_cfg'),
                    step=0.1,
                    interactive=True
                )
    with gr.Row():
        guidance_btn = gr.Checkbox(
            label="Enable distilled guidance", value=False,
            visible=False
        )
        guidance = gr.Slider(
            label="Guidance",
            minimum=0,
            maximum=30,
            value=3.5,
            step=0.1,
            interactive=True,
            visible=False
        )
    with gr.Row():
        flow_shift_btn = gr.Checkbox(
            label="Enable Flow Shift", value=False,
            visible=False
        )
        flow_shift = gr.Number(
            label="Flow Shift",
            minimum=1.0,
            maximum=12.0,
            value=3.0,
            interactive=True,
            step=0.1,
            visible=False
        )

    # Return the dictionary with all UI components
    return {
        'in_sampling': sampling,
        'in_steps': steps,
        'in_scheduler': scheduler,
        'in_width': width,
        'in_height': height,
        'in_batch_count': batch_count,
        'in_cfg': cfg,
        'in_guidance_btn': guidance_btn,
        'in_guidance': guidance,
        'in_flow_shift_btn': flow_shift_btn,
        'in_flow_shift': flow_shift
    }


def create_upscl_ui():
    """Create the upscale UI"""
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)

    with gr.Accordion(
        label="Upscale", open=False
    ):
        upscl_bool = gr.Checkbox(
            label="Enable Upscale", value=False
        )
        upscl = gr.Dropdown(
            label="Upscaler",
            choices=get_models(config.get('upscl_dir')),
            value="",
            allow_custom_value=True,
            interactive=False
        )
        with gr.Row():
            reload_upscl_btn = gr.Button(
                value=RELOAD_SYMBOL,
                interactive=False
            )
            clear_upscl_btn = gr.ClearButton(
                upscl,
                interactive=False)
        upscl_rep = gr.Slider(
            label="Upscaler repeats",
            minimum=1,
            maximum=5,
            value=1,
            step=1,
            interactive=False
        )

    upscale_comp = [upscl, reload_upscl_btn, clear_upscl_btn, upscl_rep]

    reload_upscl_btn.click(
        reload_models, inputs=[upscl_dir_txt], outputs=[upscl]
    )

    upscl_bool.change(
        partial(update_interactivity, len(upscale_comp)),
        inputs=upscl_bool,
        outputs=[upscl, reload_upscl_btn, clear_upscl_btn, upscl_rep]
    )
    
    return {
        'in_upscl_bool': upscl_bool,
        'in_upscl': upscl,
        'in_upscl_rep': upscl_rep
    }


def create_cnnet_ui():
    """Create the ControlNet UI"""
    cnnet_dir_txt = gr.Textbox(value=config.get('cnnet_dir'), visible=False)

    with gr.Accordion(
        label="ControlNet", open=False
    ):
        cnnet_enabled = gr.Checkbox(
            label="Enable ControlNet", value=False
        )
        with gr.Group():
            cnnet = gr.Dropdown(
                label="ControlNet",
                choices=get_models(config.get('cnnet_dir')),
                value=None,
                interactive=True
            )
            with gr.Row():
                reload_cnnet_btn = gr.Button(value=RELOAD_SYMBOL)
                gr.ClearButton(
                    cnnet
                )
        control_img = gr.Image(
            sources="upload", type="filepath"
        )
        control_strength = gr.Slider(
            label="ControlNet strength",
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.9)
        cnnet_cpu = gr.Checkbox(label="ControlNet on CPU")
        canny = gr.Checkbox(label="Canny (edge detection)")

    reload_cnnet_btn.click(
        reload_models,
        inputs=[cnnet_dir_txt],
        outputs=[cnnet]
    )

    # Return the dictionary with all UI components
    return {
        'in_cnnet_enabled': cnnet_enabled,
        'in_cnnet': cnnet,
        'in_control_img': control_img,
        'in_control_strength': control_strength,
        'in_cnnet_cpu': cnnet_cpu,
        'in_canny': canny
    }


def create_chroma_ui():
    """Create Chroma specific UI"""
    with gr.Accordion(
        label="Chroma settings", open=False
    ):
        with gr.Row():
            disable_dit_mask = gr.Checkbox(
                label="Disable DiT mask for Chroma",
            )
        with gr.Row():
            enable_t5_mask = gr.Checkbox(
                label="Enable T5 mask for Chroma",
            )
            t5_mask_pad = gr.Slider(
                label="T5 mask pad size for Chroma",
                minimum=0,
                maximum=1024,
                value=1,
                step=1,
            )
        return {
            'in_disable_dit_mask': disable_dit_mask,
            'in_enable_t5_mask': enable_t5_mask,
            'in_t5_mask_pad': t5_mask_pad
        }


def create_extras_ui():
    """Create the extras UI"""
    with gr.Accordion(
        label="Extra", open=False
    ):
        threads = gr.Number(
            label="Threads",
            minimum=0,
            maximum=os.cpu_count(),
            value=0
        )
        offload_to_cpu = gr.Checkbox(
            label="Offload to CPU")
        vae_tiling = gr.Checkbox(label="VAE Tiling")
        vae_cpu = gr.Checkbox(label="VAE on CPU")
        clip_cpu = gr.Checkbox(label="CLIP on CPU")
        rng = gr.Dropdown(
            label="RNG",
            choices=["std_default", "cuda"],
            value="cuda"
        )
        output = gr.Textbox(
            label="Output Name (optional)", value=""
        )
        color = gr.Checkbox(
            label="Color", value=True
        )
        flash_attn = gr.Checkbox(
            label="Flash Attention", value=config.get('def_flash_attn')
        )
        diffusion_conv_direct = gr.Checkbox(
            label="Conv2D Direct for diffusion",
            value=config.get('def_diffusion_conv_direct')
        )
        vae_conv_direct = gr.Checkbox(
            label="Conv2D Direct for VAE",
            value=config.get('def_vae_conv_direct')
        )
        verbose = gr.Checkbox(label="Verbose")

    # Return the dictionary with all UI components
    return {
        'in_threads': threads,
        'in_offload_to_cpu': offload_to_cpu,
        'in_vae_tiling': vae_tiling,
        'in_vae_cpu': vae_cpu,
        'in_clip_cpu': clip_cpu,
        'in_rng': rng,
        'in_output': output,
        'in_color': color,
        'in_flash_attn': flash_attn,
        'in_diffusion_conv_direct': diffusion_conv_direct,
        'in_vae_conv_direct': vae_conv_direct,
        'in_verbose': verbose
    }


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
        'env_GGML_VK_DISABLE_COOPMAT': disable_vk_coopmat,
        'env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT': disable_vk_int_dot
    }


def create_experimental_ui():
    """Create experimental UI"""
    with gr.Accordion(
        label="Experimental", open=False
    ):
        predict = gr.Dropdown(
            label="Prediction (WIP: PR #334)",
            choices=PREDICTION,
            value=config.get('def_predict')
        )
        preview_mode = gr.Dropdown(
            label="Preview mode (WIP: PR #522)",
            choices=PREVIEW,
            value="none"
        )
        preview_interval = gr.Number(
            label="Preview interval (PR #522)",
            value=1,
            minimum=1,
            interactive=True
        )
        preview_taesd = gr.Checkbox(
            label="TAESD for preview only (WIP: PR #522)"
        )
    return {
        'in_predict': predict,
        'in_preview_mode': preview_mode,
        'in_preview_interval': preview_interval,
        'in_preview_taesd': preview_taesd
    }


def create_folders_opt_ui():
    """Create the folder options UI"""
    with gr.Row():
        # Folders Accordion
        with gr.Accordion(
            label="Folders", open=False
        ):
            ckpt_dir_txt = gr.Textbox(
                label="Checkpoint folder",
                value=config.get('ckpt_dir'),
                interactive=True
            )
            unet_dir_txt = gr.Textbox(
                label="UNET folder",
                value=config.get('unet_dir'),
                interactive=True
            )
            vae_dir_txt = gr.Textbox(
                label="VAE folder",
                value=config.get('vae_dir'),
                interactive=True
            )
            clip_dir_txt = gr.Textbox(
                label="clip folder",
                value=config.get('clip_dir'),
                interactive=True
            )
            emb_dir_txt = gr.Textbox(
                label="Embeddings folder",
                value=config.get('emb_dir'),
                interactive=True
            )
            lora_dir_txt = gr.Textbox(
                label="Lora folder",
                value=config.get('lora_dir'),
                interactive=True
            )
            taesd_dir_txt = gr.Textbox(
                label="TAESD folder",
                value=config.get('taesd_dir'),
                interactive=True
            )
            phtmkr_dir_txt = gr.Textbox(
                label="PhotoMaker folder",
                value=config.get('phtmkr_dir'),
                interactive=True
            )
            upscl_dir_txt = gr.Textbox(
                label="Upscaler folder",
                value=config.get('upscl_dir'),
                interactive=True
            )
            cnnet_dir_txt = gr.Textbox(
                label="ControlNet folder",
                value=config.get('cnnet_dir'),
                interactive=True
            )
            txt2img_dir_txt = gr.Textbox(
                label="txt2img outputs folder",
                value=config.get('txt2img_dir'),
                interactive=True
            )
            img2img_dir_txt = gr.Textbox(
                label="img2img outputs folder",
                value=config.get('img2img_dir'),
                interactive=True
            )
            any2video_dir_txt = gr.Textbox(
                label="any2video output folder",
                value=config.get('any2video_dir'),
                interactive=True
            )

    # Return the dictionary with all UI components
    return {
        'ckpt_dir_txt': ckpt_dir_txt,
        'unet_dir_txt': unet_dir_txt,
        'vae_dir_txt': vae_dir_txt,
        'clip_dir_txt': clip_dir_txt,
        'emb_dir_txt': emb_dir_txt,
        'lora_dir_txt': lora_dir_txt,
        'taesd_dir_txt': taesd_dir_txt,
        'phtmkr_dir_txt': phtmkr_dir_txt,
        'upscl_dir_txt': upscl_dir_txt,
        'cnnet_dir_txt': cnnet_dir_txt,
        'txt2img_dir_txt': txt2img_dir_txt,
        'img2img_dir_txt': img2img_dir_txt,
        'any2video_dir_txt': any2video_dir_txt
    }
