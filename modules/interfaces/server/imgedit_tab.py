"""sd.cpp-webui - Image edit UI"""

import gradio as gr

from modules.core.server.sdcpp_server import (
    imgedit_api, cancel_active_job
)
from modules.interfaces.server.controls import (
    create_listen_ui, create_server_controls, bind_server_controls
)
from modules.shared_instance import config
from modules.utils.image_utils import size_updater
from modules.utils.ui_events import (
    get_ordered_inputs, bind_generation_pipeline,
    refresh_all_options
)
from modules.ui.models import create_imgedit_model_sel_ui
from modules.ui.loras import (
    create_lora_sel_ui, bind_lora_events
)
from modules.ui.prompts import create_prompts_ui
from modules.ui.generation_area import create_generation_area
from modules.ui.model_presets import (
    create_model_presets_ui, bind_model_presets_events
)
from modules.ui.generation_settings import (
    create_quant_ui, create_generation_settings_ui,
    create_bottom_generation_settings_ui
)
from modules.ui.upscale import create_upscl_ui
from modules.ui.controlnet import create_cnnet_ui
from modules.ui.adetailer import create_adetailer_ui
# from modules.ui.qwen import create_qwen_ui
from modules.ui.circular import create_circular_ui
from modules.ui.photomaker import create_photomaker_ui
from modules.ui.eta import create_eta_ui
from modules.ui.ref_img import create_ref_img_ui
from modules.ui.qwen_layered import create_qwen_layered_ui
from modules.ui.taesd import create_taesd_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.cache import create_cache_ui
from modules.ui.extra import create_extras_ui
from modules.ui.output import create_output_ui
from modules.ui.performance import create_performance_ui
from modules.ui.deprecated import create_deprecated_ui
from modules.ui.environment import create_env_ui
# from modules.ui.experimental import create_experimental_ui


with gr.Blocks() as imgedit_server_block:
    inputs_map = {}
    diffusion_mode = gr.Number(value=1, visible=False)
    # Directory Textboxes
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)

    # Title
    imgedit_title = gr.Markdown("# Image Edit")

    with gr.Accordion(
        label="Server configuration", open=False
    ):
        with gr.Accordion(
            label="Models selection", open=False
        ):
            # Model & VAE Selection
            model_ui = create_imgedit_model_sel_ui()
            inputs_map.update(model_ui)
            inputs_map['in_diffusion_mode'] = diffusion_mode

            # Model Type Selection
            quant_ui = create_quant_ui()
            inputs_map.update(quant_ui)

            # Model Presets
            model_presets_ui = create_model_presets_ui()

        with gr.Accordion(
            label="Server Settings", open=False
        ):
            with gr.Tab("Image Enhancement"):

                # ControlNet
                cnnet_ui = create_cnnet_ui(inputs=False)
                cnnet_ui = {'in_cnnet': cnnet_ui['in_cnnet']}
                inputs_map.update(cnnet_ui)

                # ADetailer
                adetailer_ui = create_adetailer_ui(inputs=False)
                adetailer_ui = {'in_ad_model': adetailer_ui['in_ad_model']}
                inputs_map.update(adetailer_ui)

                # Qwen
                # qwen_ui = create_qwen_ui()
                # inputs_map.update(qwen_ui)

                # Circular padding
                circular_ui = create_circular_ui()
                inputs_map.update(circular_ui)

                # PhotoMaker
                photomaker_ui = create_photomaker_ui()
                inputs_map.update(photomaker_ui)

            with gr.Tab("Advanced Settings"):

                # TAESD
                taesd_ui = create_taesd_ui()
                inputs_map.update(taesd_ui)

                # Extra Settings
                extras_ui = create_extras_ui(generation=False)
                del extras_ui['in_sigmas']
                del extras_ui['in_extra_sample_args']
                inputs_map.update(extras_ui)

                # Performance Settings
                performance_ui = create_performance_ui()
                inputs_map.update(performance_ui)

                # Environment Variables
                env_ui = create_env_ui()
                inputs_map.update(env_ui)

                # Deprecated placement flags
                deprecated_ui = create_deprecated_ui()
                inputs_map.update(deprecated_ui)

            listen_ui = create_listen_ui()
            inputs_map.update(listen_ui)

    server_controls = create_server_controls()

    # Loras
    lora_ui = create_lora_sel_ui()
    inputs_map.update(lora_ui)

    # Prompts
    prompts_ui = create_prompts_ui(nprompt_support=False)
    inputs_map.update(prompts_ui)

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            with gr.Tab("Generation Settings"):

                generation_settings_ui = create_generation_settings_ui(unet_mode=True)
                inputs_map.update(generation_settings_ui)

                bottom_generation_settings_ui = create_bottom_generation_settings_ui()
                inputs_map.update(bottom_generation_settings_ui)

            with gr.Tab("Image Enhancement"):

                # Upscale
                upscl_ui = create_upscl_ui()
                inputs_map.update(upscl_ui)

                # ControlNet
                cnnet_inputs_ui = create_cnnet_ui(model_sel=False)
                del cnnet_inputs_ui['in_cnnet']
                inputs_map.update(cnnet_inputs_ui)

                # ADetailer
                adetailer_inputs_ui = create_adetailer_ui(model_sel=False)
                del adetailer_inputs_ui['in_ad_bool']
                del adetailer_inputs_ui['in_ad_model']
                inputs_map.update(adetailer_inputs_ui)

                # ETA
                eta_ui = create_eta_ui()
                inputs_map.update(eta_ui)

                # Reference Image Settings
                ref_img_ui = create_ref_img_ui()
                inputs_map.update(ref_img_ui)

                # Qwen Image Layered
                qwen_layered_ui = create_qwen_layered_ui()
                inputs_map.update(qwen_layered_ui)

            with gr.Tab("Advanced Settings"):

                # VAE Tiling
                vae_tiling_ui = create_vae_tiling_ui()
                inputs_map.update(vae_tiling_ui)

                # Cache
                cache_ui = create_cache_ui()
                inputs_map.update(cache_ui)

                # Output
                output_ui = create_output_ui()
                inputs_map.update(output_ui)

                # Extra Settings
                extras_gen_ui = create_extras_ui(launch=False)
                extras_gen_ui = {
                    k: extras_gen_ui[k]
                    for k in ('in_sigmas', 'in_extra_sample_args')
                }
                inputs_map.update(extras_gen_ui)

            # Experimental
            # experimental_ui = create_experimental_ui()
            # inputs_map.update(experimental_ui)

            with gr.Row():
                refresh_choices = gr.Button(
                    value="Refresh sd choices"
                )

        # Output
        with gr.Column(scale=1):
            with gr.Row():
                ref_img_imgedit_server = gr.Gallery(
                    label="Reference Images",
                    interactive=True,
                    type="filepath",
                    object_fit="contain",
                    height="auto"
                )
                inputs_map['in_ref_img'] = ref_img_imgedit_server
            (gen_btn, kill_btn, queue_tracker, progress_bar, img_final,
             stats, command) = create_generation_area(server=True)

    ordered_keys, ordered_components = get_ordered_inputs(inputs_map)

    _server_config_keys = (
        list(model_ui.keys())
        + list(quant_ui.keys())
        + list(cnnet_ui.keys())
        + list(adetailer_ui.keys())
        + list(circular_ui.keys())
        + list(photomaker_ui.keys())
        + list(taesd_ui.keys())
        + list(extras_ui.keys())
        + list(performance_ui.keys())
        + list(env_ui.keys())
        + list(deprecated_ui.keys())
        + ['in_ip', 'in_port']
    )

    bind_server_controls(
        imgedit_server_block, server_controls, gen_btn,
        [progress_bar, stats, command, img_final, queue_tracker],
        _server_config_keys, inputs_map, ordered_keys, ordered_components
    )

    bind_lora_events(lora_ui, prompts_ui)

    bind_model_presets_events(
        model_presets_ui, model_ui, generation_settings_ui
    )

    ui_outputs = {
        'gen_btn': gen_btn,
        'command': command,
        'progress_bar': progress_bar,
        'stats': stats,
        'img_final': img_final,
        'queue_tracker': queue_tracker
    }

    bind_generation_pipeline(
        imgedit_api, ordered_keys, ordered_components, ui_outputs
    )

    kill_btn.click(
        cancel_active_job,
        inputs=[],
        outputs=[]
    )

    refresh_choices.click(
        refresh_all_options,
        inputs=[],
        outputs=[
            generation_settings_ui['in_sampling'],
            generation_settings_ui['in_scheduler'],
            extras_ui['in_predict']
        ]
    )

    ref_img_imgedit_server.change(
        size_updater,
        inputs=ref_img_imgedit_server,
        outputs=[
            generation_settings_ui['in_width'],
            generation_settings_ui['in_height']
        ]
    )

    width_imgedit_server = generation_settings_ui['in_width']
    height_imgedit_server = generation_settings_ui['in_height']
