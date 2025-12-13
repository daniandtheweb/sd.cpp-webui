"""sd.cpp-webui - UI constants"""

from modules.shared_instance import sd_options


QUANTS = [
    "Default", "f32", "f16", "q8_0", "q6_K", "q5_K", "q5_1",
    "q5_0", "q4_K", "q4_1", "q4_0", "q3_K", "q2_K"
]
FIELDS = [
    "pprompt", "nprompt", "width", "height", "steps",
    "sampling", "scheduler", "cfg", "seed"
]
SAMPLERS = sd_options.get_opt("samplers")
SCHEDULERS = sd_options.get_opt("schedulers")
MODELS = [
    "Checkpoint", "UNET", "VAE", "clip_g", "clip_l", "t5xxl", "llm",
    "TAESD", "Lora", "Embeddings", "Upscaler", "ControlNet"
]
RNG = sd_options.get_opt("rng")
SAMPLER_RNG = sd_options.get_opt("sampler_rng")
PREVIEW = sd_options.get_opt("previews")
PREDICTION = ['Default'] + sd_options.get_opt("prediction")
RELOAD_SYMBOL = '\U0001F504'
RANDOM_SYMBOL = '\U0001F3B2'
SWITCH_V_SYMBOL = '\u2195'
