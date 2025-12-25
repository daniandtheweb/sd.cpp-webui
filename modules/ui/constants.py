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
CIRCULAR_PADDING= ["None", "Circular", "Circular X", "Circular Y"]
RNG = sd_options.get_opt("rng")
SAMPLER_RNG = sd_options.get_opt("sampler_rng")
PREVIEW = sd_options.get_opt("previews")
PREDICTION = ['Default'] + sd_options.get_opt("prediction")
CACHE_MODE = ["easycache", "ucache", "dbcache", "taylorseer", "cache-dit"]
CACHE_DIT_PRESET = ["none","slow", "medium", "fast", "ultra"]
SCM_POLICY = ["none", "dynamic", "static"]
RELOAD_SYMBOL = '\U0001F504'
RANDOM_SYMBOL = '\U0001F3B2'
SWITCH_V_SYMBOL = '\u2195'
SORT_OPTIONS = [
    "Date (Oldest First)",
    "Date (Newest First)",
    "Name (A-Z)",
    "Name (Z-A)"
]
