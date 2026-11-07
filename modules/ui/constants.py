"""sd.cpp-webui - UI constants"""

from modules.shared_instance import sd_choices


QUANTS = [
    "Default", "f32", "f16", "q8_0", "q6_K", "q5_K", "q5_1",
    "q5_0", "q4_K", "q4_1", "q4_0", "q3_K", "q2_K"
]
FIELDS = [
    "pprompt", "nprompt", "width", "height", "steps",
    "sampling", "scheduler", "cfg", "seed"
]
SAMPLERS = ['Default'] + sd_choices.get_choices("samplers")
SCHEDULERS = ['Default'] + sd_choices.get_choices("schedulers")
MODELS = [
    "Checkpoint", "UNET", "VAE", "clip_g", "clip_l", "t5xxl", "llm",
    "tokenizer", "TAESD", "Lora", "Embeddings", "Upscaler", "ControlNet"
]
BUILTIN_UPSCALERS = [
    "Lanczos", "Nearest", "Latent", "Latent (nearest)",
    "Latent (nearest-exact)", "Latent (antialiased)",
    "Latent (bicubic)", "Latent (bicubic antialiased)"
]
CIRCULAR_PADDING = ["None", "Circular", "Circular X", "Circular Y"]
RNG = ['Default'] + sd_choices.get_choices("rng")
SAMPLER_RNG = ['Default'] + sd_choices.get_choices("sampler_rng")
PREVIEW = sd_choices.get_choices("previews")
PREDICTION = ['Default'] + sd_choices.get_choices("prediction")
CONTROL_TYPES = sd_choices.get_choices("control_types")
VAE_FORMAT = ["Default", "flux", "sd3", "flux2", "wan"]
CACHE_MODE = [
    "easycache", "ucache", "dbcache",
    "taylorseer", "cache-dit", "spectrum"
]
SCM_POLICY = ["none", "dynamic", "static"]
AUTO_ON_OFF = ["auto", "on", "off"]
ON_OFF = ["on", "off"]
AD_MASK_MODES = ["none", "merge", "merge_invert"]
PULID_EMBEDDING_EXT = (".pulidembd",)
LOG_LEVEL = [
    "debug", "verbose", "info",
    "warn", "error"
]

RELOAD_SYMBOL = '\U0001F504'
RANDOM_SYMBOL = '\U0001F3B2'
REUSE_SYMBOL = '\u267B\uFE0F'
SWITCH_V_SYMBOL = '\u2195'
SORT_OPTIONS = [
    "Date (Oldest First)",
    "Date (Newest First)",
    "Name (A-Z)",
    "Name (Z-A)"
]
THEMES = [
    "base", "default", "origin", "citrus",
    "monochrome", "soft", "glass", "ocean"
]
