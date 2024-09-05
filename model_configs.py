# Global image settings that apply to any model configuration
GLOBAL_IMAGE_SETTINGS = {
    "PROMPT_TO_CREATE": "a happy cat",
    "NUMBER_OF_IMAGES_TO_CREATE": 5,
    "NUM_INFERENCE_STEPS": 25,
    "OPEN_IMAGE_AFTER_CREATION": False,
    "IMAGES_DIRECTORY": "created_images",
    "FILENAME_TEMPLATE": "{model_prefix}_{timestamp}.png",
    "TIMESTAMP_FORMAT": "%Y%m%d_%H%M%S_%f",
    "ADD_SAFETY_CHECKER": False,
    "DEFAULT_MODEL_ID": "stablediffusionapi/disney-pixar-cartoon",
}

# Shared Pillow post-processing configuration
PILLOW_CONFIG = {
    "UPSAMPLE_FACTOR": 2,
    "SHARPNESS_ENHANCEMENT_FACTOR": 2.0,
    "CONTRAST_ENHANCEMENT_FACTOR": 1.5,
}

# LoRA model settings
GLOBAL_LORA_MODEL_LIST = [
    {"repo": "nerijs/pixel-art-xl", "weight_name": "pixel-art-xl.safetensors", "adapter_name": "pixel", "trigger_phrase": "pixel art", "strength": 0.1},
    {"repo": "CiroN2022/toy-face", "weight_name": "toy_face_sdxl.safetensors", "adapter_name": "toy", "trigger_phrase": "toy_face", "strength": 1.5},
    {"repo": "Blib-la/caricature_lora_sdxl", "weight_name": "caricature_sdxl_v2.safetensors", "adapter_name": "caricature", "trigger_phrase": "caricature", "strength": 2.4},
    {"repo": "ntc-ai/SDXL-LoRA-slider.nice-hands", "weight_name": "nice hands.safetensors", "adapter_name": "nice hands", "trigger_phrase": "nice hands", "strength": 2.0},
    {"repo": "ntc-ai/SDXL-LoRA-slider.captivating-eyes", "weight_name": "captivating eyes.safetensors", "adapter_name": "captivating eyes", "trigger_phrase": "captivating eyes", "strength": 2.0},
    {"repo": "ntc-ai/SDXL-LoRA-slider.huge-anime-eyes", "weight_name": "huge anime eyes.safetensors", "adapter_name": "huge anime eyes", "trigger_phrase": "huge anime eyes", "strength": 1.75},
    {"repo": "ntc-ai/SDXL-LoRA-slider.micro-details-fine-details-detailed", "weight_name": "micro details, fine details, detailed.safetensors", "adapter_name": "micro", "trigger_phrase": "detailed", "strength": 2.4},
]

# Model-specific configurations stored in a dictionary
MODEL_CONFIGS = {
    "RealVisXL V3.0 Turbo": {
        "MODEL_ID": "SG161222/RealVisXL_V3.0_Turbo",
        "MODEL_PREFIX": "realVizXLv3",
    },
    "RealLife": {
        "MODEL_ID": "Yntec/RealLife",
        "MODEL_PREFIX": "RealLife",
    },
    "pixar": {
        "MODEL_ID": "stablediffusionapi/disney-pixar-cartoon",
        "MODEL_PREFIX": "pixar",
    },
    "rv4": { # sdxl lightning trained
        "MODEL_ID": "SG161222/RealVisXL_V4.0",
        "MODEL_PREFIX": "rv4",
    },
    "j9": { # sdxl1
        "MODEL_ID": "stablediffusionapi/juggernaut-xl-v9",
        "MODEL_PREFIX": "j9",
    },
    # Adding new models
    "ProteusV0.4": {
        "MODEL_ID": "dataautogpt3/ProteusV0.4",
        "MODEL_PREFIX": "proteusV04",
    },
    "JuggernautXv10": {
        "MODEL_ID": "RunDiffusion/Juggernaut-X-v10",
        "MODEL_PREFIX": "jxv10",
    },
    "JuggernautXLLightning": {
        "MODEL_ID": "RunDiffusion/Juggernaut-XL-Lightning",
        "MODEL_PREFIX": "jxlLight",
    },
    "DreamshaperXLV2Turbo": {
        "MODEL_ID": "Lykon/dreamshaper-xl-v2-turbo",
        "MODEL_PREFIX": "dsxlv2",
    },
    "CounterfeitXL": {
        "MODEL_ID": "gsdf/CounterfeitXL",
        "MODEL_PREFIX": "cfxl",
    }
}

# Extend global and Pillow settings to each model configuration
for config in MODEL_CONFIGS.values():
    config.update(GLOBAL_IMAGE_SETTINGS)
    config.update(PILLOW_CONFIG)

# List of required Python packages for executing the script which are common for all models
REQUIRED_PACKAGES = [
    "pillow",  # For image manipulation
    "diffusers",  # For text-to-image diffusion models
    "transformers",  # For transformer models from Hugging Face
    "accelerate",  # For speeding up model computation
    "safetensors",  # For safe serialization of tensors
]