# Ensure all necessary modules are imported at the beginning
import os
import platform
import subprocess
import sys
from datetime import datetime
import time
from PIL import Image, ImageEnhance
import torch
from diffusers import DiffusionPipeline
from compel import Compel
import model_configs

# Add imports for space checking
from pathlib import Path
import shutil
import glob

# Define constants and functions needed for checking free space
CACHE_DIRECTORY = os.path.join(Path.home(), ".cache", "huggingface", "hub")
REQUIRED_FREE_SPACE_GB = 20

def check_free_space(directory):
    total, used, free = shutil.disk_usage(directory)
    free_gb = free / (2**30)
    return free_gb

def free_up_space(directory, required_gb):
    while check_free_space(directory) < required_gb:
        folders = [f for f in glob.glob(os.path.join(directory, '*')) if os.path.isdir(f)]
        if not folders:
            print("No more folders to delete in the cache directory.")
            break
        oldest_folder = min(folders, key=os.path.getmtime)
        print(f"Deleting oldest folder: {oldest_folder}")
        shutil.rmtree(oldest_folder)
    print("Sufficient space is now available.")

GLOBAL_NEGATIVE_PROMPT = "watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy, watermark, signature, text, logo, negativeV2,"
GLOBAL_GUIDANCE_VALUE = 7
GLOBAL_LORA_ENABLED = True

# Import GLOBAL_LORA_MODEL_LIST from model_configs
GLOBAL_LORA_MODEL_LIST = model_configs.GLOBAL_LORA_MODEL_LIST

def prepend_lora_trigger_phrases(prompt, lora_model):
    if GLOBAL_LORA_ENABLED:
        trigger_phrase = lora_model['trigger_phrase']
        prompt_with_trigger = f"{trigger_phrase}, {prompt}"
        return prompt_with_trigger
    else:
        return prompt

def create_pipe_with_lora(model_id: str, device: str, lora_model):
    # Load the pipeline without any adapters loaded
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    # Check if GLOBAL_LORA_ENABLED is true and lora_model is not None
    if GLOBAL_LORA_ENABLED and lora_model:
        print(f"Loading LoRA model: {lora_model['repo']}")
        pipe.load_lora_weights(
            lora_model["repo"],
            weight_name=lora_model["weight_name"],
            adapter_name=lora_model["adapter_name"],
            scale=lora_model.get("strength", 1.0)  # Default strength to 1.0 if not specified
        )
    else:
        print("GLOBAL_LORA_ENABLED is False or no LoRA model selected. Proceeding without loading LoRA model.")

    return pipe

# Define cache paths using the standard location for Windows
DIFFUSERS_CACHE_PATH = os.path.join(Path.home(), ".cache", "huggingface", "diffusers")
HUB_CACHE_PATH = os.path.join(Path.home(), ".cache", "huggingface", "hub")

def generate_with_long_prompt(pipe, cfg, device, modified_prompt):
    negative_prompt = cfg.get("NEGATIVE_PROMPT", GLOBAL_NEGATIVE_PROMPT)  # Use user-defined negative prompt or the global default negative prompt
    
    print(f"Processing long prompt of length {len(modified_prompt)}")

    try:
        if hasattr(pipe, "tokenizer") and hasattr(pipe, "text_encoder"):
            compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            parsed_prompt = compel.parse_prompt_string(modified_prompt)
            print(f"Parsed Prompt1: {parsed_prompt}")
        elif hasattr(pipe, 'text_model') and hasattr(pipe.text_model, 'tokenizer'):
            compel = Compel(tokenizer=pipe.text_model.tokenizer, text_encoder=pipe.text_model)
            parsed_prompt = compel.parse_prompt_string(modified_prompt)
            print(f"Parsed Prompt: {parsed_prompt}")
        else:
            raise AttributeError("The pipeline object does not have the expected tokenizer or text_encoder attributes.")

        print(f"Compel initialized with tokenizer and text encoder from the pipeline object.")
        print(f"Generating conditioning tensor with Compel...")
        conditioning = compel.build_conditioning_tensor(modified_prompt)
        print(f"Conditioning tensor shape: {conditioning.shape}")

        if conditioning.ndim == 3:
            print(f"Embeddings tensor has correct shape: {conditioning.shape}")
        else:
            print(f"WARNING: Embeddings tensor does not have the expected shape: {conditioning.shape}")

        print("Generating image with Compel generated conditioning tensor...")
        image = pipe(prompt=cfg["PROMPT_TO_CREATE"],
                     negative_prompt=cfg["NEGATIVE_PROMPT"],
                     num_inference_steps=cfg["NUM_INFERENCE_STEPS"],
                     guidance_scale=GLOBAL_GUIDANCE_VALUE).images[0]

        print("Image generated successfully with Compel.")
        return image

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Falling back to default generation process.")
        image = pipe(prompt=modified_prompt, negative_prompt=negative_prompt, num_inference_steps=cfg["NUM_INFERENCE_STEPS"]).images[0]
        print("Image generated with default process due to exception.")
        return image

def list_cached_models():
    print("Cached models:")

    list_cache(DIFFUSERS_CACHE_PATH, "Diffusers models")
    list_cache(HUB_CACHE_PATH, "Hub models")

def list_cache(cache_path, description):
    print(f"\n{description}:")
    if not os.path.isdir(cache_path):
        print(f"No cache directory found at '{cache_path}'.")
    else:
        model_directories = [d for d in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, d))]
        if not model_directories:
            print("No cached models found.")
        else:
            for idx, model_dir in enumerate(model_directories, 1):
                model_dir_path = os.path.join(cache_path, model_dir)
                print(f"{idx}. {model_dir} (Location: {model_dir_path})")

def list_models_and_choose():
    global_settings = model_configs.GLOBAL_IMAGE_SETTINGS
    print("Global configurations:")
    print(f"   - Prompt: {global_settings['PROMPT_TO_CREATE']}")
    print(f"   - Number of images to create: {global_settings['NUMBER_OF_IMAGES_TO_CREATE']}")
    print(f"   - Inference steps: {global_settings['NUM_INFERENCE_STEPS']}")
    print()

    model_keys = list(model_configs.MODEL_CONFIGS.keys())
    print("Available models and their configurations:")
    for idx, model_name in enumerate(model_keys, 1):
        model_config = model_configs.MODEL_CONFIGS[model_name]
        print(f"{idx}. {model_name}")
        if 'MODEL_ID' in model_config:
            print(f"   - Model ID: {model_config['MODEL_ID']}")
        else:
            # For refiner models, print base and refiner model IDs
            print(f"   - Base Model ID: {model_config['MODEL_ID_BASE']}")
            print(f"   - Refiner Model ID: {model_config['MODEL_ID_REFINER']}")
        print()

    selected_config = None
    while selected_config is None:
        user_input = input("Select a model by number: ")
        try:
            model_idx = int(user_input) - 1  # Adjust for 0-based indexing
            if model_idx < 0 or model_idx >= len(model_keys):
                print("Invalid selection. Please try again.")
            else:
                selected_model_key = model_keys[model_idx]
                selected_config = model_configs.MODEL_CONFIGS[selected_model_key]
                print(f"You have selected: {selected_model_key}")

        except ValueError:
            print("Invalid input. Please enter a number.")

    return selected_config

def list_lora_models_and_choose():
    print("\nAvailable LoRA models:")
    for idx, lora_model in enumerate(GLOBAL_LORA_MODEL_LIST, 1):
        print(f"{idx}. {lora_model['adapter_name']} (Repo: {lora_model['repo']})")

    selected_lora_model = None
    while selected_lora_model is None:
        user_input = input("\nSelect a LoRA model by number: ")
        try:
            lora_model_idx = int(user_input) - 1  # Adjust for 0-based indexing
            if lora_model_idx < 0 or lora_model_idx >= len(GLOBAL_LORA_MODEL_LIST):
                print("Invalid selection. Please try again.")
            else:
                selected_lora_model = GLOBAL_LORA_MODEL_LIST[lora_model_idx]
                print(f"You have selected: {selected_lora_model['adapter_name']}")

        except ValueError:
            print("Invalid input. Please enter a number.")

    return selected_lora_model

def prompt_for_global_settings():
    prompt_to_create = ""
    while not prompt_to_create.strip():
        prompt_to_create = input("\nEnter the prompt to create: ")

    number_of_images_to_create = -1
    while number_of_images_to_create <= 0:
        try:
            number_of_images_to_create = int(input("Enter the number of images to create: "))
            if number_of_images_to_create <= 0:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    num_inference_steps = -1
    while num_inference_steps <= 0:
        try:
            num_inference_steps = int(input("Enter the number of inference steps: "))
            if num_inference_steps <= 0:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    negative_prompt = ""
    while not negative_prompt.strip():
        negative_prompt = input("Enter the negative prompt: ")

    # Update the GLOBAL_IMAGE_SETTINGS with new values
    model_configs.GLOBAL_IMAGE_SETTINGS.update({
        "PROMPT_TO_CREATE": prompt_to_create,
        "NUMBER_OF_IMAGES_TO_CREATE": number_of_images_to_create,
        "NUM_INFERENCE_STEPS": num_inference_steps,
        "NEGATIVE_PROMPT": negative_prompt  # Add the negative prompt here
    })

    # Also update the CURRENT_CONFIG with new values if required
    model_configs.CURRENT_CONFIG.update({
        "PROMPT_TO_CREATE": prompt_to_create,
        "NUMBER_OF_IMAGES_TO_CREATE": number_of_images_to_create,
        "NUM_INFERENCE_STEPS": num_inference_steps,
        "NEGATIVE_PROMPT": negative_prompt  # Add the negative prompt here
    })


def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def format_time(seconds):
    return f"{int(seconds // 60)} minutes {seconds % 60:.2f} seconds"

def open_image(path):
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", path], check=True)
        elif sys.platform == "win32":  # Windows
            os.startfile(path)
        elif sys.platform.startswith('linux'):  # Linux
            subprocess.run(["xdg-open", path], check=True)
        else:
            print("Platform not supported for opening image.")
    except Exception as e:
        print(f"Failed to open image: {e}")

def post_process_image(image):
    config_values = model_configs.CURRENT_CONFIG
    factors = (config_values["UPSAMPLE_FACTOR"], config_values["SHARPNESS_ENHANCEMENT_FACTOR"],
               config_values["CONTRAST_ENHANCEMENT_FACTOR"])

    print("Resizing the image...")
    image = image.resize((image.width * factors[0], image.height * factors[0]), Image.LANCZOS)

    print("Enhancing image sharpness...")
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(factors[1])

    print("Increasing image contrast...")
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(factors[2])

    print("Post-processing complete.")
    return image

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_device_memory = torch.cuda.get_device_properties(0).total_memory
        cuda_device_memory_gb = cuda_device_memory / (1024 ** 3)
        hardware_summary = {
            "Device Type": "GPU",
            "Device Name": cuda_device_name,
            "Device Memory (GB)": f"{cuda_device_memory_gb:.2f}",
            "CUDA Version": torch.version.cuda,
        }
    else:
        device = torch.device("cpu")
        cpu_threads = torch.get_num_threads()
        hardware_summary = {
            "Device Type": "CPU",
            "Available Threads": cpu_threads,
        }

    print(f"PyTorch version: {torch.__version__}")
    device_info = f"Using device: {hardware_summary['Device Name']} with {hardware_summary['Device Memory (GB)']} GB of GPU memory and CUDA version {hardware_summary['CUDA Version']}" if "GPU" in hardware_summary["Device Type"] else f"Using device: CPU with {hardware_summary['Available Threads']} threads"
    print(device_info)

    return device, hardware_summary

def main():
    start_time = time.time()

    # Check and ensure sufficient free space before downloading models
    free_space_gb = check_free_space(CACHE_DIRECTORY)
    print(f"Currently available free space: {free_space_gb:.2f} GB")
    if free_space_gb < REQUIRED_FREE_SPACE_GB:
        print("Not enough free space, initiating cleanup...")

        free_up_space(CACHE_DIRECTORY, REQUIRED_FREE_SPACE_GB)
    else:
        print("Enough free space is available, no cleanup needed.")

    # Prompt user to choose a model first
    model_configs.CURRENT_CONFIG = list_models_and_choose()

    # Prompt user to choose a LoRA model
    selected_lora_model = list_lora_models_and_choose()

    # Prompt user for global settings
    prompt_for_global_settings()

    # List cached models (now knowing what the user has chosen)
    list_cached_models()

    device, hardware_summary = setup_device()
    cfg = model_configs.CURRENT_CONFIG
    os.makedirs(model_configs.GLOBAL_IMAGE_SETTINGS["IMAGES_DIRECTORY"], exist_ok=True)
    use_refiner = "MODEL_ID_REFINER" in cfg

    if use_refiner:
        pass  # Your existing logic for refiner models

    else:
        pipeline_kwargs = {
            "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
            **cfg.get("ADDITIONAL_PIPELINE_ARGS", {}),
            "safety_checker": None if not cfg["ADD_SAFETY_CHECKER"] else None
        }

        pipe = DiffusionPipeline.from_pretrained(cfg["MODEL_ID"], **pipeline_kwargs).to(device)

        if GLOBAL_LORA_ENABLED:
            print("LoRA is enabled. Attempting to load specified LoRA weights with individual strength.")

            # Print the user-entered settings
            print(f"Settings used:")
            print(f"   - Prompt: {model_configs.GLOBAL_IMAGE_SETTINGS['PROMPT_TO_CREATE']}")
            print(f"   - Negative Prompt: {model_configs.GLOBAL_IMAGE_SETTINGS['NEGATIVE_PROMPT']}")
            print(f"   - Number of images to create: {model_configs.GLOBAL_IMAGE_SETTINGS['NUMBER_OF_IMAGES_TO_CREATE']}")
            print(f"   - Inference steps: {model_configs.GLOBAL_IMAGE_SETTINGS['NUM_INFERENCE_STEPS']}")
            
            try:
                print(f"Loading LoRA model from {selected_lora_model['repo']}"
                    f" with adapter '{selected_lora_model.get('adapter_name')}'"
                    f" and strength '{selected_lora_model.get('strength')}'.")
                pipe.load_lora_weights(
                    selected_lora_model["repo"],
                    weight_name=selected_lora_model["weight_name"],
                    adapter_name=selected_lora_model.get("adapter_name"),
                    scale=selected_lora_model.get("strength", 1.0)  # Default strength to 1.0 if not specified
                )
                print(f"Successfully loaded LoRA weights '{selected_lora_model.get('adapter_name')}' with strength {selected_lora_model.get('strength')}.")
            except Exception as e:
                print(f"Failed to load LoRA weights for '{selected_lora_model.get('adapter_name')}' due to an error: {e}")
        else:
            print("LoRA is disabled. Proceeding without loading any LoRA weights.")
            print("The model will generate images based on the original configuration.")

        seed = model_configs.GLOBAL_IMAGE_SETTINGS.get("SEED")
        if seed is None:
            seed = int(time.time())
            model_configs.GLOBAL_IMAGE_SETTINGS["SEED"] = seed

        torch.manual_seed(seed)
        print(f"Current seed value: {seed}")

    generation_times = []

    for i in range(model_configs.GLOBAL_IMAGE_SETTINGS["NUMBER_OF_IMAGES_TO_CREATE"]):
        start_gen_time = time.time()
        print(f"Processing image {i+1} of {model_configs.GLOBAL_IMAGE_SETTINGS['NUMBER_OF_IMAGES_TO_CREATE']}...")

        additional_args = cfg.get("ADDITIONAL_PIPELINE_ARGS_BASE", {}) if use_refiner else cfg.get("ADDITIONAL_PIPELINE_ARGS", {})

        timestamp = datetime.now().strftime(model_configs.GLOBAL_IMAGE_SETTINGS["TIMESTAMP_FORMAT"])
        model_prefix = cfg.get("MODEL_PREFIX", "")

        base_filename = model_configs.GLOBAL_IMAGE_SETTINGS["FILENAME_TEMPLATE"].format(
            model_prefix=model_prefix,
            timestamp=timestamp
        )

        cfg = model_configs.CURRENT_CONFIG  

        modified_prompt = prepend_lora_trigger_phrases(cfg["PROMPT_TO_CREATE"], selected_lora_model)

        if use_refiner:
            pass  # Add your specialized logic for handling refiner models
        else:
            if len(modified_prompt) > 77:
                print("Prompt exceeds token limit. Engaging long prompt handling function...")
                image = generate_with_long_prompt(pipe, cfg, device, modified_prompt)
            else:
                print("Prompt within token limit. Proceeding with regular generation process...")
                image = pipe(prompt=cfg["PROMPT_TO_CREATE"],
                            negative_prompt=cfg["NEGATIVE_PROMPT"],
                            num_inference_steps=cfg["NUM_INFERENCE_STEPS"],
                            guidance_scale=GLOBAL_GUIDANCE_VALUE).images[0]

        print("Starting post-processing with Pillow...")
        image = post_process_image(image)

        final_filename = base_filename  # This already has the correct format
        final_img_path = os.path.join(cfg["IMAGES_DIRECTORY"], final_filename)
        image.save(final_img_path)

        gen_time = time.time() - start_gen_time
        generation_times.append(gen_time)

        print(f"Image {i+1}/{cfg['NUMBER_OF_IMAGES_TO_CREATE']} saved in '{cfg['IMAGES_DIRECTORY']}' as {final_filename}")
        print(f"Full path: {os.path.abspath(final_img_path)}")
        print(f"Single image generation time: {format_time(gen_time)}")

        if model_configs.GLOBAL_IMAGE_SETTINGS["OPEN_IMAGE_AFTER_CREATION"]:
            open_image(final_img_path)

    total_time = time.time() - start_time
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0

    print("==== SUMMARY ====")
    print(f"Total execution time: {format_time(total_time)}")
    print(f"Average generation time per image: {format_time(avg_time)}")
    
    for key, val in hardware_summary.items():
        print(f"{key}: {val}")

    print("\n--- Configuration Details ---")
    config_values = model_configs.CURRENT_CONFIG
    for key, val in config_values.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    main()