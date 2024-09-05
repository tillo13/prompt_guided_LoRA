# LoRA Prompter

## Overview
The LoRA Prompter is a Python application designed to generate high-quality images using advanced generative models like Stable Diffusion and its variants. It uniquely incorporates Low-Rank Adaptation (LoRA) models to fine-tune and enhance the output quality. This application is user-friendly and packed with features such as prompt customization, long prompt handling, robust post-processing, and efficient cache management.

## Features

### 1. **Multiple Model Integration**
The LoRA Prompter supports various pre-configured models known for generating a wide range of artistic styles. Users can easily select from the list of available models, each with its specific settings. This flexibility allows you to create images that align precisely with your creative vision.

### 2. **LoRA (Low-Rank Adaptation) Model Support**
LoRA models are specialized to enhance specific aspects of the image generation process. The LoRA Prompter can load these LoRA models and append their trigger phrases to your input prompts. This integration allows for more nuanced and detailed image generation. Users can adjust the strength of these LoRA models to control their impact on the final output.

### 3. **Prompt and Negative Prompt Handling**
When running the application, users are asked to provide both a main prompt and a negative prompt. The main prompt describes what you want to create, while the negative prompt helps filter out undesired elements from generated images, thereby improving the overall quality and relevance of the results.

### 4. **Long Prompt Handling with Compel**
The Compel library helps manage prompt strings that exceed token limits. It parses these long prompts and builds conditioning tensors that ensure the entire input is considered during image generation. This feature is particularly useful for complex and detailed descriptions that might otherwise be truncated.

### 5. **Post-Processing with Pillow**
After generating an image, the application uses the Pillow library for post-processing. This includes resizing the image to enhance its resolution and applying filters to improve sharpness and contrast. These enhancements ensure that the final output is visually appealing and of high quality.

### 6. **Effective Cache Management**
Disk space management is crucial for applications dealing with large models and datasets. The LoRA Prompter checks the available disk space in the cache directory and deletes the oldest cached folders if necessary. This process helps maintain sufficient free space for new models and data.

### 7. **Device Setup and Optimization**
The application automatically detects the available hardware and optimizes its operations accordingly. If a GPU is available, it will leverage CUDA (a parallel computing platform) for faster processing. Otherwise, it will use the CPU. This optimization ensures efficient use of system resources.

### 8. **Flexible Configuration via Model Configs**
Configurations for each model are maintained in a separate file (`model_configs.py`). This setup allows for easy modifications and additions of new models without the need to alter the main application code.

### 9. **User Interaction and Feedback**
The application provides a user-friendly command line interface, prompting users for necessary inputs and giving detailed feedback on the applied settings and generation process. This interaction ensures that users are informed and in control throughout the entire process.

### 10. **Automatic Image Saving and Viewing**
Generated images are automatically saved in a specified directory with filenames based on a customizable template. Additionally, users have the option to open and view the images immediately after creation.

## Setup and Installation

### Prerequisites
Before running the LoRA Prompter, ensure that the following prerequisites are met:
1. **Python**: Make sure you have Python 3.7 or later installed.
2. **Pip**: Ensure pip is installed for managing Python packages.
3. **Git**: Optionally, you may need Git to clone certain repositories.

### Installation
1. **Clone the Repository**
   ```sh
   git clone <repository_url>
   cd <repository_directory>
Install Required Python Packages Run the provided Python script to install all necessary libraries:

python -m pip install -r requirements.txt
If requirements.txt is not provided:

python -m pip install pillow diffusers transformers accelerate safetensors
Running the Application
Navigate to the Application Directory

cd <repository_directory>
Run the Application

python lora_prompter.py
User Inputs
Upon running the application, you will be prompted to provide the following inputs:

Model Selection: Choose the desired model from the available list.
LoRA Model Selection: Choose a LoRA model to enhance the base model.
Prompt: Enter the main prompt describing what you want to generate.
Negative Prompt: Enter the negative prompt to exclude unwanted elements.
Number of Images: Specify the number of images to generate.
Inference Steps: Define the number of inference steps for generating images.
Example Run
python lora_prompter.py

# Output example:
Currently available free space: 100.00 GB
Select a model by number:
1. RealVisXL V3.0 Turbo
2. RealLife
...
You have selected: RealLife

Available LoRA models:
1. pixel
2. toy
...
Select a LoRA model by number: 1
You have selected: pixel

Enter the prompt to create: "A futuristic cityscape"
Enter the negative prompt: "fog, noise"
Enter the number of images to create: 3
Enter the number of inference steps: 50
...
Detailed Configuration
model_configs.py
The model_configs.py file stores global image settings and model-specific configurations. Here’s an overview:

GLOBAL_IMAGE_SETTINGS: Contains the global settings that apply to all configurations, such as prompt, number of images, inference steps, etc.
PILLOW_CONFIG: Defines the post-processing settings for the generated images.
GLOBAL_LORA_MODEL_LIST: List of available LoRA models with their respective settings.
MODEL_CONFIGS: Dictionary of model-specific configurations, each with unique settings and identifiers.
# Example content from model_configs.py
GLOBAL_IMAGE_SETTINGS = {
    "PROMPT_TO_CREATE": "a happy lady",
    "NUMBER_OF_IMAGES_TO_CREATE": 5,
    ...
}

PILLOW_CONFIG = {
    "UPSAMPLE_FACTOR": 2,
    "SHARPNESS_ENHANCEMENT_FACTOR": 2.0,
    ...
}

GLOBAL_LORA_MODEL_LIST = [
    {"repo": "nerijs/pixel-art-xl", "weight_name": "pixel-art-xl.safetensors", "adapter_name": "pixel", "trigger_phrase": "pixel art", "strength": 0.1},
    ...
]

MODEL_CONFIGS = {
    "RealVisXL V3.0 Turbo": {
        "MODEL_ID": "SG161222/RealVisXL_V3.0_Turbo",
        ...
    },
    ...
}
Next Step Ideas
Graphical User Interface (GUI)

Develop a GUI to make the application more accessible to non-technical users, allowing them to interact with the application more intuitively.
Advanced Image Editing Features

Integrate advanced image editing tools such as color correction, filters, and landscape generation, giving users more control over the final output.
Automated Model Updates

Implement a feature to automatically check and update models, ensuring users always have access to the latest and most optimized versions.
Batch Processing

Add batch processing capabilities to generate multiple image sets with different prompts and settings in one run, streamlining large-scale image creation projects.
Prompt Optimization Suggestions

Introduce an intelligent system that analyzes previous successful generations to suggest optimizations for future prompts, enhancing the quality and relevance of generated images.
Integration with Online Databases

Allow users to fetch prompts, models, and configurations from online repositories and databases, expanding the variety of available resources.
Interactive Prompt Refinement

Create an interactive tool that guides users in crafting more effective prompts through a series of questions and suggestions, improving the final output.
Custom Model Training

Enable users to train their own models using specific datasets, tailoring the application to meet unique and personalized image generation requirements.
Logging and Analytics

Implement detailed logging and analytics features to track generation statistics, user preferences, and performance metrics, aiding in continuous improvement and user satisfaction.
Cloud Integration

Develop cloud integration capabilities to leverage cloud-based GPUs for faster processing, providing more scalable solutions and ensuring accessibility for users with varying hardware capabilities.
Conclusion
The LoRA Prompter is a comprehensive tool for generating high-quality images using state-of-the-art models. With its integration of LoRA models, advanced prompt handling, and user-focused features, it caters to both creative and technical users. The suggested next step ideas provide a roadmap for further enhancements, ensuring the LoRA Prompter continues to evolve and meet the diverse needs of its users.