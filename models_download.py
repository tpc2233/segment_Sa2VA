import os
import argparse
from huggingface_hub import snapshot_download

# Define the models we want to be able to download
# We map a simple name (like "26B") to the full Hugging Face repository ID.
MODELS = {
    "26B": "ByteDance/Sa2VA-26B",
    "8B": "ByteDance/Sa2VA-8B",
    "4B": "ByteDance/Sa2VA-4B",
}

# Define the main directory where all models will be stored
DOWNLOAD_DIR = "models_downloads"

def download_model(model_id: str):
    """
    Downloads a model from the Hugging Face Hub to a local directory.

    Args:
        model_id (str): The repository ID of the model on Hugging Face (e.g., "ByteDance/Sa2VA-26B").
    """
    # Create the main download directory if it doesn't exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Get the simple model name (e.g., "Sa2VA-26B") to use as the folder name
    model_name = model_id.split('/')[-1]
    local_model_path = os.path.join(DOWNLOAD_DIR, model_name)
    
    # Check if the model is already downloaded by looking for a key file like config.json
    # This prevents re-downloading the entire model if it's already present.
    if os.path.exists(os.path.join(local_model_path, 'config.json')):
        print(f"✅ Model '{model_name}' already exists in '{local_model_path}'. Skipping.")
        return

    print(f"⬇️ Downloading model '{model_id}' to '{local_model_path}'...")
    print("   This may take a while depending on the model size and your internet connection.")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,  # Use False for better compatibility (especially on Windows)
            # You can add ignore_patterns here if you want to skip certain file types
            # For example: ignore_patterns=["*.safetensors", "*.onnx"]
        )
        print(f"✅ Successfully downloaded '{model_name}'.")
    except Exception as e:
        print(f"❌ Failed to download '{model_name}'. Error: {e}")


def main():
    """
    Main function to parse command-line arguments and trigger downloads.
    """
    parser = argparse.ArgumentParser(
        description="Download Sa2VA models from Hugging Face Hub.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model_version",
        type=str,
        choices=list(MODELS.keys()) + ["all"],
        help=(
            "The model version to download.\n"
            "Choices are:\n"
            "  '26B' - Download the 26B parameter model.\n"
            "  '8B'  - Download the 8B parameter model.\n"
            "  '4B'  - Download the 4B parameter model.\n"
            "  'all' - Download all available models."
        )
    )
    
    args = parser.parse_args()
    
    if args.model_version == "all":
        print("Starting download for all models...")
        for version_key in MODELS:
            model_repo_id = MODELS[version_key]
            download_model(model_repo_id)
        print("\nAll model downloads attempted.")
    else:
        model_repo_id = MODELS[args.model_version]
        download_model(model_repo_id)

if __name__ == "__main__":
    main()
