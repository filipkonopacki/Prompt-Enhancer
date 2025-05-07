from pathlib import Path
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

ROOT_DIR = Path(__file__).resolve().parent.parent.__str__()
output_dir = ROOT_DIR + fr"\training\LoRAs\outputs\flan-t5-prompt-enhancer--{timestamp}"
final_lora = output_dir + r"\final_lora"
logging_dir = ROOT_DIR + fr"\training\LoRAs\logs\flan-t5-prompt-enhancer--{timestamp}"
tokenized_dataset_t5_path = ROOT_DIR + r"\data\tokenized_dataset\t5"
tokenized_dataset_mistral_path = ROOT_DIR + r"\data\tokenized_dataset\mistral"

instruction_templates = [
    "Enhance this image generation prompt:",
    "Rewrite this into a richly detailed image prompt:",
    "Make this prompt more vivid and imaginative:",
    "Turn this into a scene description with strong visual details:",
    "Upgrade this prompt into a detailed visual scene:",
    "Elaborate this into a descriptive image prompt:",
    "Expand this idea into a richly detailed visual scene:",
    "Transform this brief prompt into an immersive description:",
    "Develop this into a full and vivid image prompt:",
    "Craft a detailed and imaginative scene from this:",
    "Generate a rich, atmospheric description from this prompt:",
    "Describe this concept with as much visual detail as possible:",
    "Paint a vivid picture based on this simple prompt:"
]
