from pathlib import Path
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

ROOT_DIR = Path(__file__).resolve().parent.parent.__str__()
output_dir = ROOT_DIR + fr"\training\LoRAs\outputs\flan-t5-prompt-enhancer--{timestamp}"
final_lora = output_dir + r"\final_lora"
logging_dir = ROOT_DIR + fr"\training\LoRAs\logs\flan-t5-prompt-enhancer--{timestamp}"
tokenized_dataset_path = ROOT_DIR + r"\data\tokenized_dataset"

instruction_templates = [
    "Enhance this image generation prompt:",
    "Rewrite this into a richly detailed image prompt:",
    "Make this prompt more vivid and imaginative:",
    "Turn this into a scene description with strong visual details:",
    "Upgrade this prompt into a detailed visual scene:",
    "Elaborate this into a descriptive image prompt:"
]
