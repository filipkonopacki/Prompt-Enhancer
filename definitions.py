import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = ROOT_DIR + f"/training/LoRAs/outputs/flan-t5-prompt-enhancer--{timestamp}"
final_lora = output_dir + "/final_lora"
logging_dir = ROOT_DIR + f"/training/LoRAs/logs/flan-t5-prompt-enhancer--{timestamp}"
tokenized_dataset_path = ROOT_DIR + "/data/tokenized_dataset"
