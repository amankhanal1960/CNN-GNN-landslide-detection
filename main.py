# main.py
import argparse
import yaml
from src.train import train

def main(config_path):
    # 1. Load YAML Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("--- Starting Cloud Landslide Mapping Pipeline ---")
    print(f"Loaded config from: {config_path}")

    # 2. Extract configuration dictionary
    paths = config["paths"]
    hparams = config["hyperparameters"]

    # 3. Call training routine
    trained_model = train(
        img_dir=paths["img_dir"],
        mask_dir=paths["mask_dir"],
        num_epochs=hparams["epochs"],
        batch_size=hparams["batch_size"],
        lr=hparams["learning_rate"],
        save_path=paths["save_path"],
        graph_type=hparams["graph_type"],
        resume=hparams["resume"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landslide Detection Training Entrypoint")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml", 
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    main(args.config)