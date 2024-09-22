import torch
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

if __name__ == "__main__":
    logging.info("Initializing control layer files")
    control_layer_filenames = sys.argv[1]
    control_layer_filenames = control_layer_filenames.split(",")

    experiment_control_layers = [
        [[i for i in range(54)]]
        ]

    assert len(experiment_control_layers) == len(control_layer_filenames)
    for i, control_layers in enumerate(experiment_control_layers):
        logging.info(f"Initializing control layer file: {control_layer_filenames[i]} with layers {control_layers}", )
        torch.save(torch.tensor(control_layers), control_layer_filenames[i])
