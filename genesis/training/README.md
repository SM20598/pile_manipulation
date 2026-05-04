# CNN Training Script

This script trains a Convolutional Neural Network (CNN) for image classification using PyTorch, incorporating additional parameters from configuration files.

## Prerequisites

- Python 3.x
- PyTorch and torchvision (already in requirements.txt)
- PyYAML (for loading config files)

## Data Setup

The script expects image data organized in subdirectories under `../data/cubes/`, with each class folder containing images and a `0_config.yaml` file with parameters.

Example structure:

```
data/cubes/
├── chickpeas_on_glass/
│   ├── 0_config.yaml
│   ├── image1.jpg
│   └── ...
└── chickpeas_on_wood/
    ├── 0_config.yaml
    └── ...
```

The config file should contain parameters like plate size and friction, which are extracted and fed into the network.

## Usage

Run the script with:

```bash
python cnn.py
```

The script will train the model for 10 epochs by default and print training loss and validation accuracy.

## Model Architecture

The model combines:
- CNN layers for image processing
- FC layers for parameter processing
- Concatenation of features before final classification

## Customization

- Adjust `param_dim` in the model and dataset for different numbers of parameters.
- Modify parameter extraction in `ImageParamDataset` to suit your config format.
- Change data paths or transformations as needed.

## Troubleshooting

- If data loading fails, ensure images are present in the class folders and configs exist.
- For GPU training, add `model.to('cuda')` and move tensors to GPU.
- If parameters are not loading correctly, check the YAML structure and extraction logic.