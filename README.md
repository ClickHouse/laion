## Pre-requisites

- Conda e.g. miniconda
- Python >= 3.10.10

## Installation

```bash
    conda install --yes -c pytorch torchvision
    pip install git+https://github.com/openai/CLIP.git
    pip install -r requirements.txt

```

For machines with a nvidia GPU:

```bash
conda install --yes -c cudatoolkit=11.0
```