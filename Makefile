# --------------------------------------
# Detect Operating System
# --------------------------------------
UNAME_S := $(shell uname -s)

# --------------------------------------
# Virtual Environment
# --------------------------------------
VENV_DIR = .venv
PYTHON = python3
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip

# --------------------------------------
# Project folders
# --------------------------------------
DATASET_DIR = yolo/datasets
MODELS_DIR = yolo/oxford_base_models
UTILS_DIR = utils

# --------------------------------------
# Google Drive files
# --------------------------------------
OXFORD_DATASET_ZIP = https://drive.google.com/uc?export=download&id=1_4VnNFuru6qSex1QjDyZSgtEPQLV12yO
OXFORD_MODELS_ZIP  = https://drive.google.com/uc?export=download&id=1pnvq4RVjcC1VHXDUi4mbb9yxsidIvwP_

# --------------------------------------
# Default target
# --------------------------------------
all: prepare venv install_torch install_requirements download_all
	@echo "Environment setup complete!"

# --------------------------------------
# Create folders
# --------------------------------------
prepare:
	@echo "Creating project directories..."
	mkdir -p $(DATASET_DIR)
	mkdir -p $(MODELS_DIR)
	mkdir -p $(UTILS_DIR)

# --------------------------------------
# Create Python Virtual Environment
# --------------------------------------
venv:
	@echo "Creating Python virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"

# --------------------------------------
# Install PyTorch depending on OS
# --------------------------------------
install_torch:
ifeq ($(UNAME_S), Linux)
	@echo "Installing PyTorch + CUDA 13.0 (Linux)..."
	$(VENV_PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu130
else ifeq ($(UNAME_S), Darwin)
	@echo "Installing CPU-only PyTorch (macOS)..."
	$(VENV_PIP) install torch torchvision
else
	@echo "Installing PyTorch + CUDA 13.0 (Windows assumed via Git Bash)..."
	$(VENV_PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu130
endif

# --------------------------------------
# Install Python dependencies
# --------------------------------------
install_requirements:
	@echo "Installing project dependencies into venv..."
	$(VENV_PIP) install -r requirements.txt

# --------------------------------------
# Download dataset + models
# --------------------------------------
download_all: download_dataset download_models

download_dataset:
	@echo "Downloading Oxford Town dataset..."
	cd $(DATASET_DIR) && curl -L -o oxford_town.zip "$(OXFORD_DATASET_ZIP)"
	cd $(DATASET_DIR) && unzip -o oxford_town.zip

download_models:
	@echo "Downloading pretrained models..."
	cd $(MODELS_DIR) && curl -L -o oxford_models.zip "$(OXFORD_MODELS_ZIP)"
	cd $(MODELS_DIR) && unzip -o oxford_models.zip

# --------------------------------------
# Cleanup
# --------------------------------------
clean:
	rm -rf $(DATASET_DIR)/*
	rm -rf $(MODELS_DIR)/*
	rm -rf $(VENV_DIR)
	@echo "Cleanup complete."
