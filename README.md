# Computer Vision Studies

This repository contains the experimental code developed for my Master's research at **UFRPE (Universidade Federal Rural de Pernambuco)**.
The goal of this work is to **compare several neural network architectures focused on computer vision for embedded devices**, evaluating their behavior under different **model compression techniques**.

The experiments include pruning, quantization, distillation, and other strategies designed to reduce computational cost while maintaining accuracy as much as possible.
All tests, measurements, and analysis scripts used throughout the research are organized in this repository.

---

## 🧪 Research Status

Below is the current progress of the evaluated architectures and the compression techniques applied:

### **YOLO**

| Compression Technique  | Status |
| ---------------------- | ------ |
| Structural Pruning     | ✔️     |
| Unstructured Pruning   | ✔️     |
| Quantization           | ❌      |
| Knowledge Distillation | ❌      |

---

### **RetinaNet**

| Compression Technique | Status |
| --------------------- | ------ |
| All techniques        | ❌      |

---

### **Faster R-CNN**

| Compression Technique | Status |
| --------------------- | ------ |
| All techniques        | ❌      |

---

### **Spiking Neural Networks (SNN)**

| Compression Technique | Status |
| --------------------- | ------ |
| All techniques        | ❌      |

---

## 📂 Repository Structure

Currently, the only fully structured implementation is for **YOLO**.
The directory layout is as follows:

```
utils/
 └── metrics.py               # Script for evaluating memory usage, parameter count, etc.

yolo/
 ├── pruning/
 │    ├── structured_pruning/ # Structural pruning implementation
 │    └── unstructured_pruning.py
 ├── datasets/                # NOT committed: should contain images and annotations
 └── oxford_base_models/      # NOT committed: contains pretrained models
```

### ⚠️ Important Notice on Missing Files

The **datasets** and **pretrained models** are **not included** in this repository due to size limitations.

However, **all pretrained YOLO models used in the experiments are available via Google Drive**, and a download link will be provided in the experiment documentation.

For the **oxford_town dataset**, both the images and annotations can be downloaded from the link below:

📦 **Oxford Town – Images & Annotations**  
https://drive.google.com/file/d/1_4VnNFuru6qSex1QjDyZSgtEPQLV12yO/view?usp=sharing

The **trained YOLO models** derived from this dataset are also available through the following link:

🤖 **Trained Models (Oxford Town)**  
https://drive.google.com/file/d/1pnvq4RVjcC1VHXDUi4mbb9yxsidIvwP_/view?usp=sharing

---

## 🚀 Running the Experiments

Before executing any script, you must install the dependencies listed in:

```
requirements.txt
```

The project makes use of PyTorch with CUDA acceleration (when available).
Below are the installation commands for each operating system.

### **Windows**

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### **Linux**

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### **macOS**

> macOS does **not** support CUDA.
> Install the CPU version instead:

```bash
pip3 install torch torchvision
```

After installing PyTorch and the project dependencies, you can execute any experiment script inside the architecture folders (e.g., YOLO pruning scripts).

---

## 📘 YOLO

### 📄 Overview

The YOLO architecture is the first model being analyzed in this research due to its popularity in real-time object detection and suitability for embedded applications.
All compression techniques being studied are applied to YOLO first, serving as a baseline for how more complex architectures may behave under similar conditions.

### 📁 Folder Description

Inside the `yolo/` directory you will find:

* `pruning/structured_pruning/`
  Full implementation of structural pruning methods for YOLO.

* `pruning/unstructured_pruning.py`
  Script responsible for performing unstructured pruning.

* `datasets/` *(not committed)*
  Must contain the training/validation images and annotation files.

* `oxford_base_models/` *(not committed)*
  Directory expected to contain pretrained YOLO models used as baselines.

### ▶️ How to Run YOLO Scripts

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Install PyTorch (CUDA or CPU depending on your OS).
3. Download pretrained models from Google Drive and place them inside:

   ```
   yolo/oxford_base_models/
   ```
4. Prepare the dataset folder:

   ```
   yolo/datasets/
   ```
5. Run any pruning or evaluation script. Example:

   ```bash
   python -m yolo.pruning.unstructured_pruning
   ```

---

## 📫 Contact

For questions regarding the experiments, dataset structure, or pretrained models, feel free to reach out.

