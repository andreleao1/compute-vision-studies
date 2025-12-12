import os

def get_model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def count_parameters(model):
    """Conta o número total de parâmetros e parâmetros não-zero"""
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    return total_params, nonzero_params

def get_model_memory_size(model):
    total_size = 0
    for param in model.parameters():
        # Calcula bytes: número de elementos * bytes por elemento
        total_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    size_mb = total_size / (1024 * 1024)
    return size_mb