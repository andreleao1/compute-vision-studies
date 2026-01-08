import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import Counter # Útil se houverem várias classes no mesmo arquivo

# --- A CLASSE CUSTOMIZADA REVISADA ---
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # Lista todos os nomes de arquivo de imagem (assumindo que o nome do label corresponde)
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 1. Carrega a Imagem (converte para Grayscale)
        # O 'L' é para carregar a imagem em 8-bit pixels, black and white
        image = Image.open(img_path).convert('L') 

        # 2. Carrega e Processa a Label (YOLO Format)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        class_ids = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        # O primeiro elemento é o ID da classe
                        class_ids.append(int(parts[0]))
        except FileNotFoundError:
             # Se o arquivo de label não existe, podemos assumir uma classe 'background' ou pular
             return None, None 

        # DECISÃO: Simplificar para uma ÚNICA Label de Imagem (Classificação)
        if not class_ids:
            # Imagem vazia (sem detecções) - Retorna uma classe específica para background (e.g., 99 ou -1)
            # Para simplificar, vou retornar uma classe de background (você deve definir qual é)
            label_data = torch.tensor(99, dtype=torch.long) # Exemplo: 99 para "background"
        else:
            # Retorna a classe mais frequente (mais robusto)
            # Se você souber que todos os objetos são da mesma classe, basta class_ids[0]
            most_common_class = Counter(class_ids).most_common(1)[0][0]
            label_data = torch.tensor(most_common_class, dtype=torch.long)
        
        # 3. Aplica Transformações
        if self.transform:
            image = self.transform(image)
        
        # Retorna a imagem transformada e a label (agora um tensor único para a classificação)
        return image, label_data

# --- Função de Collate para lidar com amostras "None" (opcional, mas recomendado) ---
# O DataLoader não consegue empacotar tuplas que contêm None se o label_data for None
def collate_fn(batch):
    # Filtra amostras "corrompidas" ou aquelas que retornaram (None, None)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)