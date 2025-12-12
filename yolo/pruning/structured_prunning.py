from ultralytics import YOLO
import torch.nn as nn
import torch.nn.utils.prune as prune
from utils import metrics
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
oxford_town_base_model = os.path.join(root, 'yolo', 'oxford_base_models', 'trained_by_yolo11x.pt')

# Verifica se o arquivo do modelo existe antes de tentar usá-lo
if not os.path.exists(oxford_town_base_model):
    print(f"Modelo não encontrado em: {oxford_town_base_model}")
    print("Coloque o arquivo do modelo na pasta 'oxford_base_models' ou ajuste o caminho no script.")
    raise SystemExit(1)

print("=" * 60)
print("TAMANHO DO MODELO ORIGINAL")
print("=" * 60)
original_size = metrics.get_model_size(oxford_town_base_model)
print(f"Tamanho do arquivo em disco: {original_size:.2f} MB")

model = YOLO(oxford_town_base_model)
model_nn = model.model

# Calcula tamanho do modelo original em memória
memory_size_orig = metrics.get_model_memory_size(model_nn)
print(f"Tamanho em memória (RAM): {memory_size_orig:.2f} MB")

# Conta parâmetros originais
total_params_orig, nonzero_params_orig = metrics.count_parameters(model_nn)
print(f"Total de parâmetros: {total_params_orig:,}")
print(f"Parâmetros não-zero: {nonzero_params_orig:,}")
print(f"Densidade: {(nonzero_params_orig/total_params_orig)*100:.2f}%")
print()

conv_layers = [(name, module) for name, module in model_nn.named_modules() if isinstance(module, nn.Conv2d)]

print("=" * 60)
print("APLICANDO PRUNING ESTRUTURAL")
print("=" * 60)
print(f"Total de camadas Conv2d encontradas: {len(conv_layers)}")

# Pulando as duas primeiras e duas últimas camadas Conv2d foi obtido melhores resultados no mAP
skip_first = 2
skip_last = 2

for idx, (name, module) in enumerate(conv_layers):
    if idx < skip_first or idx >= len(conv_layers) - skip_last:
        print(f"Pulando camada {name}")
        continue
    print(f"Aplicando pruning estrutural em {name}")
    
    prune.ln_structured(
        module,                 # camada alvo
        name='weight',          # parâmetro a ser podado
        amount=0.3,             # percentual de filtros a remover
        n=2,                    # norma L2
        dim=0                   # 0 = remove filtros inteiros (saídas da conv)
    )

# Removendo os reparametrizadores para consolidar o pruning
for name, module in model_nn.named_modules():
    if isinstance(module, nn.Conv2d) and hasattr(module, "weight_orig"):
        prune.remove(module, 'weight')

# Verifica tamanho após pruning (antes de salvar)
print("=" * 60)
print("ESTATÍSTICAS APÓS PRUNING ESTRUTURAL (EM MEMÓRIA)")
print("=" * 60)

# Calcula tamanho em memória após pruning
memory_size_pruned = metrics.get_model_memory_size(model_nn)
print(f"Tamanho em memória (RAM): {memory_size_pruned:.2f} MB")
print(f"Tamanho original em memória: {memory_size_orig:.2f} MB")
print(f"Diferença em memória: {memory_size_orig - memory_size_pruned:.2f} MB ({((memory_size_orig - memory_size_pruned)/memory_size_orig)*100:.2f}%)")
print()

total_params_pruned, nonzero_params_pruned = metrics.count_parameters(model_nn)
print(f"Total de parâmetros: {total_params_pruned:,}")
print(f"Parâmetros não-zero: {nonzero_params_pruned:,}")
print(f"Densidade: {(nonzero_params_pruned/total_params_pruned)*100:.2f}%")
print(f"Parâmetros removidos: {total_params_orig - nonzero_params_pruned:,}")
print(f"Redução de parâmetros: {((total_params_orig - nonzero_params_pruned)/total_params_orig)*100:.2f}%")
print()

output_path = 'test_structured.pt'
model.save(output_path)

print("=" * 60)
print("TAMANHO DO MODELO SALVO EM DISCO")
print("=" * 60)
final_size = metrics.get_model_size(output_path)
print(f"Tamanho do arquivo: {final_size:.2f} MB")
print(f"Tamanho original em disco: {original_size:.2f} MB")
print(f"Diferença em disco: {original_size - final_size:.2f} MB ({((original_size - final_size)/original_size)*100:.2f}%)")
print()

print(f"Modelo salvo com pruning estrutural aplicado em '{output_path}'")
print("=" * 60)