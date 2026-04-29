"""
Módulo de manipulação de dados para o dataset GTSRB.
Este script carrega dados da biblioteca Hugging Face Datasets e os converte em
Dataloaders do PyTorch, aplicando as etapas de pré-processamento (redimensionamento, augmentation e normalização).
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import PIL.Image

HF_DATASET_ID = "tanganke/gtsrb"
IMAGE_SIZE = 224  # Tamanho esperado pelo MobileNetV2 / ResNet
BATCH_SIZE = 32


class GTSRBDataset(Dataset):
    """
    Classe Wrapper que converte as amostras do Hugging Face Dataset para
    tensores consumíveis pelo PyTorch, lidando com conversão RGB e transformações.
    """

    def __init__(self, hf_dataset, transform=None):
        """
        Inicializa o wrapper do dataset.
        
        Args:
            hf_dataset: Objeto Dataset do Hugging Face.
            transform: Transformações do torchvision para aplicar.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        """Retorna o número fixo de instâncias do dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """Busca o elemento da vez e aplica conversões e augmentations."""
        item = self.hf_dataset[idx]
        image = item["image"]

        # Garante o formato RGB, prevenindo falha de tensores de 1 canal (Greyscale)
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
            
        image = image.convert("RGB")
        
        # Hugging Face pode definir a flag de classe como 'label' ou 'id'. 
        # A forma abaixo garante fallback resiliente.
        label = item.get("label", item.get("id"))
        if label is None:
            raise KeyError("A chave 'label' ou 'id' não foi encontrada no dataset HF.")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(batch_size=BATCH_SIZE, num_workers=2, use_augmentations=True):
    """
    Realiza o download de `tanganke/gtsrb`, aplica split caso não haja divisão feita e gera Dataloaders.
    
    Args:
        batch_size (int): Tamanho do lote.
        num_workers (int): Número de processos paralelos para carregar imagens na CPU.
        use_augmentations (bool): Se verdadeiro, aplica rotação e color jitter.
        
    Returns:
        tuple: Um par (train_loader, test_loader)
    """
    print(f"[*] Carregando o dataset do Hugging Face: '{HF_DATASET_ID}'")
    dataset = load_dataset(HF_DATASET_ID)

    # A normalização utiliza a média e desvio padrão estabelecido do modelo original ImageNet.
    # Aplicaremos transfer learning (MobileNetV2), então isso é mandatório para pesos pre-trained.
    if use_augmentations:
        train_transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(15),  # Data Augmentation simples de rotação
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Invariância de iluminação 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_transforms = {
        'train': train_transforms,
        'test': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    if 'validation' in dataset:
        print("[*] Encontrado dataset de Validação/Teste pré-divido.")
        train_hf = dataset['train']
        test_hf = dataset['validation']
    elif 'test' in dataset:
        print("[*] Encontrado dataset de Teste pré-divido.")
        train_hf = dataset['train']
        test_hf = dataset['test']
    else:
        print("[*] Apenas 'train' encontrado, executando split manual (80/20)...")
        split = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_hf = split['train']
        test_hf = split['test']

    # Instanciamos o custom helper de PyTorch Datasets
    train_dataset_full = GTSRBDataset(train_hf, transform=data_transforms['train'])
    test_dataset_full = GTSRBDataset(test_hf, transform=data_transforms['test'])

    # Para garantir a execução rápida (redução solicitada), usaremos apenas 5% dos dados
    from torch.utils.data import Subset
    import random
    
    train_indices = random.sample(range(len(train_dataset_full)), int(0.05 * len(train_dataset_full)))
    test_indices = random.sample(range(len(test_dataset_full)), int(0.05 * len(test_dataset_full)))
    
    train_dataset = Subset(train_dataset_full, train_indices)
    test_dataset = Subset(test_dataset_full, test_indices)

    print(f"[+] Conjunto de Treino (5% Amostra Rápida): {len(train_dataset)} instâncias.")
    print(f"[+] Conjunto de Teste (5% Amostra Rápida): {len(test_dataset)} instâncias.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Otimiza a latência da cópia do tensor de página (Host) -> GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
