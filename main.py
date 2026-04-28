"""
Script Principal do Projeto (Entrypoint).
Orquestra o carregamento de dados, a instanciação da arquitetura, 
o loop de treinamento e o relatório final de avaliação.
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Import dos módulos baseados na arquitetura limpa (PEP 8)
from src.data_setup import create_dataloaders
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model

def get_args():
    """
    Constrói a interface CLI com utilitários para trocar os hiperparâmetros.
    """
    parser = argparse.ArgumentParser(description="Treino de CNN em Sinais de Trânsito GTSRB")
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas de treinamento completas (default: 10).')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamanho do lote que transita pela GPU/CPU (default: 32).')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Taxa de aprendizagem inicial aplicada ao Adam (default: 1e-3).')
    parser.add_argument('--num-workers', type=int, default=2, help='Número de subprocessos em prontidão para os DataLoaders (default: 2).')
    return parser.parse_args()


def main():
    args = get_args()
    
    print("=" * 60)
    print("Projeto GTSRB - Reconhecimento de Sinais de Trânsito")
    print("=" * 60)
    print(f"[ Configuração ] Épocas={args.epochs} | Batch={args.batch_size} | LR={args.learning_rate} | Workers={args.num_workers}")

    # 1. Definição Inteligente do Hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Acelerador alocado para os Tensores: '{device}'")

    # 2. Carregamento de Dados (Hugging Face -> PyTorch DataLoaders)
    print("\n>>> ETAPA 1/4: Preparação dos Pipeline de Dados")
    try:
        train_loader, test_loader = create_dataloaders(
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
    except Exception as e:
        print(f"\n[!] Falha fatal capturada durante o setup da Hugging Face API: {e}")
        sys.exit(1)

    # 3. Construção do Modelo via Transfer Learning (Congelado)
    print("\n>>> ETAPA 2/4: Instanciando Arquitetura Base")
    model = get_model(num_classes=43, freeze_features=True)
    model = model.to(device)

    # 4. Definição do Critério de Perda Penal e Algoritmo Otimizador 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate
    )

    # 5. Pipeline Extensivo de Treinamento
    print("\n>>> ETAPA 3/4: Loop de Treinamento e Otimização")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader, 
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        save_path="models/gtsrb_mobilenet_best.pth"
    )

    # 6. Avaliação Científica Final
    print("\n>>> ETAPA 4/4: Certificação de Métricas")
    evaluate_model(model, test_loader, device=device)
    
    print("\n" + "=" * 60)
    print("Processo Concluído Transversalmente com Sucesso.")
    print("O classificador treinado foi persistido na raiz do pacote.")
    print("=" * 60)


if __name__ == "__main__":
    main()
