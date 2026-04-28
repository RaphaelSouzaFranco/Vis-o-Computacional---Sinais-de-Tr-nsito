"""
Módulo de treinamento.
Gerencia o loop de épocas (Forward, Backward, Optimization) e prevê o log histórico, permitindo
identificar Overfitting usando a técnica de monitoramento na base de validação.
"""

import copy
import os
import time
import torch
from tqdm import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, save_path="best_model.pth"):
    """
    Executa o treinamento e validação da CNN, salvando o melhor estado alcançado em disco.
    
    Args:
        model (nn.Module): Modelo PyTorch instanciado que passará pelos cálculos reversos.
        train_loader (DataLoader): Loader da base de treinamento contendo augmentations.
        val_loader (DataLoader): Loader da base de validação/teste pura, sem distorções no dado original.
        criterion: Função de convergência (Sugerido: CrossEntropyLoss do PyTorch).
        optimizer: Modulador de passos (Sugerido: Adam).
        device (torch.device): Local do treinamento ('cpu' ou alocação Cuda).
        num_epochs (int): Quantidade de varreduras gerais nos conjutos completos.
        save_path (str): Arquivo para cache/salvamento do modelo. Ex: 'gtsrb_res_best.pth'
        
    Returns:
        tuple: (model (carregado com a melhor epoch), history (dict de arrays para plotar tabelas))
    """
    since = time.time()
    
    # Criamos dicionários/history dict para manter as curvas a fim de expor na Análise de Métricas do final.
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # Faz backup virtual para caso quebre, aponte para um default seguro
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iteração Macro pelas Épocas
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch + 1}/{num_epochs}")
        print("-" * 15)

        # Iteração interna obrigatória garantindo cálculo dos gradients (Treino) e o repouso deles (Validação)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Liga o modo treinando (vital para as flags de Dropout e Batch Norm reagirem)
                dataloader = train_loader
            else:
                model.eval()  # Liga o modo avaliando (silencia os dinamistas do Dropout/Batch Norm)
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterar nas partições por lotes (batch) contendo a UI visual da flag TQDM
            loop = tqdm(dataloader, total=len(dataloader), desc=phase.capitalize(), leave=True)
            
            for inputs, labels in loop:
                # Assegura a locação para o destino final (CPU/GPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Resseta sumário de gradientes da backpropagation residual (impede intersecções)
                optimizer.zero_grad()

                # Processa fluxo Forward. Setamos Gradien apenas quando em Treino
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Pega o argmax probabilístico
                    loss = criterion(outputs, labels)

                    # Backward step + Optimizer só podem ser habilitados se e apenas se no modo de aprendizagem.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Contabilidade de penalizações
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Feedback live in console UI (Tqdm suffix)
                loop.set_postfix(loss=loss.item())

            # Tabula cálculo macro contábil do Batch Geral daquela época
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"| Resumo > {phase.capitalize()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} |")
            
            # Ancoramos resultado no history array
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # Verificação profunda para capturar a Melhor Epoch ("Early Checkpoint")
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # Cria o diretório de destino se ele não existir
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        
                    torch.save(model.state_dict(), save_path)
                    print(f"[*] NOVO RECORDE de Acurácia Capturado ({best_acc:.4f})! Salvo no HD em '{save_path}'.")

    time_elapsed = time.time() - since
    print(f"\nTreinamento Finalizado. Tempo Total: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Melhor Acurácia de Validação observada no Ciclo: {best_acc:.4f}")

    # Restaura explicitamente o ambiente do Modelo à Época ideal. (Inibe efeitos prejudiciais da Ultima Epoca ruim).
    model.load_state_dict(best_model_wts)
    return model, history
