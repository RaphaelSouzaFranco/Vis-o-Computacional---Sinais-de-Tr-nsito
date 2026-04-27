"""
Módulo de Verificação Científica e Avaliação de Performance.
Possui utilitários acoplados a Scikit-Learn que recebem o modelo, produzem predições generalizadas
sobre o DataLoader de Validação inteiro, e destilam Acurácia, F1-Score e plotam a Matriz de Confusão.
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(model, test_loader, device):
    """
    Roda uma passagem completa (feed-forward analítico) sobre os dados de prova
    para tabular os acertos e extrair a Matriz de Confusão para análise de desvios.
    
    Args:
        model (nn.Module): O modelo pré-treinado finalista.
        test_loader (DataLoader): DataLoader da pipeline livre de bias.
        device (torch.device): 'cpu' ou 'cuda'
        
    Returns:
        dict: Métricas numéricas estruturadas de Acc e F1 Score.
    """
    
    model.eval()  # Congela dinâmicas estocásticas
    all_preds = []
    all_labels = []

    print("\n[*] Iniciando Inferência Total no conjunto de Teste...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Converter para vetores numpy clássicos para que a stack do Sklearn interprete sem erros.
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print("\n[+] -- RELATÓRIO DO EXPERIMENTO --")
    print(f"    -> Acurácia Global Exata     : {acc * 100:.2f}%")
    print(f"    -> F1-Score (Macro Média)    : {f1_macro:.4f}")
    print(f"    -> F1-Score (Ponderada)      : {f1_weighted:.4f}")

    # Plotando Matriz de Confusão 
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(16, 12)) # Matriz grande devido a 43 classes do sinal de transito
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.title('Matriz de Confusão - Classificação de Sinais de Trânsito GTSRB')
    plt.ylabel('Classe Verdadeira (Base do Hugging Face)')
    plt.xlabel('Classe Predita (Saída do Modelo CNN)')
    
    cm_path = "results/matriz_confusao.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"[*] Visualização Gráfica da Matriz exposta e salva em: {cm_path}")
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }
