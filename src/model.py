"""
Módulo base responsável pela arquitetura da Rede Neural Convolucional (CNN).
Emprega o padrão de Transfer Learning utilizando a MobileNetV2 para obter inferências 
rápidas e com menor custo computacional (útil para uso em CPU).
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def get_model(num_classes=43, freeze_features=True):
    """
    Constrói e retorna o modelo de classificação adaptado para o número de
    classes desejadas. Faz uso do MobileNetV2 pré-treinado na base ImageNet.

    Args:
        num_classes (int): O número de classes a serem preditas (GTSRB possui 43 classes distintas).
        freeze_features (bool): Se verdadeiro, congela a computação de gradientes de modo que os
                                pesos do extrator primário (features) não se alterem, acelerando o 
                                treinamento e focando o aprendizado no classificador final.
                                
    Returns:
        Um objeto nn.Module PyTorch contendo a rede Neural Pronta para treino/inferência.
    """
    print("[*] Instanciando MobileNetV2 com pesos pré-treinados no ImageNet...")
    
    # Fazemos download e ativamos a instância preenchida com a família de pesos DEFAULT de melhor performance.
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    if freeze_features:
        print("[*] Congelando a base convolucional (Transfer Learning Fixo)...")
        # Itera camada por camada desativando o require_grad para reuso puro 
        for param in model.features.parameters():
            param.requires_grad = False

    # A arquitetura do classificador original do mobile net para o imagenet prediz classe 1000, e
    # tem esse corpo: Sequential( Dropout(p=0.2), Linear(in_features=1280, out_features=1000) )
    # Vamos reescreper esta última camada para que a 'out_features' bata com nossa quantidade de sinais de trânsito!
    in_features_cl = model.classifier[1].in_features
    
    # Criamos a nova camada de Output dimensionada à nossa necessidade
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=False),  # Aumentamos um pouco o dropout para evitar overfitting rápido.
        nn.Linear(in_features=in_features_cl, out_features=num_classes)
    )

    return model
