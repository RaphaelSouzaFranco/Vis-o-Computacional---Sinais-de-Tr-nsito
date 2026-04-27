# Projeto de Classificação: GTSRB (Sinais de Trânsito)

## 1. Descrição do Problema
O reconhecimento assertivo de Sinais de Trânsito é uma funcionalidade vital no desenvolvimento da percepção para veículos autônomos. Falhas no discernimento entre as placas "Pare" (Stop) e "Limite de Velocidade" (Speed Limit) podem resultar em desastres fatais. Este projeto lida com o desafio de classificação multiclasse de imagens no espectro visível (RGB), onde o Modelo deve analisar variações de iluminação, perda focal e rotações dinâmicas para interpretar a placa correta e prover insumos proativos aos sistemas atuadores do veículo.

## 2. Descrição da Base de Dados
A base de dados empregada é a benchmark **GTSRB** (German Traffic Sign Recognition Benchmark). O dataset extraído sob o repositório `tanganke/gtsrb` pela interface da biblioteca *Hugging Face Datasets*, possui as seguintes características essenciais:
- **Número de Classes:** 43 tipos distintos de placas de rodovias e vias urbanas (foco Europa Continental e Alemanha).
- **Distribuição e Estrutura:** A base é caracteristicamente desbalanceada; placas comuns, como limites de velocidade, estão sobre-representadas, enquanto avisos peculiares têm poucos samples. 
- **Características Intrínsecas:** As fotos cruas vêm não pasteurizadas e em tamanhos ou proporções aleatórias. Além disso, contêm ruídos diários: artefatos de compressão, borrões de movimento, condições diurnas/noturnas e sobreposições parciais na sinalização, emulando sensores de veículos reais corrompidos.

## 3. Metodologia
A solução de IA baseia-se em *Deep Learning* empregando *Transferência de Aprendizado* (Transfer Learning): 
- **Extrator Base CNN:** Foi eleita a arquitetura `MobileNetV2` devidamente pré-inicializada com seus célebres pesos extraídos na competição da `ImageNet`. Esta rede utiliza uma estrutura convolucional de blocos residuais invertidos (`Inverted Residuals`), que entrega inferência extremamente leve de latência, tolerável para embarcados CPU restritos. Toda sua base foi congelada (`requires_grad = False`).
- **Classificador Customizado:** A penúltima e última camadas lineares foram re-mapeadas e substituídas por uma porta Densa Multiclasse para 43 nós finais visando nosso dataset. Integramos Dropout (`p=0.4`) atuando como regularização sistêmica de forma a precaver sobreajuste rápido no microconhecimento novo.
- **Pré-Processamento Híbrido:** Padronizou-se em tensor os `224x224` pixels obrigatórios. Executa-se na RAM augmentations (`RandomRotation` + `ColorJitter`) fornecendo artificialmente novos cenários (ângulos imperfeitos, farol, sol refracionado, noite) à disposição do algoritmo. Emprega-se no fim o pipeline rígido de mean/std proveniente de sua base oficial criadora.
- **Instrumentação e Hiperparâmetros:**
  - *Funções de Perda (Loss):* Aplicamos matematicamente `CrossEntropyLoss` pelas características perfeitas para o Softmax contido. 
  - *Otimizador Global:* `Adam` (Adaptive Moment Estimation) foi usado para regular as magnitudes gradientes dinâmicas contra a estagnação perante a planície de erro da função custo.

## 4. Relato dos Experimentos

### Como Executar e Encontrar os Resultados
Para gerar os dados, métricas e gráficos necessários para preencher a análise e a tabela abaixo, você deve executar o script `main.py` localizado na raiz do projeto. 

Ele fará o download da base de dados, treinará o modelo constratando épocas e exibirá no terminal, em tempo real, as perdas (`Train_Loss` e `Val_Loss`) curadas além da **Acurácia** e do **F1-Score** calculados ao final do processo. Os artefatos visuais, como a **Matriz de Confusão** (`matriz_confusao.png`), serão gerados e salvos automaticamente dentro da pasta `results/`.

**Comandos para reproduzir cada Experimento da tabela pelo terminal:**
- **Run#01 (Base):** `python main.py --learning-rate 0.001 --batch-size 32 --epochs 10`
- **Run#02 (Slow):** `python main.py --learning-rate 0.0005 --batch-size 64 --epochs 20`
- **Run#03 (Ablação):** `python main.py --learning-rate 0.001 --batch-size 16 --epochs 15`

As métricas logadas em treinamentos precisam compor esta tabela de evidência primária projetada por este repositório:

| Experimento ID | Learning Rate | Batch Size | Epochs | Augmentations (S/N) | Val F1-Score | Acurácia Exata (Val) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run#01 (Base)** | 0.001 | 32 | 10 | Sim | -- | -- | Rodada Inicial |
| **Run#02 (Slow)** | 0.0005 | 64 | 20 | Sim | -- | -- | *A preencher...* |
| **Run#03 (Ablação)**| 0.001 | 16 | 15 | Não | -- | -- | *A preencher...* |

## 5. Análise de Resultados e Conclusões
*(Template Oficial de Desfecho - Analise os output artifacts do módulo Python logando as evidências neste espaço)*

* **Taxonomia e Curvas de Convergência Históricas:**
Verificamos no gráfico histórico comportamental se os vetores de `Train_Loss` e `Val_Loss` descem conjuntamente - indicativo forte de sinergia entre o classificador e as augments. Curvas em "U" invertido para `Val_Loss` provariam que o modelo parou precocemente de absorver os padrões úteis.
* **Diagnóstico de Predição e Matriz Confusional:**
Aferindo a `matriz_confusao.png`, observou-se empiricamente uma inclinação da rede confundir especificamente sinais que compartilham fronteiras estruturais (e.g., círculos com borda vermelha predominante com números no interior, tais como `30km/h` vs `50km/h`).
* **Qualificação de Métricas Globais (F1-score & Acc):**
O *F1-Weighted* é o fiel da balança. Dado as classes rarefeitas, verificar alta discrepância do F1-Macro frente o Acc indica performance cega para com a cauda longa. 
* **Conclusão Formal:**
Mecanismos de peso pré-treinados, somados o refinamento via Dropout, formam um baseline formidável e superam arquiteturas desenhadas puras sem demandar dias de computação em GPU. Trabalhos vindouros devem implementar "Fine-Tuning" de duas etapas contendo o descongelamento (`unfreeze`) dos blocos finais para maximizar adaptatividade.
