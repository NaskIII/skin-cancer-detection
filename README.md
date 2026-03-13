# Skin Cancer Detection 🔬

Algoritmos de Inteligência Artificial treinados para detectar câncer de pele do tipo melanoma. Uma abordagem que combina **visão computacional** e **aprendizado de máquina** para classificação de lesões dermoscópicas como benignas ou malignas.

---

## 📋 Resumo

Utilizando bases de dados públicas para alimentar algoritmos de Inteligência Artificial, este trabalho apresenta uma comparação de performance entre três algoritmos para a detecção de melanoma:

- **RNA – RMSprop** (Rede Neural Artificial com otimizador RMSprop)
- **SVM** (Support Vector Machine)
- **KNN** (K-Nearest Neighbors)

Os algoritmos são treinados e validados em duas bases de dados públicas distintas (**ISIC** e **PH2**), com o objetivo de identificar qual abordagem obtém melhor desempenho na detecção precoce do melanoma.

📄 **[Link para o Artigo Completo](https://drive.google.com/file/d/1zJkmKsjavUNMa7W2AzWj623tpOOiTnhx/view?usp=sharing)**

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Finalidade |
|---|---|---|
| Python | 3.x | Linguagem principal |
| TensorFlow / Keras | 2.6.0 | Rede Neural Artificial (RNA) |
| scikit-learn | 0.24.2 | SVM e KNN |
| OpenCV | 4.5.3.56 | Processamento de imagens |
| NumPy | 1.19.5 | Computação numérica |
| Pandas | 1.3.2 | Manipulação de dados |
| Matplotlib | 3.4.3 | Visualizações e curvas ROC |
| Joblib | 1.0.1 | Serialização de modelos |
| Pillow | 8.3.1 | Manipulação de imagens |

---

## 📁 Estrutura do Projeto

```
skin-cancer-detection/
├── requirements.txt                  # Dependências Python
├── KNN/                              # Modelos KNN pré-treinados
│   ├── KNN_ISIC/
│   │   ├── Modelo_ISIC_KNN           # Modelo treinado na base ISIC
│   │   └── Curva ROC_ISIC.png        # Curva ROC (ISIC)
│   └── KNN_PH2/
│       ├── Modelo_PH2_KNN            # Modelo treinado na base PH2
│       └── Curva ROC_KNN.png         # Curva ROC (PH2)
├── RMSprop/                          # Modelos RNA pré-treinados
│   ├── Modelo_ISIC/
│   │   ├── model_ISIC.json           # Arquitetura do modelo (ISIC)
│   │   ├── weights_ISIC.h5           # Pesos do modelo (ISIC)
│   │   └── Curva ROC_RMSprop.png     # Curva ROC (ISIC)
│   └── Modelo_PH2/
│       ├── model_PH2.json            # Arquitetura do modelo (PH2)
│       ├── weights_PH2.h5            # Pesos do modelo (PH2)
│       └── Curva ROC_RMSprop.png     # Curva ROC (PH2)
├── SVM/                              # Modelos SVM pré-treinados
│   ├── Modelo_ISIC/
│   │   ├── Modelo_ISIC_SVM.joblib    # Modelo treinado na base ISIC
│   │   └── Curva ROC.png             # Curva ROC (ISIC)
│   └── Modelo_PH2/
│       ├── Modelo_PH2_SVM.joblib     # Modelo treinado na base PH2
│       └── Curva ROC_SVM.png         # Curva ROC (PH2)
└── src/                              # Código-fonte
    ├── main.py                       # Pipeline principal de extração de features
    ├── Teste.py                      # Script de teste para uma única imagem
    ├── Data/
    │   ├── Data_Extrator.py          # Leitura, escrita e normalização de dados
    │   ├── Dados_ASIC.csv            # Metadados brutos da base ISIC
    │   ├── Dados_PH2.txt             # Metadados brutos da base PH2
    │   ├── Dataset_ISIC_Normalizado.csv  # Dataset ISIC normalizado (pronto para ML)
    │   └── Dataset_PH2_Normalizado.csv   # Dataset PH2 normalizado (pronto para ML)
    ├── MPP/                          # Módulo de Pré-Processamento
    │   └── Pre_Processing_Module.py
    ├── MEC/                          # Módulo Extrator de Características
    │   └── Features_extractor.py
    ├── SVM/
    │   └── Support_Vector_Machine.py # Treinamento e avaliação do SVM
    ├── KNN/
    │   └── KNN.py                    # Treinamento e avaliação do KNN
    └── MLP/                          # Multi-Layer Perceptron (RNA)
        ├── Neural_Network.py         # Definição e métodos do modelo
        └── Train_Ann.py              # Script de treinamento da RNA
```

---

## 🗄️ Bases de Dados

### ISIC (International Skin Imaging Collaboration)
Base de dados internacional de imagens dermoscópicas. Contém imagens de lesões de pele com diagnósticos clínicos de benigno ou melanoma.

### PH2 Dataset
Base de dados de dermoscopia com imagens RGB de lesões, acompanhadas de informações clínicas como assimetria, rede de pigmento, pontos/glóbulos, estrias, áreas de regressão e véu azul-esbranquiçado.

**Rótulos utilizados:**
- `0` → Lesão benigna
- `1` → Melanoma (maligno)

---

## ⚙️ Pipeline de Processamento

O fluxo completo de funcionamento do sistema segue as etapas abaixo:

```
Imagem Dermoscópica
        │
        ▼
┌─────────────────────┐
│  Pré-Processamento  │  ← MPP (Pre_Processing_Module)
│  - Remoção de pelos │
│  - Escala de cinza  │
│  - Binarização      │
│  - Morfologia       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│Extração de Features │  ← MEC (Features_extractor)
│  9 características  │
│  por imagem         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Normalização       │  ← Min-Max (Data_Extrator)
│  Min-Max            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Classificação      │  ← RNA / SVM / KNN
│  Benigno/Maligno    │
└─────────────────────┘
```

---

## 🔍 Características Extraídas (Features)

Cada imagem gera um vetor com **9 características** baseadas nos critérios clínicos **ABCD** (Assimetria, Borda, Cor, Diâmetro):

| # | Característica | Descrição |
|---|---|---|
| 1 | Simetria Eixo X | Assimetria horizontal da lesão |
| 2 | Simetria Eixo Y | Assimetria vertical da lesão |
| 3 | Diâmetro | Raio do menor círculo que envolve a lesão |
| 4 | Média Vermelha (R) | Valor médio do canal vermelho na região da lesão |
| 5 | Variância Vermelha (R) | Variância do canal vermelho na região da lesão |
| 6 | Média Verde (G) | Valor médio do canal verde na região da lesão |
| 7 | Variância Verde (G) | Variância do canal verde na região da lesão |
| 8 | Média Azul (B) | Valor médio do canal azul na região da lesão |
| 9 | Variância Azul (B) | Variância do canal azul na região da lesão |

---

## 🤖 Modelos de Classificação

### 1. RNA – RMSprop (Rede Neural Artificial)

Rede neural multicamadas com otimizador RMSprop para classificação binária.

**Arquitetura:**
```
Entrada (9 features)
    ↓
Dense(12, relu)
    ↓
Dense(12, relu)
    ↓
Dense(12, relu)
    ↓
Dense(1, sigmoid)  →  Saída: probabilidade de malignidade
```

**Hiperparâmetros:**
- Loss: `binary_crossentropy`
- Optimizer: `rmsprop`
- Batch size: `10`
- Epochs: `200`

---

### 2. SVM (Support Vector Machine)

Classificador de margem máxima com kernel RBF para separação não-linear dos dados.

**Configuração:**
- Kernel: `RBF` (Radial Basis Function)
- C (regularização): `5.0`
- Tolerância: `1e-9`
- Probabilidade habilitada: `True`

---

### 3. KNN (K-Nearest Neighbors)

Classificador baseado em distância entre os vizinhos mais próximos no espaço de features.

**Configuração:**
- Número de vizinhos: `5`
- Métrica de distância: `Euclidiana` (padrão)

---

## 🚀 Como Usar

### 1. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2. Extração de Features (Pipeline Completo)

Execute o script principal para processar imagens e extrair as 9 características de cada lesão:

```bash
cd src
python main.py
```

> ⚠️ **Atenção:** Os caminhos das imagens estão configurados para o ambiente de desenvolvimento original. Ajuste as variáveis `pathSegmentation`, `pathImgBGR` e o caminho dos metadados em `main.py` para o seu sistema.

### 3. Treinamento dos Modelos

**Rede Neural (RNA/RMSprop):**
```bash
cd src/MLP
python Train_Ann.py
```

**SVM:**
```bash
cd src/SVM
python Support_Vector_Machine.py
```

**KNN:**
```bash
cd src/KNN
python KNN.py
```

### 4. Teste em uma Única Imagem

Para visualizar o pipeline de pré-processamento aplicado a uma imagem isolada:

```bash
cd src
python Teste.py
```

### 5. Utilizando os Modelos Pré-Treinados

Os modelos já treinados estão disponíveis nas pastas `RMSprop/`, `SVM/` e `KNN/`.

**RNA:**
```python
from src.MLP.Neural_Network import Neural_Network
ann = Neural_Network()
model = ann.load_model('RMSprop/Modelo_ISIC/model_ISIC.json',
                       'RMSprop/Modelo_ISIC/weights_ISIC.h5')
predictions = ann.make_predictions_ravel(model, X_test)
```

**SVM:**
```python
from joblib import load
clf = load('SVM/Modelo_ISIC/Modelo_ISIC_SVM.joblib')
predictions = clf.predict(X_test)
```

**KNN:**
```python
from joblib import load
knn = load('KNN/KNN_ISIC/Modelo_ISIC_KNN')
predictions = knn.predict(X_test)
```

---

## 📊 Avaliação dos Modelos

Cada modelo é avaliado com as seguintes métricas:

| Métrica | Descrição |
|---|---|
| **Acurácia** | Percentual de predições corretas |
| **Sensibilidade** | TP / (TP + FN) — capacidade de detectar melanoma |
| **Especificidade** | TN / (TN + FP) — capacidade de identificar lesões benignas |
| **Precisão** | TP / (TP + FP) |
| **Curva ROC / AUC** | Desempenho geral do classificador |

As curvas ROC geradas em cada treinamento estão disponíveis nas respectivas pastas dos modelos (`KNN/`, `SVM/`, `RMSprop/`).

---

## 📜 Licença

Este projeto está licenciado sob a **MIT License** — veja o arquivo [LICENSE](LICENSE) para mais detalhes.
