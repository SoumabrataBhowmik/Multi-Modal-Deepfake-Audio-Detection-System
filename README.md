**AudioFusion-DF: High-Performance Deepfake Audio Detection**
**Dual-Branch Fusion of Audio Transformers and Spectrogram CNNs**

This project introduces AudioFusion-DF, a high-performance architecture designed to detect deepfake audio. By combining the temporal feature extraction of WavLM with the spatial "image-like" analysis of ResNet50 on dual-channel Mel-CQT spectrograms, the model effectively identifies synthetic speech across four distinct classes.

**Key Features**

Dual-Branch Fusion: Integrates a WavLM-Base-Plus transformer branch with a ResNet50 convolutional branch. 


Multi-Modal Inputs: Processes raw audio waveforms and dual-channel spectrograms (Mel-Spectrogram + Constant-Q Transform). 


Advanced Data Augmentation: Implements Gaussian noise, Polarity Inversion, Time Shifting, Band-pass Filtering, and SpecAugment to ensure high generalization. 


Performance: Achieved a peak Validation F1-Score of 0.9912.

**Model Architecture**

The architecture leverages a 2,816-dimensional fused feature vector to make final classification decisions.

Technical Specifications

**Audio Branch (WavLM):** Captures temporal dependencies and phonetic structures ($f_{audio}$). The last 4 transformer layers are fine-tuned. 

**Spectrogram Branch (ResNet50):** Analyzes dual-channel spectral textures ($f_{spec}$). The last 2 blocks of the ResNet are unfrozen for task-specific learning. 

**Fusion Head:** Concatenates features ($768_{WavLM} + 2048_{ResNet} = 2816_{Total}$) into a dense network: 2816 → 1024 → 512 → 4. 

**Dataset & Preprocessing**
The model was trained on the Deepfake Audio Dataset (Comsys Hackathon Task A).

Total Unique Files: 47,333. 

Classes: 4 (Multi-class classification). 

Precomputation: Features (Mel and CQT) are precomputed and stored as .pt files to optimize training speed. 

**Training Results**

The model utilized a Cosine Annealing Learning Rate scheduler and reached near-perfect validation metrics. 

| Epoch | Train Loss | Train F1-Score | Val Loss | Val F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.3559 | 0.4466 | 0.8827 | 0.7920 |
| 2 | 0.9650 | 0.7116 | 0.7972 | 0.8389 |
| 3 | 0.8395 | 0.8251 | 0.7365 | 0.9029 |
| 5 | 0.7344 | 0.9119 | 0.6619 | 0.9720 |
| 11 | 0.6682 | 0.9566 | 0.6265 | 0.9774 |
| 17 | 0.6373 | 0.9756 | 0.6266 | 0.9850 |
| 23 | 0.6225 | 0.9862 | 0.6103 | 0.9888 |
| **27** | **0.6162** | **0.9902** | **0.6064** | **0.9912** |

**Best Validation F1-Score: 0.9912**

**Installation & Usage**

Clone the Repository:

Bash
git clone https://github.com/SoumabrataBhowmik/Multi-Modal-Deepfake-Audio-Detection-System.git

cd Multi-Modal-Deepfake-Audio-Detection-System

Install Dependencies:

Bash
pip install torch torchvision torchaudio transformers librosa scikit-learn pandas tqdm

Run Training:

The project includes a run_training function that handles precomputation, data splitting, and the training loop automatically.

**Training Configuration**

Optimizer: AdamW (Learning Rate: $8 \times 10^{-5}$) 

Loss Function: CrossEntropy with Label Smoothing (0.1) 

Batch Size: 32 

Patience: 12 epochs (Early Stopping) 


# Deepfake Audio Detection 

This repository contains the training pipeline and model architecture for a 4-class Deepfake Audio Detection task. The system utilizes a dual-branch fusion architecture that simultaneously processes raw waveforms and spectrograms to capture both temporal acoustic features and frequency-domain representations.

## System Architecture

The model combines a pre-trained Audio Transformer (`WavLM-base-plus`) and a Convolutional Neural Network (`ResNet-50`) to extract and fuse multimodal audio features.

```mermaid
flowchart TD
    %% Node Class Definitions 
    classDef inputNode fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#0d47a1,rx:8px,ry:8px
    classDef processNode fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20,rx:8px,ry:8px
    classDef modelNode fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#b71c1c,rx:8px,ry:8px
    classDef embedNode fill:#fff8e1,stroke:#fbc02d,stroke-width:2px,color:#f57f17,rx:8px,ry:8px
    classDef matchNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c,rx:8px,ry:8px
    classDef outputNode fill:#e0f7fa,stroke:#0097a7,stroke-width:2px,color:#006064,rx:8px,ry:8px
    
    %% Global Edge styling
    linkStyle default stroke:#78909c,stroke-width:2px

    %% Input Data
    InputData[/"Input Audio Data<br/><b>WAV files</b>"/]:::inputNode

    %% Preprocessing Subgraph
    subgraph Preprocessing ["Data Preprocessing & Augmentation"]
        direction TB
        
        WavPrep("<b>Raw Waveform Pipeline</b><br/>• Resample to 16kHz<br/>• Pad/Truncate (Max 5s)<br/>• Augmentations (Train): Noise, Gain,<br/>Polarity, Time Shift, BandPass, Clipping"):::processNode
        
        SpecPrep("<b>Spectrogram Pipeline</b><br/>• Channel 1: Mel Spectrogram (128 mels)<br/>• Channel 2: CQT (84 bins)<br/>• Augmentation (Train): SpecAugment<br/>(Time & Freq Masking)"):::processNode
    end

    InputData ==> WavPrep
    InputData ==> SpecPrep

    %% Architecture Subgraph
    subgraph Architecture ["Dual-Branch Feature Extraction"]
        direction TB
        
        WavBranch{{"<b>AudioTransformerBranch</b><br/>• Base: microsoft/wavlm-base-plus<br/>• Frozen except last 4 layers<br/>• Mean Pooling (masking padding)<br/>• Output: 768-Dim Vector"}}:::modelNode
        
        SpecBranch{{"<b>SpectrogramNet</b><br/>• Base: ResNet-50 (Pretrained)<br/>• Mod: conv1 (in_channels=2)<br/>• Frozen except last 2 blocks<br/>• AdaptiveAvgPool2d<br/>• Output: 2048-Dim Vector"}}:::modelNode
        
        WavBranchOut[/"WavLM Features (768)"/]:::embedNode
        SpecBranchOut[/"ResNet Features (2048)"/]:::embedNode
        
        WavBranch --> WavBranchOut
        SpecBranch --> SpecBranchOut
    end

    WavPrep ==>|1D Audio Tensor| WavBranch
    SpecPrep ==>|2-Channel Tensor| SpecBranch

    %% Head Subgraph
    subgraph Head ["Fusion & Classification Head"]
        direction TB
        
        Concat{"<b>Concatenation</b><br/>(2816-Dimensional)"}:::matchNode
        
        Dense1("Linear (2816 → 1024)<br/>ReLU + BatchNorm1d + Dropout(0.5)"):::modelNode
        Dense2("Linear (1024 → 512)<br/>ReLU + BatchNorm1d + Dropout(0.5)"):::modelNode
        Dense3("Linear (512 → 4)<br/>Logits"):::modelNode
        
        Concat ==> Dense1 ==> Dense2 ==> Dense3
    end

    WavBranchOut ==> Concat
    SpecBranchOut ==> Concat

    %% Outputs
    LossFunction("<b>Training Configuration</b><br/>• CrossEntropyLoss (Label Smoothing 0.1)<br/>• Optimizer: AdamW (wd=0.01)<br/>• Scheduler: Cosine Warmup<br/>• AMP Enabled"):::processNode
    
    OutputMetrics("<b>Final Prediction (4 Classes)</b><br/>Evaluated via Macro F1-Score"):::outputNode

    Dense3 ==> LossFunction
    LossFunction ==> OutputMetrics
