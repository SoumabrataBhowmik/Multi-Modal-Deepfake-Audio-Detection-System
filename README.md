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
