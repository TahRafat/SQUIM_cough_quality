# Cough Audio Quality Assessment using SQUIM

This project explores adapting **Torchaudio’s SQUIM (Speech Quality and Intelligibility Metrics)** model to evaluate the quality of **cough audio recordings** from the Coswara dataset.

The goal is to **filter low-quality cough recordings** before using them in machine learning pipelines such as cough-based disease detection.


## Dataset

This project uses the Coswara cough dataset available on Hugging Face.

Dataset:
szzs1693/coswara-data

The dataset is automatically downloaded when running the notebook.


## Method

The workflow:

1. Load cough recordings from the dataset  
2. Convert audio to waveform format  
3. Resample audio to **16 kHz**  
4. Apply the pretrained **SQUIM Objective model**  
5. Extract quality metrics:
   - STOI
   - PESQ
   - SI-SDR  
6. Classify recordings as **good** or **bad** quality using predefined thresholds.

## Quality Thresholds

| Metric | Threshold |
|------|------|
| STOI | ≥ 0.8 |
| PESQ | ≥ 2.0 |
| SI-SDR | ≥ 10 |

Recordings meeting all thresholds are saved as **good-quality coughs**.


## Installation

```bash
pip install -r requirements.txt