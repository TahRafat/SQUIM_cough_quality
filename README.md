# Cough Audio Quality Assessment using SQUIM

This project adapts **Torchaudio’s SQUIM (Speech Quality and Intelligibility Metrics)** to evaluate and filter the quality of **cough audio recordings** from the Coswara dataset.

The goal is to remove low-quality cough recordings before using them in downstream machine learning pipelines such as cough-based disease detection.

## Dataset

- Source: `szzs1693/coswara-data` (Hugging Face)
- Used audio types:
  - `cough-heavy`
  - `cough-shallow`

Other audio types (e.g., breathing, speech) are excluded to focus only on cough signals.


## Thresholds

### Original (Speech-Based)
| Metric | Threshold |
|--------|----------|
| STOI   | ≥ 0.8 |
| PESQ   | ≥ 2.0 |
| SI-SDR | ≥ 10 |

### Adapted for Cough (Used in this Project)
| Metric | Threshold |
|--------|----------|
| STOI   | ≥ 0.6 |
| PESQ   | ≥ 1.15 |
| SI-SDR | ≥ -12 |

## Results (1000 Samples)
- Processed: 977  
- Good: 222  
- Bad: 755  

### Average Metrics
| Label | STOI | PESQ | SI-SDR |
|------|------:|------:|-------:|
| Bad  | 0.43 | 1.18 | -14.93 |
| Good | 0.76 | 1.41 | -1.97 |

These results demonstrate clear separation between high- and low-quality cough recordings.

## Output

- `cough_quality_results.csv` → per-sample metrics and labels  

The CSV includes:
- sample index  
- audio type  
- STOI  
- PESQ  
- SI-SDR  
- predicted label  

## Installation

```bash
pip install torch torchaudio soundfile datasets pandas