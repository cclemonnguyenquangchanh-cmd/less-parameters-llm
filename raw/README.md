# Raw Experiment Files

This directory contains the raw Jupyter notebooks for all experiments conducted in the "Saving 77% of the Parameters in Large Language Models" research project. Each notebook includes complete training code, model configurations, and experimental results.

## 📚 Experiment Overview

All experiments were conducted using:
- **Base Model**: Microsoft's [phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Training Dataset**: [LaMini-instruction](https://huggingface.co/datasets/MBZUAI/LaMini-instruction)
- **Hardware**: Single NVIDIA L4 GPU
- **Training Duration**: ~3 days per experiment

## 🔬 Main Experiments

### Baseline Experiments (phi-3)

#### JP47D54C - Baseline (2 Transformer Layers)
- **Notebook**: [JP47D54C_Baseline_2T.ipynb](JP47D54C_Baseline_2T.ipynb)
- **Model**: phi-3 with 2 transformer decoder layers
- **Parameters**: 227M (non-embedding)
- **Intermediate Dimensions**: 8192
- **Results**: Train Loss 1.08, Validation Loss 1.58
- **HuggingFace Model**: [schuler/experimental-JP47D54C](https://huggingface.co/schuler/experimental-JP47D54C)

### Optimized Experiments (kphi-3)

#### JP47D55C - Optimized 2-Layer (15% Parameters)
- **Notebook**: [JP47D55C_kphi3_2T.ipynb](JP47D55C_kphi3_2T.ipynb)
- **Model**: kphi-3 with 2 transformer decoder layers
- **Parameters**: **35M** (non-embedding) - **15% of baseline**
- **Intermediate Dimensions**: 9216
- **Results**: Train Loss 1.26, Validation Loss 1.60
- **HuggingFace Model**: [schuler/experimental-JP47D55C](https://huggingface.co/schuler/experimental-JP47D55C)
- **Key Achievement**: 85% parameter reduction with minimal performance impact

#### JP47D56C - Optimized 3-Layer (23% Parameters)
- **Notebook**: [JP47D56C_kphi3_3T.ipynb](JP47D56C_kphi3_3T.ipynb)
- **Model**: kphi-3 with 3 transformer decoder layers
- **Parameters**: **53M** (non-embedding) - **23% of baseline**
- **Intermediate Dimensions**: 9216
- **Results**: Train Loss 1.21, **Validation Loss 1.57** (best)
- **HuggingFace Model**: [schuler/experimental-JP47D56C](https://huggingface.co/schuler/experimental-JP47D56C)
- **Key Achievement**: 77% parameter reduction with comparable validation loss to baseline
- **Interactive Demo**: [Chat with JP47D56C](https://huggingface.co/spaces/schuler/kphi3-talk-to-JP47D56C)

## 📋 Other Experiment Variants

The directory also contains additional experiment variants (B-suffix versions and non-suffixed versions) that represent intermediate iterations during the research process:
- `JP47D54_Baseline_2T.ipynb`, `JP47D54B_Baseline_2T.ipynb` - Earlier baseline iterations
- `JP47D55_kphi3_2T.ipynb`, `JP47D55B_kphi3_2T.ipynb` - Earlier 2-layer optimized iterations
- `JP47D56_kphi3_3T.ipynb`, `JP47D56B_kphi3_3T.ipynb` - Earlier 3-layer optimized iterations

## 🚀 How to Use These Notebooks

### Prerequisites

```bash
pip install -q -U transformers==4.51.3
pip install -q -U accelerate==1.10.1
pip install -q -U flash-attn==2.7.3 --no-build-isolation
pip install -q -U bitsandbytes
pip install -q -U datasets
```

### Running an Experiment

1. **Open the notebook** in Jupyter, Google Colab, or your preferred environment
2. **Install dependencies** (first cells contain installation commands)
3. **Configure parameters** (CONTEXT_LENGTH, REPO_NAME, etc.)
4. **Load the dataset** (LaMini-instruction from HuggingFace)
5. **Train the model** (follow the training cells sequentially)
6. **Evaluate results** (validation loss, qualitative outputs)

### Hardware Requirements

- **Minimum**: NVIDIA L4 GPU (or equivalent with 16GB+ VRAM)
- **Recommended**: NVIDIA A100 or V100 for faster training
- **Training Time**: ~3 days on L4 GPU per experiment

## 📊 Key Results Summary

| Experiment | Model | Layers | Non-Emb. Params | % of Baseline | Train Loss | Val. Loss |
|:----------:|:-----:|:------:|:---------------:|:-------------:|:----------:|:---------:|
| JP47D54C | phi-3 | 2 | 227M | 100% | **1.08** | 1.58 |
| JP47D55C | kphi-3 | 2 | **35M** | **15%** | 1.26 | 1.60 |
| JP47D56C | kphi-3 | 3 | **53M** | **23%** | 1.21 | **1.57** |

## 🔑 Key Innovations

The optimized kphi-3 architecture achieves dramatic parameter reduction through:
- **Grouped Pointwise Convolutions**: Replacing dense layers with more efficient convolution operations
- **Optimized Subnetworks**: Strategic parameter sharing while maintaining representational capacity
- **Architectural Efficiency**: Maintaining transformer benefits with reduced computational overhead

## 📖 Learn More

- [Full Technical Report (PDF)](https://www.researchgate.net/publication/388835829_SAVING_77_OF_THE_PARAMETERS_IN_LARGE_LANGUAGE_MODELS_TECHNICAL_REPORT)
- [Main Repository README](../README.md)
- [Source Code](../src/)
- [HuggingFace Model Collection](https://huggingface.co/schuler/)

## 📄 Citation

If you use these experiments or the kphi-3 model in your research, please cite:

```bibtex
@article{SchulerRojas_2025,
  title={Saving 77% of the Parameters in Large Language Models Technical Report},
  url={https://www.researchgate.net/publication/388835829_SAVING_77_OF_THE_PARAMETERS_IN_LARGE_LANGUAGE_MODELS_TECHNICAL_REPORT},
  author={Schwarz Schuler, Joao Paulo and Rojas Gómez, Alejandra},
  year={2025}
}
```

## 💡 Contributing

These notebooks represent reproducible research experiments. For questions or discussions about the experiments, please open an issue in the [main repository](https://github.com/joaopauloschuler/less-parameters-llm).

## 📝 License

Please refer to the [LICENSE](../LICENSE) file in the main repository for licensing information.
