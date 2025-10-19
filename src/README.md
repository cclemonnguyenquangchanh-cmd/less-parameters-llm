# KPhi-3 Source Code

This directory contains the source code implementation of the **KPhi-3** (K-Phi-3) model architecture, an optimized variant of Microsoft's Phi-3 that achieves up to **77% parameter reduction** while maintaining comparable performance.

## 📋 Overview

KPhi-3 is a transformer-based large language model that implements a novel parameter reduction technique using **grouped pointwise convolutions**. This approach, originally developed for computer vision, has been successfully adapted to reduce the parameters in dense layers of transformer models without significant loss in performance.

### Key Innovation

The core innovation replaces traditional dense linear layers with an optimized subnetwork containing grouped pointwise convolutions, dramatically reducing parameters while maintaining the model's learning capacity and representational power.

## 📁 Files in This Directory

### Core Model Files

| File | Description |
|------|-------------|
| **modeling_kphi3.py** | Main model implementation (1955 lines) containing all model classes, attention mechanisms, and the KPhi-3 architecture |
| **configuration_kphi3.py** | Configuration class for KPhi-3 models, defining architecture parameters and hyperparameters |
| **config.json** | Sample configuration file for a KPhi-3 model instance |
| **generation_config.json** | Text generation configuration (token IDs, generation parameters) |

### Media Files

| File | Description |
|------|-------------|
| **video-banner.jpg** | Banner image for introduction video |
| **banner-bigger-smarter.png** | Banner image for additional video content |

## 🏗️ Architecture Components

### Key Classes in `modeling_kphi3.py`

#### Core Model Classes
- **`KPhi3ForCausalLM`**: Main causal language model class for text generation
- **`KPhi3Model`**: Base transformer model without language modeling head
- **`KPhi3ForSequenceClassification`**: Model wrapper for sequence classification tasks
- **`KPhi3ForTokenClassification`**: Model wrapper for token classification tasks

#### Transformer Components
- **`KPhi3DecoderLayer`**: Single transformer decoder layer
- **`KPhi3Attention`**: Standard attention mechanism
- **`KPhi3FlashAttention2`**: Optimized Flash Attention 2.0 implementation
- **`KPhi3SdpaAttention`**: Scaled Dot-Product Attention implementation
- **`KPhi3MLP`**: Multi-layer perceptron with grouped convolutions

#### Parameter Reduction Components
- **`GroupedLinear`**: Grouped linear layer implementation
- **`GroupedLinearFast`**: Optimized fast version of grouped linear layers
- **`GroupedPointwiseConvolutionBlock`**: Core grouped pointwise convolution block
- **`GroupedPointwiseConvolutionBlockIO`**: I/O variant of grouped convolution block
- **`InterleaveChannels`**: Channel interleaving layer for efficient computation
- **`InterleaveChannelsFast`**: Optimized fast channel interleaving

#### Supporting Components
- **`Phi3RMSNorm`**: RMS normalization layer
- **`Phi3RotaryEmbedding`**: Rotary position embeddings (RoPE)
- **`Phi3LongRoPEScaledRotaryEmbedding`**: Long sequence RoPE implementation
- **`KPhi3PreTrainedModel`**: Base class with weight initialization

### Key Functions

- **`get_max_acceptable_common_divisor()`**: Helper for computing optimal grouping parameters
- **`SignedSquareRoot1()`**: Custom activation function
- **`repeat_kv()`**: Key-value repetition for grouped query attention
- **`apply_rotary_pos_emb()`**: Apply rotary position embeddings

## ⚙️ Configuration

The `KPhi3Config` class (in `configuration_kphi3.py`) supports the following key parameters:

### Model Architecture
- **`vocab_size`**: Vocabulary size (default: 32064)
- **`hidden_size`**: Hidden dimension size (default: 3072)
- **`intermediate_size`**: MLP intermediate dimension (default: 9216)
- **`num_hidden_layers`**: Number of transformer layers (default: 3)
- **`num_attention_heads`**: Number of attention heads (default: 32)
- **`num_key_value_heads`**: Number of key-value heads for grouped query attention

### Optimization Parameters
- **`min_channels_per_group`**: Minimum channels per group for grouped convolutions (default: 256)
- **`attention_dropout`**: Dropout rate for attention (default: 0.0)
- **`resid_pdrop`**: Dropout for MLP outputs (default: 0.0)
- **`embd_pdrop`**: Dropout for embeddings (default: 0.0)

### Position Embeddings
- **`max_position_embeddings`**: Maximum sequence length (default: 128000)
- **`rope_theta`**: RoPE base period (default: 10000.0)
- **`rope_scaling`**: Optional RoPE scaling configuration
- **`sliding_window`**: Optional sliding window size

### Other Parameters
- **`hidden_act`**: Activation function (default: "silu")
- **`initializer_range`**: Weight initialization std (default: 0.02)
- **`rms_norm_eps`**: RMS normalization epsilon (default: 1e-5)
- **`use_cache`**: Enable KV caching for generation

## 🚀 Usage

### Loading a Pre-trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "schuler/experimental-JP47D56C"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  # Optional: use Flash Attention 2
)
model.to('cuda')

# Generate text
prompt = "<|user|>\nHello, how are you?\n<|end|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.5)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

### Creating a Custom KPhi-3 Model

```python
from configuration_kphi3 import KPhi3Config
from modeling_kphi3 import KPhi3ForCausalLM

# Create custom configuration
config = KPhi3Config(
    vocab_size=32064,
    hidden_size=3072,
    intermediate_size=9216,
    num_hidden_layers=3,
    num_attention_heads=32,
    min_channels_per_group=256
)

# Initialize model with custom config
model = KPhi3ForCausalLM(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Training

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("MBZUAI/LaMini-instruction")

# Setup training arguments
training_args = TrainingArguments(
    output_dir="./kphi3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=100,
    save_steps=1000,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Train
trainer.train()
```

## 📊 Performance Characteristics

### Parameter Efficiency

Compared to the baseline Phi-3 architecture:
- **2-layer KPhi-3**: 35M parameters (15% of baseline)
- **3-layer KPhi-3**: 53M parameters (23% of baseline)
- **Baseline Phi-3**: 227M parameters (100%)

### Memory and Compute

- Reduced memory footprint proportional to parameter reduction
- Faster inference due to fewer parameters
- Training time comparable to baseline on single GPU
- Compatible with Flash Attention 2 for further optimization

## 🔬 Technical Background

The KPhi-3 architecture is based on research in parameter reduction techniques:

1. **Grouped Pointwise Convolutions**: [Grouped Pointwise Convolutions Reduce Parameters in Convolutional Neural Networks](https://www.researchgate.net/publication/360226228_Grouped_Pointwise_Convolutions_Reduce_Parameters_in_Convolutional_Neural_Networks)

2. **EfficientNet Application**: [Grouped Pointwise Convolutions Significantly Reduces Parameters in EfficientNet](https://www.researchgate.net/publication/355214501_Grouped_Pointwise_Convolutions_Significantly_Reduces_Parameters_in_EfficientNet)

3. **Base Architecture**: Microsoft's [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

## 🔗 Related Resources

- **📄 Full Technical Report**: [Saving 77% of the Parameters in Large Language Models](https://www.researchgate.net/publication/388835829_SAVING_77_OF_THE_PARAMETERS_IN_LARGE_LANGUAGE_MODELS_TECHNICAL_REPORT)
- **🤗 Model Checkpoints**: [HuggingFace Collection](https://huggingface.co/schuler/)
- **📊 Experiment Notebooks**: [Raw Experiments](../raw/)
- **💬 Interactive Demos**:
  - [Chat with JP47D56C](https://huggingface.co/spaces/schuler/kphi3-talk-to-JP47D56C)
  - [KPhi-3 Nano](https://huggingface.co/spaces/schuler/experimental-KPhi-3-nano-4k-instruct)

## 📦 Dependencies

Required packages:
```bash
pip install transformers==4.51.3
pip install accelerate==1.10.1
pip install torch==2.0.0
pip install flash-attn==2.7.3
```

## 🛠️ Development

### Code Structure

```
src/
├── modeling_kphi3.py          # Model implementation (~2000 lines)
│   ├── Parameter reduction layers
│   ├── Attention mechanisms
│   ├── Transformer components
│   └── Model variants
├── configuration_kphi3.py      # Configuration class
├── config.json                 # Example config
└── generation_config.json      # Generation settings
```

### Extending the Model

To create a custom variant:

1. Modify `KPhi3Config` to add new parameters
2. Update `KPhi3MLP` or attention layers for architectural changes
3. Adjust `min_channels_per_group` for different parameter trade-offs
4. Test with different `intermediate_size` values

## 📝 License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.

This implementation modifies Microsoft's Phi-3 model, which is also licensed under Apache 2.0.

## 📖 Citation

If you use KPhi-3 in your research, please cite:

```bibtex
@article{SchulerRojas_2025,
  title={Saving 77% of the Parameters in Large Language Models Technical Report},
  url={https://www.researchgate.net/publication/388835829_SAVING_77_OF_THE_PARAMETERS_IN_LARGE_LANGUAGE_MODELS_TECHNICAL_REPORT},
  author={Schwarz Schuler, Joao Paulo and Rojas Gómez, Alejandra},
  year={2025}
}
```

## 💡 Contributing

For questions, issues, or contributions, please visit the [main repository](https://github.com/joaopauloschuler/less-parameters-llm).

## ⚠️ Note

- This is a research implementation demonstrating parameter reduction techniques
- Models are trained on limited datasets (LaMini) for proof of concept
- Production use may require additional fine-tuning and validation
- Always use `trust_remote_code=True` when loading models from HuggingFace
