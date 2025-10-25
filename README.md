# transformer-translator-pytorch

This project implements a **machine translation model** based on the original Transformer architecture introduced in *“Attention is All You Need”* (Vaswani et al., 2017) [1].

The model is trained on the **English–Italian parallel corpus** from *OPUS Books* [2], but you can easily replace it with any other bilingual dataset.

---

## ⚙️ Configurations

All hyperparameters and file paths can be configured in the `config` section of the main script or configuration file.

### **Data Parameters**

| Argument | Type | Description | Default |
|-----------|------|-------------|----------|
| datasource | str | Dataset name | 'opus_books' |
| lang_src | str | Source language | 'en' |
| lang_tgt | str | Target language | 'it' |
| seq_len | int | Maximum sequence length | 350 |
| batch_size | int | Batch size | 8 |
| tokenizer_file | str | Tokenizer filename pattern | 'tokenizer_{0}.json' |

---

### **Transformer Parameters**

| Argument | Type | Description | Default |
|-----------|------|-------------|----------|
| d_model | int | Hidden dimension | 512 |
| d_ff | int | Feed-forward hidden dimension | 2048 |
| num_heads | int | Number of attention heads | 8 |
| num_layers | int | Number of encoder/decoder layers | 6 |
| dropout | float | Dropout probability | 0.1 |
| lr | float | Learning rate | 1e-4 |
| num_epochs | int | Number of training epochs | 8 |

---

### **Training and Checkpoint Parameters**

| Argument | Type | Description | Default |
|-----------|------|-------------|----------|
| model_folder | str | Directory to save weights | 'weights' |
| model_basename | str | Base name for model checkpoints | 'tmodel_' |
| preload | str | Which model to preload ('latest' or specific epoch) | 'latest' |
| experiment_name | str | TensorBoard experiment path | 'runs/tmodel' |

---

## 🧠 Model Overview

The model follows the **Encoder–Decoder Transformer** structure:
- Scaled **multi-head attention**
- **Positional encoding**
- **Residual connections + Layer normalization**
- **Feed-forward blocks**
- Implemented in **PyTorch (object-oriented structure)**

It includes:
- Custom WordLevel tokenizers (Hugging Face `tokenizers`)
- Sentence masking and padding logic
- Greedy decoding for inference
- BLEU, WER, CER metrics via TorchMetrics

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
