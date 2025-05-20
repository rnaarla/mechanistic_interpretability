
# Mechanistic Interpretability: A Practical Guide

This repository provides a comprehensive practical guide to **mechanistic interpretability** — the process of reverse-engineering and understanding the internal workings of modern neural networks, especially transformer-based language models.

---

## Table of Contents

- [Introduction](#introduction)  
- [Why Mechanistic Interpretability?](#why-mechanistic-interpretability)  
- [Core Techniques](#core-techniques)  
- [Tools and Frameworks](#tools-and-frameworks)  
- [Practical Implementation Guide](#practical-implementation-guide)  
- [Summary Table of Tools and Techniques](#summary-table-of-tools-and-techniques)  
- [Recommendations](#recommendations)  
- [Further Resources](#further-resources)  

---

## Introduction

Modern deep neural networks, especially large transformers like GPT and BERT, achieve remarkable capabilities but often behave as “black boxes” with inscrutable internal logic. Mechanistic interpretability aims to **illuminate the causal inner mechanisms**—neurons, attention heads, and circuits—that drive model behavior.

Understanding these mechanisms is critical to improve model transparency, diagnose failures and biases, and ultimately build safer and more accountable AI systems.

---

## Why Mechanistic Interpretability?

- **Accountability:** Understand *why* models make specific decisions.  
- **Debugging:** Identify internal sources of bias or failure modes.  
- **Transparency:** Move from opaque “black boxes” to explainable “glass boxes.”  
- **Control:** Enable targeted interventions on specific model components.  

---

## Core Techniques

| Technique               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Attention Visualization** | Interactive maps of attention weights between tokens.                      |
| **Attribution Methods**     | Algorithms (e.g., Integrated Gradients) assigning importance scores to inputs or neurons. |
| **Neuron / Activation Analysis** | Programmatic hooks to inspect and intervene in neuron activations.           |
| **Ablation Studies**         | Disabling neurons or attention heads to test causal impact on outputs.      |
| **Activation Patching**      | Swapping activations between inputs to test causal influence.                |
| **Circuit Discovery**        | Identifying subnetworks or neuron groups performing specific functions.      |
| **Causal Mediation Analysis**| Statistical methods to quantify causal effects of components.                |
| **Visualization**            | Tools and dashboards for heatmaps, embeddings, and attention patterns.      |

---

## Tools and Frameworks

| Tool / Library         | Description                                                       | Installation                  | Notes                          |
|-----------------------|-------------------------------------------------------------------|------------------------------|-------------------------------|
| **OpenAI Microscope**   | Web-based tool to explore neuron activations and attention.      | No install; web-based         | [https://microscope.openai.com/](https://microscope.openai.com/) |
| **BERTViz**             | Interactive attention visualization for transformers.            | `pip install bertviz`         | Works well in Jupyter notebooks |
| **Captum**              | PyTorch interpretability library for attribution methods.        | `pip install captum`          | Supports Integrated Gradients, DeepLIFT, etc. |
| **TransformerLens**     | Programmatic API for neuron-level inspection and activation patching. | `pip install transformer_lens` | Formerly Huggingface-Explainability |
| **TensorBoard**         | Visualization tool for activations, embeddings, and scalars.     | Included with TensorFlow       | Can be integrated with PyTorch |
| **Weights & Biases**    | Experiment tracking and custom visualization dashboards.         | `pip install wandb`           | Excellent for collaborative projects |
| **NetworkX**            | Graph analysis and visualization library.                        | `pip install networkx`        | Useful for circuit discovery |
| **DoWhy / CausalNex**   | Causal inference and mediation analysis libraries.                | `pip install dowhy causalnex` | Statistical causal analysis   |
| **Lucid**               | Feature visualization for neural nets (Google Brain).             | See repo: https://github.com/tensorflow/lucid | Mostly TensorFlow-based |

---

## Practical Implementation Guide

### 1. Attention Visualization with BERTViz

```python
from transformers import BertTokenizer, BertModel
from bertviz import head_view

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

sentence = "The cat sat on the mat."
inputs = tokenizer.encode_plus(sentence, return_tensors='pt')
outputs = model(**inputs)
attentions = outputs.attentions

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
head_view(attentions, tokens)
```

### 2. Attribution Using Captum (Integrated Gradients)

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import IntegratedGradients

model_name = 'textattack/bert-base-uncased-SST-2'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

text = "The movie was absolutely wonderful!"
inputs = tokenizer(text, return_tensors='pt')

def forward_func(input_ids):
    outputs = model(input_ids)
    return outputs.logits[:, 1]

ig = IntegratedGradients(forward_func)
attributions, delta = ig.attribute(inputs['input_ids'], target=1, return_convergence_delta=True)

print("Attributions:", attributions)
```

### 3. Using TransformerLens for Activation Inspection and Patching

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained('gpt2-small')

text_a = "The cat is happy."
text_b = "The dog is angry."

tokens_a = model.to_tokens(text_a)
tokens_b = model.to_tokens(text_b)

# Capture activations for tokens_a and tokens_b
activations_a = {}
activations_b = {}

def save_hook(name):
    def hook_fn(tensor, hook):
        activations_a[name] = tensor.clone()
    return hook_fn

def save_hook_b(name):
    def hook_fn(tensor, hook):
        activations_b[name] = tensor.clone()
    return hook_fn

model.run_with_hooks(tokens_a, fwd_hooks=[("blocks.0.attn.hook_attn_result", save_hook("attn_0"))])
model.run_with_hooks(tokens_b, fwd_hooks=[("blocks.0.attn.hook_attn_result", save_hook_b("attn_0"))])

def patch_hook(tensor, hook):
    return activations_b["attn_0"]

logits_patched = model.run_with_hooks(tokens_a, fwd_hooks=[("blocks.0.attn.hook_attn_result", patch_hook)])
```

### 4. Ablation of Attention Heads in Huggingface Transformers

```python
from transformers import GPT2Model, GPT2Tokenizer

model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

input_ids = tokenizer.encode("Hello world!", return_tensors='pt')

def zero_head_hook(module, input, output):
    output[:, 0, :, :] = 0  # Zero out head 0 attention scores
    return output

hook = model.transformer.h[0].attn.register_forward_hook(zero_head_hook)
outputs = model(input_ids)
hook.remove()
```

### 5. Visualization Example with Plotly

```python
import plotly.express as px
import numpy as np

attention = np.random.rand(10, 10)  # Example attention matrix
fig = px.imshow(attention, color_continuous_scale='Viridis')
fig.update_layout(title='Attention Map')
fig.show()
```

---

## Summary Table of Tools and Techniques

| Technique               | Tools / Libraries              | Purpose                                               |
|-------------------------|-------------------------------|------------------------------------------------------|
| Attention Visualization | BERTViz, OpenAI Microscope    | Visualize attention heads and patterns               |
| Attribution Methods     | Captum, ELI5                  | Compute feature and neuron attributions              |
| Activation Analysis     | TransformerLens, Lucid        | Inspect neuron activations and intervene programmatically |
| Ablation Studies        | Huggingface Hooks             | Disable neurons or heads to test causal effects      |
| Activation Patching     | TransformerLens               | Swap activations between inputs                       |
| Circuit Discovery       | NetworkX, TSNE, Clustering    | Identify functional neuron groups                     |
| Causal Mediation        | DoWhy, CausalNex              | Quantify causal effects                               |
| Visualization           | TensorBoard, W&B, Plotly      | Interactive visualizations and dashboards             |

---

## Recommendations

- Start with **attention visualization** tools like BERTViz or OpenAI Microscope to gain intuition.  
- Use **TransformerLens** for in-depth neuron-level analysis and causal interventions.  
- Leverage **Captum** for attribution methods on PyTorch models.  
- Employ ablation hooks in Huggingface Transformers to test causal impact of components.  
- Combine multiple techniques to validate mechanistic hypotheses.  
- Track experiments and visualizations with **Weights & Biases** for reproducibility and collaboration.

---

## Advanced Implementations in Mechanistic Interpretability

This section presents advanced code examples and practical implementations relevant to mechanistic interpretability and model optimization.

---

### 1. Efficient Fine-Tuning Techniques

#### LoRA (Low-Rank Adaptation) Example Using PEFT Library

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,              # Low-rank dimension
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.shape)
```

---

### 2. Performance and Scalability

#### Using FlashAttention (PyTorch Integration Snippet)

```python
# FlashAttention is often integrated at a lower level, but PyTorch users can enable it via:
import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

# Example: Using flash attention within custom attention function for speed
# (Requires flash_attn installed and CUDA environment)
# Note: Full FlashAttention integration requires specific setup and is often embedded in custom libraries or frameworks.
```

---

### 3. Automated Circuit Discovery

#### Simple Neuron Clustering Using UMAP and HDBSCAN

```python
import torch
import umap
import hdbscan
import matplotlib.pyplot as plt

# Assume neuron_activations: Tensor of shape (num_samples, num_neurons)
neuron_activations = torch.randn(1000, 768).numpy()

# Dimensionality reduction with UMAP
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(neuron_activations.T)

# Clustering with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(embedding)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
plt.title("Neuron Clusters")
plt.show()
```

---

### 4. Formal Verification and Hybrid Approaches

#### Using Z3 SMT Solver for Simple Logic Verification

```python
from z3 import *

# Example: Verify a simple logical property of neuron outputs
x = Bool('x')
y = Bool('y')

solver = Solver()
solver.add(Implies(x, y))
solver.add(x == True)

if solver.check() == sat:
    print("Property holds:", solver.model())
else:
    print("Property violated")
```

---

### 5. Expanded Practical Examples

#### Bias Detection via Activation Patching (Simplified Example)

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained('gpt2-small')

# Define two inputs
input_biased = model.to_tokens("The doctor said he was tired.")
input_neutral = model.to_tokens("The person said they were tired.")

# Store activations from biased input
activations_biased = {}

def save_hook(name):
    def hook_fn(tensor, hook):
        activations_biased[name] = tensor.clone()
    return hook_fn

model.run_with_hooks(input_biased, fwd_hooks=[("blocks.0.attn.hook_attn_result", save_hook("attn_0"))])

# Patch activations in neutral input with biased activations
def patch_hook(tensor, hook):
    return activations_biased["attn_0"]

logits_patched = model.run_with_hooks(input_neutral, fwd_hooks=[("blocks.0.attn.hook_attn_result", patch_hook)])
```

---

## Further Resources

- [OpenAI Microscope](https://microscope.openai.com/) — Interactive visualization of GPT-2 neurons  
- [BERTViz GitHub](https://github.com/jessevig/bertviz) — Attention visualization library  
- [Captum Documentation](https://captum.ai/) — PyTorch interpretability toolkit  
- [TransformerLens GitHub](https://github.com/neelnanda-io/TransformerLens) — Programmatic transformer introspection  
- [Lucid GitHub](https://github.com/tensorflow/lucid) — Feature visualization for TensorFlow  
- [DoWhy Documentation](https://microsoft.github.io/dowhy/) — Causal inference library  

---

> *“Mechanistic interpretability is the forensic analysis of neural networks — dissecting their inner workings to understand, explain, and control.”*

---

## License

This work is provided under the MIT License.

---

## Acknowledgments

Thanks to the open-source community and researchers advancing the field of mechanistic interpretability and AI transparency.

---

# Happy dissecting and exploring the neural black box! 🚀
