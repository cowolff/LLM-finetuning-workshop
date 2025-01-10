# Workshop on LLM Fine-Tuning

This repository contains scripts and notebooks designed for a hands-on workshop on fine-tuning large language models (LLMs). The included materials demonstrate both **predefined training classes** for simplicity and **from-scratch implementations** for deeper insights into model fine-tuning techniques.

---

## Repository Contents

### Notebooks

1. **`LoRa_simple.ipynb`**
   - A straightforward implementation of low-rank adaptation (LoRA) using predefined classes and utilities.
   - Ideal for participants looking for an easy-to-follow introduction to LoRA.

2. **`LoRa_from_scratch.ipynb`**
   - A detailed, step-by-step notebook that implements LoRA from scratch.
   - Recommended for users interested in understanding the inner workings of LoRA.

3. **`create_dataset_wiki.ipynb`**
   - A script to generate datasets from Wikipedia for training and evaluation.
   - Useful for building custom datasets tailored to your specific fine-tuning needs.

4. **`knowledge_distillation.ipynb`**
   - Demonstrates knowledge distillation for transferring knowledge from a larger model to a smaller one.
   - Great for optimizing models for resource-constrained environments.

5. **`supervised_simple.ipynb`**
   - Provides a simple example of naive supervised fine-tuning using predefined training classes.
   - A good starting point for beginners exploring supervised learning techniques for LLMs.

---

### Scripts

1. **`inference.py`**
   - A script for running inference on fine-tuned models.
   - Supports evaluating model performance on custom datasets.

2. **`rlhf_example.py`**
   - An implementation of Reinforcement Learning with Human Feedback (RLHF).
   - Builds from scratch for participants looking to learn about RLHF fundamentals.

3. **`supervised_ewc_self.py`**
   - A script showcasing supervised fine-tuning with Elastic Weight Consolidation (EWC).
   - Useful for scenarios where catastrophic forgetting needs to be mitigated.

---
## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-repo/llm-fine-tuning-workshop.git
cd llm-fine-tuning-workshop
```

2. Explore the notebooks for guided workflows, or execute the scripts for specific tasks.

---

## Contribution

Contributions and feedback are welcome! Please feel free to fork this repository and create pull requests with your suggestions or enhancements.

---

## License

This repository is open-source and available under the [MIT License](LICENSE).

---