# Sexism Detection via Few-Shot Prompting with LLMs

This repository contains the code and resources for the "Sexism Detection via Few-Shot Prompting with LLMs" project, an NLP assignment investigating binary sexism detection using Large Language Models (LLMs) on the EDOS Task A dataset.

## Author

* Nadia Farokhpay - University of Bologna, Master’s Degree in Artificial Intelligence
    * Email: nadia.farokhpay@studio.unibo.it

## Abstract

This report investigates binary sexism detection using Large Language Models (LLMs) on the EDOS Task A dataset. We compare the performance of Phi-2 and Llama 3.1 under zero-shot, two-shot, and four-shot prompting. Results show that few-shot prompting generally improves accuracy for Llama 3.1, with the two-shot configuration yielding the best performance. The study highlights the LLMs’ ability to follow instructions and the critical role of prompt design and example selection in sensitive text classification tasks.

## Introduction

Online sexism is a pervasive issue. This study evaluates the performance of two prominent open-source LLMs, Phi-2 and Llama 3.1, for binary sexism detection (EDOS Task A). The objective is to classify an input text sentence as either “sexist” or “not sexist” by exploring zero-shot and few-shot (two-shot and four-shot) prompting strategies without extensive fine-tuning.

## Methodology

The system comprises four main components: model initialization, prompt setup, inference pipeline, and evaluation.

### Model Initialization

* **Models Used:** Phi-2 (`microsoft/phi-2`) and Llama 3.1 (`meta-llama/Llama-3.1-8B`) were downloaded from Hugging Face.
* **Optimization:** Both models were loaded with 4-bit quantization using `BitsAndBytesConfig` and `torch.bfloat16` for optimized inference in hardware-limited environments.
* **Generation Configuration:** Text generation pipelines were configured with `max_new_tokens=30`, `do_sample=False`, and `temperature=0.0` for deterministic and concise responses.

### Data Loading

Evaluation was performed on a balanced subset of the EDOS Task A dataset, comprising 300 samples for testing (`a2_test.csv`) and 1000 samples for demonstrations (`demonstrations.csv`).

### Prompt Setup

* **Zero-shot:** A `zeroshot_prompt` template was defined, instructing LLMs as "sexism detection annotators" to respond with "YES" or "NO."
* **Few-shot:** A `fewshot_template` dynamically injected a specified number of balanced examples from `demonstrations.csv` into the structured instruction prompts.

### Inference and Evaluation

The inference pipeline processed tokenized input data, generated responses, and parsed them into “sexist” (1) or “not sexist” (0) labels. Model performance was evaluated using:

* **Accuracy:** Percentage of correct predictions.
* **Fail-ratio:** Proportion of responses that failed to follow the prompt format.

## Experimental Results

The performance of Phi-2 and Llama 3.1 across zero-shot, 2-shot, and 4-shot configurations is summarized below:

| Model     | Prompt Type | Accuracy | Fail Ratio |
| :-------- | :---------- | :------- | :--------- |
| Phi-2     | Zero-shot   | 0.55     | 0.00       |
| Phi-2     | 2-shot      | 0.57     | 0.00       |
| Phi-2     | 4-shot      | 0.47     | 0.00       |
| Llama 3.1 | Zero-shot   | 0.54     | 0.00       |
| Llama 3.1 | 2-shot      | 0.61     | 0.00       |
| Llama 3.1 | 4-shot      | 0.59     | 0.00       |

## Discussion

* **Accuracy Analysis:** Both models showed modest performance (47% to 61% accuracy). Llama 3.1 consistently improved with few-shot prompting, with the 2-shot configuration achieving the highest accuracy (0.61), indicating better task context understanding. Phi-2 showed a slight improvement with 2-shot but a notable decrease with 4-shot (0.47), suggesting increased examples might introduce noise or lead to overfitting for Phi-2.
* **Fail Ratio:** A consistent 0.0 fail ratio across all experiments for both models highlights effective prompt engineering in guiding LLMs to produce desired “YES” or “NO” responses.
* **Error Analysis:** Classification reports showed both models had higher recall for the “Sexist” class and lower precision for the “Not Sexist” class. This means models were more prone to false positives (identifying non-sexist content as sexist). This bias was more pronounced in zero-shot setups and generally reduced in few-shot configurations.

## Conclusion

The assignment successfully demonstrated the application of LLMs for binary text classification using zero-shot and few-shot prompting. While the models consistently adhered to the response format (zero fail-ratio), their classification accuracy for sexism detection remains moderate. Llama 3.1 benefited from few-shot examples, while Phi-2’s performance can degrade with more examples, emphasizing the sensitivity of LLMs to prompt design and example selection.

## Future Work

Further work could involve more sophisticated prompt engineering, exploring different LLM architectures, or fine-tuning the models on a larger, task-specific dataset to improve classification performance.

## References

* [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
* [Meta Llama-3.1](https://huggingface.co/meta-llama/Llama-3.1-8B)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
* [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
