## ClinicalTrialBioBert_FineTuned

This repository contains a fine-tuned version of the [domenicrosati/ClinicalTrialBioBert-NLI4CT](https://huggingface.co/domenicrosati/ClinicalTrialBioBert-NLI4CT) model, tailored specifically for tasks within the clinical trial domain. The model has been trained on the [Clinical Trial Texts Dataset](https://huggingface.co/datasets/domenicrosati/clinical_trial_texts), a comprehensive collection of text data extracted from clinical trials available on [ClinicalTrials.gov](https://ClinicalTrials.gov) as of December 3rd, 2022. With a vast corpus comprising 434,977 trials, this dataset serves as a rich source of information for training language models in both biomedical and clinical trial domains.

### Files

Within this repository, you will find the following files:

- **DownstreamTasks**: A directory containing Python notebooks implementing downstream tasks using the fine-tuned model.
- **LoRA_ClinicalBioBert.py**: Python code responsible for fine-tuning the model using the LoRA (Low-Rank Adaptation of Large Language Models) technique.
- **out_acc.txt**: A text file containing accuracy metrics after every 5 epochs of fine-tuning.
- **input.csv**: A CSV file containing new trial data (50439 trials) used as input for fine-tuning the model.

### Finetuning

#### Input

The dataset used for fine-tuning was enriched by identifying an additional 50439 clinical trials available on [ClinicalTrials.gov](https://ClinicalTrials.gov) but not present in the original dataset used to train ClinicalTrialBioBert. This supplementary dataset was then converted to CSV format and used as input for further processing.

#### Preprocessing

Prior to feeding the data into the model, extensive preprocessing was conducted. This preprocessing included tokenization of the text and generation of input_ids, token_type_ids, and attention masks. These preprocessed inputs played a crucial role in fine-tuning the model on the Masked Language Model (MLM) task.

### Masked Language Model (MLM)

The Masked Language Model (MLM) task is a key component of the pre-training process for BERT-based models like ClinicalTrialBioBert. In MLM, a certain percentage of tokens in the input text are randomly masked, and the model is trained to predict these masked tokens based on the surrounding context. This task enables the model to learn bidirectional contextual representations of words, which is crucial for understanding and generating natural language.

During fine-tuning, the masked tokens are replaced with a special token, `[MASK]`, and the model is trained to predict the original tokens. This fine-tuning process adapts the model to specific downstream tasks, such as classification or sequence labeling, by adjusting the model's weights to better capture task-specific features in the data.

### LoRA (Low-Rank Adaptation of Large Language Models)

LoRA is a novel technique for fine-tuning large language models that offers several advantages over traditional fine-tuning approaches. In traditional fine-tuning, all the weights of the pre-trained model are updated during the fine-tuning process, which can be computationally expensive and memory-intensive, especially for large models like ClinicalTrialBioBert.

LoRA addresses this challenge by introducing a low-rank approximation to the weight matrices of the pre-trained model. Instead of updating all the weights, LoRA fine-tunes a smaller number of parameters, resulting in faster training times and reduced memory footprint. This low-rank approximation is achieved by decomposing the weight matrices into smaller matrices that approximate the original weights.

By fine-tuning only a subset of parameters, LoRA allows for more efficient and scalable fine-tuning of large language models. This makes it particularly well-suited for scenarios where computational resources are limited or where fine-tuning needs to be performed on a large scale. Additionally, LoRA produces smaller model weights, making the fine-tuned model easier to deploy and share.


### Downstream Tasks

To evaluate the performance of the fine-tuned model, two downstream tasks were conducted:

1. **Multi-label Classification**: Utilizing the [opentargets/clinical_trial_stop_reasons](https://huggingface.co/datasets/opentargets/clinical_trial_reason_to_stop) dataset, reasons for early termination of clinical trials were classified. This task provides valuable insights into the factors influencing the outcome of clinical trials.

2. **Multi-class Classification**: Gender classification on clinical trial texts was performed using the [Kira-Asimov/gender_clinical_trial dataset](https://huggingface.co/datasets/Kira-Asimov/gender_clinical_trial). This task aids in understanding the representation of gender in clinical trials and its implications.

### Results

#### Multi-label Classification Results

The precision, recall, and F1-score values for each class in both non-fine-tuned and fine-tuned models are provided in the tables below.

**Non-Fine-Tuned Model:**

| Label                   | Precision | Recall | F1-score | Support |
|-------------------------|-----------|--------|----------|---------|
| Another_Study           | 0.9613    | 0.8744 | 0.9158   | 199     |
| Business_Administrative | 0.9712    | 0.8939 | 0.9310   | 792     |
| Covid19                 | 1.0000    | 1.0000 | 1.0000   | 183     |
| Endpoint_Met            | 0.0000    | 0.0000 | 0.0000   | 51      |
| Ethical_Reason          | 0.0000    | 0.0000 | 0.0000   | 17      |
| Insufficient_Data       | 0.0000    | 0.0000 | 0.0000   | 39      |
| Insufficient_Enrollment | 0.9675    | 0.9684 | 0.9679   | 1075    |
| Interim_Analysis        | 0.0000    | 0.0000 | 0.0000   | 28      |
| Invalid_Reason          | 0.9174    | 0.8880 | 0.9024   | 250     |
| Logistics_Resources     | 0.9196    | 0.6080 | 0.7320   | 301     |
| Negative                | 0.9535    | 0.8913 | 0.9213   | 368     |
| No_Context              | 0.0000    | 0.0000 | 0.0000   | 83      |
| Regulatory              | 0.0000    | 0.0000 | 0.0000   | 112     |
| Safety_Sideeffects      | 0.9558    | 0.8199 | 0.8827   | 211     |
| Study_Design            | 0.9254    | 0.6019 | 0.7294   | 309     |
| Study_Staff_Moved       | 0.9630    | 0.9455 | 0.9541   | 165     |
| Success                 | 0.0000    | 0.0000 | 0.0000   | 21      |
| Micro Avg               | 0.9588    | 0.7978 | 0.8709   | 4204    |
| Macro Avg               | 0.5609    | 0.4995 | 0.5257   | 4204    |
| Weighted Avg            | 0.8770    | 0.7978 | 0.8319   | 4204    |
| Samples Avg             | 0.8890    | 0.8455 | 0.8591   | 4204    |

**Fine-Tuned Model:**

| Label                   | Precision | Recall | F1-score | Support |
|-------------------------|-----------|--------|----------|---------|
| Another_Study           | 0.9744    | 0.9548 | 0.9645   | 199     |
| Business_Administrative | 0.9712    | 0.9811 | 0.9761   | 792     |
| Covid19                 | 0.9946    | 1.0000 | 0.9973   | 183     |
| Endpoint_Met            | 0.9474    | 0.7059 | 0.8090   | 51      |
| Ethical_Reason          | 1.0000    | 0.1176 | 0.2105   | 17      |
| Insufficient_Data       | 1.0000    | 0.1795 | 0.3043   | 39      |
| Insufficient_Enrollment | 0.9861    | 0.9898 | 0.9879   | 1075    |
| Interim_Analysis        | 1.0000    | 0.8214 | 0.9020   | 28      |
| Invalid_Reason          | 0.9790    | 0.9320 | 0.9549   | 250     |
| Logistics_Resources     | 0.9148    | 0.9269 | 0.9208   | 301     |
| Negative                | 0.9782    | 0.9755 | 0.9769   | 368     |
| No_Context              | 1.0000    | 0.7952 | 0.8859   | 83      |
| Regulatory              | 0.9717    | 0.9196 | 0.9450   | 112     |
| Safety_Sideeffects      | 0.9857    | 0.9810 | 0.9834   | 211     |
| Study_Design            | 0.9394    | 0.9029 | 0.9208   | 309     |
| Study_Staff_Moved       | 1.0000    | 0.9636 | 0.9815   | 165     |
| Success                 | 1.0000    | 0.0952 | 0.1739   | 21      |
| Micro Avg               | 0.9733    | 0.9441 | 0.9585   | 4204    |
| Macro Avg               | 0.9790    | 0.7790 | 0.8173   | 4204    |
| Weighted Avg            | 0.9738    | 0.9441 | 0.9524   | 4204    |
| Samples Avg             | 0.9654    | 0.9564 | 0.9577   | 4204    |


#### Multi-class Classification Results

Similar to the multi-label classification results, the precision, recall, and F1-score values for each class in both non-fine-tuned and fine-tuned models are provided in the tables below.

**Non-Fine-Tuned Model:**
|  Class| Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| All   | 0.9899    | 0.9952 | 0.9925   | 34451   |
| None  | 0.0000    | 0.0000 | 0.0000   | 58      |
| Male  | 0.9688    | 0.9259 | 0.9469   | 1646    |
| Female| 0.9644    | 0.9516 | 0.9579   | 3986    |
| Micro Avg | 0.9865 | 0.9865 | 0.9865 | 40141   |
| Macro Avg | 0.7308 | 0.7182 | 0.7243 | 40141   |
| Weighted Avg | 0.9850 | 0.9865 | 0.9858 | 40141 |
| Samples Avg | 0.9865 | 0.9865 | 0.9865 | 40141   |


**Fine-Tuned Model:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| All   | 0.9911    | 0.9960 | 0.9935   | 34451   |
| None  | 0.0000    | 0.0000 | 0.0000   | 58      |
| Male  | 0.9722    | 0.9350 | 0.9532   | 1646    |
| Female| 0.9718    | 0.9604 | 0.9661   | 3986    |
| Micro Avg | 0.9885 | 0.9885 | 0.9885 | 40141   |
| Macro Avg | 0.7338 | 0.7228 | 0.7282 | 40141   |
| Weighted Avg | 0.9870 | 0.9885 | 0.9877 | 40141 |
| Samples Avg | 0.9885 | 0.9885 | 0.9885 | 40141   |


### Running the Code

To replicate our experiments or fine-tune the model further, you can use the provided Python code. Simply execute the following command:
```bash
python LoRA_ClinicalBioBert.py
```


### Conclusion

The fine-tuned ClinicalTrialBioBert model, augmented with LoRA, demonstrates superior performance in both multi-label and multi-class classification tasks compared to the non-fine-tuned model. This repository serves as a valuable resource for researchers and practitioners in the clinical trial domain seeking state-of-the-art language models for various NLP tasks.

