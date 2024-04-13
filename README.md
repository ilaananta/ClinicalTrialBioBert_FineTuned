## ClinicalTrialBioBert_FineTuned

Welcome to the ClinicalTrialBioBert_FineTuned repository! Here, you'll find a fine-tuned version of the dmis-lab/biobert-v1.1 model specifically designed for tasks in the clinical trial domain. This model is trained on the Clinical Trial Texts Dataset, which comprises text data extracted from clinical trials available on [ClinicalTrials.gov](https://ClinicalTrials.gov) as of December 3rd, 2022. With a total of 434,977 trials and over 2.1 billion tokens, this dataset provides a rich source of information for training models in the biomedical and clinical trial domains.

### Files

In this repository, you'll find the following files:

- **DownstreamTasks**: Python notebooks containing implementations of downstream tasks using the fine-tuned model.
- **LoRA_ClinicalBioBert.py**: Python code for fine-tuning the model using the LoRA (Low-Rank Adaptation of Large Language Models) technique.
- **out_acc.txt**: A text file containing accuracy metrics after every 5 epochs of fine-tuning.
- **input.csv**: A CSV file containing new trial data (50439 trials) used as input for fine-tuning the model.

### Finetuning

#### Input

To enrich the dataset used for fine-tuning, we identified 50439 clinical trials available on ClinicalTrials.gov that were not present in the original dataset used to train ClinicalTrialBioBert. This additional dataset was then converted to CSV format and used as input for further processing.

#### Preprocessing

Before feeding the data into the model, we performed extensive preprocessing. This involved tokenizing the text and generating input_ids, token_type_ids, and attention masks. These preprocessed inputs were crucial for fine-tuning the model on the Masked Language Model (MLM) task.

#### Masked Language Model (MLM)

The Masked Language Model (MLM) task involves masking certain tokens in the input and training the model to predict these masked tokens. BERT-based models like ClinicalTrialBioBert are trained using MLM among other techniques to learn contextual representations of tokens.

#### LoRA (Low-Rank Adaptation of Large Language Models)

LoRA is a lightweight training technique that fine-tunes only a smaller number of weights in the model, significantly reducing memory consumption and speeding up training. Instead of finetuning all the weights, LoRA focuses on fine-tuning two smaller matrices that approximate the larger weight matrix of the pre-trained model. This technique makes fine-tuning more efficient and practical for large language models like ClinicalTrialBioBert.

### Downstream Tasks

To evaluate the performance of the fine-tuned model, we conducted two downstream tasks:

1. **Multi-label Classification**: Using the opentargets/clinical_trial_stop_reasons dataset, we classified reasons for early termination of clinical trials. This task provides valuable insights into the factors influencing the outcome of clinical trials.

2. **Multi-class Classification**: We performed gender classification on clinical trial texts using the Kira-Asimov/gender_clinical_trial dataset. This task helps in understanding the representation of gender in clinical trials and its implications.

### Results

#### Multi-label Classification Results

The precision, recall, and F1-score values for each class in both non-fine-tuned and fine-tuned models are provided in the tables below.

**Non-Fine-Tuned Model:**

| Label                     | Precision | Recall | F1-score | Support |
|---------------------------|-----------|--------|----------|---------|
| Another_Study             | 0.9613    | 0.8744 | 0.9158   | 199     |
| ...                       | ...       | ...    | ...      | ...     |


**Fine-Tuned Model:**

| Label                     | Precision | Recall | F1-score | Support |
|---------------------------|-----------|--------|----------|---------|
| Another_Study             | 0.9744    | 0.9548 | 0.9645   | 199     |
| ...                       | ...       | ...    | ...      | ...     |

#### Multi-class Classification Results

Similar to the multi-label classification results, the precision, recall, and F1-score values for each class in both non-fine-tuned and fine-tuned models are provided in the tables below.

**Non-Fine-Tuned Model:**

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| All         | 0.9800    | 0.9895 | 0.9847   | 30954   |
| ...         | ...       | ...    | ...      | ...     |

**Fine-Tuned Model:**

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| All         | 0.9911    | 0.9960 | 0.9935   | 34451   |
| ...         | ...       | ...    | ...      | ...     |

### Running the Code

To replicate our experiments or fine-tune the model further, you can use the provided Python code. Simply execute the following command:


 
