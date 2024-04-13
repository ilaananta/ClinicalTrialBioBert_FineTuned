import pandas as pd
import re
from datasets import load_dataset, Dataset,DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from peft import get_peft_model
import numpy as np
from transformers import EarlyStoppingCallback


def clean_text(text):
    # Define regex pattern for special characters
    pattern = r'[\[\]{}():@#$%^&*]'
    # Remove special characters using regex
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def tokenize_function(examples):
    return tokenizer(examples["text"],truncation=True, max_length=2109)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def compute_metrics(labels,predictions):
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]

    # Calculate accuracy
    #print(predictions,labels)
    accuracy = (predictions == labels).sum() / len(labels)
    return {"accuracy": accuracy}



def compute_accuracy(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy=[]
    model.to(device)
    model.eval()
    for val in lm_datasets["validation"] :
        #print(val)
        formatted_input = data_collator([val])
        
        formatted_input = {key: value.to(device) for key, value in formatted_input.items()}
        lables=formatted_input['labels']
        with torch.no_grad():
        # Ensure all intermediate tensors are on the same device as the model
            for key, value in formatted_input.items():
                formatted_input[key] = value.to(device)
        
            # Perform inference
            outputs = model(**formatted_input)
        predictions = outputs.logits.argmax(dim=-1)
        acc=compute_metrics(lables,predictions)
        #print(acc
        accuracy.append(np.float64(acc['accuracy']))
    
    
    #print(accuracy)
    return(np.mean(accuracy))


# Apply cleaning function to 'text' column
block_size = 128

df=pd.read_csv('./clinical_bio_bert/input.csv',index_col=0)
df['text'] = df['text'].apply(clean_text)
train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=42)
# Creating Dataset objects from the train and test sets
train_dataset = Dataset.from_dict({'text': train_dataset['text']})
val_dataset = Dataset.from_dict({'text': test_dataset['text'][0:500]})
test_dataset = Dataset.from_dict({'text': test_dataset['text'][500:]})

# Creating a DatasetDict
new_dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})
tokenizer = AutoTokenizer.from_pretrained('domenicrosati/ClinicalTrialBioBert', use_fast=True)
tokenized_datasets = new_dataset.map(tokenize_function, batched=True, num_proc=4,remove_columns=["text"])
print('Tokenized dataset created')
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=16,
    num_proc=4,
)
print('Dataset Divided into chunks')
model =AutoModelForMaskedLM.from_pretrained("domenicrosati/ClinicalTrialBioBert")
print('Model Initialized')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15,mlm='True')

common_training_args = {
    "evaluation_strategy": "no",
    "num_train_epochs": 5,
    "learning_rate": 3e-06,
    "lr_scheduler_type": "linear",
    "max_grad_norm": 1.0,
    "optim": "adamw_hf",
    "adafactor": False,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-08,
    "save_strategy": "epoch",
    "save_total_limit": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 1,
    "report_to": ["tensorboard"]
}


lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, r=1, lora_alpha=1, lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=lm_datasets["train"],
#     eval_dataset=lm_datasets["validation"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )


accur=[]
for i in range(10):
    print('Training for 5 epochs')
    output_dir = f"Checkpoint_LoRA_run_{i}"

    # Update the output directory in training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        **common_training_args
    )
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

    trainer.train()
    acc=compute_accuracy(model)
    print(acc)
    accur.append(acc)


with open('./clinical_bio_bert/out_acc.txt','w') as f:
    for i in range(10):
     f.write(f'Accuracy after {i*5} epoch = {accur[i]}\n')
    f.close()    
    
    
    
