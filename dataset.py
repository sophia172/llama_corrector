from datasets import load_dataset

dataset = load_dataset("Intel/orca_dpo_pairs")

print("keys", dataset["train"][100].keys())
print("system", dataset["train"][100]['system'])
print("question", dataset["train"][100]['question'])
print("datastet", dataset.keys())



# you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:
from transformers import AutoTokenizer


model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, token=True)
#
#
def tokenize_function(examples):
    return tokenizer(examples["question"], padding=True, max_length=1000, truncation=True)

# result = tokenize_function(dataset["train"][100])
tokenized_datasets = dataset.map(tokenize_function, batched=True)


## Remove the text column because the model does not accept raw text as an input:
tokenized_datasets = tokenized_datasets.remove_columns(["rejected"])
#
# # Rename the label column to labels because the model expects the argument to be named labels:
tokenized_datasets = tokenized_datasets.rename_column("chosen", "labels")
#
# # Set the format of the dataset to return PyTorch tensors instead of lists:
tokenized_datasets.set_format("torch")
#
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
#
# # Create a DataLoader for your training and test datasets so you can iterate over batches of data:
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)


# Load your model with the number of expected labels:
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model)



# Optimizer and learning rate scheduler
# Create an optimizer and learning rate scheduler to fine-tune the model. Let’s use the AdamW optimizer from PyTorch:


from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
# Create the default learning rate scheduler from Trainer:
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
# Lastly, specify device to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes.

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
# To keep track of your training progress, use the tqdm library to add a progress bar over the number of training steps:

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# # Evaluate
# # Just like how you added an evaluation function to Trainer, you need to do the same when you write your own training loop. But instead of calculating and reporting the metric at the end of each epoch, this time you’ll accumulate all the batches with add_batch and calculate the metric at the very end.
#
# import evaluate
#
# metric = evaluate.load("accuracy")
# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)
#
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])
#
# metric.compute()