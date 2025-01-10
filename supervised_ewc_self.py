from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, PreTrainedTokenizer, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn as nn
from datasets import load_dataset
import os

class EWC(nn.Module):
    """
    Implementation of Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting
    """
    def __init__(self, model, dataset, importance=1000000, fisher_path='fisher_info.pth'):
        super().__init__()
        self.model = model
        self.importance = importance
        self.fisher_path = fisher_path

        # Load Fisher Information Matrix if it exists, otherwise calculate and save it
        self.fisher_info = self._load_or_calculate_fisher(dataset)

        # Store a copy of the initial model parameters
        self.old_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters() if p.requires_grad
        }

    def _load_or_calculate_fisher(self, dataset):
        """Load Fisher Information Matrix from file or calculate it if not found."""
        if os.path.exists(self.fisher_path):
            try:
                fisher_info = torch.load(self.fisher_path, map_location=self.model.device)
                print(f"Loaded Fisher Information Matrix from {self.fisher_path}")
            except Exception as e:
                print(f"Error loading Fisher Information Matrix: {e}")
                print("Recalculating Fisher Information Matrix...")
                fisher_info = self._calculate_and_save_fisher(dataset)
        else:
            print("Fisher Information Matrix not found. Calculating...")
            fisher_info = self._calculate_and_save_fisher(dataset)
        return fisher_info

    def _calculate_and_save_fisher(self, dataset):
        """Calculate Fisher Information Matrix and save it to a file."""
        fisher_info = self._calculate_fisher(dataset)
        try:
            fisher_info_cpu = {n: f.to('cpu') for n, f in fisher_info.items()}  # Save to CPU for portability
            torch.save(fisher_info_cpu, self.fisher_path)
            print(f"Fisher Information Matrix saved to {self.fisher_path}")
        except Exception as e:
            print(f"Failed to save Fisher Information Matrix: {e}")
        return fisher_info

    def _calculate_fisher(self, dataset):
        """Calculate Fisher Information Matrix using the original dataset."""
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters() if p.requires_grad
        }

        self.model.eval()
        for batch in tqdm(dataset, desc="Calculating Fisher Information"):
            self.model.zero_grad()
            output = self.model(**batch)
            loss = output.loss
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += p.grad.data ** 2 / len(dataset)
        return fisher
    
    def ewc_loss(self):
        """Calculate EWC penalty loss"""
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # Add L2 penalty weighted by Fisher Information
                loss += (self.fisher_info[n] * (p - self.old_params[n]) ** 2).sum()
        return self.importance * loss

# Define a custom dataset class
class QADataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Pre-tokenize the data
def tokenize_data(dataframe, tokenizer: PreTrainedTokenizer, max_seq_length: int, device: str):
    tokenized_data = []
    for _, row in dataframe.iterrows():
        question = row['Question']
        answer = row['Answer']

        # Tokenize the question
        inputs = tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )

        # Tokenize the answer as labels
        labels = tokenizer(
            answer,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )['input_ids'].squeeze()

        tokenized_data.append({
            'input_ids': inputs['input_ids'].squeeze().to(device),
            'attention_mask': inputs['attention_mask'].squeeze().to(device),
            'labels': labels.to(device)
        })

    return tokenized_data

def tokenize_wiki(dataset, tokenizer: PreTrainedTokenizer, max_seq_length: int, device: str):
    tokenized_data = []
    for row in dataset:
        data = row['text']
        # Split data in chunks according to max_seq_length
        chunks = [data[i:i + max_seq_length] for i in range(0, len(data), max_seq_length)]
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )

            tokenized_data.append({
                'input_ids': inputs['input_ids'].squeeze().to(device),
                'attention_mask': inputs['attention_mask'].squeeze().to(device),
                'labels': inputs['input_ids'].squeeze().to(device)
            })
    # Only return 10 percent of the data
    tokenized_data = tokenized_data[:int(len(tokenized_data) * 0.1)]
    return tokenized_data

device = "mps"
# 3. Load Pre-trained Model and Tokenizer
print("Loading model...")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Change this to any pre-trained model name, e.g., "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)  # Move model to device (CPU or GPU)

exp_name = "Supervised"
timestamp = time.time()

writer = SummaryWriter(log_dir=f"./tensorboard_logs/{exp_name}_{timestamp}")

print("Loading dataset...")
dataframe = pd.read_csv('dataset.csv')

wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")

max_seq_length = 512
tokenized_based = tokenize_wiki(wikitext, tokenizer, max_seq_length, device)

tokenized_data = tokenize_data(dataframe, tokenizer, max_seq_length, device)

wiki_dataset = QADataset(tokenized_based)
train_loader_wiki = DataLoader(wiki_dataset, batch_size=8, shuffle=False)

print("Loading EWC...")
ewc = EWC(model, dataset=train_loader_wiki)
del train_loader_wiki

# Create dataset
qa_dataset = QADataset(tokenized_data)
# Create DataLoader
train_loader = DataLoader(qa_dataset, batch_size=8, shuffle=True)

print("Loading optimizer and warming up...")
# 5. Define Optimizer and Training Parameters
optimizer = AdamW(
        model.parameters(),
        lr=5e-6,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )
epochs = 3
logging_steps = 100
warmup_steps = 200
gradient_steps = 4

total_steps = len(train_loader) * epochs

# 6. Training Loop
model.train()
global_step = 0
optimizer.zero_grad()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)

        ewc_penalty = ewc.ewc_loss()

        task_loss = outputs.loss
        loss = task_loss + ewc_penalty
        print(ewc_penalty, task_loss)
        loss.backward()

        writer.add_scalar('Loss/Total', loss.item(), global_step)
        writer.add_scalar('Loss/Task', task_loss.item(), global_step)
        writer.add_scalar('Loss/EWC', ewc_penalty.item(), global_step)
        global_step += 1

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        if step % logging_steps == 0 and step > 0:
            print(f"Step {step}: Loss = {loss.item()}")

        if (step % gradient_steps == 0 and step > 0) or step == len(train_loader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.1)
            optimizer.step()
            optimizer.zero_grad()

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

# 7. Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model fine-tuned and saved to ./fine_tuned_model")