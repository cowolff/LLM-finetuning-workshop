import os
import random
import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class QnAEnvironment:
    """
    A toy single-step environment for Q&A.
    
    - State: A tuple of (question, context, reference_answer).
    - Action: A generated answer string.
    - Reward: Computed via BLEURT similarity to the reference answer.
    
    We'll cycle through the dataset sample by sample.
    """
    def __init__(self, dataset, bleurt_metric):
        self.dataset = dataset
        self.bleurt = bleurt_metric
        self.size = len(dataset)
        self.current_idx = 0

    def reset(self):
        """Shuffle the dataset and reset the index to 0."""
        self.dataset = self.dataset.shuffle(seed=42)
        self.current_idx = 0

    def get_state(self):
        """Return the next sample as the 'state'."""
        if self.current_idx >= self.size:
            return None, None, None  # signal that we are out of samples
        sample = self.dataset[self.current_idx]
        self.current_idx += 1
        
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]
        return input_ids, attention_mask, labels

    def compute_reward(self, generated_answer, reference_answer):
        """
        Compute the reward by comparing generated_answer with reference_answer using BLEURT.
        """
        scores = self.bleurt.compute(
            references=[reference_answer], 
            predictions=[generated_answer]
        )
        # We could use the BLEURT score directly, or clip/scale it, etc.
        return scores["scores"][0]


class QnAPolicy(nn.Module):
    """
    Wrap a causal language model (like GPT-2) as a 'policy'.
    
    We'll store:
        - The model (for forward passes).
        - A method to generate an answer from (question, context).
        - A method to compute log probabilities of the generated tokens.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", lr=1e-5, device="mps"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def generate_answer(self, input_ids, attention_mask, max_length=50, seed=42):
        """
        Generate an answer from the model with a fixed random seed
        for reproducibility.
        """
        random.seed(seed)

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=len(input_ids) + max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_beams=1
        )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract the segment after "Answer:"
        if "Answer:" in generated_text:
            answer_candidate = generated_text.split("Answer:")[-1].strip()
        else:
            answer_candidate = generated_text
        return answer_candidate

    def compute_log_prob_of_sequence(self, prompt_text, generated_answer):
        """
        Compute the log probability (negative cross-entropy) of the generated_answer
        under the model, given the prompt_text. This is used for REINFORCE.
        
        Return: scalar (log_prob), shape ()
        """
        # Full text = prompt + generated answer
        full_text = prompt_text + generated_answer
        
        input_ids = self.tokenizer.encode(
            full_text, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs.logits
        
        prompt_ids = self.tokenizer.encode(prompt_text)
        prompt_len = len(prompt_ids)
        
        # The relevant portion of logits for the answer:
        # logits[:, prompt_len-1 : -1, :]  (shifting by 1 for next-token prediction)
        # But simpler here: we’ll just do next-token prediction for the entire sequence
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # This yields a per-token loss
        per_token_loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Negative cross-entropy is the log_prob. We want the sum or mean?
        # For REINFORCE, we typically sum log probabilities of the chosen tokens.
        # We’ll just do the mean for demonstration.
        log_prob = -per_token_loss.mean()
        return log_prob

    def reinforce_update(self, log_prob, reward):
        """
        Single-step REINFORCE update:
            loss = - (reward * log_prob)
        
        We'll treat log_prob as a *scalar*, which is the average log prob of all
        tokens in the generated sequence.
        """
        loss = -(reward * log_prob)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

# Define a custom dataset class
class QADataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def shuffle(self, seed=None):
        random.shuffle(self.data)
        return self

def tokenize_data(dataframe, tokenizer: PreTrainedTokenizer, max_seq_length: int, device: str):
    tokenized_data = []
    for _, row in dataframe.iterrows():
        question = row['Question']
        answer = row['Answer']

        try:
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
                f"{question}: {answer}",
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )['input_ids'].squeeze(0)  # shape: [max_seq_length]

            tokenized_data.append({
                'input_ids': inputs['input_ids'].squeeze(0).to(device),
                'attention_mask': inputs['attention_mask'].squeeze(0).to(device),
                'labels': labels.to(device)
            })
        except Exception as e:
            print(f"Error tokenizing: {e}")
            continue

    return tokenized_data

if __name__ == "__main__":

    device = "mps"
    
    bleurt = evaluate.load("bleurt", "bleurt-20")
    
    policy = QnAPolicy(lr=1e-5)

    dataframe = pd.read_csv('dataset.csv')
    max_seq_length = 512

    tokenized_data = tokenize_data(dataframe, policy.tokenizer, max_seq_length, device)
    qa_dataset = QADataset(tokenized_data)

    env = QnAEnvironment(qa_dataset, bleurt)
    
    EPOCHS = 1    # for demonstration
    SAMPLES = 20  # train on a small subset
    
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch} ===")
        env.reset()  # shuffle the dataset each epoch
        
        for i in range(SAMPLES):
            # 1) Get environment state
            input_ids, attention_mask, label = env.get_state()
            if input_ids is None:
                break  # no more samples
            
            answer1 = policy.generate_answer(input_ids, attention_mask, seed=42)
            answer2 = policy.generate_answer(input_ids, attention_mask, seed=43)
            
            label_text = policy.tokenizer.decode(label, return_tensors="pt")
            prompt_text = policy.tokenizer.decode(input_ids, return_tensors="pt")

            r1 = env.compute_reward(answer1, label_text)
            r2 = env.compute_reward(answer2, label_text)

            if r1 >= r2:
                chosen_answer = answer1
                chosen_reward = r1
                chosen_seed = 42
            else:
                chosen_answer = answer2
                chosen_reward = r2
                chosen_seed = 43
            
            log_prob = policy.compute_log_prob_of_sequence(prompt_text, chosen_answer)
            
            loss_value = policy.reinforce_update(log_prob, chosen_reward)

            if i % 5 == 0:
                print(f"Sample {i} | Chosen seed: {chosen_seed} | "
                      f"Reward: {chosen_reward:.3f} | Loss: {loss_value:.4f}")

    print("Training complete!")
