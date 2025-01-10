import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

def generate_response(
    model, 
    tokenizer, 
    prompt: str, 
    device: str = "mps", 
    max_length: int = 128, 
    temperature: float = 1.0, 
    top_k: int = 50, 
    top_p: float = 0.95, 
    do_sample: bool = True
):
    """Helper function to run inference on a single model."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compare_models(
    prompt: str,
    original_model_name: str = "Qwen/Qwen2.5-0.5B",   # Replace with whichever base model you used
    fine_tuned_model_dir: str = "./fine_tuned_model",
    device: str = "mps"
):
    """
    Loads the original (pretrained) model and the fine-tuned model, 
    generates responses to the same prompt, and prints both.
    """

    # 1. Load the original model
    print(f"Loading original model: {original_model_name}")
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    original_model = AutoModelForCausalLM.from_pretrained(original_model_name).to(device)
    original_model.eval()

    # 2. Load the fine-tuned model
    print(f"Loading fine-tuned model from: {fine_tuned_model_dir}")
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_dir).to(device)
    fine_tuned_model.eval()

    # 3. Generate response from the original model
    print("\nGenerating response from the original model...")
    original_response = generate_response(
        model=original_model,
        tokenizer=original_tokenizer,
        prompt=prompt,
        device=device
    )

    # 4. Generate response from the fine-tuned model
    print("Generating response from the fine-tuned model...")
    fine_tuned_response = generate_response(
        model=fine_tuned_model,
        tokenizer=fine_tuned_tokenizer,
        prompt=prompt,
        device=device
    )

    # 5. Print results side by side
    print("\nPROMPT:")
    print(prompt)
    print("\nORIGINAL MODEL RESPONSE:")
    print(original_response)
    print("\nFINE-TUNED MODEL RESPONSE:")
    print(fine_tuned_response)


if __name__ == "__main__":
    # Example usage:
    test_prompt = "What are some new cake decorating products that became available in the late 20th century?"
    compare_models(
        prompt=test_prompt,
        original_model_name="Qwen/Qwen2.5-0.5B",
        fine_tuned_model_dir="./fine_tuned_model",
        device="mps"
    )
