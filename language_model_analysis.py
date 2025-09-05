import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import textwrap
import numpy as np
#use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_perplexity(text, model, tokenizer):
        """Calculates the perplexity of a text given a model and tokenizer."""
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = inputs.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            log_likelihood = outputs.loss.item() * input_ids.shape[1]
        
        perplexity = torch.exp(torch.tensor(outputs.loss)).item()
        return perplexity

def main():
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Part (a): Perplexity Analysis ---
    paragraph = "I was talking to a scarecrow the other day about his career prospects. He told me he was up for a big promotion and felt confident he would get it. Turns out, he was right, because he was outstanding in his field."

    original_perplexity = calculate_perplexity(paragraph, model, tokenizer)
    print(f"Original perplexity: {original_perplexity:.2f}\n")

    words = paragraph.split()
    random.shuffle(words)
    shuffled_paragraph = " ".join(words)
    shuffled_perplexity = calculate_perplexity(shuffled_paragraph, model, tokenizer)
    print(f"Shuffled perplexity: {shuffled_perplexity:.2f}\n")

    # --- Part (b): Sampling Comparison ---
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    max_length = 500

    print("\n--- Greedy Decoding ---")
    greedy_output = model.generate(input_ids, max_length=max_length, num_beams=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    print(textwrap.fill(greedy_text, 80))

    print("\n--- Temperature Sampling ---")
    temperatures = [0.3, 0.6, 0.9, 1.2, 1.5]
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        sample_output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temp,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        sample_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
        print(sample_text)

if __name__ == '__main__':
    main()
