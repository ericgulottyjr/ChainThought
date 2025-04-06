import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Define the model identifier
model_name = "deepseek-ai/deepseek-llm-7b-base"

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load a small subset of the GSM8K test set for demonstration purposes
# You can increase the split (e.g., "test") once you are ready for a full evaluation.
dataset = load_dataset("gsm8k", split="test[:10]")  # Using first 10 examples for demo

def extract_answer(generated_text):
    """
    Extracts the final answer from the model output by splitting on the "Answer:" token.
    """
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    else:
        return generated_text.strip()

def clean_text(text):
    """
    Basic cleaning: remove non-alphanumeric characters and convert to lowercase.
    This helps in comparing the generated answer with the reference answer.
    """
    return re.sub(r'\W+', '', text).lower()

num_correct = 0
num_total = len(dataset)

for example in dataset:
    # Construct a chain-of-thought prompt using the problem statement.
    # Adjust field names ('question' and 'answer') if needed.
    prompt = (
        f"Problem: {example['question']}\n"
        "Solution (chain-of-thought): Let's break it down step by step.\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the model's output
    outputs = model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    
    # Decode and extract the final answer from the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer = extract_answer(generated_text)
    
    # Clean both the generated answer and the reference answer for comparison
    gen_ans_clean = clean_text(generated_answer)
    ref_ans_clean = clean_text(example['answer'])
    
    print(f"Prompt:\n{prompt}")
    print(f"Generated Answer: {generated_answer}")
    print(f"Reference Answer: {example['answer']}")
    print("-" * 50)
    
    # Check if the generated answer exactly matches the reference answer (after cleaning)
    if gen_ans_clean == ref_ans_clean:
        num_correct += 1

# Calculate and print the baseline accuracy
accuracy = num_correct / num_total
print(f"Baseline Accuracy on {num_total} examples: {accuracy * 100:.2f}%")
