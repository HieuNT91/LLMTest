import os
import csv
import time
import psutil
import torch  # Assuming you're using a PyTorch model
from transformers import pipeline

# Initialize model and tokenizer
model_id = "/storage/hiu/llm/Llama-3.2-1B-Instruct/"
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)

# Read instructions from file
def read_instructions(file_path):
    with open(file_path, 'r') as file:
        instructions = file.readlines()
    return [line.strip() for line in instructions if line.strip()]

# Generate response using the language model
def generate_response(prompt):
    return pipe(prompt, max_new_tokens=100)[0]['generated_text']

# Measure the memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Convert to MB

# Save results to CSV
def save_results_to_csv(data, file_path):
    keys = data[0].keys()
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

# Main function
def main():
    instructions = read_instructions('instructions.txt')
    prompt_template = "<|begin_of_text|>{}"

    results = []
    model_metrics = []

    # Get model properties
    model_memory_usage = get_memory_usage()
    total_layers = len(list(pipe.model.parameters()))
    hidden_size = pipe.model.config.hidden_size

    for instruction in instructions:
        prompt = prompt_template.format(instruction)

        # Record start time and memory usage
        start_time = time.time()
        initial_memory = get_memory_usage()

        # Generate response
        response = generate_response(prompt)

        # Record end time and memory usage
        end_time = time.time()
        final_memory = get_memory_usage()

        # Calculate metrics
        time_taken = end_time - start_time
        memory_used = final_memory - initial_memory

        # Log the result
        result = {
            "Prompt": prompt,
            "Instruction": instruction,
            "Response": response,
            "TimeTaken": time_taken,
            "MemoryUsage": memory_used
        }
        results.append(result)

    # Save results
    model_name = model_id.replace("/", "-")
    os.makedirs(model_name, exist_ok=True)
    save_results_to_csv(results, os.path.join(model_name, 'results.csv'))

    # Save model metrics
    model_metrics.append({
        "ModelMemoryUsage": model_memory_usage,
        "TotalLayers": total_layers,
        "HiddenSize": hidden_size
    })
    save_results_to_csv(model_metrics, os.path.join(model_name, 'model_metrics.csv'))

if __name__ == "__main__":
    main()
