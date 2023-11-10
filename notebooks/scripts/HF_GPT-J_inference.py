import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# Copy to the device and use FP16.
assert torch.cuda.is_available()
device = torch.device("cuda:1")
model.half().to(device)
model.eval()

# Generate the sentence.
for i in range(10):
    torch.cuda.nvtx.range_push(f"Iteration_{i}")
    output = model.generate(input_ids=None, max_length=32, num_return_sequences=1)
    torch.cuda.nvtx.range_pop()
    
# Output the text.
for sentence in output:
    sentence = sentence.tolist()
    text = tokenizer.decode(sentence, clean_up_tokenization_spaces=True)
    print(text)