"""Check if LLaDA diffusion algorithm is working correctly."""

# Step 1: Check dllm's generate function
print("Checking dllm LLaDA generation implementation...")
print("="*70)

import inspect
from dllm.pipelines.llada import generate

# Check function signature
sig = inspect.signature(generate)
print(f"\nGenerate function signature:")
print(f"  {sig}")

# Check source location
print(f"\nSource file: {inspect.getfile(generate)}")

# Step 2: Read the actual implementation
print("\n" + "="*70)
print("Reading generate() implementation...")
print("="*70)

with open(inspect.getfile(generate), 'r') as f:
    content = f.read()
    
# Find the generate function
import re
match = re.search(r'def generate\(.*?\):\s*"""(.*?)"""', content, re.DOTALL)
if match:
    print(f"\nDocstring:\n{match.group(1)}")

# Check for key diffusion steps
print("\nKey implementation checks:")
print(f"  Has 'steps' parameter: {'steps' in content}")
print(f"  Has 'max_new_tokens': {'max_new_tokens' in content}")
print(f"  Has diffusion loop: {'for step in' in content or 'for t in' in content}")
print(f"  Has masking: {'mask' in content.lower()}")
print(f"  Has denoising: {'denoise' in content.lower() or 'unmask' in content.lower()}")

# Step 3: Check what the function actually returns
print("\n" + "="*70)
print("Testing actual generation...")
print("="*70)

from dllm.utils import get_tokenizer, get_model
from dataclasses import dataclass

@dataclass
class ModelArgs:
    model_name_or_path: str = "/scratch/si2356/models/llada-1.5-8b"

model_args = ModelArgs()
tokenizer = get_tokenizer(model_args=model_args)
model = get_model(model_args=model_args).eval()
if tokenizer.mask_token_id is None:
    tokenizer.mask_token_id = model.config.mask_token_id

# Test prompt
test_prompt = "The capital of France is"
print(f"\nTest prompt: '{test_prompt}'")

input_ids = tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(model.device)
print(f"Input IDs shape: {input_ids.shape}")
print(f"Input IDs: {input_ids[0].tolist()[:10]}... (first 10)")

# Generate with different steps
for steps in [4, 8, 16, 32]:
    print(f"\n--- Testing with steps={steps} ---")
    
    output = generate(
        model,
        tokenizer,
        [input_ids[0]],
        steps=steps,
        max_new_tokens=20,
        block_length=20,
        temperature=0.7,
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Output IDs: {output[0].tolist()[:15]}... (first 15)")
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Decoded: '{response}'")
    
    # Check if it's just echoing
    if response.strip() == test_prompt.strip():
        print("  ⚠️  ECHOING INPUT!")
    elif response.startswith(test_prompt):
        new_text = response[len(test_prompt):].strip()
        if len(new_text) > 5:
            print(f"  ✓ Generated: '{new_text[:50]}...'")
        else:
            print(f"  ⚠️  Very short generation: '{new_text}'")
    else:
        print(f"  ? Unexpected output")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Step 4: Check if the problem is in how we're calling it
print("\nPossible issues:")
print("1. max_new_tokens might not be working")
print("2. Output might include input tokens")
print("3. Diffusion steps might be too low")
print("4. Temperature/sampling not working")
print("\nRecommendation:")
print("- Check if output includes input IDs")
print("- Verify diffusion loop is actually running")
print("- Test with very high steps (64+)")

