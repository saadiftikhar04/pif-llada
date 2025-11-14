"""Patch to fix generation parameters - just updates the generation calls."""

# Original problematic generation:
# output = generate(model, tokenizer, [input_ids[0]], 
#                  steps=4, max_new_tokens=100, block_length=32, temperature=0.0)

# Fixed generation (paste this into run_pif_llada_CORRECTED.py):

def generate_response(model, tokenizer, input_ids, max_new_tokens=100):
    """Generate with proper parameters."""
    from dllm.pipelines.llada import generate
    
    output = generate(
        model, 
        tokenizer, 
        [input_ids[0]],
        steps=16,  # Increased from 4 to 16 for better quality
        max_new_tokens=max_new_tokens,
        block_length=32,
        temperature=0.7,  # Add some randomness
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

print("Use steps=16 and temperature=0.7 for better generation")
