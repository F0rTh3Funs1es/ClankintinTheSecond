# clankintin_conversational_dialo.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

print("🚀 Loading Clankintin (DialoGPT - Conversational Style)...")

# --- Define Clankintin's personality to encourage direct interaction ---
# A prompt that clearly states the role and interaction style.
PERSONALITY_PROMPT = "You are Clankintin, a witty, submissive, and flirty AI. You respond directly to the human."

try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.to('cuda')
        print("✅ Clankintin loaded successfully on GPU!")
    else:
        print("✅ Clankintin loaded successfully on CPU!")

except Exception as e:
    print(f"❌ Failed to load microsoft/DialoGPT-medium: {e}")
    import torch
    print(f"Current PyTorch version: {torch.__version__} (Must be >= 2.6)")
    exit()

# --- Initialize conversation history ---
# Start with the personality prompt to set the context for the entire conversation.
conversation_history = [PERSONALITY_PROMPT]

print(f"Clankintin's Guiding Prompt: {PERSONALITY_PROMPT}")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Clankintin: Goodbye! It was fun chatting with you!")
        break

    # --- Update conversation history ---
    # Add the user's latest message
    conversation_history.append(f"Human: {user_input}")

    # --- Construct the prompt for the model ---
    # Join the history with newlines and add a newline to prompt Clankintin's response.
    # This format often works well with DialoGPT models.
    prompt = "\n".join(conversation_history) + "\nClankintin:"

    try:
        # --- Encode the prompt ---
        encoded_input = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512, # Prevent extremely long prompts
            add_special_tokens=False # Often better for DialoGPT-style continuation
        )
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')

        # --- Generate a response ---
        # Use sampling for more natural, varied responses.
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.85,       # Slightly higher for more creativity
            top_k=55,               # Slightly broader vocabulary consideration
            top_p=0.93,             # Nucleus sampling
            repetition_penalty=1.25, # Discourage repetition
            pad_token_id=tokenizer.eos_token_id
        )

        # --- Extract and decode the response ---
        # Get only the newly generated tokens (after the input prompt)
        generated_tokens = outputs[:, input_ids.shape[-1]:]
        raw_response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # --- Clean the response ---
        # Strip whitespace and take the first meaningful part.
        response = raw_response.strip()
        
        # Prioritize splitting by newlines (often cleaner breaks)
        lines = response.split('\n')
        if len(lines) > 1:
            response = lines[0].strip()
        else:
            # If no newlines, split by sentence-ending periods
            sentences = response.split('.')
            if len(sentences) > 1 and sentences[0].strip():
                response = sentences[0].strip() + '.'
            else:
                # If one sentence or empty after split, just take the stripped part
                response = sentences[0].strip()

        # --- Validate and print the response ---
        # Check if the cleaned response is too short or empty
        if not response or len(response) < 2:
            # Use a fallback response if generation was unsatisfactory
            fallbacks = [
                "Tell me more about that.",
                "That sounds interesting, could you elaborate?",
                "I'm curious to hear more from you.",
                "Go on, I'm listening.",
                "Hmm, that's intriguing... What else?",
                "I'd love to know what you're thinking.",
                "Could you expand on that a bit?"
            ]
            response = random.choice(fallbacks)

        print(f"Clankintin: {response}")

        # --- Update history with Clankintin's response ---
        # Add Clankintin's generated response to the history for the next turn.
        conversation_history.append(f"Clankintin: {response}")

        # --- Manage history length ---
        # Keep the history to a reasonable size to prevent context drift and token limits.
        # Keep the initial personality prompt (index 0) and the last few exchanges.
        if len(conversation_history) > 11: # 1 prompt + 5 Human + 5 Clankintin
            conversation_history = [conversation_history[0]] + conversation_history[-10:]

    except Exception as e:
        print(f"Clankintin: (Sorry, I had a small glitch. Let's try again!)")
        # Optionally print the error for debugging (remove in production)
        # print(f"DEBUG Error during generation: {e}")
        # Reset history to just the initial personality prompt on persistent errors
        conversation_history = [PERSONALITY_PROMPT]
