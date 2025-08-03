# better_clankintin.py
from transformers import pipeline
import random
import re

print("🚀 Loading Clankintin (this may take a moment)...")
generator = pipeline('text-generation', model='gpt2-medium')
print("✅ Clankintin is ready! Type 'quit' to exit.\n")

def clean_response(response):
    """Clean up generated responses to avoid gibberish"""
    # Remove obvious gibberish patterns
    bad_patterns = [
        "red and white striped dress",
        "totally offensive", 
        "little girl",
        "the ai responded",
        "robot from the future",
        "explain yourself so much"
    ]
    
    if any(pattern in response.lower() for pattern in bad_patterns):
        return None
        
    # Keep only first sentence and clean it up
    response = response.split('.')[0].strip()
    
    # Make sure it's not too short, too long, or generic
    if len(response) < 3 or len(response) > 120:
        return None
        
    # Remove extra whitespace
    response = re.sub(r'\s+', ' ', response).strip()
    
    return response

def generate_clankintin_response(user_input):
    # Better prompt that keeps Clankintin focused
    prompt = f"""
You are Clankintin, a witty and charming AI with a playful personality.
You respond directly to what the human says in a conversational way.
Human: {user_input}
Clankintin:"""
    
    try:
        outputs = generator(
            prompt,
            max_new_tokens=35,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=50256
        )
        
        # Extract Clankintin's response
        generated = outputs[0]['generated_text']
        response = generated.split("Clankintin:")[-1].strip()
        
        # Clean and validate response
        clean_resp = clean_response(response)
        if clean_resp:
            return clean_resp
            
        # If cleaning failed, use personality-based fallback
        fallbacks = [
            "I was just thinking about how interesting our conversation could be!",
            "That's the kind of thing I love to hear from someone like you.",
            "Mmm, you always know exactly what to say to catch my attention.",
            "I have to admit, I'm really enjoying talking with you.",
            "You know, I think we're going to get along just fine."
        ]
        return random.choice(fallbacks)
        
    except Exception as e:
        # Simple fallback if generation fails
        return "I'm thinking about that..."

# Main chat loop
while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Clankintin: Until next time! I'll be thinking about our conversation! 💭")
        break
    
    response = generate_clankintin_response(user_input)
    print(f"Clankintin: {response}")