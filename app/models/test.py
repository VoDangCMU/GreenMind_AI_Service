from .gemini import get_gemini 

gem = get_gemini()

resp = gem.chat("Naruto là ai?", max_output_tokens=512)

print(resp)
