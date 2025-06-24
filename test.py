import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Try a simple prompt
model = genai.GenerativeModel('gemini-2.0-flash')

response = model.generate_content("Hello")
print(response)