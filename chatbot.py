import pandas as pd
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Load the dataset
df = pd.read_csv("parquetreader.csv")
questions = df['question'].astype(str).tolist()
answers = df['answers'].astype(str).tolist()

# Load multilingual model
print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encode all questions in dataset
print("Encoding questions...")
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Ask user for preferred language
def get_user_language():
    print("Please enter your preferred language code (e.g., 'en' for English, 'hi' for Hindi, 'bn' for Bengali):")
    lang = input("Language: ").strip().lower()
    if not lang:
        lang = 'en'
    return lang

# Function to get response
def get_response(user_input, user_lang):
    query_embedding = model.encode([user_input], convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    best_match_index = similarities.argmax().item()
    best_score = similarities[best_match_index].item()

    if best_score < 0.3:
        return "माफ़ कीजिए, मैं उस प्रश्न को समझ नहीं पाया।" if user_lang == 'hi' else "Sorry, I don't understand that question."

    response = answers[best_match_index]

    if user_lang != 'en':
        try:
            response = GoogleTranslator(source='en', target=user_lang).translate(response)
        except:
            pass

    return response

# Command-line chatbot
if __name__ == "__main__":
    print("Multilingual Chatbot is ready! Type 'exit' to quit.")
    user_lang = get_user_language()
    while True:
        inp = input("You: ")
        if inp.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        print("Bot:", get_response(inp, user_lang))
