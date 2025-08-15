import pandas as pd
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# ----------------------------
# 1. Load the dataset
# ----------------------------
df = pd.read_csv("parquetreader.csv")

# Convert columns to strings (avoiding NaN issues) and store as lists
questions = df['question'].astype(str).tolist()
answers = df['answers'].astype(str).tolist()

# ----------------------------
# 2. Load the multilingual model
# ----------------------------
print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ----------------------------
# 3. Encode all dataset questions once (for fast search)
# ----------------------------
print("Encoding questions...")
question_embeddings = model.encode(questions, convert_to_tensor=True)

# ----------------------------
# 4. Ask user for preferred language
# ----------------------------
def get_user_language():
    """
    Ask the user for preferred language code (ISO format).
    Defaults to English if left blank.
    """
    print("Please enter your preferred language code (e.g., 'en' for English, 'hi' for Hindi, 'bn' for Bengali):")
    lang = input("Language: ").strip().lower()
    if not lang:
        lang = 'en'
    return lang

# ----------------------------
# 5. Core function to get chatbot response
# ----------------------------
def get_response(user_input, user_lang):
    """
    Processes user input, finds best matching question in dataset,
    and returns the translated answer.
    """
    # Step 1: Translate user input to English for better matching
    if user_lang != 'en':
        try:
            translated_input = GoogleTranslator(source=user_lang, target='en').translate(user_input)
        except:
            translated_input = user_input  # Fallback to original if translation fails
    else:
        translated_input = user_input

    # Step 2: Encode translated input into embeddings
    query_embedding = model.encode([translated_input], convert_to_tensor=True)

    # Step 3: Compute similarity with all questions
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]

    # Step 4: Find best match
    best_match_index = similarities.argmax().item()
    best_score = similarities[best_match_index].item()

    # Step 5: Low similarity → don't understand
    if best_score < 0.3:
        return "माफ़ कीजिए, मैं उस प्रश्न को समझ नहीं पाया।" if user_lang == 'hi' else "Sorry, I don't understand that question."

    # Step 6: Get matching answer
    response = answers[best_match_index]

    # Step 7: Translate answer to user's preferred language (if needed)
    if user_lang != 'en':
        try:
            response = GoogleTranslator(source='en', target=user_lang).translate(response)
        except:
            pass  # Keep English answer if translation fails

    return response

# ----------------------------
# 6. Main chatbot loop
# ----------------------------
if __name__ == "__main__":
    print("Multilingual Chatbot is ready! Type 'exit' to quit.")
    user_lang = get_user_language()

    while True:
        inp = input("You: ")
        if inp.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        print("Bot:", get_response(inp, user_lang))
