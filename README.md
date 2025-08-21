# Multilingual Agriculture FAQ Chatbot

A simple **multilingual question-answering chatbot** that uses **Sentence Transformers** for semantic search and **Google Translator** for real-time language translation.  
It can understand questions in multiple languages, find the closest matching FAQ from the dataset and reply in the user's preferred language.

## Features

- **Multilingual Support** – Works with any language supported by Google Translate.
- **Semantic Search** – Uses [SentenceTransformers](https://www.sbert.net/) for high-quality question matching.
- **Fast Matching** – Pre-encodes dataset questions for quick responses.
- **Custom Dataset** – Works with own CSV of questions and answers.


## Project Structure

```
├── parquetreader.csv       # Dataset file (questions & answers)
├── chatbot.py              # Main chatbot script
└── README.md               # Documentation
````

## Dataset Format

The `dataset.csv` file should have **two columns**:

| question                 | answers               |
|--------------------------|-----------------------|
| What is your name?       | My name is Chatbot.   |
| How can I reset my password? | You can reset it from the settings page. |


### Example Interaction

```
Please enter your preferred language code (e.g., 'en' for English, 'hi' for Hindi, 'bn' for Bengali):
Language: hi
Multilingual Chatbot is ready! Type 'exit' to quit.
You: पासवर्ड कैसे रीसेट करें?
Bot: आप इसे सेटिंग्स पेज से रीसेट कर सकते हैं।
```


## How It Works

1. **Load Dataset** – Reads `dataset.csv` into memory.
2. **Sentence Embeddings** – Encodes all questions using `paraphrase-multilingual-MiniLM-L12-v2`.
3. **User Input** – Translates input into English for better matching.
4. **Semantic Search** – Finds the most similar question from the dataset.
5. **Answer Translation** – Translates the answer back to the user's preferred language.

## Requirements

* [pandas](https://pandas.pydata.org/)
* [sentence-transformers](https://www.sbert.net/)
* [deep-translator](https://pypi.org/project/deep-translator/)

## Notes

* Default similarity threshold is **0.3** – you can adjust it in `get_response()` if needed.
* If translation fails, the bot falls back to the original text.
* Works best with **clear and direct FAQs**.


## Future Improvements

* Add **vector database** support for larger datasets.
* Integrate **voice input/output** for a conversational feel.
* Deploy as a **web app** with Flask/FastAPI.
