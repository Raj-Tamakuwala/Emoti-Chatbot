import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from textblob import TextBlob
import pandas as pd
import requests
from io import StringIO
import gradio as gr
import speech_recognition as sr
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader

# --- Data Cleaning and Preprocessing ---
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)         # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)                     # Remove @ and #
    text = re.sub(r'[^a-z\s]', '', text)                     # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()                 # Normalize spaces
    return text

# --- Load datasets ---
df = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=14D_HcvTFL63-KffCQLNFxGH-oY_knwmo",
    delimiter=';', header=None, names=['sentence', 'label']
)
ts_df = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1Vmr1Rfv4pLSlAUrlOCxAcszvlxJOSHrm",
    delimiter=';', header=None, names=['sentence', 'label']
)
df = pd.concat([df, ts_df], ignore_index=True)
df.drop_duplicates(inplace=True)
df['clean_sentence'] = df['sentence'].apply(clean_text)

# --- Build Vocabulary ---
tokenized = df['clean_sentence'].apply(str.split)
vocab = Counter([token for sentence in tokenized for token in sentence])
vocab = {word: i+2 for i, (word, _) in enumerate(vocab.most_common())}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def encode(text):
    return [vocab.get(word, vocab['<UNK>']) for word in text]

encoded_texts = tokenized.apply(encode)

# --- Pad Sequences ---
MAX_LEN = 32
def pad_sequence(seq):
    return seq[:MAX_LEN] + [vocab['<PAD>']] * max(0, MAX_LEN - len(seq))
padded = encoded_texts.apply(pad_sequence).tolist()

# --- Encode Labels ---
le = LabelEncoder()
labels = le.fit_transform(df['label'])

# --- Dataset + DataLoader ---
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(padded, labels, test_size=0.2, stratify=labels, random_state=42)
train_loader = DataLoader(EmotionDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(EmotionDataset(X_val, y_val), batch_size=16)

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# --- Transformer Model with Masking + Dropout for Bayesian Inference ---
class EmotionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        mask = (x == vocab['<PAD>'])
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.dropout(x.mean(dim=1))  # mean pooling
        return self.fc(x)

# --- Train the Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionTransformer(len(vocab), embed_dim=64, num_heads=4, num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Accuracy: {correct / total:.4f}")

# Save model
torch.save(model.state_dict(), "emotion_transformer_model.pth")

# --- Load Solutions CSV ---
file_id = "1yVJh_NVL4Y4YqEXGym47UCK5ZNZgVZYv"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
csv_text = response.text

if csv_text.strip().startswith('<'):
    raise Exception("ERROR: Google Drive link is not returning CSV! Check your sharing settings.")

solutions_df = pd.read_csv(StringIO(csv_text), header=0, on_bad_lines='skip')

used_solutions = {emotion: set() for emotion in solutions_df['emotion'].unique()}
negative_words = [
    "not", "bad", "sad", "anxious", "anxiety", "depressed", "upset", "shit", "stress",
    "worried", "unwell", "struggling", "low", "down", "terrible", "awful",
    "nervous", "panic", "afraid", "scared", "tense", "overwhelmed", "fear", "uneasy"
]

responses = {
    "sadness": [
        "It’s okay to feel down sometimes. I’m here to support you.",
        "I'm really sorry you're going through this. Want to talk more about it?",
        "You're not alone — I’m here for you."
    ],
    "anger": [
        "That must have been frustrating. Want to vent about it?",
        "It's okay to feel this way. I'm listening.",
        "Would it help to talk through it?"
    ],
    "love": [
        "That’s beautiful to hear! What made you feel that way?",
        "It’s amazing to experience moments like that.",
        "Sounds like something truly meaningful."
    ],
    "happiness": [
        "That's awesome! What’s bringing you joy today?",
        "I love hearing good news. 😊",
        "Yay! Want to share more about it?"
    ],
    "neutral": [
        "Got it. I’m here if you want to dive deeper.",
        "Thanks for sharing that. Tell me more if you’d like.",
        "I’m listening. How else can I support you?"
    ]
}

relaxation_resources = {
    "exercise": "Try this 5-4-3-2-1 grounding method:\n- 5 things you see\n- 4 you can touch\n- 3 you hear\n- 2 you smell\n- 1 you taste",
    "video": "Here’s a short calming video that might help: https://youtu.be/O-6f5wQXSu8"
}

help_keywords = ["suggest", "help", "calm", "exercise", "relax", "how can i", "any tips", "can u", "can you"]
thank_you_inputs = ["thank", "thanks", "thank you"]
bye_inputs = ["bye", "goodbye", "see you", "take care", "ok bye", "exit", "quit"]

def correct_spelling(text):
    return str(TextBlob(text).correct())

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def is_negative_input(text):
    text_lower = text.lower()
    return any(word in text_lower for word in negative_words)

def get_unique_solution(emotion):
    available = solutions_df[solutions_df['emotion'] == emotion]
    unused = available[~available['solution'].isin(used_solutions[emotion])]
    if unused.empty:
        used_solutions[emotion] = set()
        unused = available
    solution_row = unused.sample(1).iloc[0]
    used_solutions[emotion].add(solution_row['solution'])
    return solution_row['solution']

def preprocess_input(text):
    tokens = text.lower().split()
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    padded = encoded[:MAX_LEN] + [vocab['<PAD>']] * max(0, MAX_LEN - len(encoded))
    return torch.tensor([padded], dtype=torch.long).to(next(model.parameters()).device)

def get_emotion(user_input):
    if is_negative_input(user_input):
        return "sadness"
    sentiment = get_sentiment(user_input)
    x = preprocess_input(user_input)
    model.train()
    with torch.no_grad():
        probs = torch.stack([F.softmax(model(x), dim=1) for _ in range(5)])
        avg_probs = probs.mean(dim=0)
        prob, idx = torch.max(avg_probs, dim=1)
    pred_emotion = le.classes_[idx.item()]
    if prob.item() < 0.6:
        return "neutral"
    if sentiment < -0.25 and pred_emotion == "happiness":
        return "sadness"
    if sentiment > 0.25 and pred_emotion == "sadness":
        return "happiness"
    return pred_emotion

def audio_to_text(audio_file):
    if audio_file is None:
        return ""
    recog = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recog.record(source)
    try:
        text = recog.recognize_google(audio)
        return text
    except Exception:
        return ""

# LLM API function
def call_llm_api(user_text):
    api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
    headers = {
    "Authorization": f"Bearer YOUR KEY"
}
    payload = {"inputs": user_text}
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=15)
        output = resp.json()
        if isinstance(output, dict) and 'error' in output:
            return "API error: " + str(output['error'])
        return str(output)
    except Exception as e:
        return f"API call failed: {e}"

GLOBAL_CONVO_HISTORY = []
USER_FEEDBACK_STATE = {}

def emoti_chat(audio, text, history_json=""):
    # --- Get user input from voice or text ---
    if text and text.strip():
        user_input = text
    elif audio is not None:
        user_input = audio_to_text(audio)
    else:
        user_input = ""
    if not user_input.strip():
        return "Please say something or type your message.", json.dumps(GLOBAL_CONVO_HISTORY[-5:], indent=2), ""

    user_input = correct_spelling(user_input)

    # --- Exit logic ---
    exit_phrases = ["exit", "quit", "goodbye", "bye", "close"]
    if user_input.lower().strip() in exit_phrases:
        return "Take care! I’m here whenever you want to talk. 👋", json.dumps(GLOBAL_CONVO_HISTORY[-5:], indent=2), gr.update(visible=False)

    # --- HuggingFace LLM API call for "fun fact" or "more about" ---
    if "fun fact" in user_input.lower() or "more about" in user_input.lower() or "api" in user_input.lower():
        api_reply = call_llm_api("Tell me a fun fact about AI.")
        return f"(LLM API response)\n{api_reply}", json.dumps(GLOBAL_CONVO_HISTORY[-5:], indent=2), ""

    # Feedback logic
    user_id = "default_user"
    state = USER_FEEDBACK_STATE.get(user_id, {"emotion": None, "pending": False})

    if state["pending"]:
        feedback = user_input.lower().strip()
        GLOBAL_CONVO_HISTORY[-1]["feedback"] = feedback
        if feedback == "no":
            suggestion = get_unique_solution(state["emotion"])
            reply = f"Here's another suggestion for you: {suggestion}\nDid this help? (yes/no/skip)"
            USER_FEEDBACK_STATE[user_id]["pending"] = True
            return reply, json.dumps(GLOBAL_CONVO_HISTORY[-5:], indent=2), ""
        else:
            USER_FEEDBACK_STATE[user_id] = {"emotion": None, "pending": False}
            return "How can I help you further?", json.dumps(GLOBAL_CONVO_HISTORY[-5:], indent=2), ""

    # Normal user message: get emotion, give suggestion
    pred_emotion = get_emotion(user_input)
    support = random.choice(responses.get(pred_emotion, responses["neutral"]))
    try:
        suggestion = get_unique_solution(pred_emotion)
    except Exception:
        suggestion = get_unique_solution("neutral")

    reply = f"{support}\n\nHere's a suggestion for you: {suggestion}\nDid this help? (yes/no/skip)"
    GLOBAL_CONVO_HISTORY.append({
        "user_input": user_input,
        "emotion": pred_emotion,
        "bot_support": support,
        "bot_suggestion": suggestion,
        "feedback": ""
    })
    USER_FEEDBACK_STATE[user_id] = {"emotion": pred_emotion, "pending": True}
    return reply, json.dumps(GLOBAL_CONVO_HISTORY[-5:], indent=2), ""

# ---- Gradio Web Interface ----
iface = gr.Interface(
    fn=emoti_chat,
    inputs=[
        gr.Audio(type="filepath", label="🎤 Speak your message"),
        gr.Textbox(lines=2, placeholder="Or type your message here...", label="💬 Type message"),
        gr.Textbox(lines=1, value="", visible=False)  # Hidden, passes history state
    ],
    outputs=[
        gr.Textbox(label="EmotiBot Reply"),
        gr.Textbox(label="Hidden", visible=False)
    ],
    title="EmotiBot Connect",
    description="Talk to EmotiBot using your voice or by typing. Detects your emotion, gives dynamic suggestions, remembers your feedback, and keeps a conversation history! Type 'fun fact' or 'api' for an AI-generated fact."
)

iface.launch()