import scipy
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertModel
from collections import defaultdict
from tqdm import tqdm
import numpy as np
# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

# Apply preprocessing to the text column
df['processed_text'] = df['text'].apply(preprocess_text)

# Top 10 lexical collocates analysis

# 1. Initialize BERT with Fast tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Using Fast tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. Text Cleaning
from nltk.corpus import stopwords
import re

# Initialize stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=True):
    """Enhanced cleaning for BERT analysis with stop word control"""
    text = str(text)

    # 1. Basic cleaning
    text = re.sub(r'[^\w\s.,!?:;]', '', text)  # Remove special chars except punctuation
    text = text.replace('\n', ' ').replace('\t', ' ')  # Remove line breaks

    # 2. Conditional stop word removal
    if remove_stopwords:
        # Tokenize preserving punctuation
        tokens = []
        current_word = []
        for char in text:
            if char.isalnum():
                current_word.append(char.lower())
            else:
                if current_word:
                    word = ''.join(current_word)
                    if word not in stop_words:  # Filter stopwords
                        tokens.append(word)
                    current_word = []
                if char in {'.', ',', '?', '!', ':', ';'}:  # Keep punctuation
                    tokens.append(char)

        # Add last word if exists
        if current_word:
            word = ''.join(current_word)
            if word not in stop_words:
                tokens.append(word)

        text = ' '.join(tokens)

    # 3. Final cleanup
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse multiple spaces

    return text

# Apply with stopword removal
df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x, remove_stopwords=True))

# 3. BERT Co-occurrence Analysis with Fast Tokenizer
def analyze_bert_cooccurrences(texts, target_word="data", window_size=3, batch_size=8):
    target_word = target_word.lower()
    cooccurrence_counts = defaultdict(int)

    for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing texts"):
        batch = texts[i:i+batch_size]

        # Tokenize with Fast tokenizer features
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,  # Now available
            return_attention_mask=True
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )

        # Process each text in batch
        for j in range(len(batch)):
            # Get word alignments
            word_ids = inputs.word_ids(batch_index=j)  # New Fast tokenizer feature
            input_ids = inputs.input_ids[j]

            # Reconstruct whole words from WordPieces
            words = []
            word_positions = {}  # map word index -> position in reconstructed list
            current_tokens = []
            current_word_id = None
            reconstructed_index = -1

            for k, word_id in enumerate(word_ids):
                if word_id is None:  # skip [CLS], [SEP], padding
                    continue

                if word_id != current_word_id:
                    if current_tokens:
                        words.append("".join(current_tokens))
                        #reconstructed_index += 1
                        #word_positions[current_word_id] = reconstructed_index
                    current_tokens = []

                token_text = tokenizer.convert_ids_to_tokens(int(input_ids[k]))
                token_text = token_text.replace("##", "")
                current_tokens.append(token_text)
                current_word_id = word_id

            # add the final word
            if current_tokens:
                words.append("".join(current_tokens))
                #reconstructed_index += 1
                #word_positions[current_word_id] = reconstructed_index

            # find positions of the target word in reconstructed words
            target_positions = [
        idx for idx, w in enumerate(words) if w.lower() == target_word
    ]

            # analyze context
            for pos in target_positions:
                context_words = set()
                for k in range(max(0, pos-window_size), min(len(words), pos+window_size+1)):
                    if k == pos:
                        continue
                    word = words[k].lower()
                    if word.isalpha() and word != target_word:
                        context_words.add(word)


                for word in context_words:
                    cooccurrence_counts[word] += 1

    return cooccurrence_counts

# 4. Run Analysis
cooccurrence_counts = analyze_bert_cooccurrences(df['cleaned_text'].tolist())

# 5. Display Results
top_cooccurring = sorted(cooccurrence_counts.items(),
                        key=lambda x: (-x[1], x[0]))[:10]

print("\nTop 10 words co-occurring with 'data':")
print("{:<15} {:<10}".format("Word", "Count"))
print("-"*25)
for word, count in top_cooccurring:
    print("{:<15} {:<10}".format(word, count))
