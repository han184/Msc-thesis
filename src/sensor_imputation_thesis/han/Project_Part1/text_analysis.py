import torch
import re
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizerFast, BertModel

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Initialize BERT with Fast tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. Minimal Text Cleaning (with stopword removal)
def clean_text(text, remove_stopwords=True):
    text = str(text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_stopwords:
        tokens = text.split()
        tokens = [w for w in tokens if w.lower() not in stop_words]
        text = " ".join(tokens)

    return text

df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x, remove_stopwords=True))

# 3. BERT Co-occurrence Analysis
def analyze_bert_cooccurrences(texts, target_word="data", window_size=3, batch_size=8):
    target_word = target_word.lower()
    cooccurrence_counts = defaultdict(int)

    for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing texts"):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )

        for j, text in enumerate(batch):
            word_ids = inputs.word_ids(batch_index=j)
            input_ids = inputs.input_ids[j]

            words = []
            current_tokens = []
            current_word_id = None

            for k, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                if word_id != current_word_id:
                    if current_tokens:
                        words.append("".join(current_tokens))
                    current_tokens = []

                token_text = tokenizer.convert_ids_to_tokens(int(input_ids[k]))
                token_text = token_text.replace("##", "")
                current_tokens.append(token_text)
                current_word_id = word_id

            if current_tokens:
                words.append("".join(current_tokens))

            target_positions = [idx for idx, w in enumerate(words) if w.lower() == target_word]

            for pos in target_positions:
                context_words = set()
                for k in range(max(0, pos-window_size), min(len(words), pos+window_size+1)):
                    if k == pos:
                        continue
                    word = words[k].lower()
                    if word.isalpha() and word != target_word and word not in stop_words:
                        context_words.add(word)

                for word in context_words:
                    cooccurrence_counts[word] += 1

    return cooccurrence_counts

# 4. Run Analysis
cooccurrence_counts = analyze_bert_cooccurrences(df['cleaned_text'].tolist())

top_cooccurring = sorted(cooccurrence_counts.items(), key=lambda x: (-x[1], x[0]))[:30]
print("\nTop 30 words co-occurring with 'data':")
print("{:<15} {:<10}".format("Word", "Count"))
print("-"*25)
for word, count in top_cooccurring:
    print("{:<15} {:<10}".format(word, count))

# 5. Contextual Embeddings (using word_ids for alignment)
def get_contextual_embeddings(texts, target_word="data"):
    target_embeddings = []

    for text in tqdm(texts, desc="Extracting embeddings"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )

        word_ids = inputs.word_ids(batch_index=0)
        input_ids = inputs.input_ids[0]

        words = []
        word_to_token_idxs = {}
        current_tokens = []
        current_word_id = None
        reconstructed_index = -1

        for k, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            if word_id != current_word_id:
                if current_tokens:
                    words.append("".join(current_tokens))
                    reconstructed_index += 1
                    word_to_token_idxs[reconstructed_index] = token_positions
                current_tokens = []
                token_positions = []

            token_text = tokenizer.convert_ids_to_tokens(int(input_ids[k]))
            token_text = token_text.replace("##", "")
            current_tokens.append(token_text)
            token_positions.append(k)
            current_word_id = word_id

        if current_tokens:
            words.append("".join(current_tokens))
            reconstructed_index += 1
            word_to_token_idxs[reconstructed_index] = token_positions

        for idx, w in enumerate(words):
            if w.lower() == target_word:
                for token_idx in word_to_token_idxs[idx]:
                    target_embeddings.append(outputs.last_hidden_state[0, token_idx].cpu().numpy())

    return np.array(target_embeddings)
