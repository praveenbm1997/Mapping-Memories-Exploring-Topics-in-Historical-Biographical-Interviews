
from flask import Flask, render_template, request
import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel
import umap.umap_ as umap
import hdbscan
import re

app = Flask(__name__)

# Load the llama model and tokenizer
llama_model_name = 'meta-llama/Llama-3-8B'  # can change the model based on the computation power
llama_model = LlamaForCausalLM.from_pretrained(llama_model_name)
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)

# Load the sentence embedding model and tokenizer
embedding_model_name = 'jinaai/jina-embeddings-v2-base-de'
embedding_model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True)
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)

# List of common filler words
filler_words = {"ja", "hm", "mhm", "ach", "gut", "und", "eben", "ne", "ok", "aha", "ach so", "nicht", "ja", "nein", "wieder", "schon", "naja", "wieso", "wieso nicht", "wieso nicht"}

# Filtering criteria
min_words = 2
min_unique_words = 2
min_characters = 5

def has_repeated_filler_patterns(sentence, filler_words):
    """
    Check if any filler word appears repeatedly either consecutively
    or separated by commas or spaces in the sentence.
    """
    pattern = r'\b(' + '|'.join(re.escape(word) for word in filler_words) + r')\b(?:[\s,]+)+\1\b'
    
    if re.search(pattern, sentence):
        return True
    return False

def filter_short_and_filler_sentences(text, filler_words, min_words, min_unique_words, min_characters):
    """
    Filter sentences from the text based on length, unique words,
    minimum characters, and absence of repeated filler patterns.
    """
    sentences = text.split('. ')
    filtered_sentences = []

    for sentence in sentences:
        words = sentence.split()
        
        if len(words) >= min_words and len(set(words)) >= min_unique_words and len(sentence) >= min_characters:
            if not has_repeated_filler_patterns(sentence, filler_words):
                filtered_sentences.append(sentence)
    
    return '. '.join(filtered_sentences)

def generate_topic_name(sentences, model, tokenizer):
    """
    Generate a topic name for a cluster based on its sentences using llama or similar models.
    """
    # Join sentences into a single text
    text = " ".join(sentences)
    
    # Craft the prompt
    prompt = f"""You are a text summarization expert. Your task is to generate a concise and descriptive topic name in one or two words for the following text. 
    The topic name should capture the core idea or theme of the text. Text: "{text}" Topic Name:"""
   
    # Prepare input for the model
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)

    # Generate text using the model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,  # Limit the number of new tokens generated
            num_beams=5,
            early_stopping=True,
            temperature=0.7,  # Adjust temperature for creativity
            eos_token_id=tokenizer.eos_token_id  # Ensure generation stops appropriately
        )
    
    # Decode the generated text
    topic_name = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Extract only the topic name by removing prompt text if it reappears
    if "Text:" in topic_name:
        topic_name = topic_name.split("Text:")[-1].strip()
    if "Topic Name:" in topic_name:
        topic_name = topic_name.split("Topic Name:")[-1].strip()

    return topic_name


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        
        # Retrieve UMAP and HDBSCAN parameters from the form
        n_neighbors = int(request.form.get('n_neighbors', 4))
        n_components = int(request.form.get('n_components', 5))
        min_dist = float(request.form.get('min_dist', 0.1))
        min_cluster_size = int(request.form.get('min_cluster_size', 0))
        cluster_selection_epsilon = float(request.form.get('cluster_selection_epsilon', 0.2))

        # Filter and process the input text
        filtered_paragraph = filter_short_and_filler_sentences(text, filler_words, min_words, min_unique_words, min_characters)
        sentences = filtered_paragraph.split('. ')
        embeddings = []

        batch_size = 32
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            inputs = embedding_tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = embedding_model(**inputs).last_hidden_state
                batch_embeddings = outputs.mean(dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)
        
        # Apply UMAP for dimensionality reduction
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric='cosine',
            low_memory=True
        )
        umap_embeddings = umap_model.fit_transform(embeddings)
        
        # Apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        clusters = clusterer.fit_predict(umap_embeddings)
        
        # Create a list to hold clusters with their sentences and topic names
        clusters_list = []
        cluster_colors = ['#FFDDC1', '#FFABAB', '#FFC3A0', '#D5AAFF', '#B9FBC0', '#FFCBF2', '#E0F7FA', '#C5E1A5', '#FFAB91', '#CE93D8']
        
        for cluster in np.unique(clusters):
            cluster_id = int(cluster)
            sentences_in_cluster = [sentence for i, sentence in enumerate(sentences) if clusters[i] == cluster and "prompt" not in sentence]
            # Generate a topic name for the cluster
            topic_name = generate_topic_name(sentences_in_cluster, llama_tokenizer, llama_model)
            
            cluster_dict = {
                "cluster_id": cluster_id,
                "sentences": sentences_in_cluster,
                "topic_name": topic_name  # Add topic name to cluster data
            }
            clusters_list.append(cluster_dict)
            

        return render_template('clusters.html', clusters=clusters_list, cluster_colors=cluster_colors)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)