from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from gensim import corpora
from langdetect import detect_langs, LangDetectException, DetectorFactory
import os
import glob
import gc
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go  # Import plotly's graph objects
from typing import List, Union
from IPython.display import display, HTML
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set a fixed seed for langdetect to make results deterministic
DetectorFactory.seed = 42

def load_data():
    """Load and prepare publication and working paper data"""
    # load data
    os.chdir('/Users/lhampton/Documents/Local_Projects/ailandscapeproject')

    os.chdir('/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/constructed/combined_topic_modeling')
    pub_output_dir = '/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/constructed/pub_topic_modeling'
    wp_output_dir = '/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/constructed/wp_topic_modeling'
    constr_wp_dir = '/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/constructed/wp_classification'
    constr_pub_dir = '/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/constructed/pub_classification'
    orig_wp_dir = "/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/original/working_papers"
    orig_pub_dir = "/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/original/published_papers"

    pub_data_full = pd.read_csv(f"{orig_pub_dir}/no_duplicates.csv", index_col = 0).reset_index()
    pub_data_full = pub_data_full[['Abstract', 'Article Title', 'Authors or Inventors']]
    pub_data_full = pub_data_full.rename(columns = {'Abstract': 'abstract', 'Authors or Inventors' : 'author', 'Article Title': 'title'})

    pub_data = pd.read_csv(f'{constr_pub_dir}/relevant_pubs.csv', index_col = 0)
    pub_data = pub_data[['Abstract', 'DOI', 'Cited by', 'Publication Year']]
    pub_data = pub_data.rename(columns={'Abstract': 'abstract'})

    pub_data = pd.merge(pub_data, pub_data_full, on = 'abstract', how = 'left')
    pub_data = pub_data.rename(columns = {'Publication Year' : 'year'})

    wp_data_both = pd.read_csv(f"{constr_wp_dir}/classed_wps_3.csv", index_col = 0).reset_index()
    wp_data_As = wp_data_both[wp_data_both['GPT_class'] == "A"][['abstract']].drop_duplicates('abstract')
    wp_data_full = pd.read_csv(f"{orig_wp_dir}/working_papers.csv", index_col = 0).drop_duplicates('abstract')
    wp_data_As = pd.merge(wp_data_As, wp_data_full, how = 'left', on = 'abstract')
    print("length of base data: ", len(wp_data_As))

    # add citations
    wp_data_cits_dois = pd.read_csv(f"{orig_wp_dir}/citations.csv", index_col=0).drop_duplicates(subset = ['doi'])
    wp_data = pd.merge(wp_data_As, wp_data_cits_dois, on = 'doi', how = 'left')

    wp_data_cits_abstracts = pd.read_csv(f"{orig_wp_dir}/citations_abstracts.csv", index_col=0).drop_duplicates(subset=['abstract'])
    wp_data_cits_abstracts = wp_data_cits_abstracts.rename(columns = {'abstract' : 'abstract_short'}) # this contains only the first 500 letters of the abstract
    wp_data['abstract_short'] = wp_data['abstract'].str[:500]
    wp_data = pd.merge(wp_data, wp_data_cits_abstracts, on = 'abstract_short', how = 'left') 

    wp_data['Cited by'] = wp_data['citation_count_x'].fillna(wp_data['citation_count_y'])
    wp_data = wp_data.drop(columns=['citation_count_x', 'citation_count_y'])

    # drop duplicates, and drop abstracts with no citations
    wp_data = wp_data.drop_duplicates('abstract')
    wp_data = wp_data.dropna(subset = 'Cited by')
    print("length of data with citations: ", len(wp_data))

    # change column names to be consistent with pub_data
    wp_data = wp_data.rename(columns = {'doi' : 'DOI'})
    wp_data = wp_data.drop(columns = ['abstract_short', 'ID', 'Source', 'link'])

    data = pd.concat([wp_data, pub_data])
    data = data.drop_duplicates('abstract')
    print(data)
    
    return data

def is_english(text, threshold=0.60):
    """
    Determines whether a text is predominantly English.

    Parameters:
        text (str): The text to analyse.
        threshold (float): The minimum total probability for English for the text to be considered English.

    Returns:
        bool: True if the cumulative probability of English is at least the threshold, otherwise False.
    """
    try:
        # Skip empty or missing abstracts
        if pd.isna(text) or text.strip() == '':
            return False
        # Obtain the list of detected languages with their probabilities
        lang_probs = detect_langs(text)
        # Calculate the total probability of the text being English
        en_prob = sum(lang.prob for lang in lang_probs if lang.lang == 'en')
        return en_prob >= threshold
    except LangDetectException:
        # If there is an error during language detection, assume it's not English
        return False

def preprocess_data(data):
    """Preprocess data, including language detection and cleaning"""
    # apply to first 50 words of each abstract
    data['Abstract_Short'] = data['abstract'].apply(lambda x: ' '.join(str(x).split()[:50]) if pd.notna(x) else '')
    english_mask = data['Abstract_Short'].apply(lambda x: is_english(x) if pd.notna(x) else False)

    # apply to last 50 words of each abstract
    data['Abstract_Short_last'] = data['abstract'].apply(lambda x: ' '.join(str(x).split()[-50:]) if pd.notna(x) else '')
    english_mask_last = data['Abstract_Short_last'].apply(lambda x: is_english(x) if pd.notna(x) else False)

    # Create a dataframe of non-English abstracts 
    non_english_abstracts = data[~(english_mask & english_mask_last)][['abstract', 'Abstract_Short']]

    # Print the short abstracts of all removed papers
    print("Short abstracts of removed non-English papers:")
    for idx, row in non_english_abstracts.iterrows():
        print(f"Index {idx}: {row['abstract']}")

    print(f"Removing {len(data) - len(data[english_mask & english_mask_last])} non-English abstracts")
    data = data[english_mask & english_mask_last]
    data.reset_index(drop=True, inplace=True)

    documents = data['abstract']
    documents = documents.reset_index(drop=True)
    documents = documents.tolist()
    documents = [str(doc) for doc in documents if isinstance(doc, str)]
    
    return documents, data

# Setup GPU Embedder
class GPUEmbedder:
    def __init__(self, model_name="all-roberta-large-v1"):
        # Check if MPS (Metal Performance Shaders) is available
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) for GPU acceleration")
            print(f"PyTorch version: {torch.__version__}")
            print(f"Current device: {device}")
            print(f"MPS device available: {torch.backends.mps.is_available()}")
            print(f"MPS device built: {torch.backends.mps.is_built()}")
        else:
            device = torch.device("cpu")
            print("MPS not available, using CPU")
            
        self.model = SentenceTransformer(model_name).to(device)
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
        
    def __call__(self, documents):
        return self.model.encode(documents, show_progress_bar=True)

def generate_embeddings(documents):
    """Generate embeddings for documents using the GPUEmbedder"""
    print("\nInitializing embedder and generating embeddings...")
    embedder = GPUEmbedder()
    embeddings = embedder(documents)
    embeddings = embeddings.astype(np.float64)
    print(f"Embeddings shape: {embeddings.shape}")
    print("Embeddings generated successfully!")
    return embeddings

def create_topic_model(documents, min_cluster_size, n_neighbors, n_components, embeddings):
    """Create a topic model with specified parameters using pre-computed embeddings"""
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=-1
    )

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric='cosine',
        random_state=42,
        low_memory=False,  # Set to False for GPU usage
        n_jobs=-1
    )

    # Initialise the BERTopic model
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(stop_words="english"),
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    )

    # Fit the model on your data
    topics, probs = topic_model.fit_transform(documents, embeddings)

    # return topics, probs, topic model
    return topics, probs, topic_model

def calculate_coherence_score(topic_model, docs):
    """Calculate coherence score for a topic model"""
    # Set device appropriately for your Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Preprocess documents
    cleaned_docs = topic_model._preprocess_text(docs)
    
    # Extract vectorizer from BERTopic
    vectorizer = topic_model.vectorizer_model
    
    # Initialize tokenizer
    tokenizer = vectorizer.build_tokenizer()
    
    # Tokenize documents
    tokens = []
    batch_size = 100  # Process in batches to show progress
    for i in range(0, len(cleaned_docs), batch_size):
        batch = cleaned_docs[i:i+batch_size]
        batch_tokens = [tokenizer(doc) for doc in batch]
        tokens.extend(batch_tokens)
    
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    
    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()
    
    # Get topic words 
    topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                    for topic in range(len(set(topic_model.topics_))-1)]
    
    # Create coherence model
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    # Compute coherence
    coherence = coherence_model.get_coherence()
    
    return coherence

def run_grid_search(documents, embeddings):
    """Run grid search for hyperparameter tuning"""
    # Define parameter range for min_cluster_size
    min_cluster_sizes = range(10, 41, 10)
    n_neighbors_list = [10, 15, 20]
    n_components_list = [5, 10, 15]
    
    results = []

    print("Starting hyperparameter grid search...")
    for min_cluster_size in min_cluster_sizes:
        for n_neighbors in n_neighbors_list:
            for n_components in n_components_list:
                print(f"Testing: min_cluster={min_cluster_size}, n_neighbors={n_neighbors}, n_components={n_components}")
                try:
                    # Get topic model
                    topics, probs, topic_model = create_topic_model(documents, min_cluster_size, n_neighbors, n_components, embeddings)

                    # Get coherence and uncategorized topics
                    coherence = calculate_coherence_score(topic_model, documents)
                    uncategorized_count = topics.count(-1)
                    num_topics_val = len(set(topics)) - (1 if -1 in topics else 0)  # Number of topics excluding -1

                    # Append to list
                    results.append({
                        'min_cluster_size': min_cluster_size,
                        'n_neighbors': n_neighbors,
                        'n_components': n_components,
                        'coherence': coherence,
                        'num_topics': num_topics_val,
                        'uncategorized': uncategorized_count
                    })
                    print(f"  -> Coherence: {coherence:.4f}, Topics: {num_topics_val}, Uncategorized: {uncategorized_count}")

                    # Clean up memory
                    del topic_model, topics, probs
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                except Exception as e:
                    print(f"  -> Error: {e}")
                    results.append({
                        'min_cluster_size': min_cluster_size,
                        'n_neighbors': n_neighbors,
                        'n_components': n_components,
                        'coherence': None,
                        'num_topics': None,
                        'uncategorized': None,
                        'error': str(e)
                    })

    # Create DataFrame and display results
    results_df = pd.DataFrame(results)
    print("\nGrid Search Results:")
    display(results_df)
    
    return results_df

def train_final_model(documents, embeddings, best_params):
    """Train the final model with the best hyperparameters"""
    # Extract best parameters
    best_min_cluster_size = int(best_params['min_cluster_size'])
    best_n_neighbors = int(best_params['n_neighbors'])
    best_n_components = int(best_params['n_components'])

    print(f"\nTraining final model with best parameters: min_cluster_size={best_min_cluster_size}, n_neighbors={best_n_neighbors}, n_components={best_n_components}")
    topics, probs, topic_model = create_topic_model(documents, best_min_cluster_size, best_n_neighbors, best_n_components, embeddings)
    topic_df = topic_model.get_topic_info()
    print("\nFinal Topic Model Info:")
    print(topic_df)
    
    return topics, probs, topic_model, topic_df

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    # Step 1: Load data
    data = load_data()
    
    # Step 2: Preprocess data
    documents, processed_data = preprocess_data(data)
    
    # Step 3: Generate embeddings (ONCE!)
    embeddings = generate_embeddings(documents)
    
    # Step 4: Run grid search
    results_df = run_grid_search(documents, embeddings)
    
    # Step 5: Find the best parameters based on coherence
    best_idx = results_df['coherence'].idxmax()
    best_params = results_df.loc[best_idx]
    print("\nBest Hyperparameters based on Coherence:")
    print(best_params)
    
    # Step 6: Train final model with best parameters
    topics, probs, topic_model, topic_df = train_final_model(documents, embeddings, best_params)
