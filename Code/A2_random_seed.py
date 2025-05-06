from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect_langs, LangDetectException, DetectorFactory
import os
import gc
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Set a fixed seed for langdetect to make results deterministic
DetectorFactory.seed = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def create_topic_model(documents, embeddings, random_seed):
    """Create a topic model with specified parameters and random seed"""
    # Fixed hyperparameters
    min_cluster_size = 20
    n_neighbors = 15
    n_components = 5
    
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
        random_state=random_seed,  # Using the provided random seed
        low_memory=False,
        n_jobs=-1
    )

    # Initialize the BERTopic model
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(stop_words="english")
    )

    # Fit the model on your data
    topics, probs = topic_model.fit_transform(documents, embeddings)

    return topics, probs, topic_model

def match_topics_between_runs(topic_keywords1, topic_keywords2, embedder=None):
    """
    Match topics between two runs using the Hungarian algorithm based on embedding similarity
    
    Parameters:
    - topic_keywords1: Dict with {topic_id: [(word, score), ...]} from run 1
    - topic_keywords2: Dict with {topic_id: [(word, score), ...]} from run 2
    - embedder: Sentence embedder model (will create one if None)
    
    Returns:
    - Dictionary mapping topic_ids from run1 to the most similar topic_ids in run2
    - Similarity scores between matched topics
    """
    # Skip outlier topic (-1)
    topic_ids1 = [tid for tid in topic_keywords1.keys() if tid != -1]
    topic_ids2 = [tid for tid in topic_keywords2.keys() if tid != -1]
    
    # Initialize embedder if not provided
    if embedder is None:
        # Choose device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        print("Creating sentence embedder for topic similarity computation...")
        embedder = SentenceTransformer("all-roberta-large-v1").to(device)
    
    # Get embeddings for topic keywords
    topic_vectors1 = {}
    topic_vectors2 = {}
    
    # Create representative text for each topic by concatenating top keywords
    for topic_id in topic_ids1:
        # Get top 20 keywords (or all if less than 20)
        words = [word for word, _ in topic_keywords1[topic_id][:20]]
        # Create a text representation
        topic_text = " ".join(words)
        # Embed the text
        embedding = embedder.encode(topic_text)
        topic_vectors1[topic_id] = embedding
        
    for topic_id in topic_ids2:
        words = [word for word, _ in topic_keywords2[topic_id][:20]]
        topic_text = " ".join(words)
        embedding = embedder.encode(topic_text)
        topic_vectors2[topic_id] = embedding
    
    # Calculate cosine similarity between topic embeddings
    similarity_matrix = np.zeros((len(topic_ids1), len(topic_ids2)))
    
    for i, topic_id1 in enumerate(topic_ids1):
        for j, topic_id2 in enumerate(topic_ids2):
            # Get embeddings
            emb1 = topic_vectors1[topic_id1]
            emb2 = topic_vectors2[topic_id2]
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarity_matrix[i, j] = similarity
    
    # Apply Hungarian algorithm to find best matching
    # The algorithm minimizes cost, so we negate similarity
    cost_matrix = 1 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping from topic_ids in run1 to topic_ids in run2
    mapping = {}
    similarities = []
    
    for i, j in zip(row_ind, col_ind):
        mapping[topic_ids1[i]] = topic_ids2[j]
        similarities.append(similarity_matrix[i, j])
    
    return mapping, similarities

def create_document_co_occurrence_matrix(topic_assignments_list, document_count):
    """
    Create a document co-occurrence matrix across multiple runs
    
    Parameters:
    - topic_assignments_list: List of topic assignments for each run
    - document_count: Total number of documents
    
    Returns:
    - Co-occurrence matrix
    """
    # Initialize co-occurrence matrix
    co_occurrence = np.zeros((document_count, document_count))
    
    # For each run
    for topics in topic_assignments_list:
        # Create a dictionary mapping topic IDs to document indices
        topic_to_docs = {}
        for doc_idx, topic_id in enumerate(topics):
            if topic_id not in topic_to_docs:
                topic_to_docs[topic_id] = []
            topic_to_docs[topic_id].append(doc_idx)
        
        # Update co-occurrence matrix
        for topic_id, doc_indices in topic_to_docs.items():
            if topic_id == -1:
                continue  # Skip outlier topics
                
            # For each pair of documents in the same topic
            for i in range(len(doc_indices)):
                for j in range(i+1, len(doc_indices)):
                    doc1, doc2 = doc_indices[i], doc_indices[j]
                    co_occurrence[doc1, doc2] += 1
                    co_occurrence[doc2, doc1] += 1
    
    # Normalize by number of runs
    co_occurrence /= len(topic_assignments_list)
    
    return co_occurrence

def create_reference_based_co_occurrence(all_topics, document_count, reference_run_index):
    """
    Create a co-occurrence matrix that compares how documents clustered with the reference seed
    are clustered in other runs.
    
    Parameters:
    - all_topics: List of topic assignments for each run
    - document_count: Total number of documents
    - reference_run_index: Index of the reference run (e.g., seed 42)
    
    Returns:
    - Reference-based stability matrix
    """
    reference_topics = all_topics[reference_run_index]
    
    # Create a dictionary mapping topic IDs to document indices for reference run
    ref_topic_to_docs = {}
    for doc_idx, topic_id in enumerate(reference_topics):
        if topic_id == -1:  # Skip outlier topics
            continue
        if topic_id not in ref_topic_to_docs:
            ref_topic_to_docs[topic_id] = []
        ref_topic_to_docs[topic_id].append(doc_idx)
    
    # Initialize stability matrix
    stability_matrix = np.zeros((document_count, document_count))
    
    # For each topic in the reference run
    for ref_topic_id, doc_indices in ref_topic_to_docs.items():
        # For each pair of documents in this reference topic
        for i in range(len(doc_indices)):
            for j in range(i+1, len(doc_indices)):
                doc1, doc2 = doc_indices[i], doc_indices[j]
                
                # Count how many times this pair stays together in other runs
                co_occurrence_count = 0
                run_count = 0
                
                for run_idx, topics in enumerate(all_topics):
                    if run_idx == reference_run_index:  # Skip reference run
                        continue
                    
                    # Skip if either document is an outlier in this run
                    if topics[doc1] == -1 or topics[doc2] == -1:
                        continue
                        
                    run_count += 1
                    # Check if they're in the same topic in this run
                    if topics[doc1] == topics[doc2]:
                        co_occurrence_count += 1
                
                # Calculate stability as proportion of runs where they stay together
                if run_count > 0:
                    stability = co_occurrence_count / run_count
                    stability_matrix[doc1, doc2] = stability
                    stability_matrix[doc2, doc1] = stability
    
    return stability_matrix

def save_topic_results_to_excel(all_topic_models, all_topics, documents, random_seeds, n_top_words=20, n_samples=5):
    """
    Save the topic information from each random seed run to a separate sheet in an Excel file.
    
    Parameters:
    - all_topic_models: List of BERTopic models from different runs
    - all_topics: List of topic assignments for each run
    - documents: List of document texts
    - random_seeds: List of random seeds used for each run
    - n_top_words: Number of top words to include for each topic
    - n_samples: Number of sample documents to include for each topic
    """
    print("\nSaving topic information to Excel file...")
    
    # Create a new Excel workbook
    filename = 'topic_model_results_by_seed.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # For each seed run
        for i, (model, topics, seed) in enumerate(zip(all_topic_models, all_topics, random_seeds)):
            # Get topic info DataFrame
            topic_info = model.get_topic_info()
            
            # Add top words for each topic
            all_keywords = []
            for topic_id in topic_info['Topic']:
                if topic_id == -1:  # Skip outlier topic
                    all_keywords.append("Outlier")
                    continue
                    
                keywords = ", ".join([word for word, _ in model.get_topic(topic_id)[:n_top_words]])
                all_keywords.append(keywords)
            
            topic_info['Top Words'] = all_keywords
            
            # Add sample documents for each topic
            topic_docs = {}
            for doc_idx, topic_id in enumerate(topics):
                if topic_id not in topic_docs:
                    topic_docs[topic_id] = []
                # Only store limited number of samples per topic
                if len(topic_docs[topic_id]) < n_samples:
                    # Truncate long documents for readability
                    doc_text = documents[doc_idx][:300] + "..." if len(documents[doc_idx]) > 300 else documents[doc_idx]
                    topic_docs[topic_id].append(doc_text)
            
            # Add sample documents to DataFrame
            sample_docs = []
            for topic_id in topic_info['Topic']:
                if topic_id in topic_docs and topic_docs[topic_id]:
                    sample_docs.append("\n\n".join(topic_docs[topic_id]))
                else:
                    sample_docs.append("")
                    
            topic_info['Sample Documents'] = sample_docs
            
            # Save to Excel sheet named with the seed
            sheet_name = f"Seed_{seed}"
            topic_info.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Adjust column widths for better readability
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(topic_info.columns):
                max_len = max(
                    topic_info[col].astype(str).map(len).max(), 
                    len(col)
                )
                # Cap column width to avoid very wide columns
                adjusted_width = min(max_len + 2, 50)
                worksheet.column_dimensions[chr(65 + idx)].width = adjusted_width
    
    print(f"Topic information saved to {filename}")

def run_seed_stability_analysis(documents, embeddings, n_seeds=20):
    """
    Run multiple BERTopic models with different random seeds and analyze stability
    
    Parameters:
    - documents: List of document texts
    - embeddings: Pre-computed document embeddings
    - n_seeds: Number of different random seeds to try
    
    Returns:
    - Dataframe with stability metrics
    - Document co-occurrence matrix
    - Reference-based stability matrix
    - DataFrame with all topic assignments
    """
    print(f"\nRunning seed stability analysis with {n_seeds} different random seeds...")
    
    # List to store topics for each run
    all_topics = []
    all_topic_models = []
    np.random.seed(123)  # Fixed seed for reproducibility
    candidate_seeds = np.random.choice([i for i in range(1, 101) if i != 42], size=50, replace=False)
    random_seeds = sorted([42] + list(candidate_seeds[:n_seeds-1]))
    
    # Run BERTopic with different random seeds
    for seed in tqdm(random_seeds, desc="Running models with different seeds"):
        topics, probs, topic_model = create_topic_model(documents, embeddings, random_seed=seed)
        all_topics.append(topics)
        all_topic_models.append(topic_model)
        
        # Print topic counts for this run
        topic_counts = pd.Series(topics).value_counts()
        n_outliers = topic_counts.get(-1, 0)
        n_topics = len(topic_counts) - (1 if -1 in topic_counts.index else 0)
        print(f"Seed {seed}: Found {n_topics} topics, {n_outliers} outliers")
    
    # Create DataFrame with all topic assignments
    doc_labels_df = pd.DataFrame(np.column_stack(all_topics), 
                                columns=[f"seed_{seed}" for seed in random_seeds])
    
    # Calculate pairwise NMI and ARI between all runs
    n_runs = len(all_topics)
    nmi_matrix = np.zeros((n_runs, n_runs))
    ari_matrix = np.zeros((n_runs, n_runs))
    
    for i in range(n_runs):
        for j in range(n_runs):
            if i != j:
                nmi_matrix[i, j] = normalized_mutual_info_score(all_topics[i], all_topics[j])
                ari_matrix[i, j] = adjusted_rand_score(all_topics[i], all_topics[j])
    
    # Calculate average NMI and ARI for each run
    avg_nmi = np.mean(nmi_matrix, axis=1)
    avg_ari = np.mean(ari_matrix, axis=1)
    
    # Create stability metrics dataframe
    metrics_df = pd.DataFrame({
        'seed': random_seeds,
        'avg_nmi': avg_nmi,
        'avg_ari': avg_ari,
        'n_topics': [len(set(topics)) - (1 if -1 in topics else 0) for topics in all_topics],
        'n_outliers': [list(topics).count(-1) for topics in all_topics]
    })
    
    # Create document co-occurrence matrix
    co_occurrence_matrix = create_document_co_occurrence_matrix(all_topics, len(documents))
    
    # Find the index of the run with seed 42 (our reference seed)
    reference_seed = 42
    try:
        reference_run_index = random_seeds.index(reference_seed)
    except ValueError:
        print(f"Warning: Seed {reference_seed} not found in random seeds. Using first run as reference.")
        reference_run_index = 0
        
    # Create reference-based stability matrix
    reference_stability_matrix = create_reference_based_co_occurrence(all_topics, len(documents), reference_run_index)
    
    # Create embedder for topic similarity
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print("Creating sentence embedder for topic similarity computation...")
    topic_embedder = SentenceTransformer("all-roberta-large-v1").to(device)
    
    print(f"Using run with seed {random_seeds[reference_run_index]} as reference point for topic comparisons")
    
    # Analyze topic keyword similarity across runs
    reference_model = all_topic_models[reference_run_index]
    reference_topics = all_topics[reference_run_index]
    
    # Get topic keywords for reference run
    reference_keywords = {}
    for topic_id in set(reference_topics):
        if topic_id != -1:  # Skip outlier topic
            reference_keywords[topic_id] = reference_model.get_topic(topic_id)
    
    # Compare keywords with other runs
    topic_similarity_data = []
    for i, (topics, model) in enumerate(zip(all_topics, all_topic_models)):
        if i == reference_run_index:
            continue
            
        # Get topic keywords for this run
        run_keywords = {}
        for topic_id in set(topics):
            if topic_id != -1:  # Skip outlier topic
                run_keywords[topic_id] = model.get_topic(topic_id)
        
        # Match topics and calculate similarity
        topic_mapping, similarities = match_topics_between_runs(reference_keywords, run_keywords, embedder=topic_embedder)
        
        # Record results
        topic_similarity_data.append({
            'run': i,
            'seed': random_seeds[i],
            'avg_keyword_similarity': np.mean(similarities) if similarities else 0,
            'min_keyword_similarity': np.min(similarities) if similarities else 0,
            'max_keyword_similarity': np.max(similarities) if similarities else 0,
        })
    
    topic_similarity_df = pd.DataFrame(topic_similarity_data)
    print("\nTopic Keyword Similarity (compared to reference run with seed 42):")
    print(topic_similarity_df)
    
    # Make sure we return all the data needed for the Excel output
    return metrics_df, co_occurrence_matrix, reference_stability_matrix, doc_labels_df, topic_similarity_df, all_topic_models, all_topics, random_seeds

def visualize_stability_results(metrics_df, co_occurrence_matrix, reference_stability_matrix, doc_labels_df, topic_similarity_df):
    """
    Visualize stability analysis results
    
    Parameters:
    - metrics_df: DataFrame with stability metrics
    - co_occurrence_matrix: Document co-occurrence matrix
    - reference_stability_matrix: Reference-based stability matrix
    - doc_labels_df: DataFrame with all topic assignments
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 15))
    
    # Plot 1: NMI and ARI by seed
    ax1 = plt.subplot(221)
    ax1.plot(metrics_df['seed'], metrics_df['avg_nmi'], 'o-', label='Average NMI')
    ax1.plot(metrics_df['seed'], metrics_df['avg_ari'], 'o-', label='Average ARI')
    ax1.set_xlabel('Random Seed')
    ax1.set_ylabel('Score')
    ax1.set_title('Clustering Stability Metrics by Seed')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Number of topics and outliers by seed with split y-axis
    ax2 = plt.subplot(222)
    color1 = 'tab:blue'
    color2 = 'tab:red'
    
    # Primary axis for topics (left)
    ax2.set_xlabel('Random Seed')
    ax2.set_ylabel('Number of Topics', color=color1)
    ax2.plot(metrics_df['seed'], metrics_df['n_topics'], 'o-', color=color1, label='Number of Topics')
    ax2.tick_params(axis='y', labelcolor=color1)
    
    # Secondary axis for outliers (right)
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Number of Outliers', color=color2)
    ax2_twin.plot(metrics_df['seed'], metrics_df['n_outliers'], 'o-', color=color2, label='Number of Outliers')
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    
    # Add title and legend
    ax2.set_title('Topic Count and Outliers by Seed (Split Axis)')
    
    # Create a combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 3: Heatmap of reference-based document co-occurrence
    ax3 = plt.subplot(223)
    # Sample the matrix if it's too large (>1000 documents)
    if reference_stability_matrix.shape[0] > 1000:
        sample_size = 1000
        indices = np.random.choice(reference_stability_matrix.shape[0], sample_size, replace=False)
        sampled_matrix = reference_stability_matrix[np.ix_(indices, indices)]
        sns.heatmap(sampled_matrix, cmap='viridis', ax=ax3, cbar_kws={'label': 'Co-occurrence Stability'})
        ax3.set_title(f'Seed 42 Document Cluster Stability (Sampled {sample_size} documents)')
    else:
        sns.heatmap(reference_stability_matrix, cmap='viridis', ax=ax3, cbar_kws={'label': 'Co-occurrence Stability'})
        ax3.set_title('Seed 42 Document Cluster Stability')
    ax3.set_xlabel('Document Index')
    ax3.set_ylabel('Document Index')
    
    # Plot 4: Topic keyword similarity across runs
    ax4 = plt.subplot(224)
    ax4.plot(topic_similarity_df['seed'], topic_similarity_df['avg_keyword_similarity'], 'o-', 
             label='Average Keyword Similarity')
    ax4.fill_between(topic_similarity_df['seed'], 
                     topic_similarity_df['min_keyword_similarity'],
                     topic_similarity_df['max_keyword_similarity'], alpha=0.3)
    ax4.set_xlabel('Random Seed')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Topic Embedding Similarity Across Runs')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('umap_seed_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional visualization: Topic assignment consistency
    # Create a visualization of how consistently documents are assigned to topics
    doc_consistency = np.zeros(len(doc_labels_df))
    
    # For each document, count how many unique topics it was assigned to
    for i in range(len(doc_labels_df)):
        doc_consistency[i] = doc_labels_df.iloc[i].nunique()
    
    plt.figure(figsize=(10, 6))
    plt.hist(doc_consistency, bins=range(1, int(max(doc_consistency))+2), alpha=0.7)
    plt.xlabel('Number of Different Topics Assigned')
    plt.ylabel('Number of Documents')
    plt.title('Document Topic Assignment Consistency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('document_topic_consistency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return

def main():
    """Main function to execute the analysis"""
    print("Starting UMAP Random Seed Stability Analysis for BERTopic")
    
    # Step 1: Load data
    data = load_data()
    
    # Step 2: Preprocess data
    documents, processed_data = preprocess_data(data)
    
    # Step 3: Generate embeddings
    embeddings = generate_embeddings(documents)
    
    # Step 4: Run seed stability analysis
    metrics_df, co_occurrence_matrix, reference_stability_matrix, doc_labels_df, topic_similarity_df, all_topic_models, all_topics, random_seeds = run_seed_stability_analysis(
        documents, embeddings, n_seeds=20
    )
    
    # Step 5: Visualize results
    visualize_stability_results(metrics_df, co_occurrence_matrix, reference_stability_matrix, doc_labels_df, topic_similarity_df)
    
    # Step 6: Save topic information to Excel
    save_topic_results_to_excel(all_topic_models, all_topics, documents, random_seeds)
    
    # Save results
    metrics_df.to_csv('umap_seed_stability_metrics.csv', index=False)
    topic_similarity_df.to_csv('umap_seed_topic_similarity.csv', index=False)
    
    print("\nAnalysis complete! Results have been saved to CSV files, Excel file, and visualizations.")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()