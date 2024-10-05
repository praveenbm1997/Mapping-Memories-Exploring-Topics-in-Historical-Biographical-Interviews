# Master Thesis : Mapping Memories: Exploring Topics in Historical-Biographical Interviews
## Overview
This repository provides the implementation of a topic modelling approach applied to large-scale text documents, specifically focusing on historical-biographical interviews in the German language. The dataset, provided by the University of Hagen (Fern Universität), consists of video transcripts capturing conversations around German history, culture, events, and lifestyle. The primary goal is to extract meaningful topics from these transcripts using both traditional and modern machine learning techniques for topic modelling.

### Key Components:
- **Traditional Methods**: 
    - Latent Dirichlet Allocation (LDA)
    - Non-Negative Matrix Factorization (NMF)
    - Latent Semantic Analysis (LSA)
- **Modern Methods**: 
    - BERTopic
    - Prompt-based Topic Models using Large Language Models (LLMs)
  
### Unique Approach:
A hybrid approach was developed combining:
1. **Sentence Embedding**: Captures the semantic meaning at a sentence level.
2. **Clustering**: Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) is applied to cluster the embedded sentences.
3. **Large Language Models (LLMs)**: Prompting techniques with models like GPT-3.5 and Google Gemini are used to generate human-readable topic names from the clustered sentences.

## Dataset
The dataset comprises video transcripts of historical-biographical interviews in German. The data includes real-time conversations, often noisy, containing irrelevant sentences or poorly structured dialogues. The steps used to preprocess the data include:
- Noise removal
- Filtering shorter sentences
- Embedding the cleaned sentences for clustering

### Noise in Data:
- Despite preprocessing, some noise remains, and a few clusters contain irrelevant information due to this noise. In future iterations, better noise-handling strategies should be explored.

## Methods Used

### Traditional Approaches:
- **LDA, NMF, LSA**: These methods work by grouping words based on their co-occurrence patterns but fall short in understanding the semantic relationships between words.

### Advanced Approach:
- **BERTopic & Prompt-Based Topic Modelling**:
    - The sentence embedding technique ensures semantic coherence at the sentence level.
    - **Clustering**: HDBSCAN clusters the embedded sentences, and the LLM is used to assign names to the topics based on the clusters.
    - The final topic names are verified by comparing them across different LLMs (e.g., GPT-3.5, Google Gemini) to ensure consistency and relevance.

## Evaluation & Validation
The generated topic names are evaluated by human experts and cross-checked using other LLMs. This manual evaluation ensures that the generated topics are accurate and aligned with the thematic content of the transcript dataset.

## Future Work
- **Noise Handling**: Further work is needed to reduce noise in the data and improve clustering precision.
- **Improved Embedding Models**: More powerful sentence embedding models can be explored, and the current models can be fine-tuned for better semantic understanding.
- **Hyperparameter Optimization**: Fine-tuning clustering and LLM parameters can yield more accurate topic results.

