# Modern Approaches to Entity Resolution: Leveraging NLP and Graph-Based Models for Financial Data Integration at the European Central Bank

This project aims to detect duplicate corporate records within RIAD (Register of Institutions and Affiliates Database) at the European Central Bank.

The complete Entity Resolution pipeline is based on both [Ditto](https://github.com/megagonlabs/ditto/) and [GraLMatch](https://github.com/FernandoDeMeer/GraLMatch/) implementations. Please refer to both original repositories and cite their publications.
- [Deep Entity Matching with Pre-Trained Language Models](https://arxiv.org/abs/2004.00584)
- [GraLMatch: Matching Groups of Entities with Graphs and Language Models](https://arxiv.org/abs/2406.15015)

The pipeline is divided into three main phases:

1.   **Blocking (Candidate Generation):** To overcome the $\mathcal{O}(n^{2})$ computational bottleneck of evaluating all possible pairs, several filtering strategies were developed and benchmarked. These include lexical vectorization (TF-IDF), a hybrid approach combining DistilBERT tokenization with Jaro-Winkler distance, and semantic blocking using Bi-Encoders alongside FAISS for approximate nearest neighbor search.
2.    **Matching (Classification):** Candidate pairs are evaluated using a Cross-Encoder architecture based on the DITTO framework. By fine-tuning a `distilbert-base-cased` model, the system analyzes the concatenated sequences of entity attributes to predict the probability of a match.
3.   **Graph Topology Refinement:** To consolidate records into unified entity groups, the system applies transitivities to the pair predictions, forming a graph. A post-processing cleanup based on the GraLMatch framework is applied to detect and prune false positives in oversized subgraphs using minimum edge cut and edge betweenness centrality techniques.



## Repository Structure

Based on the project's organization, the repository includes the following key components:

*   `blocking/`: Contains the implementations of the various candidate generation strategies (TF-IDF, Hybrid, Bi-Encoder).
*   `em/`: Contains the modules and scripts related to the Entity Matching (Cross-Encoder) architectures.
*   `blockingBenchmark.ipynb`: A Jupyter Notebook detailing the experimental evaluation, hyperparameter tuning, and performance comparison of the different blocking techniques.
*   `EntityMatcher.ipynb`: A Jupyter Notebook used for the ground truth generation.
*   `requirements.txt`: The list of Python dependencies required to run the notebooks and models.



## Setup & Installation

To run the exploratory notebooks and models locally, it is recommended to set up a dedicated virtual environment.

```bash
# Create and activate a new virtual environment
conda create -y -n "entity_resolution" python=3.12
conda activate entity_resolution

# Install the required dependencies
conda install -c conda-forge nvidia-apex
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

#### Dataset Configuration
Ensure your target dataset is explicitly registered in `configs.json` so it can be resolved via the `--task` flag. Refer to the original publication for task-specific hyperparameter recommendations (e.g., matching the correct `--lm` transformer architecture to the task).


### 1. Training Ditto

```bash
python -m src.models.run_training \
  --dataset_name riad_entities \
  --model_name distilbert \
  --experiment_name exp_riad_ditto \
  --num_epochs 5 \
  --batch_size 16 \
  --max_seq_length 256 \
  --learning_rate 2e-5 \
  --warmup_steps 200 \
  --weight_decay 0.01 \
  --pos_weight 2.5 \
  --use_validation_set \
  --save_model
```

### 2. Matching & Evaluation (Trees)

Use the following scripts to run full graph matching and generate performance metrics.

#### Run Full Matching
```bash
python -m scripts.run_full_matching \
  --experiment_name exp_riad_ditto \
  --epoch 5 \
  --threshold 0.85
```

#### Evaluate and Generate Scores
```bash
python -m scripts.get_scores_matching \
  --dataset_name "riad_entities" \
  --experiment_names_list "exp_riad_ditto" \
  --ground_truth_path "data/processed/riad_entities/seed_44/test__pre_split__all_matches.csv"
```
