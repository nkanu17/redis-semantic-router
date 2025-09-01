# News Classification: Baseline LLM vs Redis Semantic Router

A proof-of-concept comparing LLM-based news classification with Redis semantic routing for improved latency and cost efficiency.

## Overview

This project implements two approaches for classifying BBC news articles into 5 categories (**business**, **entertainment**, **politics**, **sport**, **tech**):

1. **Baseline LLM**: Uses LiteLLM with Claude for direct text classification
2. **Redis Semantic Router**: Uses vector embeddings and Redis for fast similarity-based classification

Dataset: Pre-split BBC News articles:
- **Training**: 1,117 articles (`train_data.csv`) 
- **Validation**: 373 articles (`validation_data.csv`)
- **Test**: 735 articles (`BBC News Test.csv`)

## Setup

### Prerequisites
- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **Docker** (for Redis)

### 1. Install Dependencies
```bash
# Install all Python dependencies
uv sync
```

### 2. Start Redis Stack
```bash
# Start RedisVL with extensions
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Verify Redis is running
docker ps | grep redis-stack
```

### 3. Verify Setup
```bash
# Test Redis connection
docker exec -it redis-stack redis-cli ping
# Should return: PONG

# Check dataset is in place
ls bbc-news-articles-labeled/
# Should show: BBC News Train.csv, BBC News Test.csv (split files auto-created)
```

## Usage

All commands run from the `src/` directory:

```bash
cd src
```

### Basic Commands

```bash
# Check system status
python main.py status

# Train semantic router (one-time setup)
python main.py train_router

# Run baseline LLM classification
python main.py llm_classifier

# Run semantic router classification
python main.py semantic_router

# Compare results from both approaches
python main.py evaluate

# Clear all routes from Redis
python main.py clear-routes
```

### Advanced Usage

```bash
# Use custom config file
python main.py llm_classifier --config custom_config.yaml

# Force retrain semantic router
python main.py train_router --force-retrain

# Run on training data instead of test/validation data
python main.py llm_classifier --train-articles
python main.py semantic_router --train-articles
```

## Architecture

### Design Philosophy

The codebase follows a **modular pipeline architecture** where each component can be executed independently or composed together. This design enables:

- **Independent execution**: Each classifier runs as a standalone pipeline
- **Composable workflows**: Pipelines can be chained for comparison analysis
- **Pluggable components**: Easy to swap classifiers or add new ones
- **Stateless operations**: Each pipeline is self-contained with its own configuration
- **Result persistence**: Each pipeline saves results independently for later analysis

### Pipeline Flow

```mermaid
graph TB
    A[News Articles Data] --> B{Pipeline Selection}
    
    B -->|LLM Classification| C[LLM Classification Pipeline]
    B -->|SemanticRoute Registration (Train)| D[SemanticRoute Registration Pipeline]
    B -->|SemanticRoute Matching (Classify)| F[SemanticRoute Matching Pipeline]
    B -->|Compare Results| E[Evaluation Pipeline]
    
    subgraph "LLM Path"
        C --> C1[Load Test Data]
        C1 --> C2[Batch Articles]
        C2 --> C3[LLM API Calls]
        C3 --> C4[Classification Results]
        C4 --> C5[Save LLM Results]
    end
    
    subgraph "Semantic Router Path"
        subgraph "Phase 1: Route Registration (Train)"
            D --> D1[Load Training Data]
            D1 --> D2[Sample Articles per Class]
            D2 --> D3[Generate Embeddings]
            D3 --> D4[Store in Redis Vector Index]
            D4 --> D5[Optimize Thresholds]
        end
        
        subgraph "Phase 2: Route Matching (Classify)"
            F --> F1[Load Test Data]
            F1 --> F2[Generate Article Embeddings]
            F2 --> F3[Fetch Existing Router]
            F3 --> F4[Vector Similarity Search]
            F4 --> F5[Classification Results]
            F5 --> F6[Save Results]
        end
        
        D5 -.->|Router Ready| F
    end
    
    subgraph "Comparison"
        C5 --> E
        F6 --> E
        E --> E1[Load Both Results]
        E1 --> E2[Calculate Metrics]
        E2 --> E3[Generate Comparison]
        E3 --> E4[Save Comparison Report]
    end
```

### Classification Workflows

**LLM Classifier Flow:**
```
Articles → Batch → LLM Prompt → Async LiteLLM API → Classification
```

**Semantic Router Flow:**
```
Training: Articles → Sample → Embeddings → Redis Vector Index
Runtime:  Article → Embed → Vector Search → Classification
                           
```

### Project Structure
```
redis_semantic_router/
├── README.md                  # Project documentation
├── pyproject.toml             # Python project configuration
├── config/                    # Configuration files
│   └── pipeline_config.yaml   # Pipeline configuration
├── data/                      # Input datasets (gitignored)
│   └── bbc-news-articles-labeled/
├── results/                   # Generated results (gitignored)
│   ├── llm_classifier/        # LLM classification results
│   ├── semantic_router/       # Semantic router results
│   └── comparison/            # Evaluation results
└── src/                       # Source code
    ├── main.py                # CLI orchestrator
    ├── pipelines/             # Pipeline orchestration
    ├── llm_classifier/        # LLM classification components
    ├── semantic_router/       # Vector-based classification
    ├── shared/                # Common utilities and abstractions
    └── utils/                 # Configuration, data loading, logging
```

## Configuration

Edit `config/pipeline_config.yaml`:
> **Note:** Classification categories are defined in the `NewsCategory` enum in `shared/data_types.py`. To add new categories, update this enum rather than configuration files.

```yaml
# Dataset settings  
data:
  dataset_path: "data/bbc-news-articles-labeled"
  train_file: "train_data.csv"
  validation_file: "validation_data.csv"

# LLM Classifier settings
llm_classifier:
  model_name: "claude-sonnet-4-20250514"
  batch_size: 10
  max_concurrent: 20
  temperature: 0
  max_tokens: 50000
  save_results: true
  results_dir: "results/llm_classifier"

# Redis semantic router settings
semantic_router:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  router_name: "news-classification-router"
  route_config:
    samples_per_class: 100
    initial_threshold: 0.6
    optimize_thresholds: true
  save_results: true
  results_dir: "results/semantic_router"
```

## Results

### Output Structure
```
results/
├── llm_classifier/            # LLM classification results
├── semantic_router/           # Semantic router results  
└── comparison/                # Side-by-side comparison
    └── 2025-08-30_22-15-04_7ff67441/
        ├── metrics.json       # Performance metrics
        ├── classifications.csv # Article-level predictions
        └── run_info.json      # Run metadata
```

### Viewing Results
```bash
# Check latest results
python main.py status

# Redis web interface
open http://localhost:8001
```


## Performance Analysis at Scale

Based on actual test results with 373 validation articles, here's the projected performance at 100,000 samples:

### LLM Classifier
- **Total cost**: $182 (at $0.00182 per article)
- **Total processing time**: 7.1 hours (257ms per article)
- **Accuracy**: 97.6%
- **F1 macro**: 97.6%

### Semantic Router
- **Total cost**: $0 (no API costs after training)
- **Total processing time**: 29.8 minutes (17.9ms per article)  
- **Accuracy**: 94.1%
- **F1 macro**: 94.1%

### Trade-offs at Scale
- **Cost savings**: 100% ($182 → $0)
- **Speed improvement**: 14.4x faster (7.1 hours → 29.8 minutes)
- **Accuracy trade-off**: 3.5% accuracy loss (97.6% → 94.1%)
- **F1 macro trade-off**: 3.5% F1 loss (97.6% → 94.1%)

The semantic router provides massive operational benefits for high-volume classification scenarios, with the accuracy trade-off often acceptable for real-time applications.

## Troubleshooting

```bash
# Redis not running
docker restart redis-stack

# Clear Redis data
python main.py clear-routes

# Import errors - ensure you're in src/
cd src

# Memory issues - reduce sample sizes in config
```

This implementation provides a complete comparison framework for evaluating LLM vs semantic routing approaches to news classification.