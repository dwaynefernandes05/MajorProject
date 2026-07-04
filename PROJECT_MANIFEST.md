# 📋 PROJECT MANIFEST & FILE INDEX

**Cable Specification Matcher - Prototype v1.0**

## 📂 Project Structure

```
MajorProject/
│
├── 🚀 QUICK START & SETUP
│   ├── quickstart.py              - Automated setup & validation (RUN THIS FIRST!)
│   ├── requirements.txt           - Python dependencies
│   └── test_prototype.py          - Test suite to validate setup
│
├── 🎨 APPLICATION
│   ├── app.py                     - Main Gradio interface (600+ lines)
│   │   └── Contains:
│   │       • Lazy loading of SBERT model
│   │       • Hybrid matching engine (50/30/20 weights)
│   │       • 3 sample cable specifications
│   │       • Gradio UI with 3 tabs
│   │       • Full documentation & comments
│   │
│   ├── utils.py                   - Utility functions & validators
│   │   └── Contains:
│   │       • SpecificationValidator (6 field validators)
│   │       • BatchProcessor (CSV/JSON I/O)
│   │       • MetricsComputer (IR metrics)
│   │       • ExperimentAnalyzer (7 experiment handlers)
│   │       • ReportGenerator (markdown reports)
│   │
│   ├── spaces_app.py              - HF Spaces entry point
│   │   └── Optimized for HF Spaces environment
│   │
│   └── batch_process.py           - Batch processing script
│       └── Contains:
│           • CSV input/output handlers
│           • Batch matching with metrics
│           • Markdown report generation
│           • Sample data creation
│
├── 📚 DOCUMENTATION
│   ├── README.md (THIS DOCUMENT)
│   ├── DEPLOYMENT_README.md       - Complete user guide (6000+ words)
│   │   └── Contains:
│   │       • Features overview
│   │       • Quick start (local & HF)
│   │       • Usage examples (manual & API)
│   │       • All 7 experiment explanations
│   │       • Project structure walkthrough
│   │       • Known limitations & future work
│   │
│   ├── HF_SPACES_DEPLOYMENT.md   - Step-by-step HF Spaces guide
│   │   └── Contains:
│   │       • Pre-deployment checklist
│   │       • 9-step deployment process
│   │       • Troubleshooting guide
│   │       • Performance optimization tips
│   │       • File size reference
│   │       • Security & privacy notes
│   │
│   └── PROJECT_MANIFEST.md        - This file
│
├── 🧠 MODEL ARTIFACTS (obj1_model_artifacts/)
│   ├── model_config.json          - Model weights & configuration
│   │   └── Contains:
│   │       • SBERT model name (all-mpnet-base-v2)
│   │       • Hybrid weights (50/30/20)
│   │       • Structured field lists
│   │       • Baseline metrics
│   │
│   ├── product_embeddings_mpnet.npy    - Pre-computed product vectors (280 MB)
│   ├── rfp_embeddings_mpnet.npy        - Pre-computed RFP vectors (280 MB)
│   ├── product_df.parquet              - Product catalog (200 columns, 50 MB)
│   └── rfp_df_with_gt.parquet          - RFP specs with ground truth (15 MB)
│
├── 📊 ORIGINAL DATASETS (for reference)
│   ├── datasets/
│   │   ├── product_catalog.csv         - ~7000 cable SKUs
│   │   └── rfp_specs_7000.csv          - 7000 RFP specifications
│   │
│   └── OBJ2/ (7 Experiment Notebooks)
│       ├── EXP1/ - Numeric Sensitivity Test
│       ├── EXP2/ - Unit Variation Test
│       ├── EXP3/ - Missing Parameters Test
│       ├── EXP4/ - Noise Injection Test
│       ├── EXP5/ - Long Document Bias Test
│       ├── EXP6/ - Partial Specification Test
│       └── EXP7/ - Positional Bias Test
│
└── 🏆 OBJECTIVE 1 BASELINE
    └── OBJ1/
        ├── OBJECTIVE_1_BASELINES_AND_SAVING.ipynb - Baseline comparison
        └── OBJECTIVE_1_FIXED.ipynb                 - Hybrid model training
```

## 📄 FILE DESCRIPTIONS

### Core Application Files

#### `quickstart.py` ⭐ START HERE!
- **Purpose**: Automated setup and validation
- **Size**: ~5 KB
- **Usage**: `python quickstart.py`
- **Does**: Checks Python version, project structure, installs dependencies, runs tests
- **Recommended**: Run this first to validate your environment

#### `app.py` - Main Application
- **Purpose**: Gradio web interface + hybrid matching engine
- **Size**: ~40 KB
- **Lines**: 600+
- **Key Functions**:
  - `load_sentence_transformer()` - Lazy load SBERT model
  - `load_artifacts()` - Lazy load pre-trained artifacts
  - `structured_score()` - Field-based matching
  - `semantic_score()` - SBERT embedding similarity
  - `standards_score()` - Compliance verification
  - `hybrid_match()` - Weighted combination (50/30/20)
  - `create_interface()` - Gradio UI builder
- **Features**:
  - 3 tabs: Manual Input, Samples, Experiments Info
  - 3 pre-loaded sample specifications
  - Automatic form filling
  - Detailed score breakdowns
- **Latency**: 2-3 seconds per query (CPU), <1s with GPU

#### `utils.py` - Utilities & Helpers
- **Purpose**: Reusable components for validation, processing, reporting
- **Size**: ~20 KB
- **Classes**:
  - `SpecificationValidator` - Validates voltage, conductor, insulation, etc.
  - `BatchProcessor` - Loads CSV, saves results
  - `MetricsComputer` - Top-k, MRR, NDCG@5 metrics
  - `ExperimentAnalyzer` - Analyze results from 7 experiments
  - `ReportGenerator` - Generate markdown reports
- **Use Cases**:
  - Batch processing CSV files
  - Validating input specifications
  - Computing IR metrics
  - Generating analysis reports

#### `batch_process.py` - Batch Processing
- **Purpose**: Process multiple specifications and generate reports
- **Size**: ~15 KB
- **Usage**:
  ```bash
  python batch_process.py --create-sample
  python batch_process.py --csv input.csv --output results/
  ```
- **Features**:
  - Load CSV specifications
  - Run hybrid matching on all rows
  - Save results in CSV/JSON
  - Generate markdown report
  - Compute Top-1/Top-5 accuracy (if ground truth available)
- **Output**: `results.csv`, `results.json`, `report.md`

#### `spaces_app.py` - HF Spaces Launcher
- **Purpose**: Entry point for HF Spaces environment
- **Size**: ~2 KB
- **Simply imports and launches app.py with HF-optimized settings**

### Documentation Files

#### `DEPLOYMENT_README.md` 📖 COMPREHENSIVE GUIDE
- **Purpose**: Complete user guide and reference
- **Size**: ~150 KB
- **Sections**:
  - ✅ Features & performance metrics
  - ✅ Quick start (local & HF)
  - ✅ Model comparison table
  - ✅ Usage examples (UI & API)
  - ✅ Detailed 7 experiment explanations
  - ✅ Project structure walkthrough
  - ✅ API usage examples
  - ✅ Known limitations
  - ✅ Future enhancements
  - ✅ File reference guide
- **Read this for**: Complete understanding of the system

#### `HF_SPACES_DEPLOYMENT.md` 🚀 DEPLOYMENT GUIDE
- **Purpose**: Step-by-step HF Spaces deployment
- **Size**: ~50 KB
- **Sections**:
  - ✅ Prerequisites & checklist
  - ✅ 9-step deployment process
  - ✅ Troubleshooting (5 common issues)
  - ✅ Performance optimization
  - ✅ File size reference
  - ✅ Security & privacy
  - ✅ Custom domain setup
  - ✅ Success checklist
- **Read this for**: Deploying to HF Spaces

#### `PROJECT_MANIFEST.md` (This File)
- **Purpose**: Index & overview of all files
- **Size**: ~20 KB
- **Sections**:
  - Project structure
  - File descriptions
  - Usage instructions
  - Key metrics
  - Integration points

### Dependency Files

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Size**: ~200 bytes
- **Key Packages**:
  - `gradio>=4.0.0` - Web UI framework
  - `sentence-transformers>=2.2.0` - SBERT embeddings
  - `torch>=2.0.0` - Deep learning backend
  - `pandas>=2.0.0` - Data manipulation
  - `numpy>=1.24.0` - Numerical computing
  - `scikit-learn>=1.3.0` - ML utilities
  - `rank-bm25>=0.2.2` - BM25 baseline

### Model Artifacts (in `obj1_model_artifacts/`)

#### `model_config.json` (~2 KB)
```json
{
  "sbert_model": "all-mpnet-base-v2",
  "hybrid_weights": {"structured": 0.5, "semantic": 0.3, "standards": 0.2},
  "K": 5,
  "exact_fields": ["voltage_rating", "conductor_material", ...],
  "numeric_fields": {"core_count": "exact", "size_sqmm": "ladder_adjacent"},
  "baseline_metrics": { ... }
}
```

#### `product_embeddings_mpnet.npy` (~280 MB)
- Pre-computed SBERT embeddings for all 7000 products
- Shape: (7000, 768) - 7000 products × 768-dim vectors
- Format: NumPy binary (normalized L2)
- Used for: Semantic similarity computation

#### `rfp_embeddings_mpnet.npy` (~280 MB)
- Pre-computed SBERT embeddings for all 7000 RFPs
- Shape: (7000, 768)
- Format: NumPy binary (normalized L2)
- Used for: Query embedding matching

#### `product_df.parquet` (~50 MB)
- Product catalog with 7000 rows × ~200 columns
- Columns: sku_id, product_name, voltage_rating, conductor_material, etc.
- Format: Apache Parquet (compressed)
- Used for: Structured field retrieval

#### `rfp_df_with_gt.parquet` (~15 MB)
- RFP specifications with ground truth labels
- Columns: rfp_id, spec_text_raw, voltage_rating, core_count, ground_truth_sku
- Format: Apache Parquet
- Used for: Evaluation & ground truth matching

## 🎯 QUICK REFERENCE

### Running Locally
```bash
# 1. One-time setup
python quickstart.py

# 2. Launch web interface
python app.py

# 3. Open browser to http://127.0.0.1:7860
```

### Batch Processing
```bash
# Create sample CSV
python batch_process.py --create-sample

# Process specifications
python batch_process.py --csv sample_batch.csv --output results/

# Check results
ls results/
```

### Testing
```bash
python test_prototype.py
```

### Deploying to HF Spaces
```bash
# See HF_SPACES_DEPLOYMENT.md for detailed steps
# Quick summary:
# 1. Create Space on huggingface.co
# 2. Git clone Space repo
# 3. Copy all files
# 4. Git push
```

## 📊 KEY METRICS

### Model Performance
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 17.8% |
| Top-5 Accuracy | 44.3% |
| Mean Reciprocal Rank (MRR) | 0.309 |
| NDCG@5 | 0.312 |
| Inference Time (CPU) | 2-3 seconds |
| Inference Time (GPU) | <1 second |

### Robustness (from 7 experiments)
| Experiment | Robustness | Insight |
|------------|-----------|---------|
| EXP1: Numeric Sensitivity | ✅ High | Handles ±10% value changes |
| EXP2: Unit Variation | ✅ High | Invariant to notation (kV vs V) |
| EXP3: Missing Params | ✅ Good | Works with incomplete specs |
| EXP4: Noise | ✅ Good | Filters ~10 words of noise |
| EXP5: Long Docs | ✅ High | Handles 500-1300 word docs |
| EXP6: Partial Specs | ✅ Good | Matches with 2+ tokens |
| EXP7: Positional Bias | ✅ High | Position-invariant |

### File Sizes
| Component | Size | Purpose |
|-----------|------|---------|
| app.py | 40 KB | Main application |
| utils.py | 20 KB | Utilities |
| Model artifacts | ~625 MB | Pre-trained embeddings & catalogs |
| Documentation | ~200 KB | Guides & references |
| **Total** | **~675 MB** | Complete deployment |

## 🔌 INTEGRATION POINTS

### Using as Python Module
```python
from app import hybrid_match, load_artifacts

artifacts = load_artifacts()
products = artifacts['product_df'].to_dict('records')

results = hybrid_match(rfp_spec, products, top_k=5)
```

### Using Batch Processor
```python
from utils import BatchProcessor, MetricsComputer

specs = BatchProcessor.load_csv('input.csv')
# Process specs...
results = [...]
BatchProcessor.save_results(results, 'output.csv')
```

### Validating Input
```python
from utils import SpecificationValidator

spec = {'voltage_rating': '11kV', ...}
is_valid, errors = SpecificationValidator.validate_spec(spec)
```

## 🎓 EXPERIMENT REFERENCE

The prototype incorporates findings from 7 robustness experiments:

1. **EXP1 - Numeric Sensitivity** (OBJ2/EXP1/)
   - Tests ±10% numeric perturbations
   - Shows structured matching is critical

2. **EXP2 - Unit Variation** (OBJ2/EXP2/)
   - Tests unit notation changes (kV vs V)
   - Justifies ladder-based size matching

3. **EXP3 - Missing Parameters** (OBJ2/EXP3/)
   - Tests 1-4 missing token types
   - Shows field redundancy value

4. **EXP4 - Noise Injection** (OBJ2/EXP4/)
   - Tests in/out-of-domain noise
   - Shows structured score anchoring

5. **EXP5 - Long Document Bias** (OBJ2/EXP5/)
   - Tests 500-1300 word documents
   - Shows position-invariant benefits

6. **EXP6 - Partial Specification** (OBJ2/EXP6/)
   - Tests 1, 2, 4, 8 token specs
   - Defines minimum viable spec length

7. **EXP7 - Positional Bias** (OBJ2/EXP7/)
   - Tests spec position (start/mid/end)
   - Shows primacy bias solution

## 🚀 DEPLOYMENT OPTIONS

### Option 1: Local Development ✅
- Run `python app.py`
- No setup required
- Perfect for testing & iteration

### Option 2: Hugging Face Spaces ⭐ RECOMMENDED
- Free hosting, no credit card needed
- Accessible via URL to anyone
- Auto-scaling & monitoring included
- See HF_SPACES_DEPLOYMENT.md

### Option 3: Docker Containerization (Future)
- Would enable cloud deployment
- Available on request

### Option 4: REST API (Future)
- Would enable programmatic access
- Could be built with FastAPI

## ⚠️ KNOWN LIMITATIONS

1. **Model Size**: SBERT all-mpnet-base-v2 is ~400MB
2. **CPU-only by Default**: HF Spaces free tier is CPU-only
3. **Real-time Catalog Updates**: Need model retraining for new products
4. **Language**: Only English tested
5. **Scalability**: Works for <100k products; may need indexing beyond that

## 🔮 FUTURE ENHANCEMENTS

1. **Active Learning**: Collect feedback to improve matching
2. **Multi-modal**: Add product images & technical specs
3. **Batch API**: FastAPI endpoint for bulk processing
4. **Advanced Filtering**: Price, lead time, availability
5. **Multilingual**: Support other languages
6. **Product Updates**: Dynamic product catalog management

## 📞 SUPPORT

- **Documentation**: See DEPLOYMENT_README.md
- **Setup Help**: Run quickstart.py for validation
- **Deployment**: Follow HF_SPACES_DEPLOYMENT.md
- **Testing**: Run test_prototype.py
- **Batch Processing**: See batch_process.py examples

---

**Document Version**: 1.0  
**Last Updated**: May 2026  
**Status**: Production Ready for Faculty Demo & HF Spaces  
**Maintainer**: Your Project Team
