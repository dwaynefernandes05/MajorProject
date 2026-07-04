# Hugging Face Spaces Deployment Guide

## Step-by-Step Deployment Instructions

### Prerequisites
- Hugging Face account (free at https://huggingface.co)
- Git knowledge (basic commands)
- All project files ready

### Step 1: Prepare Your Project Files

Ensure you have all these files in your local project directory:

```
MajorProject/
├── app.py                          ✅ Required (main app)
├── utils.py                        ✅ Required (utilities)
├── requirements.txt                ✅ Required (dependencies)
├── spaces_app.py                   ⚠️  Optional (use this if app.py fails)
├── obj1_model_artifacts/           ✅ CRITICAL (pre-trained models)
│   ├── model_config.json
│   ├── product_embeddings_mpnet.npy
│   ├── rfp_embeddings_mpnet.npy
│   ├── product_df.parquet
│   └── rfp_df_with_gt.parquet
├── README.md                       ✅ Optional (project description)
└── .gitignore                      ✅ Optional (ignore files)
```

### Step 2: Create a New Space on Hugging Face

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `cable-spec-matcher` (or choose your name)
   - **License**: Apache 2.0 (or your preference)
   - **SDK**: Select **Gradio**
   - **Visibility**: Public (or Private if preferred)
4. Click **"Create space"**

### Step 3: Get Your Space Git Repository

After creation, HF will show you a Git repository URL. It will look like:
```
https://huggingface.co/spaces/YOUR_USERNAME/cable-spec-matcher.git
```

### Step 4: Clone the Space Repository

```bash
cd /path/to/work
git clone https://huggingface.co/spaces/YOUR_USERNAME/cable-spec-matcher.git
cd cable-spec-matcher
```

### Step 5: Copy Your Project Files

Copy all files from your local MajorProject to this cloned space directory:

```bash
# Copy main application files
cp /path/to/MajorProject/app.py .
cp /path/to/MajorProject/utils.py .
cp /path/to/MajorProject/requirements.txt .

# Copy model artifacts
cp -r /path/to/MajorProject/obj1_model_artifacts .

# Copy optional files
cp /path/to/MajorProject/README.md .
```

### Step 6: Create README.md for Space Metadata

Create a `README.md` file in your space directory with HF metadata at the top:

```yaml
---
title: Cable Specification Matcher
emoji: 🔌
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Cable Specification Matcher

A hybrid semantic + structural retrieval system for matching electrical cable specifications with product catalogs.

## Features
- ✅ Hybrid matching (50% structured + 30% semantic + 20% standards)
- ✅ Robust across 7 real-world scenarios
- ✅ Pre-loaded with sample cable specifications
- ✅ Instant results with pre-computed embeddings

## Usage
1. Enter cable specifications or select a sample
2. Click "Search Products"
3. View top-5 matches with score breakdown

---

[See DEPLOYMENT_README.md for full documentation]
```

### Step 7: Create .gitignore (Optional but Recommended)

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp

# Large files (if any)
*.npy
*.pkl
```

### Step 8: Commit and Push to Hugging Face

```bash
# Initialize git (if not already done)
git config user.email "your_email@example.com"
git config user.name "Your Name"

# Add all files
git add .

# Commit
git commit -m "Initial prototype deployment: Cable Specification Matcher"

# Push to HF Spaces
git push
```

### Step 9: Monitor Deployment

1. Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/cable-spec-matcher`
2. Watch the "Build" logs (usually takes 3-5 minutes)
3. Once complete, you'll see a green ✅ "Running" badge
4. Click the Gradio interface link to access your app

## 🎯 Expected Behavior

### First Load (First time accessing the space)
- ⏳ Takes 30-60 seconds (model loading + initialization)
- SBERT model (~400MB) downloads on first access
- Subsequent loads are instant

### Normal Operation
- Manual input search: ~2-3 seconds
- Sample data fills: Instant
- Results display: ~0.5 seconds per result

## ⚠️ Troubleshooting

### "Module not found: app"
- Ensure `app.py` is in the root directory
- Check file names for typos
- Verify all imports are available in requirements.txt

### "No module named 'sentence_transformers'"
- Check `requirements.txt` is present
- Verify it includes `sentence-transformers>=2.2.0`
- Wait for Space rebuild (may take 5-10 minutes)

### "model_config.json not found"
- Ensure `obj1_model_artifacts/` directory is uploaded
- Check all 5 artifact files are present:
  - `model_config.json`
  - `product_embeddings_mpnet.npy`
  - `rfp_embeddings_mpnet.npy`
  - `product_df.parquet`
  - `rfp_df_with_gt.parquet`

### "CUDA out of memory" or "Out of memory"
- HF Spaces has CPU-only inference tier by default
- For GPU, upgrade to "GPU-small" or "GPU-medium" (paid)
- CPU works fine; just slower for SBERT encoding

### Empty results / No matches found
- Check product catalog loaded: `obj1_model_artifacts/product_df.parquet`
- Verify spec text is not empty
- Try a sample spec to validate system works

## 🚀 Performance Optimization

### For Fast Responses:
- Pre-computed embeddings are cached ✅
- ~2-3s per query (CPU) is expected
- Using pre-cached embeddings saves >10x time

### For Multiple Specifications:
- Use `batch_process.py` for bulk processing
- Or access via API (if you add FastAPI wrapper)

### Reducing Load Times:
- Consider GPU upgrade for production
- Cache frequently used specifications
- Pre-load model on Space startup

## 📊 File Size Reference

| File | Size | Notes |
|------|------|-------|
| app.py | ~40KB | Main application |
| utils.py | ~20KB | Utilities |
| requirements.txt | <1KB | Dependencies |
| product_embeddings_mpnet.npy | ~280MB | Pre-computed vectors |
| rfp_embeddings_mpnet.npy | ~280MB | Pre-computed vectors |
| product_df.parquet | ~50MB | Product catalog |
| rfp_df_with_gt.parquet | ~15MB | RFP specs |
| model_config.json | ~2KB | Configuration |
| **Total** | **~675MB** | Full deployment size |

⚠️ **Space storage limit**: 50GB (free tier) - Your project is ~675MB, so plenty of room!

## 🔒 Security & Privacy

### Recommendations:
1. **Set Space to Private** if handling proprietary specs
2. **No data persistence** - Each inference is ephemeral
3. **No logging of inputs** - Specs not stored or shared
4. **Model weights public** - Pre-trained model info available

### Data Handling:
- Input specs processed in-memory only
- No database storage
- Results shown in UI, not persisted
- Refresh page to clear session

## 🎓 Advanced: Custom Domain

To use a custom domain with your HF Space:

1. Go to Space settings → "Custom Domain"
2. Add your domain (requires DNS setup)
3. Follow HF's instructions for CNAME record

Example: `cable-matcher.yourcompany.com`

## 📞 Support & Debugging

### View Build Logs:
- Click "Build" tab on Space page
- See real-time build progress
- Check for error messages

### Test Locally First:
```bash
# Before deploying to HF Spaces
python app.py

# Should show:
# 🚀 Initializing Cable Spec Matcher...
# ✅ Interface ready! Launching on http://127.0.0.1:7860
```

### Common Log Messages:
```
✅ Artifacts loaded successfully       → Good!
⏳ Loading model (first time only)     → Normal, takes time
⚠️  No artifacts found               → Missing obj1_model_artifacts/
❌ Error: module 'app' not found      → Check app.py filename
```

## 🎉 Success Checklist

- ✅ All 5 artifact files present
- ✅ app.py in root directory
- ✅ requirements.txt up-to-date
- ✅ README.md with HF metadata
- ✅ Space shows "Running" with green badge
- ✅ Interface loads at `https://huggingface.co/spaces/YOUR_USERNAME/cable-spec-matcher`
- ✅ Sample buttons fill form correctly
- ✅ Search returns results with scores

## 📝 Next Steps

1. **Share with Faculty**: Send Space URL to your advisors
2. **Gather Feedback**: Use results to improve matching
3. **Add Batch API**: Extend with `/predict` endpoint
4. **Collect Annotations**: Build active learning pipeline
5. **Scale to Production**: Migrate to dedicated server if needed

---

**Deployment Time**: ~10 minutes  
**First Load Time**: ~60 seconds  
**Subsequent Loads**: ~2-3 seconds  
**Maintenance**: Minimal (HF manages infrastructure)

Good luck! 🚀
