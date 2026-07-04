"""
Test Script for Cable Specification Matcher
=============================================

Run this to verify the prototype works correctly locally
before deploying to Hugging Face Spaces.
"""

import sys
import time
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import gradio
        print("   ✅ gradio")
    except ImportError as e:
        print(f"   ❌ gradio: {e}")
        return False
    
    try:
        import sentence_transformers
        print("   ✅ sentence_transformers")
    except ImportError as e:
        print(f"   ❌ sentence_transformers: {e}")
        return False
    
    try:
        import pandas as pd
        print("   ✅ pandas")
    except ImportError as e:
        print(f"   ❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("   ✅ numpy")
    except ImportError as e:
        print(f"   ❌ numpy: {e}")
        return False
    
    return True

def test_artifacts():
    """Test that model artifacts are present and loadable"""
    print("\n🧪 Testing model artifacts...")
    
    artifact_dir = Path('obj1_model_artifacts')
    
    required_files = [
        'model_config.json',
        'product_embeddings_mpnet.npy',
        'rfp_embeddings_mpnet.npy',
        'product_df.parquet',
        'rfp_df_with_gt.parquet'
    ]
    
    all_present = True
    for fname in required_files:
        fpath = artifact_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {fname} (NOT FOUND)")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Missing artifacts! Make sure obj1_model_artifacts/ directory is in project root.")
        return False
    
    # Try loading config
    try:
        import json
        with open(artifact_dir / 'model_config.json') as f:
            config = json.load(f)
        print(f"\n   ✅ Config loaded: {len(config)} keys")
        print(f"      - Model: {config.get('sbert_model')}")
        print(f"      - Weights: {config.get('hybrid_weights')}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return False

def test_app_functions():
    """Test core app functions"""
    print("\n🧪 Testing app functions...")
    
    try:
        from app import (
            _adjacent_sizes,
            structured_score,
            standards_score
        )
        print("   ✅ Functions imported successfully")
        
        # Test _adjacent_sizes
        sizes = _adjacent_sizes(240.0)
        if 240.0 in sizes:
            print(f"   ✅ _adjacent_sizes(240) = {sizes}")
        else:
            print(f"   ❌ _adjacent_sizes failed")
            return False
        
        # Test structured_score
        rfp = {
            'voltage_rating': '11kV',
            'conductor_material': 'Copper',
            'insulation_type': 'PVC',
            'core_count': '3',
            'size_sqmm': '240',
            'sheath_type': 'PVC',
            'fire_resistance': 'FRLS',
            'armouring': 'Yes',
            'temperature_rating': '70°C',
            'mandatory_specs': 'voltage_rating;core_count;size_sqmm'
        }
        sku = {
            'voltage_rating': '11kV',
            'conductor_material': 'Copper',
            'insulation_type': 'PVC',
            'core_count': '3',
            'size_sqmm': '240',
            'sheath_type': 'PVC',
            'fire_resistance': 'FRLS',
            'armouring': 'Yes',
            'temperature_rating': '70°C'
        }
        
        score = structured_score(rfp, sku)
        if 0 <= score <= 1:
            print(f"   ✅ structured_score (exact match) = {score:.3f}")
        else:
            print(f"   ❌ structured_score out of range: {score}")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_lazy_loading():
    """Test lazy loading pattern"""
    print("\n🧪 Testing lazy loading...")
    
    try:
        from app import load_artifacts
        
        print("   ⏳ Loading artifacts (first time)...")
        start = time.time()
        artifacts = load_artifacts()
        elapsed = time.time() - start
        
        if artifacts and 'product_df' in artifacts:
            print(f"   ✅ Artifacts loaded in {elapsed:.2f}s")
            print(f"      - Products: {len(artifacts['product_df'])} items")
            return True
        else:
            print(f"   ❌ Artifacts incomplete")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_sample_specs():
    """Test sample specifications"""
    print("\n🧪 Testing sample specifications...")
    
    try:
        from app import SAMPLE_SPECS, hybrid_match, load_artifacts
        
        if len(SAMPLE_SPECS) == 3:
            print(f"   ✅ {len(SAMPLE_SPECS)} sample specs loaded")
        else:
            print(f"   ❌ Expected 3 samples, got {len(SAMPLE_SPECS)}")
            return False
        
        # Try matching first sample
        artifacts = load_artifacts()
        products = artifacts['product_df'].to_dict('records')
        
        print("   ⏳ Running hybrid match on sample 1...")
        start = time.time()
        results = hybrid_match(
            SAMPLE_SPECS[0],
            products,
            top_k=5,
            return_scores=True
        )
        elapsed = time.time() - start
        
        if results:
            print(f"   ✅ Got {len(results)} results in {elapsed:.2f}s")
            best = results[0]
            print(f"      - Top match: {best['product_name'][:40]}")
            print(f"      - Score: {best['hybrid_score']:.3f}")
            return True
        else:
            print(f"   ❌ No results returned")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_creation():
    """Test Gradio interface can be created"""
    print("\n🧪 Testing Gradio interface...")
    
    try:
        from app import create_interface
        
        print("   ⏳ Creating interface...")
        start = time.time()
        demo = create_interface()
        elapsed = time.time() - start
        
        if demo:
            print(f"   ✅ Interface created in {elapsed:.2f}s")
            return True
        else:
            print(f"   ❌ Interface creation returned None")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🚀 Cable Specification Matcher - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Artifacts", test_artifacts),
        ("App Functions", test_app_functions),
        ("Lazy Loading", test_lazy_loading),
        ("Sample Specs", test_sample_specs),
        ("Interface", test_interface_creation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
