"""
Batch Processing Example
========================

Shows how to process multiple cable specifications from a CSV file
and generate a detailed report with metrics.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict

# Import from main app
from app import load_artifacts, hybrid_match
from utils import (
    SpecificationValidator,
    BatchProcessor,
    MetricsComputer,
    ReportGenerator
)

def process_batch_from_csv(
    csv_filepath: str,
    output_dir: str = "batch_results",
    top_k: int = 5
) -> None:
    """
    Process multiple specifications from CSV and generate report
    
    CSV should have columns:
    - voltage_rating
    - conductor_material
    - insulation_type
    - core_count
    - size_sqmm
    - sheath_type
    - fire_resistance
    - armouring
    - temperature_rating
    - standards_required
    - spec_text_raw (optional)
    - ground_truth_sku (optional, for evaluation)
    """
    
    print("🚀 Starting batch processing...")
    print(f"📂 Input: {csv_filepath}")
    
    # Load specifications
    specs = BatchProcessor.load_csv(csv_filepath)
    print(f"✅ Loaded {len(specs)} specifications")
    
    # Load model artifacts
    print("⏳ Loading model artifacts...")
    artifacts = load_artifacts()
    product_catalog = artifacts['product_df'].to_dict('records')
    print(f"✅ Loaded {len(product_catalog)} products")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Process each specification
    results = []
    start_time = time.time()
    
    print(f"\n🔍 Processing {len(specs)} specifications...")
    for i, spec in enumerate(specs, 1):
        print(f"  [{i}/{len(specs)}] Processing: {spec.get('voltage_rating', 'Unknown')} "
              f"{spec.get('core_count', '?')}C {spec.get('size_sqmm', '?')}sqmm")
        
        # Validate specification
        is_valid, errors = SpecificationValidator.validate_spec(spec)
        if not is_valid:
            print(f"    ⚠️  Validation errors: {', '.join(errors)}")
        
        # Run hybrid matching
        matches = hybrid_match(spec, product_catalog, top_k=top_k, return_scores=True)
        
        # Prepare result record
        result = {
            'input_voltage': spec.get('voltage_rating'),
            'input_core_count': spec.get('core_count'),
            'input_size': spec.get('size_sqmm'),
            'top_1_sku': matches[0]['sku'] if matches else 'N/A',
            'top_1_product': matches[0]['product_name'] if matches else 'N/A',
            'top_1_score': matches[0]['hybrid_score'] if matches else 0.0,
            'top_5_skus': '|'.join([m['sku'] for m in matches[:5]]),
            'validation_errors': len(errors),
        }
        
        # Add ground truth if available
        if 'ground_truth_sku' in spec:
            result['ground_truth_sku'] = spec['ground_truth_sku']
            result['correct_in_top1'] = (result['top_1_sku'] == spec['ground_truth_sku'])
            result['correct_in_top5'] = any(
                m['sku'] == spec['ground_truth_sku'] for m in matches[:5]
            )
        
        results.append(result)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Processing complete in {elapsed_time:.2f}s ({elapsed_time/len(specs):.3f}s per spec)")
    
    # Save results
    results_csv = output_path / "results.csv"
    BatchProcessor.save_results(results, str(results_csv), format='csv')
    
    results_json = output_path / "results.json"
    BatchProcessor.save_results(results, str(results_json), format='json')
    
    # Generate report
    report_path = output_path / "report.md"
    ReportGenerator.generate_markdown_report(results, str(report_path))
    
    # Compute metrics if ground truth available
    if any('ground_truth_sku' in r for r in results):
        print("\n📊 Evaluation Metrics:")
        
        top1_correct = sum(1 for r in results if r.get('correct_in_top1', False))
        top5_correct = sum(1 for r in results if r.get('correct_in_top5', False))
        total = len(results)
        
        print(f"  Top-1 Accuracy: {top1_correct}/{total} ({100*top1_correct/total:.2f}%)")
        print(f"  Top-5 Accuracy: {top5_correct}/{total} ({100*top5_correct/total:.2f}%)")
    
    print(f"\n📁 Results saved to: {output_path}/")
    print(f"   - results.csv")
    print(f"   - results.json")
    print(f"   - report.md")


def create_sample_csv(output_file: str = "sample_batch.csv") -> None:
    """Generate sample CSV for batch processing"""
    import pandas as pd
    
    sample_data = [
        {
            'voltage_rating': '11kV',
            'conductor_material': 'Copper',
            'insulation_type': 'PVC',
            'core_count': '3',
            'size_sqmm': '240',
            'sheath_type': 'PVC',
            'fire_resistance': 'FRLS',
            'armouring': 'Yes',
            'temperature_rating': '70°C',
            'standards_required': 'IS 7098',
            'spec_text_raw': 'FRLS 11kV 3C 240sqmm Cu PVC armoured cable as per IS 7098 Part 5'
        },
        {
            'voltage_rating': '0.66kV',
            'conductor_material': 'Aluminium',
            'insulation_type': 'XLPE',
            'core_count': '4',
            'size_sqmm': '120',
            'sheath_type': 'XLPE',
            'fire_resistance': 'Standard',
            'armouring': 'No',
            'temperature_rating': '90°C',
            'standards_required': 'IEC 60502',
            'spec_text_raw': '0.66kV 4C 120sqmm Al XLPE unarmoured cable as per IEC 60502'
        },
        {
            'voltage_rating': '6.6kV',
            'conductor_material': 'Copper',
            'insulation_type': 'PVC',
            'core_count': '2',
            'size_sqmm': '50',
            'sheath_type': 'PVC',
            'fire_resistance': 'FR',
            'armouring': 'Yes',
            'temperature_rating': '70°C',
            'standards_required': 'IS 7098',
            'spec_text_raw': '6.6kV 2C 50sqmm Cu PVC armoured fire-resistant cable'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_file, index=False)
    print(f"✅ Sample CSV created: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process cable specifications")
    parser.add_argument("--csv", type=str, help="Input CSV file with specifications")
    parser.add_argument("--output", type=str, default="batch_results", 
                       help="Output directory for results")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample CSV file for testing")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of results to return per specification")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_csv()
        print("\n💡 Tip: Run batch processing with:")
        print("   python batch_process.py --csv sample_batch.csv")
    elif args.csv:
        process_batch_from_csv(args.csv, args.output, args.top_k)
    else:
        print("📖 Usage:")
        print("\n  Create sample CSV:")
        print("    python batch_process.py --create-sample")
        print("\n  Process specifications:")
        print("    python batch_process.py --csv input.csv --output results/")
        print("\nFor help, run: python batch_process.py --help")
