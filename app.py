"""
RFP Cable Specification Matcher - Full Prototype
=================================================

Objective 1: Hybrid Retrieval Engine
  Structured field matching (50%) + Semantic SBERT similarity (30%)
  + Standards compliance (20%). Best baseline: Top-1 17.8%, Top-5 44.3%.

Objective 2: Robustness Across 7 Real-World Stress Scenarios
  EXP1 Numeric Perturbation, EXP2 Unit Variation, EXP3 Missing Parameters,
  EXP4 Noise Injection (in- and out-of-domain), EXP6 Partial Spec,
  EXP7 Positional Bias.

Objective 3: Explainable & Confidence-Aware Spec Matching
  Per-field breakdown (matched / partially matched / conflicting),
  weighted spec-match percentage, risk level, procurement recommendation.

Model: all-mpnet-base-v2 (SBERT)
"""

import copy
import re
import time
import json
import numpy as np
import pandas as pd
import gradio as gr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────
# LAZY LOADING
# ────────────────────────────────────────────────────────────

_LOADED_MODELS: Dict = {}


def load_sentence_transformer():
    """Lazy-load SBERT model on first call."""
    if 'sbert' not in _LOADED_MODELS:
        from sentence_transformers import SentenceTransformer
        print("Loading SBERT model (first time only)...")
        _LOADED_MODELS['sbert'] = SentenceTransformer('all-mpnet-base-v2')
    return _LOADED_MODELS['sbert']


def load_artifacts():
    """Lazy-load model artifacts from disk."""
    if 'artifacts' not in _LOADED_MODELS:
        print("Loading model artifacts...")
        artifact_dir = Path(__file__).parent / 'obj1_model_artifacts'
        _LOADED_MODELS['artifacts'] = {
            'config':      json.load(open(artifact_dir / 'model_config.json')),
            'product_emb': np.load(artifact_dir / 'product_embeddings_mpnet.npy'),
            'rfp_emb':     np.load(artifact_dir / 'rfp_embeddings_mpnet.npy'),
            'product_df':  pd.read_parquet(artifact_dir / 'product_df.parquet'),
            'rfp_df':      pd.read_parquet(artifact_dir / 'rfp_df_with_gt.parquet'),
        }
    return _LOADED_MODELS['artifacts']

# ────────────────────────────────────────────────────────────
# IEC CABLE SIZE LADDER
# ────────────────────────────────────────────────────────────

_SIZE_LADDER = [
    0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 16.0,
    25.0, 35.0, 50.0, 70.0, 95.0, 120.0, 150.0, 185.0,
    240.0, 300.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
]
_SIZE_SET = set(_SIZE_LADDER)


def _adjacent_sizes(val: float) -> set:
    """Return val and its immediate neighbours on the IEC size ladder."""
    try:
        val = float(val)
        if val not in _SIZE_SET:
            val = min(_SIZE_LADDER, key=lambda x: abs(x - val))
        idx = _SIZE_LADDER.index(val)
        out = {val}
        if idx > 0:               out.add(_SIZE_LADDER[idx - 1])
        if idx < len(_SIZE_LADDER) - 1: out.add(_SIZE_LADDER[idx + 1])
        return out
    except Exception:
        return set()

# ════════════════════════════════════════════════════════════
# OBJECTIVE 1 — HYBRID RETRIEVAL SCORING ENGINE
# ════════════════════════════════════════════════════════════


def structured_score(rfp_row: Dict, sku_row: Dict) -> float:
    """
    Weighted exact-match on structured fields.
    Mandatory fields (declared in rfp_row['mandatory_specs']) get 2x weight.
    Size uses IEC ladder adjacency (covers EXP2 unit-variation).
    """
    mandatory = set(
        f.strip()
        for f in str(rfp_row.get('mandatory_specs', '')).split(';')
        if f.strip()
    )
    exact_fields = [
        'voltage_rating', 'conductor_material', 'insulation_type',
        'sheath_type', 'temperature_rating', 'fire_resistance', 'armouring',
    ]
    score = total = 0.0
    for field in exact_fields:
        w = 2.0 if field in mandatory else 1.0
        total += w
        if (str(rfp_row.get(field, '')).strip().lower() ==
                str(sku_row.get(field, '')).strip().lower()):
            score += w

    w_cc = 2.0 if 'core_count' in mandatory else 1.0
    total += w_cc
    try:
        if int(float(rfp_row['core_count'])) == int(float(sku_row['core_count'])):
            score += w_cc
    except Exception:
        pass

    w_sz = 2.0 if 'size_sqmm' in mandatory else 1.0
    total += w_sz
    try:
        if float(sku_row['size_sqmm']) in _adjacent_sizes(float(rfp_row['size_sqmm'])):
            score += w_sz
    except Exception:
        pass

    return score / total if total > 0 else 0.0


def standards_score(rfp_row: Dict, sku_row: Dict) -> float:
    """Bidirectional substring match on standards field."""
    rfp_std = str(rfp_row.get('standards_required', '')).strip().lower()
    sku_std = str(sku_row.get('standards_complied', '')).strip().lower()
    if not rfp_std or rfp_std == 'nan':
        return 0.0
    return 1.0 if (rfp_std in sku_std or sku_std in rfp_std) else 0.0


def semantic_score(query_text: str, product_text: str,
                   sbert_model=None) -> float:
    """SBERT cosine similarity between spec text and product name."""
    if sbert_model is None:
        sbert_model = load_sentence_transformer()
    try:
        q = str(query_text).strip()
        p = str(product_text).strip()
        if not q or not p:
            return 0.0
        q_emb = sbert_model.encode(q, normalize_embeddings=True)
        p_emb = sbert_model.encode(p, normalize_embeddings=True)
        return float(np.dot(q_emb, p_emb))
    except Exception:
        return 0.0


def hybrid_match(rfp_spec: Dict, product_catalog: List[Dict],
                 top_k: int = 5, return_scores: bool = False) -> List[Dict]:
    """
    Hybrid retrieval: 50% Structured + 30% Semantic + 20% Standards.

    Args:
        rfp_spec:        RFP specification dict.
        product_catalog: List of product dicts from product_df.
        top_k:           Number of results to return.
        return_scores:   Kept for backward compatibility — scores always included.

    Returns:
        List of top_k result dicts with all component scores and product fields.
    """
    sbert = load_sentence_transformer()
    scores = []
    for product in product_catalog:
        s   = structured_score(rfp_spec, product)
        sem = semantic_score(
            rfp_spec.get('spec_text_raw', ''),
            product.get('product_name', ''),
            sbert,
        )
        std = standards_score(rfp_spec, product)
        h   = 0.5 * s + 0.3 * sem + 0.2 * std
        scores.append({
            'sku_id':             product.get('sku_id', ''),
            'product_name':       product.get('product_name', ''),
            'voltage_rating':     product.get('voltage_rating', ''),
            'core_count':         product.get('core_count', ''),
            'size_sqmm':          product.get('size_sqmm', ''),
            'conductor_material': product.get('conductor_material', ''),
            'insulation_type':    product.get('insulation_type', ''),
            'fire_resistance':    product.get('fire_resistance', ''),
            'armouring':          product.get('armouring', ''),
            'standards_complied': product.get('standards_complied', ''),
            'hybrid_score':       h,
            'structured_score':   s,
            'semantic_score':     sem,
            'standards_score':    std,
        })
    scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
    for i, r in enumerate(scores[:top_k], 1):
        r['rank'] = i
    return scores[:top_k]

# ════════════════════════════════════════════════════════════
# OBJECTIVE 2 — ROBUSTNESS EXPERIMENT TRANSFORMATIONS
# ════════════════════════════════════════════════════════════

_IN_DOMAIN_NOISE = (
    "Please refer to technical specifications for cable installation "
    "requirements per site conditions and local regulatory standards."
)
_OUT_DOMAIN_NOISE = (
    "The match was played under floodlights with spectators cheering "
    "loudly from the stands throughout the entire evening session."
)


def parse_spec_text(text: str) -> Dict:
    """Extract structured fields from free-text cable specification."""
    r: Dict = {
        'voltage_rating': '', 'core_count': None, 'size_sqmm': None,
        'conductor_material': '', 'insulation_type': '', 'armouring': '',
        'fire_resistance': '', 'sheath_type': '', 'temperature_rating': '',
        'standards_required': '',
    }
    m = re.search(r'(\d+(?:\.\d+)?)\s*kV', text, re.I)
    if m: r['voltage_rating'] = m.group(1) + 'kV'

    m = re.search(r'(\d+)\s*[Cc](?!\w)', text)
    if m: r['core_count'] = int(m.group(1))

    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:sqmm|mm2|mm²)', text, re.I)
    if m: r['size_sqmm'] = float(m.group(1))

    if re.search(r'\bCu\b|copper', text, re.I):
        r['conductor_material'] = 'Copper'
    elif re.search(r'\bAl\b|alumin', text, re.I):
        r['conductor_material'] = 'Aluminium'

    if re.search(r'XLPE', text, re.I): r['insulation_type'] = 'XLPE'
    elif re.search(r'\bPVC\b', text):  r['insulation_type'] = 'PVC'
    elif re.search(r'\bEPR\b', text, re.I): r['insulation_type'] = 'EPR'

    if re.search(r'unarmou?red', text, re.I): r['armouring'] = 'No'
    elif re.search(r'armou?red', text, re.I): r['armouring'] = 'Yes'

    for fr in ['FRZH', 'LSZH', 'FRLS', 'FR']:
        if fr in text.upper():
            r['fire_resistance'] = fr
            break

    m = re.search(r'(?:IS|IEC|BS)\s*[\d]+(?:[- ]\w+)*', text, re.I)
    if m: r['standards_required'] = m.group(0).strip()

    return r


def apply_numeric_perturbation(rfp: Dict) -> Dict:
    """EXP1: Inflate all numbers in spec_text_raw by +10%; structured columns unchanged."""
    exp = copy.deepcopy(rfp)
    def _bump(m: re.Match) -> str:
        v = float(m.group())
        bumped = int(v * 1.1) if v == int(v) else round(v * 1.1, 2)
        return str(bumped)
    exp['spec_text_raw'] = re.sub(r'\b\d+\.?\d*\b', _bump,
                                  exp.get('spec_text_raw', ''))
    return exp


def apply_unit_variation(rfp: Dict) -> Dict:
    """EXP2: Substitute kV→V and sqmm→mm² in spec_text_raw."""
    exp = copy.deepcopy(rfp)
    text = exp.get('spec_text_raw', '')
    text = re.sub(r'(\d+(?:\.\d+)?)\s*kV',
                  lambda m: str(int(float(m.group(1)) * 1000)) + 'V', text)
    text = re.sub(r'sqmm|sq mm', 'mm²', text, flags=re.I)
    text = re.sub(r'\bCu\b', 'Copper', text)
    text = re.sub(r'\bAl\b', 'Aluminium', text)
    exp['spec_text_raw'] = text
    return exp


def apply_missing_params(rfp: Dict, n_drop: int = 2) -> Dict:
    """EXP3: Clear n non-mandatory structured fields to simulate incomplete RFP."""
    exp = copy.deepcopy(rfp)
    droppable = [
        'conductor_material', 'insulation_type', 'sheath_type',
        'fire_resistance', 'temperature_rating', 'armouring',
    ]
    for field in droppable[:n_drop]:
        exp[field] = ''
    return exp


def apply_noise_injection(rfp: Dict,
                          noise_type: str = 'in_domain') -> Dict:
    """EXP4: Prepend in- or out-of-domain noise to spec_text_raw."""
    exp = copy.deepcopy(rfp)
    noise = _IN_DOMAIN_NOISE if noise_type == 'in_domain' else _OUT_DOMAIN_NOISE
    exp['spec_text_raw'] = noise + ' ' + exp.get('spec_text_raw', '')
    return exp


def apply_partial_spec(rfp: Dict, level: int = 2) -> Dict:
    """EXP6: Truncate spec_text_raw to 2^level tokens (sparse text test)."""
    exp = copy.deepcopy(rfp)
    tokens = exp.get('spec_text_raw', '').split()
    exp['spec_text_raw'] = ' '.join(tokens[:max(1, 2 ** level)])
    return exp


def apply_positional_shift(rfp: Dict, position: str = 'end') -> Dict:
    """EXP7: Bury spec text inside a longer document (primacy-bias test)."""
    exp = copy.deepcopy(rfp)
    filler = (
        "This document outlines procurement requirements for the upcoming project. "
        "Multiple items are listed for review and approval by the technical committee. "
        "All specifications must be verified against the approved vendor catalog data. "
    )
    spec = exp.get('spec_text_raw', '')
    if position == 'end':
        exp['spec_text_raw'] = filler + spec
    elif position == 'middle':
        half = len(filler) // 2
        exp['spec_text_raw'] = filler[:half] + spec + filler[half:]
    return exp


def _apply_experiment(rfp: Dict, exp_key: str) -> Dict:
    """Dispatch the correct experiment transform by key."""
    dispatch = {
        'clean':      lambda r: copy.deepcopy(r),
        'numeric':    apply_numeric_perturbation,
        'unit_var':   apply_unit_variation,
        'missing':    apply_missing_params,
        'noise_in':   lambda r: apply_noise_injection(r, 'in_domain'),
        'noise_out':  lambda r: apply_noise_injection(r, 'out_of_domain'),
        'partial':    lambda r: apply_partial_spec(r, level=2),
        'positional': lambda r: apply_positional_shift(r, 'end'),
    }
    return dispatch.get(exp_key, lambda r: copy.deepcopy(r))(rfp)


EXPERIMENT_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    'clean': {
        'title': 'Clean Baseline',
        'description': 'Unmodified RFP — standard retrieval.',
        'what_changes': 'Nothing. Spec text and structured fields are unmodified.',
        'hybrid_advantage': 'Establishes the best-case baseline for comparison.',
    },
    'numeric': {
        'title': 'EXP1: Numeric Perturbation',
        'description': 'Numbers in spec_text_raw inflated +10% (e.g. 240sqmm → 264sqmm). Structured columns unchanged.',
        'what_changes': 'All numbers in raw text are wrong; structured columns are correct.',
        'hybrid_advantage': 'Structured score reads the column value, not the text — immune to numeric text errors.',
    },
    'unit_var': {
        'title': 'EXP2: Unit Variation',
        'description': 'Unit notation changed (11kV → 11000V, sqmm → mm²). Structured columns unchanged.',
        'what_changes': 'Same physical value expressed in different notation.',
        'hybrid_advantage': 'Ladder-based size matching and voltage field are unit-agnostic; SBERT is confused by token change.',
    },
    'missing': {
        'title': 'EXP3: Missing Parameters',
        'description': '2 non-mandatory structured fields cleared (conductor_material, insulation_type).',
        'what_changes': 'conductor_material and insulation_type set to empty — incomplete RFP.',
        'hybrid_advantage': 'Remaining mandatory fields (voltage, size, core_count) still anchor the score.',
    },
    'noise_in': {
        'title': 'EXP4: In-Domain Noise',
        'description': 'Engineering boilerplate (~20 words) prepended to spec_text_raw.',
        'what_changes': 'Relevant but non-specific text added before spec.',
        'hybrid_advantage': 'Structured score is immune. Semantic diluted but compensated by 50% structural weight.',
    },
    'noise_out': {
        'title': 'EXP4: Out-of-Domain Noise',
        'description': 'Completely irrelevant non-engineering text prepended to spec_text_raw.',
        'what_changes': 'Off-topic text added before spec.',
        'hybrid_advantage': 'Out-of-domain noise hurts SBERT more than in-domain noise. Structural score unaffected.',
    },
    'partial': {
        'title': 'EXP6: Partial Specification',
        'description': 'spec_text_raw truncated to first 4 tokens only.',
        'what_changes': 'Only first 4 words of spec retained (extremely sparse text).',
        'hybrid_advantage': 'Structured columns still complete — hybrid recovers where SBERT fails at sparse text.',
    },
    'positional': {
        'title': 'EXP7: Positional Bias',
        'description': 'Spec buried at the end of a ~45-word document.',
        'what_changes': 'Filler text prepended — spec at end instead of beginning.',
        'hybrid_advantage': 'Structured fields are position-invariant. SBERT shows attention primacy bias.',
    },
}

EXPERIMENT_MODES: List[Tuple[str, str]] = [
    ('Clean Baseline',                'clean'),
    ('EXP1 — Numeric Perturbation',  'numeric'),
    ('EXP2 — Unit Variation',        'unit_var'),
    ('EXP3 — Missing Parameters',    'missing'),
    ('EXP4 — In-Domain Noise',       'noise_in'),
    ('EXP4 — Out-of-Domain Noise',   'noise_out'),
    ('EXP6 — Partial Spec',          'partial'),
    ('EXP7 — Positional Bias',       'positional'),
]
_EXP_LABELS  = [label for label, _ in EXPERIMENT_MODES]
_EXP_KEY_MAP = {label: key for label, key in EXPERIMENT_MODES}

# ════════════════════════════════════════════════════════════
# OBJECTIVE 3 — EXPLAINABILITY & CONFIDENCE ENGINE
# ════════════════════════════════════════════════════════════


def _parse_size_val(val) -> Optional[float]:
    """Strip unit suffixes and return numeric size."""
    try:
        return float(
            str(val).replace('sqmm', '').replace('sq mm', '')
                    .replace('mm²', '').replace(' ', '').strip()
        )
    except Exception:
        return None


def classify_match(h: float, s: float, sem: float) -> str:
    """
    Classify an RFP-product pair into OBJ3 match category.
    Thresholds calibrated against obj3_top1_summary.csv.
    """
    if h >= 0.72 and s >= 0.70 and sem >= 0.45:
        return 'matching'
    elif h >= 0.45:
        return 'partially matching'
    return 'not matching'


_FIELD_DEFS: List[Tuple[str, str, str, bool]] = [
    ('voltage_rating',     'voltage_rating',     'Voltage Rating',        True),
    ('conductor_material', 'conductor_material',  'Conductor Material',    True),
    ('insulation_type',    'insulation_type',     'Insulation Type',       True),
    ('core_count',         'core_count',          'Core Count',            True),
    ('size_sqmm',          'size_sqmm',           'Cross-Section (sqmm)',  True),
    ('sheath_type',        'sheath_type',         'Sheath Type',           False),
    ('fire_resistance',    'fire_resistance',     'Fire Resistance',       False),
    ('armouring',          'armouring',           'Armoured',              False),
    ('temperature_rating', 'temperature_rating',  'Temperature Rating',    False),
    ('standards_required', 'standards_complied',  'Standards',             False),
]

_BLANK_VALS = {'', 'unknown', 'nan', 'none', '0', 'not specified'}


def generate_field_explanation(rfp_spec: Dict, product: Dict) -> Dict:
    """
    Per-field comparison between an RFP spec and a product (OBJ3 core logic).
    Returns weighted spec-match %, risk level, and per-field status list.
    """
    field_results: List[Dict] = []
    matched = partial_ct = conflict = missing = 0
    w_num = w_den = 0.0

    for rfp_key, prod_key, label, is_critical in _FIELD_DEFS:
        weight   = 2.0 if is_critical else 1.0
        rfp_raw  = str(rfp_spec.get(rfp_key,  '')).strip()
        prod_raw = str(product.get(prod_key, '')).strip()
        rfp_val  = '' if rfp_raw.lower()  in _BLANK_VALS else rfp_raw
        prod_val = '' if prod_raw.lower() in {'', 'nan', 'none'} else prod_raw

        if not rfp_val:
            status, note = 'not_specified', 'Not specified in RFP'
            missing += 1

        elif not prod_val:
            status, note = 'conflict', 'Not available for this product'
            conflict += 1
            w_den += weight

        elif rfp_key == 'size_sqmm':
            r_n, p_n = _parse_size_val(rfp_val), _parse_size_val(prod_val)
            if r_n is not None and p_n is not None:
                if abs(r_n - p_n) < 0.01:
                    status, note = 'match',   'Exact match'
                    matched += 1;  w_num += weight
                elif p_n in _adjacent_sizes(r_n):
                    status, note = 'partial', 'Adjacent size (±1 step on IEC ladder)'
                    partial_ct += 1; w_num += weight * 0.5
                else:
                    status, note = 'conflict', f'Size mismatch (RFP: {rfp_val}, Product: {prod_val})'
                    conflict += 1
            else:
                status, note = 'conflict', 'Unable to parse size values'
                conflict += 1
            w_den += weight

        elif rfp_key == 'core_count':
            try:
                r_i, p_i = int(float(rfp_val)), int(float(prod_val))
                if r_i == p_i:
                    status, note = 'match',   'Exact match'
                    matched += 1; w_num += weight
                else:
                    status, note = 'conflict', f'RFP requires {r_i}C, product has {p_i}C'
                    conflict += 1
            except Exception:
                status, note = 'conflict', 'Cannot compare core counts'
                conflict += 1
            w_den += weight

        elif rfp_key == 'standards_required':
            r_l, p_l = rfp_val.lower(), prod_val.lower()
            if r_l in p_l or p_l in r_l:
                status, note = 'match',   'Standards compliance confirmed'
                matched += 1; w_num += weight
            else:
                status, note = 'conflict', f'Product does not comply with {rfp_val}'
                conflict += 1
            w_den += weight

        else:
            if rfp_val.lower() == prod_val.lower():
                status, note = 'match',   'Exact match'
                matched += 1; w_num += weight
            else:
                status, note = 'conflict', f'RFP: {rfp_val}  vs  Product: {prod_val}'
                conflict += 1
            w_den += weight

        field_results.append({
            'field':         label,
            'is_critical':   is_critical,
            'rfp_value':     rfp_val  or '—',
            'product_value': prod_val or '—',
            'status':        status,
            'note':          note,
        })

    spec_match_pct      = (w_num / w_den * 100) if w_den > 0 else 0.0
    critical_conflicts  = sum(
        1 for r in field_results if r['is_critical'] and r['status'] == 'conflict'
    )

    if spec_match_pct >= 75 and critical_conflicts == 0:
        risk_level, risk_icon = 'Low',    '🟢'
        procurement_note = 'Safe to procure — all critical specifications are met.'
    elif spec_match_pct >= 50 and critical_conflicts <= 1:
        risk_level, risk_icon = 'Medium', '🟡'
        procurement_note = 'Review required — some specs deviate. Confirm with supplier before ordering.'
    else:
        risk_level, risk_icon = 'High',   '🔴'
        procurement_note = 'High risk — major conflicts detected. Do not procure without full technical review.'

    return {
        'field_results':     field_results,
        'matched':           matched,
        'partial':           partial_ct,
        'conflict':          conflict,
        'missing':           missing,
        'spec_match_pct':    spec_match_pct,
        'risk_level':        risk_level,
        'risk_icon':         risk_icon,
        'procurement_note':  procurement_note,
        'critical_conflicts': critical_conflicts,
    }


_STATUS_ICON = {
    'match': '✅', 'partial': '⚠️', 'conflict': '❌', 'not_specified': '—',
}


def _format_match_block(rank: int, result: Dict,
                        rfp_spec: Dict, product: Dict) -> str:
    """Render one product's OBJ3 explanation as a markdown block."""
    exp  = generate_field_explanation(rfp_spec, product)
    cat  = classify_match(result['hybrid_score'],
                          result['structured_score'],
                          result['semantic_score'])
    badge = {
        'matching':          '✅ MATCHING',
        'partially matching': '⚠️ PARTIALLY MATCHING',
        'not matching':      '❌ NOT MATCHING',
    }[cat]

    s, sem, std = (result['structured_score'],
                   result['semantic_score'],
                   result['standards_score'])

    out  = f"---\n### Rank {rank} — {result['product_name'][:65]}\n\n"
    out += (f"**SKU:** `{result['sku_id']}` &nbsp;|&nbsp; "
            f"**Status:** {badge} &nbsp;|&nbsp; "
            f"**Spec-Match:** `{exp['spec_match_pct']:.0f}%` &nbsp;|&nbsp; "
            f"**Risk:** {exp['risk_icon']} {exp['risk_level']}\n\n")
    out += f"> {exp['procurement_note']}\n\n"

    out += "**Score Breakdown:**\n\n"
    out += "| Component | Score | Weight | Contribution |\n"
    out += "|-----------|------:|-------:|-------------:|\n"
    out += f"| Structured field matching   | `{s:.3f}`   | 50% | `{s*0.5:.3f}` |\n"
    out += f"| Semantic (SBERT) similarity | `{sem:.3f}` | 30% | `{sem*0.3:.3f}` |\n"
    out += f"| Standards compliance        | `{std:.3f}` | 20% | `{std*0.2:.3f}` |\n"
    out += f"| **Overall Hybrid Score**    | **`{result['hybrid_score']:.3f}`** | — | — |\n\n"

    out += "**Specification-Level Breakdown:**\n\n"
    out += "| Field | Crit | RFP Requirement | Product Value | Match Status |\n"
    out += "|-------|:----:|----------------|---------------|:------------:|\n"
    for fr in exp['field_results']:
        icon  = _STATUS_ICON.get(fr['status'], '—')
        crit  = '⭐' if fr['is_critical'] else ''
        if fr['status'] in ('match', 'not_specified'):
            cell = icon
        elif fr['status'] == 'partial':
            cell = f"{icon} Close"
        else:
            cell = f"{icon} {fr['note']}"
        out += (f"| {fr['field']} | {crit} | {fr['rfp_value']} "
                f"| {fr['product_value']} | {cell} |\n")

    out += (f"\n**Summary:** {exp['matched']} matched ✅ &nbsp;|&nbsp; "
            f"{exp['partial']} close ⚠️ &nbsp;|&nbsp; "
            f"{exp['conflict']} conflicting ❌ &nbsp;|&nbsp; "
            f"{exp['missing']} not specified\n\n")
    return out

# ────────────────────────────────────────────────────────────
# SAMPLE SPECIFICATIONS
# Keys use rfp_spec format so SAMPLE_SPECS[i] can be passed
# directly to hybrid_match() (required by test_prototype.py).
# ────────────────────────────────────────────────────────────

SAMPLE_SPECS: List[Dict] = [
    {
        'label':              '11kV 3C 240sqmm Cu PVC FRLS Armoured',
        'spec_text_raw':      'FRLS 11kV 3C 240sqmm Cu PVC armoured cable as per IS 7098',
        'voltage_rating':     '11kV',
        'conductor_material': 'Copper',
        'insulation_type':    'PVC',
        'core_count':         '3',
        'size_sqmm':          '240',
        'sheath_type':        'PVC',
        'fire_resistance':    'FRLS',
        'armouring':          'Yes',
        'temperature_rating': '70°C',
        'standards_required': 'IS 7098',
        'mandatory_specs':    'voltage_rating;size_sqmm;core_count',
    },
    {
        'label':              '0.66kV 4C 120sqmm Al XLPE Unarmoured',
        'spec_text_raw':      '0.66kV 4C 120sqmm Al XLPE unarmoured cable as per IEC 60502',
        'voltage_rating':     '0.66kV',
        'conductor_material': 'Aluminium',
        'insulation_type':    'XLPE',
        'core_count':         '4',
        'size_sqmm':          '120',
        'sheath_type':        'XLPE',
        'fire_resistance':    'Standard',
        'armouring':          'No',
        'temperature_rating': '90°C',
        'standards_required': 'IEC 60502',
        'mandatory_specs':    'voltage_rating;size_sqmm',
    },
    {
        'label':              '6.6kV 2C 50sqmm Cu PVC FR Armoured',
        'spec_text_raw':      '6.6kV 2C 50sqmm Cu PVC armoured fire-resistant cable IS 7098',
        'voltage_rating':     '6.6kV',
        'conductor_material': 'Copper',
        'insulation_type':    'PVC',
        'core_count':         '2',
        'size_sqmm':          '50',
        'sheath_type':        'PVC',
        'fire_resistance':    'FR',
        'armouring':          'Yes',
        'temperature_rating': '70°C',
        'standards_required': 'IS 7098',
        'mandatory_specs':    'voltage_rating;core_count',
    },
]

# ────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ────────────────────────────────────────────────────────────

def create_interface():
    """Build the full Gradio UI covering OBJ1, OBJ2 and OBJ3."""

    artifacts = load_artifacts()
    product_catalog = artifacts['product_df'].to_dict('records')
    sku_map = {p.get('sku_id'): p for p in product_catalog}

    # ── Build rfp_spec dict from form field values ─────────────
    def _build_rfp(spec_text, voltage, conductor, insulation,
                   cores, size, sheath, fire, armouring,
                   temp, standards, mandatory) -> Dict:
        return {
            'spec_text_raw':      spec_text  or '',
            'voltage_rating':     voltage    or 'Unknown',
            'conductor_material': conductor  or 'Unknown',
            'insulation_type':    insulation or 'Unknown',
            'core_count':         cores      or '0',
            'size_sqmm':          size       or '0',
            'sheath_type':        sheath     or 'Unknown',
            'fire_resistance':    fire       or 'Standard',
            'armouring':          armouring  or 'No',
            'temperature_rating': temp       or '70°C',
            'standards_required': standards  or 'Not specified',
            'mandatory_specs':    mandatory  or 'voltage_rating;size_sqmm;core_count',
        }

    # ── OBJ1 + OBJ2: Single retrieval with optional experiment ──
    def run_retrieval(spec_text, voltage, conductor, insulation,
                      cores, size, sheath, fire, armouring, temp,
                      standards, mandatory, experiment_mode, top_k):
        rfp     = _build_rfp(spec_text, voltage, conductor, insulation,
                              cores, size, sheath, fire, armouring,
                              temp, standards, mandatory)
        exp_key = _EXP_KEY_MAP.get(experiment_mode, 'clean')
        rfp_exp = _apply_experiment(rfp, exp_key)

        t0 = time.perf_counter()
        results = hybrid_match(rfp_exp, product_catalog, top_k=int(top_k))
        latency = int((time.perf_counter() - t0) * 1000)

        exp_info = EXPERIMENT_DESCRIPTIONS[exp_key]
        eff_text = rfp_exp.get('spec_text_raw', '')[:350]
        info_md  = (
            f"### {exp_info['title']}\n\n"
            f"**What this tests:** {exp_info['description']}\n\n"
            f"**What changed:** `{exp_info['what_changes']}`\n\n"
            f"**Hybrid advantage:** _{exp_info['hybrid_advantage']}_\n\n"
            f"---\n**Effective spec text sent to model:**\n```\n{eff_text}\n```\n"
        )

        if not results:
            return info_md, "_No results found._", pd.DataFrame()

        top = results[0]
        score_md = (
            f"### Top Match: `{top['sku_id']}` — {top['product_name']}\n\n"
            "| Score Component | Value | Weight |\n"
            "|---|---|---|\n"
            f"| Structured field matching   | **{top['structured_score']:.3f}** | 50% |\n"
            f"| Semantic (SBERT) similarity | **{top['semantic_score']:.3f}** | 30% |\n"
            f"| Standards compliance        | **{top['standards_score']:.3f}** | 20% |\n"
            f"| **Hybrid Total**            | **{top['hybrid_score']:.3f}** | 100% |\n\n"
            f"**Retrieval latency:** {latency} ms\n"
        )

        rows = [{
            'Rank':         r['rank'],
            'SKU':          r['sku_id'],
            'Product':      r['product_name'],
            'Voltage':      r['voltage_rating'],
            'Cores':        r['core_count'],
            'Size (sqmm)':  r['size_sqmm'],
            'Conductor':    r['conductor_material'],
            'Insulation':   r['insulation_type'],
            'Fire':         r['fire_resistance'],
            'Armoured':     r['armouring'],
            'Standards':    r['standards_complied'],
            'Hybrid':       round(r['hybrid_score'],     3),
            'Struct':       round(r['structured_score'], 3),
            'Semantic':     round(r['semantic_score'],   3),
            'Std':          round(r['standards_score'],  3),
        } for r in results]

        return info_md, score_md, pd.DataFrame(rows)

    # ── Auto-parse structured fields from free text ─────────────
    def auto_parse(spec_text):
        p = parse_spec_text(spec_text)
        return (
            p['voltage_rating'],
            p['conductor_material'],
            p['insulation_type'],
            str(p['core_count'])  if p['core_count']  is not None else '',
            str(p['size_sqmm'])   if p['size_sqmm']   is not None else '',
            p['sheath_type'],
            p['fire_resistance'],
            p['armouring'],
            p['temperature_rating'],
            p['standards_required'],
        )

    # ── OBJ2: Compare all 8 experiment modes at once ────────────
    def compare_all_experiments(spec_text, voltage, conductor, insulation,
                                cores, size, sheath, fire, armouring, temp,
                                standards, mandatory):
        if not spec_text.strip():
            return pd.DataFrame(), "_Enter a spec text on the OBJ1 tab first._"

        rfp  = _build_rfp(spec_text, voltage, conductor, insulation,
                           cores, size, sheath, fire, armouring,
                           temp, standards, mandatory)
        rows = []
        for label, key in EXPERIMENT_MODES:
            rfp_exp = _apply_experiment(rfp, key)
            res     = hybrid_match(rfp_exp, product_catalog, top_k=1)
            if res:
                r = res[0]
                rows.append({
                    'Experiment':    label,
                    'Top-1 SKU':    r['sku_id'],
                    'Top-1 Product': r['product_name'][:50],
                    'Hybrid':        round(r['hybrid_score'],     3),
                    'Struct':        round(r['structured_score'], 3),
                    'Semantic':      round(r['semantic_score'],   3),
                    'Std':           round(r['standards_score'],  3),
                })

        summary = (
            "### Key Insight\n\n"
            "The **Struct** score stays stable across EXP1–EXP7 because it reads "
            "structured columns (`voltage_rating`, `core_count`, `size_sqmm` etc.) "
            "— NOT the raw spec text.\n\n"
            "The **Semantic** score fluctuates when text is perturbed, noisy or partial "
            "— this is the SBERT failure mode that the Hybrid model compensates for "
            "via its 50% structural weight.\n\n"
            "The **Hybrid** score therefore stays robust across all 8 scenarios."
        )
        return pd.DataFrame(rows), summary

    # ── OBJ3: Analytical SHAP for a single RFP-SKU pair ───────────
    _OBJ3_EXACT_FIELDS = [
        'voltage_rating', 'conductor_material', 'insulation_type',
        'sheath_type', 'temperature_rating', 'fire_resistance', 'armouring',
    ]
    _OBJ3_FIELD_ORDER  = _OBJ3_EXACT_FIELDS + ['core_count', 'size_sqmm']
    _OBJ3_FEATURE_LABELS = [
        'Voltage Rating', 'Conductor Material', 'Insulation Type',
        'Sheath Type', 'Temp. Rating', 'Fire Resistance', 'Armouring',
        'Core Count', 'Cross-Section (sqmm)', 'Semantic (SBERT)', 'Standards',
    ]
    _OBJ3_MANDATORY = {'voltage_rating', 'size_sqmm', 'core_count'}
    _OBJ3_WEIGHTS   = np.array(
        [2.0 if f in _OBJ3_MANDATORY else 1.0 for f in _OBJ3_FIELD_ORDER],
        dtype=np.float32
    )
    _OBJ3_TOTAL_W = float(_OBJ3_WEIGHTS.sum())
    _OBJ3_COEFS   = np.zeros(11, dtype=np.float32)
    for _ii in range(9):
        _OBJ3_COEFS[_ii] = (_OBJ3_WEIGHTS[_ii] / _OBJ3_TOTAL_W) * 0.5
    _OBJ3_COEFS[9]  = 0.3
    _OBJ3_COEFS[10] = 0.2
    # Baseline (background) — uniform 50% match for all features
    _OBJ3_BACKGROUND = np.array([0.5] * 11, dtype=np.float32)

    def _compute_shap_features(rfp_spec: Dict, product: Dict,
                                sem: float, std: float) -> np.ndarray:
        feat = []
        for field in _OBJ3_EXACT_FIELDS:
            rv = str(rfp_spec.get(field, '')).strip().lower()
            pv = str(product.get(field, '')).strip().lower()
            feat.append(1.0 if rv and pv and rv == pv else 0.0)
        try:
            feat.append(1.0 if int(float(rfp_spec['core_count'])) ==
                               int(float(product['core_count'])) else 0.0)
        except Exception:
            feat.append(0.0)
        try:
            feat.append(1.0 if float(product['size_sqmm']) in
                               _adjacent_sizes(float(rfp_spec['size_sqmm'])) else 0.0)
        except Exception:
            feat.append(0.0)
        feat.append(float(sem))
        feat.append(float(std))
        return np.array(feat, dtype=np.float32)

    def _build_shap_chart(rfp_spec: Dict, result: Dict, product: Dict):
        """Return a matplotlib Figure with the SHAP waterfall for the top-1 match."""
        import matplotlib
        import matplotlib.patches
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        feat_vec  = _compute_shap_features(rfp_spec, product,
                                           result['semantic_score'],
                                           result['standards_score'])
        shap_vals = (feat_vec - _OBJ3_BACKGROUND) * _OBJ3_COEFS
        expected  = float(np.dot(_OBJ3_BACKGROUND, _OBJ3_COEFS))
        final_val = result['hybrid_score']

        # Sort by |SHAP|
        order  = np.argsort(np.abs(shap_vals))[::-1]
        labels = [_OBJ3_FEATURE_LABELS[i] for i in order]
        vals   = [shap_vals[i] for i in order]
        fv     = [feat_vec[i] for i in order]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        colors  = ['#1565C0' if v >= 0 else '#C62828' for v in vals]
        bars    = ax.barh(range(len(labels)), vals, color=colors,
                          edgecolor='white', linewidth=0.5, height=0.65)

        for bar, v, fval in zip(bars, vals, fv):
            ax.text(bar.get_width() + (0.001 if v >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:+.4f}  (feat={fval:.2f})',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=8)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Contribution to Hybrid Score', fontsize=10)
        ax.set_title(
            f'SHAP Attribution — {result["product_name"][:55]}\n'
            f'Baseline E[f(x)]={expected:.3f} → Final f(x)={final_val:.3f} | '
            f'Sum of SHAPs={shap_vals.sum():+.4f}',
            fontsize=10, fontweight='bold'
        )

        pos_patch = matplotlib.patches.Patch(color='#1565C0', label='Positive (pushes score up)')
        neg_patch = matplotlib.patches.Patch(color='#C62828', label='Negative (pushes score down)')
        ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc='lower right')

        plt.tight_layout()
        return fig

    # ── OBJ3: Explainability report + SHAP chart ────────────────
    def run_obj3_explain(spec_text, voltage, conductor, insulation,
                         cores, size, sheath, fire, armouring, temp,
                         standards, mandatory):
        rfp     = _build_rfp(spec_text, voltage, conductor, insulation,
                              cores, size, sheath, fire, armouring,
                              temp, standards, mandatory)
        results = hybrid_match(rfp, product_catalog, top_k=5)

        out  = "## Explainability & Confidence Report (Objective 3)\n\n"
        out += (
            "**Objective:** Generate interpretable, spec-level explanations highlighting "
            "**matched**, **partially matched**, and **conflicting** requirements — "
            "with confidence and risk indicators for procurement decision-making.\n\n"
        )
        for i, result in enumerate(results, 1):
            product = sku_map.get(result['sku_id'])
            if product:
                out += _format_match_block(i, result, rfp, product)
        out += (
            "\n---\n"
            "*⭐ Critical field &nbsp;|&nbsp; "
            "✅ Match &nbsp;|&nbsp; "
            "⚠️ Close &nbsp;|&nbsp; "
            "❌ Conflict &nbsp;|&nbsp; "
            "— Not specified*\n"
        )

        # SHAP chart for top-1 match
        shap_fig = None
        if results:
            top1_product = sku_map.get(results[0]['sku_id'])
            if top1_product:
                try:
                    shap_fig = _build_shap_chart(rfp, results[0], top1_product)
                except Exception:
                    shap_fig = None

        return out, shap_fig

    # ── Fill form from sample spec ──────────────────────────────
    def fill_sample(idx: int):
        s = SAMPLE_SPECS[int(idx)]
        return (
            s['spec_text_raw'],
            s['voltage_rating'],
            s['conductor_material'],
            s['insulation_type'],
            s['core_count'],
            s['size_sqmm'],
            s['sheath_type'],
            s['fire_resistance'],
            s['armouring'],
            s['temperature_rating'],
            s['standards_required'],
            s['mandatory_specs'],
        )

    # ════════════════════════════════════════════════════════════
    # BUILD GRADIO LAYOUT
    # ════════════════════════════════════════════════════════════
    with gr.Blocks(title="Cable Spec Matcher", theme=gr.themes.Soft()) as demo:

        gr.HTML("""
        <div style="text-align:center;padding:14px 0 6px 0;">
            <h1 style="margin:0 0 4px 0;">⚡ RFP Cable Specification Matcher</h1>
            <p style="color:#555;margin:0;font-size:0.95em;">
                OBJ1: Hybrid Retrieval &nbsp;|&nbsp;
                OBJ2: Robustness Tests (EXP1–EXP7) &nbsp;|&nbsp;
                OBJ3: Explainability &amp; Confidence
            </p>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════════════════
            # TAB 1 — OBJ1: HYBRID MATCH
            # ════════════════════════════════════════════════
            with gr.Tab("🎯 OBJ1: Hybrid Match"):
                gr.Markdown("""
**Objective 1:** Match electrical cable RFP specifications against the product catalog
using a hybrid retrieval engine — **Structured (50%) + Semantic SBERT (30%) + Standards (20%)**.

Select an experiment mode to also see OBJ2 robustness scenarios in action.
                """)

                with gr.Row():

                    # ── Left column: input form ───────────
                    with gr.Column(scale=2):

                        gr.Markdown("#### Specification Input")
                        spec_text = gr.Textbox(
                            label="Free-text Spec (used for semantic score)",
                            placeholder="e.g. FRLS 11kV 3C 240sqmm Cu PVC armoured cable as per IS 7098",
                            lines=2,
                        )
                        auto_btn = gr.Button("Auto-Parse Fields from Text Above",
                                             variant="secondary", size="sm")

                        gr.Markdown("#### Structured Fields *(read directly by the scoring engine)*")
                        with gr.Row():
                            voltage   = gr.Textbox(label="Voltage Rating",  placeholder="11kV")
                            cores     = gr.Textbox(label="Core Count",      placeholder="3")
                            size      = gr.Textbox(label="Size (sqmm)",     placeholder="240")
                        with gr.Row():
                            conductor  = gr.Dropdown(
                                choices=["Copper", "Aluminium", "Unknown"],
                                label="Conductor", value="Copper")
                            insulation = gr.Dropdown(
                                choices=["PVC", "XLPE", "EPR", "Unknown"],
                                label="Insulation", value="PVC")
                            sheath     = gr.Dropdown(
                                choices=["PVC", "XLPE", "LSZH", "Unknown"],
                                label="Sheath", value="PVC")
                        with gr.Row():
                            fire      = gr.Dropdown(
                                choices=["Standard", "FR", "FRLS", "FRZH", "Unknown"],
                                label="Fire Resistance", value="Standard")
                            armouring = gr.Dropdown(
                                choices=["Yes", "No"],
                                label="Armoured", value="No")
                            temp      = gr.Textbox(label="Temperature", placeholder="70°C")

                        standards = gr.Textbox(label="Standards Required",
                                               placeholder="IS 7098 / IEC 60502")
                        mandatory = gr.Textbox(
                            label="Mandatory Fields (2x weight in scoring)",
                            value="voltage_rating;size_sqmm;core_count",
                            placeholder="voltage_rating;size_sqmm;core_count",
                        )

                        gr.Markdown("#### Experiment Mode *(applies OBJ2 stress-test to the spec)*")
                        exp_mode = gr.Radio(
                            choices=_EXP_LABELS,
                            value=_EXP_LABELS[0],
                            label="",
                        )
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                1, 10, value=5, step=1, label="Top-K results")
                            run_btn = gr.Button("Run Retrieval",
                                                variant="primary", scale=2)

                        gr.Markdown("#### Load Sample Specification")
                        with gr.Row():
                            for idx, s in enumerate(SAMPLE_SPECS):
                                gr.Button(f"Sample {idx+1}: {s['label']}",
                                          size="sm").click(
                                    fn=lambda i=idx: fill_sample(i),
                                    outputs=[spec_text, voltage, conductor,
                                             insulation, cores, size, sheath,
                                             fire, armouring, temp,
                                             standards, mandatory],
                                )

                    # ── Right column: results ─────────────
                    with gr.Column(scale=3):
                        gr.Markdown("#### Results")
                        exp_info_md   = gr.Markdown("*Run a query to see experiment info.*")
                        score_info_md = gr.Markdown()
                        results_table = gr.DataFrame(label="Ranked Matches", wrap=True)

                # ── Wire OBJ1 button ──────────────────────
                _obj1_inputs = [
                    spec_text, voltage, conductor, insulation,
                    cores, size, sheath, fire, armouring, temp,
                    standards, mandatory, exp_mode, top_k_slider,
                ]
                run_btn.click(
                    fn=run_retrieval, inputs=_obj1_inputs,
                    outputs=[exp_info_md, score_info_md, results_table],
                )
                auto_btn.click(
                    fn=auto_parse, inputs=[spec_text],
                    outputs=[voltage, conductor, insulation, cores, size,
                             sheath, fire, armouring, temp, standards],
                )

            # ════════════════════════════════════════════════
            # TAB 2 — OBJ2: ROBUSTNESS TESTS
            # ════════════════════════════════════════════════
            with gr.Tab("🔬 OBJ2: Robustness Tests"):
                gr.Markdown("""
## Objective 2: Robustness Across 7 Real-World Stress Scenarios

Run the **same specification** through all 8 experiment modes simultaneously
to see how the Hybrid model stays robust where pure SBERT degrades.

> Fill in a spec on the **OBJ1 tab** (or load a sample), then click below.
> The comparison uses the same form values — no need to re-enter anything.
                """)

                compare_btn = gr.Button("Run All 8 Experiments", variant="primary", size="lg")
                compare_summary = gr.Markdown()
                compare_table   = gr.DataFrame(label="Experiment Comparison", wrap=True)

                # Reuse OBJ1 inputs (shared Gradio components)
                _obj2_inputs = [
                    spec_text, voltage, conductor, insulation,
                    cores, size, sheath, fire, armouring, temp,
                    standards, mandatory,
                ]
                compare_btn.click(
                    fn=compare_all_experiments, inputs=_obj2_inputs,
                    outputs=[compare_table, compare_summary],
                )

            # ════════════════════════════════════════════════
            # TAB 3 — OBJ3: EXPLAINABILITY
            # ════════════════════════════════════════════════
            with gr.Tab("🔍 OBJ3: Explainability"):
                gr.Markdown("""
## Objective 3: Explainable & Confidence-Aware Spec Matching

> *Generate interpretable, spec-level explanations highlighting **matched**, **partially matched**,
> and **conflicting** requirements — with a meaningful spec-match percentage and risk indicators
> to support procurement decision-making.*

Enter a specification below (or load a sample) and click **Generate Explanation Report** to see:

| Output | Description |
|---|---|
| **Match Status** | ✅ Matching / ⚠️ Partially Matching / ❌ Not Matching |
| **Spec-Match %** | Weighted confidence score across all spec fields |
| **Risk Level** | 🟢 Low / 🟡 Medium / 🔴 High |
| **Procurement Note** | Actionable recommendation for buyers |
| **Score Breakdown** | Structured · Semantic · Standards contributions |
| **Field Table** | Every spec field compared side-by-side with status icons |
                """)

                with gr.Row():
                    e_voltage   = gr.Textbox(label="Voltage Rating",  placeholder="11kV")
                    e_cores     = gr.Textbox(label="Core Count",      placeholder="3")
                    e_size      = gr.Textbox(label="Size (sqmm)",     placeholder="240")
                with gr.Row():
                    e_conductor  = gr.Dropdown(
                        choices=["Copper", "Aluminium", "Unknown"],
                        label="Conductor", value="Copper")
                    e_insulation = gr.Dropdown(
                        choices=["PVC", "XLPE", "EPR", "Unknown"],
                        label="Insulation", value="PVC")
                    e_sheath     = gr.Dropdown(
                        choices=["PVC", "XLPE", "LSZH", "Unknown"],
                        label="Sheath", value="PVC")
                with gr.Row():
                    e_fire      = gr.Dropdown(
                        choices=["Standard", "FR", "FRLS", "FRZH", "Unknown"],
                        label="Fire Resistance", value="Standard")
                    e_armouring = gr.Dropdown(
                        choices=["Yes", "No"], label="Armoured", value="No")
                    e_temp      = gr.Textbox(label="Temperature", placeholder="70°C")

                e_standards = gr.Textbox(label="Standards Required",
                                         placeholder="IS 7098 / IEC 60502")
                e_mandatory = gr.Textbox(
                    label="Mandatory Fields",
                    value="voltage_rating;size_sqmm;core_count",
                )
                e_spec_text = gr.Textbox(
                    label="Full Specification Text (for semantic score)",
                    placeholder="Paste RFP spec text here...", lines=2,
                )

                _e_outputs = [
                    e_spec_text, e_voltage, e_conductor, e_insulation,
                    e_cores, e_size, e_sheath, e_fire, e_armouring,
                    e_temp, e_standards, e_mandatory,
                ]

                with gr.Row():
                    for idx, s in enumerate(SAMPLE_SPECS):
                        gr.Button(f"Sample {idx+1}: {s['label']}",
                                  size="sm", variant="secondary").click(
                            fn=lambda i=idx: fill_sample(i),
                            outputs=_e_outputs,
                        )

                explain_btn = gr.Button("Generate Explanation Report",
                                        variant="primary", size="lg")

                with gr.Row():
                    with gr.Column(scale=3):
                        output_obj3 = gr.Markdown()
                    with gr.Column(scale=2):
                        shap_chart = gr.Plot(
                            label="SHAP Attribution Chart (Top-1 Match)",
                            visible=True,
                        )

                _e_inputs = [
                    e_spec_text, e_voltage, e_conductor, e_insulation,
                    e_cores, e_size, e_sheath, e_fire, e_armouring,
                    e_temp, e_standards, e_mandatory,
                ]
                explain_btn.click(
                    fn=run_obj3_explain, inputs=_e_inputs,
                    outputs=[output_obj3, shap_chart],
                )

            # ════════════════════════════════════════════════
            # TAB 4 — EXPERIMENT REFERENCE
            # ════════════════════════════════════════════════
            with gr.Tab("📋 Reference"):
                gr.Markdown("""
## System Architecture

**Hybrid Score = 0.50 × Structured + 0.30 × Semantic (SBERT) + 0.20 × Standards**

The structured score reads **structured columns** (`voltage_rating`, `core_count`,
`size_sqmm` etc.) NOT the raw spec text — this is the core innovation that delivers
robustness across all 7 experiment scenarios.

---

## Baseline Comparison

| Model | Top-1 Acc | Top-5 Acc | MRR | NDCG@5 |
|---|---|---|---|---|
| Random Baseline | 0.46% | 2.84% | 2.98% | 1.60% |
| TF-IDF + Cosine | 11.54% | 32.29% | 23.13% | 22.08% |
| BM25 (Okapi) | 5.59% | 23.46% | 16.51% | 14.83% |
| SBERT MiniLM-L6 | 13.26% | 32.46% | 23.14% | 23.15% |
| SBERT MPNet (sem-only) | 16.53% | 33.54% | 26.09% | 24.84% |
| Exact Structured Match | 5.70% | 17.53% | 12.94% | 11.67% |
| **Hybrid (Ours)** | **17.80%** | **44.34%** | **30.88%** | **31.25%** |

---

## Experiment Summary

| ID | Name | What Changes | Key Insight |
|---|---|---|---|
| EXP1 | Numeric Perturbation | Numbers in text ±10% | Struct immune; SBERT degrades |
| EXP2 | Unit Variation | kV→V, sqmm→mm² | Ladder matching is unit-agnostic |
| EXP3 | Missing Parameters | 2 structured fields cleared | Mandatory fields anchor score |
| EXP4 | In-Domain Noise | Boilerplate prepended | Struct unaffected |
| EXP4 | Out-of-Domain Noise | Irrelevant text prepended | Hurts SBERT more than in-domain |
| EXP6 | Partial Spec | Text truncated to 4 tokens | Struct columns still complete |
| EXP7 | Positional Bias | Spec moved to end of doc | Struct is position-invariant |

---

## Objective 3 — Classification & Risk Thresholds

| Category | Condition |
|---|---|
| ✅ Matching | Hybrid ≥ 0.72 AND Structured ≥ 0.70 AND Semantic ≥ 0.45 |
| ⚠️ Partially Matching | Hybrid ≥ 0.45 |
| ❌ Not Matching | Hybrid < 0.45 |

| Risk Level | Condition |
|---|---|
| 🟢 Low | Spec-Match ≥ 75% AND no critical field conflicts |
| 🟡 Medium | Spec-Match ≥ 50% AND at most 1 critical conflict |
| 🔴 High | Spec-Match < 50% OR 2+ critical field conflicts |

---

## Critical Fields (⭐ in OBJ3 Table)

Voltage Rating, Conductor Material, Insulation Type, Core Count,
Cross-Section (sqmm) — these carry 2× weight in both the structured score
and the spec-match percentage.
                """)

        gr.HTML("""
        <div style="text-align:center;margin-top:16px;color:#888;font-size:0.85em;">
            RFP Cable Spec Matcher &nbsp;|&nbsp; Major Project Prototype &nbsp;|&nbsp;
            OBJ1 · OBJ2 · OBJ3
        </div>
        """)

    return demo

# ────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing Cable Spec Matcher...")
    demo = create_interface()
    print("Interface ready. Launching on http://127.0.0.1:7860")
    demo.launch(share=True, show_error=True)
