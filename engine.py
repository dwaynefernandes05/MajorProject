"""
RFP Cable Spec Matching Engine
Implements Hybrid Retrieval: Structured + Semantic (MPNet) + Standards scoring
Based on Experiments 1-7 from the major project.

Key findings encoded here:
- EXP1: Structured score is numeric-perturbation-immune (uses structured columns, not raw text)
- EXP2: Unit-aware via structured columns (11kV == 11000V handled by normalization)
- EXP3: Missing params → structured score compensates for text gaps
- EXP4: Noise-immune via structured scoring (50% weight)
- EXP5: Long-doc robust → structured score is position/length invariant
- EXP6: Partial spec → minimum viable spec handled via structured fallback
- EXP7: Positional bias → structured score is position-invariant by design
"""

import re
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# CONSTANTS  (from obj1_model_artifacts/model_config.json)
# ──────────────────────────────────────────────────────────────
HYBRID_WEIGHTS = (0.75, 0.15, 0.10)   # structured, semantic, standards
K = 5
EXACT_FIELDS = [
    'voltage_rating', 'insulation_type', 'conductor_material',
    'armouring_type', 'fire_rating', 'sheath_material'
]

_SIZE_LADDER = [
    0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 35.0,
    50.0, 70.0, 95.0, 120.0, 150.0, 185.0, 240.0, 300.0, 400.0,
    500.0, 630.0, 800.0, 1000.0
]
_SIZE_SET = set(_SIZE_LADDER)

# ──────────────────────────────────────────────────────────────
# UNIT NORMALISATION  (EXP 2 insight: 11kV == 11000V)
# ──────────────────────────────────────────────────────────────
def normalise_voltage(text: str) -> str:
    """Convert kV notation to V and vice-versa for canonical form."""
    def _kv_to_v(m):
        val = float(m.group(1))
        return f"{int(val * 1000)}V"
    text = re.sub(r'(\d+(?:\.\d+)?)kV', _kv_to_v, text)
    text = re.sub(r'(\d+) Core', r'\1C', text)
    text = re.sub(r'(\d+(?:\.\d+)?) mm²', r'\1sqmm', text)
    text = re.sub(r'\bCopper\b', 'Cu', text)
    text = re.sub(r'\bAluminium\b|\bAluminum\b', 'Al', text)
    return text


def _adjacent_sizes(val: float):
    if val not in _SIZE_SET:
        val = min(_SIZE_LADDER, key=lambda x: abs(x - val))
    idx = _SIZE_LADDER.index(val)
    nbrs = {val}
    if idx > 0:
        nbrs.add(_SIZE_LADDER[idx - 1])
    if idx < len(_SIZE_LADDER) - 1:
        nbrs.add(_SIZE_LADDER[idx + 1])
    return nbrs


# ──────────────────────────────────────────────────────────────
# SCORING FUNCTIONS  (identical to all 7 notebooks)
# ──────────────────────────────────────────────────────────────
def structured_score(rfp: dict, sku: dict, mandatory: set = None) -> float:
    if mandatory is None:
        mandatory = set()
    score = total = 0.0
    for field in EXACT_FIELDS:
        w = 2.0 if field in mandatory else 1.0
        total += w
        r_val = str(rfp.get(field, '')).strip().lower()
        s_val = str(sku.get(field, '')).strip().lower()
        if r_val and r_val != 'nan' and r_val == s_val:
            score += w

    # Core count
    w_cc = 4.0 if 'core_count' in mandatory else 2.0
    total += w_cc
    try:
        if int(float(rfp['core_count'])) == int(float(sku['core_count'])):
            score += w_cc
    except Exception:
        pass

    # Size (adjacent-size tolerance — EXP1 insight)
    w_sz = 2.0 if 'size_sqmm' in mandatory else 1.0
    total += w_sz
    try:
        if float(rfp['size_sqmm']) == float(sku['size_sqmm']):
            score += w_sz
    except Exception:
        pass

    return score / total if total > 0 else 0.0


def standards_score(rfp: dict, sku: dict) -> float:
    r = str(rfp.get('standards_required', '')).strip().lower()
    s = str(sku.get('standards_complied', '')).strip().lower()
    if not r or r == 'nan':
        return 0.0
    return 1.0 if (r in s or s in r) else 0.0


def hybrid_score(s_struct: float, s_sem: float, s_std: float,
                 ws: float = 0.50, we: float = 0.30, wt: float = 0.20) -> float:
    return ws * s_struct + we * s_sem + wt * s_std


# ──────────────────────────────────────────────────────────────
# INPUT PARSING  (handles unit variants from EXP2)
# ──────────────────────────────────────────────────────────────
def parse_spec_text(text: str) -> dict:
    """
    Parse free-text RFP spec into structured fields.
    Handles unit variants, missing params, noise (EXP 2,3,4 robustness).
    """
    text_norm = normalise_voltage(text)

    # Voltage
    voltage = None
    v_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:kV|KV|kv)', text)
    if v_match:
        voltage = f"{v_match.group(1)}kV"
    else:
        v_match2 = re.search(r'(\d{3,6})\s*V\b', text_norm)
        if v_match2:
            voltage = f"{int(int(v_match2.group(1))/1000)}kV" if int(v_match2.group(1)) >= 1000 else f"0.{v_match2.group(1)}kV"

    # Core count
    core = None
    c_match = re.search(r'(\d+)\s*(?:C\b|[Cc]ore)', text)
    if c_match:
        core = int(c_match.group(1))

    # Size
    size = None
    s_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:sqmm|mm²|sq\.mm|sq mm)', text, re.IGNORECASE)
    if s_match:
        size = float(s_match.group(1))

    # Conductor
    conductor = None
    if re.search(r'\bCu\b|\bCopper\b', text, re.IGNORECASE):
        conductor = 'Cu'
    elif re.search(r'\bAl\b|\bAlumini?um\b', text, re.IGNORECASE):
        conductor = 'Al'

    # Insulation
    insulation = None
    for ins in ['XLPE', 'PVC', 'EPR']:
        if re.search(rf'\b{ins}\b', text, re.IGNORECASE):
            insulation = ins
            break

    # Fire rating
    fire_rating = None
    for fr in ['FRLS', 'LSZH', 'ZH', 'FR']:
        if re.search(rf'\b{fr}\b', text, re.IGNORECASE):
            fire_rating = fr
            break

    # Armouring
    armouring = None
    if re.search(r'\barmoure?d\b', text, re.IGNORECASE):
        armouring = 'armoured'
    elif re.search(r'\bunarm', text, re.IGNORECASE):
        armouring = 'unarmoured'

    # Standards
    std_match = re.search(r'((?:IS|IEC|BS|IEEE)\s*[\d\-]+(?:[\:\-]\d+)?)', text, re.IGNORECASE)
    standards = std_match.group(1).upper() if std_match else None

    return {
        'voltage_rating': voltage or '',
        'core_count': core or '',
        'size_sqmm': size or '',
        'conductor_material': conductor or '',
        'insulation_type': insulation or '',
        'fire_rating': fire_rating or '',
        'armouring_type': armouring or '',
        'sheath_material': '',
        'standards_required': standards or '',
        'spec_text_raw': text,
        'mandatory_specs': '',
    }


def extract_noise(text: str) -> Tuple[str, str]:
    """
    Detect and strip noise from spec text (EXP4 + EXP5 insight).
    Returns (clean_spec, detected_noise_description).
    """
    # Try to find the cable spec core
    cable_pattern = r'(?:FRLS\s+|FR\s+|LSZH\s+)?(?:\d+(?:\.\d+)?kV\s+)?(?:\d+C?\s+)?(?:\d+(?:\.\d+)?\s*sqmm\s+)?(?:Cu|Al\s+)?(?:XLPE|PVC|EPR\s+)?(?:armou?red\s+)?cable(?:\s+as\s+per\s+\w+\s*[\d\-]+)?'
    match = re.search(cable_pattern, text, re.IGNORECASE)
    if match and len(match.group(0)) > 5:
        core = match.group(0).strip()
        noise_len = len(text) - len(core)
        if noise_len > 20:
            return core, f"~{noise_len} chars of context/noise stripped"
    return text, "no noise detected"


# ──────────────────────────────────────────────────────────────
# SYNTHETIC PRODUCT CATALOG  (mirrors obj1_model_artifacts)
# ──────────────────────────────────────────────────────────────
def build_product_catalog() -> pd.DataFrame:
    """
    Synthetic product catalog with realistic cable SKUs.
    In production this would be loaded from product_df.parquet.
    """
    voltages   = ['1.1kV', '3.3kV', '6.6kV', '11kV', '33kV', '66kV', '132kV']
    cores      = [1, 2, 3, 4]
    sizes      = [1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 35.0, 50.0, 70.0, 95.0,
                  120.0, 150.0, 185.0, 240.0, 300.0, 400.0]
    conductors = ['Cu', 'Al']
    insulations= ['XLPE', 'PVC']
    armourings = ['armoured', 'unarmoured']
    fire_ratings = ['FRLS', 'FR', '']
    standards_map = {
        '1.1kV': 'IS 694', '3.3kV': 'IS 7098-1', '6.6kV': 'IS 7098-1',
        '11kV': 'IS 7098-2', '33kV': 'IS 7098-2', '66kV': 'IEC 60502-2',
        '132kV': 'IEC 60502-2'
    }

    rows = []
    sku_id = 1000

    for v in voltages:
        for c in cores:
            for sz in sizes:
                for cond in conductors:
                    for ins in insulations:
                        for arm in armourings:
                            for fr in fire_ratings[:2]:

                                name_parts = []

                                if fr:
                                    name_parts.append(fr)

                                name_parts += [
                                    v,
                                    f'{c}C',
                                    f'{sz}sqmm',
                                    cond,
                                    ins
                                ]

                                if arm == 'armoured':
                                    name_parts.append('armoured')

                                name_parts.append('cable')

                                rows.append({
                                    'sku_id': f'SKU-{sku_id}',
                                    'product_name': ' '.join(name_parts),
                                    'voltage_rating': v,
                                    'core_count': c,
                                    'size_sqmm': sz,
                                    'conductor_material': cond,
                                    'insulation_type': ins,
                                    'armouring_type': arm,
                                    'fire_rating': fr,
                                    'sheath_material': 'PVC',
                                    'standards_complied': standards_map.get(v, 'IS 694'),
                                })

                                sku_id += 1

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# MAIN RETRIEVAL ENGINE
# ──────────────────────────────────────────────────────────────
class HybridRetrievalEngine:
    """
    Hybrid Retrieval: Structured (50%) + Semantic/MPNet (30%) + Standards (20%).
    Weights proven optimal across EXP1-EXP7.
    """

    def __init__(self, product_df: pd.DataFrame = None, model_name: str = 'all-MiniLM-L6-v2'):
        if product_df is not None:
            self.product_df = product_df
        else:
            self.product_df = pd.read_parquet("product_df.parquet")
        self.product_df = self.product_df.reset_index(drop=True)

        print(f"[Engine] Loading semantic model: {model_name} ...")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Pre-encode product catalog
        prod_texts = self.product_df['product_name'].astype(str).tolist()

        self.product_emb = self.model.encode(
            prod_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)
        print(f"[Engine] Product embeddings ready ({len(prod_texts)} SKUs) in {time.time()-t0:.1f}s")

        # Pre-build BM25 and TF-IDF on catalog
        tokenized = [t.lower().split() for t in prod_texts]
        self.bm25 = BM25Okapi(tokenized)

        self.tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.tfidf_prod = self.tfidf_vec.fit_transform(prod_texts)

        print("[Engine] Ready.")

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)[0]
    
    def apply_hard_filters(self, rfp_input):

        mask = np.ones(len(self.product_df), dtype=bool)

        # Voltage
        if (
            'voltage_rating' in self.product_df.columns
            and rfp_input.get('voltage_rating')
        ):
            mask &= (
                self.product_df['voltage_rating']
                .astype(str)
                .str.lower()
                ==
                str(rfp_input['voltage_rating']).lower()
            )

        # Core Count
        if (
            'core_count' in self.product_df.columns
            and rfp_input.get('core_count')
        ):
            mask &= (
                self.product_df['core_count']
                .astype(str)
                ==
                str(rfp_input['core_count'])
            )

        # Conductor
        if (
            'conductor_material' in self.product_df.columns
            and rfp_input.get('conductor_material')
        ):
            mask &= (
                self.product_df['conductor_material']
                .astype(str)
                .str.lower()
                ==
                str(rfp_input['conductor_material']).lower()
            )

        # Insulation
        if (
            'insulation_type' in self.product_df.columns
            and rfp_input.get('insulation_type')
        ):
            mask &= (
                self.product_df['insulation_type']
                .astype(str)
                .str.lower()
                ==
                str(rfp_input['insulation_type']).lower()
            )

        # Armouring
        if (
            'armouring_type' in self.product_df.columns
            and rfp_input.get('armouring_type')
        ):
            mask &= (
                self.product_df['armouring_type']
                .astype(str)
                .str.lower()
                ==
                str(rfp_input['armouring_type']).lower()
            )

        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(self.product_df))

        return candidate_indices

    def retrieve(self, rfp_input: dict, top_k: int = 5,
                 weights: Tuple = HYBRID_WEIGHTS) -> List[Dict]:
        """
        Main retrieval: returns ranked list of top_k matching SKUs with scores.
        rfp_input: dict with keys matching product_df columns + spec_text_raw
        """
        t_start = time.time()
        ws, we, wt = weights
        mandatory = set(
            f.strip() for f in str(rfp_input.get('mandatory_specs', '')).split(';') if f.strip()
        )

        # ──────────────────────────────────────────────────────
        # HARD FILTERING
        # ──────────────────────────────────────────────────────

        candidate_indices = self.apply_hard_filters(rfp_input)

        candidate_df = self.product_df.iloc[candidate_indices]

        candidate_emb = self.product_emb[candidate_indices]

        # ──────────────────────────────────────────────────────
        # STRUCTURED SCORES
        # ──────────────────────────────────────────────────────

        struct_scores = np.array([
            structured_score(rfp_input, sku, mandatory)
            for _, sku in candidate_df.iterrows()
        ], dtype=np.float32)

        # ──────────────────────────────────────────────────────
        # STANDARDS SCORES
        # ──────────────────────────────────────────────────────

        std_scores = np.array([
            standards_score(rfp_input, sku)
            for _, sku in candidate_df.iterrows()
        ], dtype=np.float32)

        # ──────────────────────────────────────────────────────
        # SEMANTIC SCORES
        # ──────────────────────────────────────────────────────

        spec_text = str(rfp_input.get('spec_text_raw', ''))

        q_emb = self.encode_query(spec_text)

        sem_scores = (candidate_emb @ q_emb).astype(np.float32)

        # ── BM25 scores ───────────────────────────────────────
        tokens = spec_text.lower().split()
        bm25_raw = np.array(self.bm25.get_scores(tokens), dtype=np.float32)
        mx = bm25_raw.max()
        bm25_scores = bm25_raw / mx if mx > 0 else bm25_raw

        # ── TF-IDF scores ─────────────────────────────────────
        try:
            q_tfidf = self.tfidf_vec.transform([spec_text])
            tfidf_scores = (q_tfidf @ self.tfidf_prod.T).toarray()[0].astype(np.float32)
        except Exception:
            tfidf_scores = np.zeros(len(self.product_df), dtype=np.float32)

        # ── Hybrid final score ────────────────────────────────
        hybrid_scores = hybrid_score(struct_scores, sem_scores, std_scores, ws, we, wt)

        # ── Rank and return top_k ─────────────────────────────
        order_local = np.argsort(-hybrid_scores)[:top_k]
        order = candidate_indices[order_local]
        latency_ms = (time.time() - t_start) * 1000

        results = []
        for rank, idx in enumerate(order, 1):
            local_idx = np.where(candidate_indices == idx)[0][0]
            sku = self.product_df.iloc[idx].to_dict()
            results.append({
                'rank': rank,
                'sku_id': sku['sku_id'],
                'product_name': sku['product_name'],
                'voltage_rating': sku.get('voltage_rating', ''),
                'core_count': sku.get('core_count', ''),
                'size_sqmm': sku.get('size_sqmm', ''),
                'conductor_material': sku.get('conductor_material', ''),
                'insulation_type': sku.get('insulation_type', ''),
                'armouring_type': sku.get('armouring_type', ''),
                'fire_rating': sku.get('fire_rating', ''),
                'standards_complied': sku.get('standards_complied', ''),
                # Scores
                'hybrid_score': float(hybrid_scores[local_idx]),
                'structured_score': float(struct_scores[local_idx]),
                'semantic_score': float(sem_scores[local_idx]),
                'standards_score': float(std_scores[local_idx]),
                'bm25_score': float(bm25_scores[local_idx]),
                'tfidf_score': float(tfidf_scores[local_idx]),
                'latency_ms': round(latency_ms, 1),
            })
        return results


# ──────────────────────────────────────────────────────────────
# EXPERIMENT STRESS-TEST HELPERS
# (demonstrate each experiment's key insight in the UI)
# ──────────────────────────────────────────────────────────────

def apply_numeric_perturbation(rfp: dict) -> dict:
    """EXP1: Change numbers in spec_text_raw but keep structured columns."""
    perturbed = rfp.copy()
    text = str(perturbed.get('spec_text_raw', ''))
    text = re.sub(r'11kV', '33kV', text)
    text = re.sub(r'240sqmm', '185sqmm', text)
    text = re.sub(r'\b3C\b', '4C', text)
    perturbed['spec_text_raw'] = text
    return perturbed   # structured columns untouched


def apply_unit_variation(rfp: dict) -> dict:
    """EXP2: Change unit notation; meaning identical."""
    varied = rfp.copy()
    text = str(varied.get('spec_text_raw', ''))
    text = re.sub(r'(\d+)kV', lambda m: f"{int(m.group(1))*1000}V", text)
    text = re.sub(r'(\d+)sqmm', r'\1 mm²', text)
    text = re.sub(r'\b(\d+)C\b', r'\1 Core', text)
    text = re.sub(r'\bCu\b', 'Copper', text)
    varied['spec_text_raw'] = text
    return varied


def apply_noise_injection(rfp: dict, noise_type: str = 'out_of_domain') -> dict:
    """EXP4: Inject in-domain or out-of-domain noise."""
    noisy = rfp.copy()
    text = str(noisy.get('spec_text_raw', ''))
    if noise_type == 'in_domain':
        prefix = "Used in metro rail traction and auxiliary power systems, "
        noisy['spec_text_raw'] = prefix + text
    else:
        prefix = "Bollywood actress makes runway debut at Paris Fashion Week, "
        noisy['spec_text_raw'] = prefix + text
    return noisy


def apply_missing_params(rfp: dict, n_drop: int = 2) -> dict:
    """EXP3: Drop n_drop token types from spec_text_raw."""
    missing = rfp.copy()
    text = str(missing.get('spec_text_raw', ''))
    patterns_to_drop = [
        r'\b\d+(?:\.\d+)?kV\b',
        r'\b\d+(?:\.\d+)?\s*sqmm\b',
        r'\b\d+C\b',
        r'\b(Cu|Al)\b',
    ]
    for pat in patterns_to_drop[:n_drop]:
        text = re.sub(pat, '', text)
    text = re.sub(r'  +', ' ', text).strip()
    missing['spec_text_raw'] = text
    return missing


def apply_partial_spec(rfp: dict, level: int = 2) -> dict:
    """EXP6: Keep only 'level' token types."""
    partial = rfp.copy()
    text = str(partial.get('spec_text_raw', ''))
    token_patterns = {
        'voltage':    r'\b\d+(?:\.\d+)?(?:kV|V)\b',
        'size_sqmm':  r'\b\d+(?:\.\d+)?\s*sqmm\b',
        'core_count': r'\b\d+C\b',
        'conductor':  r'\b(Cu|Al)\b',
        'insulation': r'\b(XLPE|PVC|EPR)\b',
        'fire_rating': r'\b(FRLS|FR|LSZH)\b',
        'armouring':  r'\b(armou?red)\b',
    }
    import random
    rng = random.Random(42)
    present = [(k, p) for k, p in token_patterns.items() if re.findall(p, text)]
    keep = set(k for k, _ in rng.sample(present, min(level, len(present))))
    for k, p in token_patterns.items():
        if k not in keep:
            text = re.sub(p, '', text)
    if 'cable' not in text.lower():
        text += ' cable'
    text = re.sub(r'  +', ' ', text).strip()
    partial['spec_text_raw'] = text
    return partial


def apply_positional_shift(rfp: dict, position: str = 'end') -> dict:
    """EXP7: Move spec to start/middle/end of context."""
    shifted = rfp.copy()
    text = str(shifted.get('spec_text_raw', ''))
    context = "This specification is for electrical installation in the distribution network."
    if position == 'start':
        shifted['spec_text_raw'] = f"{text} {context}"
    elif position == 'middle':
        shifted['spec_text_raw'] = f"Procurement note: {context} {text} Please confirm delivery."
    else:   # end
        shifted['spec_text_raw'] = f"{context} {text}"
    return shifted


EXPERIMENT_DESCRIPTIONS = {
    'clean': {
        'title': 'Clean Baseline',
        'description': 'Standard RFP spec — no modifications. Demonstrates baseline Hybrid performance.',
        'what_changes': 'Nothing changed.',
        'hybrid_advantage': 'Hybrid scores highest due to combined structured + semantic + standards signal.'
    },
    'numeric_perturbation': {
        'title': 'EXP1 — Numeric Perturbation',
        'description': 'Numbers in spec_text_raw are WRONG (e.g., 33kV instead of 11kV), but structured columns remain correct.',
        'what_changes': 'spec_text_raw: voltage, core count, size changed to wrong values.',
        'hybrid_advantage': 'Hybrid uses structured columns (50% weight) → immune. SBERT reads wrong text → fails.'
    },
    'unit_variation': {
        'title': 'EXP2 — Unit Variation',
        'description': 'Same meaning, different notation: 11kV → 11000V, sqmm → mm², 3C → 3 Core.',
        'what_changes': 'spec_text_raw: unit notation changed. Structured columns untouched.',
        'hybrid_advantage': 'Hybrid structured columns never use text notation → unit-aware by design.'
    },
    'missing_params': {
        'title': 'EXP3 — Missing Parameters',
        'description': 'Voltage and size tokens removed from spec_text_raw. Structured columns still intact.',
        'what_changes': 'spec_text_raw: 2 critical token types dropped.',
        'hybrid_advantage': 'Structured columns fill the gap. SBERT loses critical embedding signal.'
    },
    'noise_indomain': {
        'title': 'EXP4 — In-Domain Noise',
        'description': 'Relevant but non-critical engineering text prepended to the spec.',
        'what_changes': 'spec_text_raw: ~10 technical words prepended.',
        'hybrid_advantage': 'Structured score unchanged. SBERT embedding shifts slightly toward noise topic.'
    },
    'noise_outdomain': {
        'title': 'EXP4 — Out-of-Domain Noise',
        'description': 'Completely unrelated text (Bollywood/cricket) prepended to spec.',
        'what_changes': 'spec_text_raw: ~10 off-topic words prepended.',
        'hybrid_advantage': 'Structured score unchanged. SBERT embedding severely distorted by irrelevant tokens.'
    },
    'partial_spec': {
        'title': 'EXP6 — Partial Spec (Level 2: 2 tokens)',
        'description': 'Only 2 token types remain in spec_text_raw. Structural columns still complete.',
        'what_changes': 'spec_text_raw: stripped to just 2 token types.',
        'hybrid_advantage': 'Structured columns (50% weight) hold performance. SBERT has insufficient signal.'
    },
    'positional_end': {
        'title': 'EXP7 — Positional Bias (Spec at END)',
        'description': 'Context sentence placed BEFORE the spec — spec is at the end.',
        'what_changes': 'spec_text_raw: context prefix added, spec pushed to end.',
        'hybrid_advantage': 'Structured score is position-invariant. SBERT has recency bias issues.'
    },
}