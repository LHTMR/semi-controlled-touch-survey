import json
import logging
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta
from dataclasses import dataclass
from typing import List, Dict
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 1. RERANKER IMPLEMENTATION
# ==========================================


@dataclass
class RerankingResult:
    ranked_docs: List[Dict]
    semantic_scores: np.ndarray
    raw_win_matrix: np.ndarray


class TournamentReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_gpu: bool = False,
        default_batch_size: int = 32,
    ):
        """
        Initializes the Stochastic Tournament Reranker.
        Acts as the deterministic 'Referee' in the pairwise duels.
        """
        device = "cuda" if use_gpu else "cpu"
        logger.info(f"Loading Cross-Encoder model: {model_name} on {device}...")
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = default_batch_size

    def rerank(
        self,
        query: str,
        docs: List[Dict],
        tournament_rounds: int = 15,
        subset_size: int = 8,
        king_of_hill_streak: int = 3,
        beta_alpha: float = 2.0,
        beta_beta: float = 2.0,
        use_dynamic_beta_fitting: bool = True,
        content_truncation: int = 2000,
    ) -> RerankingResult:
        if not docs:
            return RerankingResult([], np.array([]), np.array([]))

        n = len(docs)
        if n == 1:
            doc = docs[0].copy()
            doc["rerank_score"] = 1.0
            return RerankingResult([doc], np.array([1.0]), np.array([[0]]))

        # 1. Initialize Matrices
        W = np.zeros((n, n))  # Win Matrix W[i, j] = wins of i over j
        M = np.zeros((n, n))  # Match Matrix M[i, j] = total matches between i and j

        doc_indices = np.arange(n)

        # 2. The Tournament Loop (Alternating Selection)
        for round_idx in range(tournament_rounds):
            k = min(subset_size, n)
            current_subset = np.random.choice(doc_indices, k, replace=False)
            queue = list(current_subset)
            current_gladiator = queue.pop(0)

            winner_stays_mode = True
            streak_counter = 0

            while queue:
                challenger = queue.pop(0)

                # Fetch text (support both our new 'raw_text' and legacy 'content')
                text_a = docs[current_gladiator].get(
                    "raw_text", docs[current_gladiator].get("content", "")
                )[:content_truncation]
                text_b = docs[challenger].get(
                    "raw_text", docs[challenger].get("content", "")
                )[:content_truncation]

                # PREDICT: Doc A vs Doc B
                pair_a = [query, text_a]
                pair_b = [query, text_b]

                scores = self.model.predict([pair_a, pair_b])

                # Update Matrices
                M[current_gladiator, challenger] += 1
                M[challenger, current_gladiator] += 1

                if scores[0] > scores[1]:
                    W[current_gladiator, challenger] += 1
                    winner, loser = current_gladiator, challenger
                else:
                    W[challenger, current_gladiator] += 1
                    winner, loser = challenger, current_gladiator

                # Alternating Selection Logic
                streak_counter += 1
                if streak_counter >= king_of_hill_streak:
                    winner_stays_mode = not winner_stays_mode
                    streak_counter = 0

                current_gladiator = winner if winner_stays_mode else loser

        # 3. Mathematical Projection (Inverse Beta Calibration)
        total_matches = np.sum(M, axis=1) + 1e-9
        total_wins = np.sum(W, axis=1)
        X = total_wins / total_matches
        X = np.clip(X, 0.01, 0.99)

        if use_dynamic_beta_fitting:
            sample_mean = np.mean(X)
            sample_var = np.var(X)
            common_term = (sample_mean * (1 - sample_mean) / (sample_var + 1e-6)) - 1

            if common_term > 0:
                dynamic_alpha = np.clip(sample_mean * common_term, 1.0, 10.0)
                dynamic_beta = np.clip((1 - sample_mean) * common_term, 1.0, 10.0)
                final_alpha, final_beta = dynamic_alpha, dynamic_beta
            else:
                final_alpha, final_beta = beta_alpha, beta_beta
        else:
            final_alpha, final_beta = beta_alpha, beta_beta

        final_scores = beta.ppf(X, final_alpha, final_beta)
        sorted_indices = np.argsort(final_scores)[::-1]

        ranked_results = []
        for rank, idx in enumerate(sorted_indices):
            doc = docs[idx].copy()
            doc["rerank_score"] = float(final_scores[idx])
            doc["tournament_win_rate"] = float(X[idx])
            doc["matches_played"] = int(total_matches[idx])
            ranked_results.append(doc)

        return RerankingResult(
            ranked_docs=ranked_results,
            semantic_scores=final_scores[sorted_indices],
            raw_win_matrix=W,
        )


# ==========================================
# 2. PHASE 5 PIPELINE
# ==========================================


def run_phase5_pipeline(
    micro_json_path: str,
    embeddings_path: str,
    anchors_path: str,
    factions_path: str,
):
    logger.info("Loading outputs from Phases 1, 2, and 4...")
    with open(micro_json_path, "r", encoding="utf-8") as f:
        micro_data = json.load(f)

    with open(anchors_path, "r", encoding="utf-8") as f:
        anchors = json.load(f)

    with open(factions_path, "r", encoding="utf-8") as f:
        factions = json.load(f)

    embeddings = np.load(embeddings_path)

    logger.info("Initializing the Tournament Reranker (Cross-Encoder)...")
    reranker = TournamentReranker(use_gpu=False)

    final_report_data = {"macro_consensus_quotes": {}, "micro_debate_quotes": {}}

    # ==========================================
    # 1. MACRO QUOTES (The Consensus Anchors)
    # ==========================================
    logger.info("--- Extracting Macro Quotes for K Latent Anchors ---")

    for anchor_name, anchor_data in anchors.items():
        query_quote = anchor_data["prototype_quote"]
        centroid = np.array(anchor_data["centroid"])

        # Fast Pre-filter: Grab the 30 geometrically closest quotes to the centroid
        dists = cdist(embeddings, [centroid], metric="cosine").flatten()
        top_30_indices = np.argsort(dists)[:30]

        candidate_docs = [micro_data[i] for i in top_30_indices]

        logger.info(f"Running Tournament for {anchor_name}...")
        rerank_res = reranker.rerank(
            query=query_quote, docs=candidate_docs, tournament_rounds=10, subset_size=5
        )

        best_doc = rerank_res.ranked_docs[0]

        final_report_data["macro_consensus_quotes"][anchor_name] = {
            "prototype_query": query_quote,
            "best_representative_quote": best_doc["raw_text"],
            "source_video_id": best_doc["video_id"],
            "rerank_score": best_doc.get("rerank_score", 0.0),
        }
        logger.info(f"Winner for {anchor_name} -> Video {best_doc['video_id']}")

    # ==========================================
    # 2. MICRO QUOTES (The Intra-Video Debates)
    # ==========================================
    logger.info("\n--- Extracting Micro Quotes for Polarizing Videos ---")

    for v_id, faction_data in factions.items():
        logger.info(f"Resolving debate representatives for Video {v_id}...")

        final_report_data["micro_debate_quotes"][v_id] = {"axes": []}

        # Run a tournament for EVERY axis!
        for axis in faction_data.get("axes", []):
            axis_result = {
                "axis_index": axis["axis_index"],
                "variance_explained": axis["variance_explained_percent"],
                "faction_positive_winner": None,
                "faction_negative_winner": None,
            }

            for side in ["positive", "negative"]:
                docs = axis[f"faction_{side}"]
                if docs:
                    query = docs[0][
                        "raw_text"
                    ]  # Most extreme view on this side of the axis
                    if len(docs) > 3:
                        res = reranker.rerank(
                            query=query, docs=docs, tournament_rounds=8, subset_size=4
                        )
                        winner = res.ranked_docs[0]
                    else:
                        winner = docs[0]

                    axis_result[f"faction_{side}_winner"] = {
                        "quote": winner["raw_text"],
                        "participant_id": winner["participant_id"],
                    }

            final_report_data["micro_debate_quotes"][v_id]["axes"].append(axis_result)

    out_path = "Analysis/RAG_analysis/phase5_final_quotes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report_data, f, indent=2)

    logger.info(
        f"\nPhase 5 Complete! All definitive quotes extracted and saved to {out_path}"
    )


if __name__ == "__main__":
    run_phase5_pipeline(
        micro_json_path="Analysis/RAG_analysis/phase1_micro_level.json",
        embeddings_path="Analysis/RAG_analysis/phase2_micro_embeddings.npy",
        anchors_path="Analysis/RAG_analysis/phase2_anchors.json",
        factions_path="Analysis/RAG_analysis/phase4_factions.json",
    )
