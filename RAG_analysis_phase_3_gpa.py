import json
import logging
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generalized_procrustes(profiles: list, max_iter: int = 15):
    """
    Aligns multiple geometric matrices to find the mathematical 'Consensus' shape.
    Re-implemented from geometric_rag_engine.py
    """
    reference = profiles[0]
    aligned_profiles = profiles.copy()

    for iteration in range(max_iter):
        mean_profile = np.mean(aligned_profiles, axis=0)
        new_aligned = []

        for p in profiles:
            # procrustes() returns: standardized ref, aligned p, and disparity
            mtx1, mtx2, disparity = procrustes(mean_profile, p)
            new_aligned.append(mtx2)

        # Check for convergence
        diff = np.linalg.norm(np.array(aligned_profiles) - np.array(new_aligned))
        aligned_profiles = new_aligned
        if diff < 1e-6:
            logger.info(f"GPA converged at iteration {iteration}")
            break

    return np.mean(aligned_profiles, axis=0), aligned_profiles


def run_phase3_pipeline(micro_json_path, embeddings_path, anchors_path):
    logger.info("Loading Phase 2 outputs...")
    with open(micro_json_path, "r", encoding="utf-8") as f:
        micro_data = json.load(f)

    with open(anchors_path, "r", encoding="utf-8") as f:
        anchors = json.load(f)

    embeddings = np.load(embeddings_path)

    # 1. Group embeddings by Video
    logger.info("Grouping participant embeddings by Video ID...")
    video_groups = {}
    for idx, entry in enumerate(micro_data):
        v_id = entry["video_id"]
        # Skip leaked/malformed video IDs just in case
        if not str(v_id).isdigit() or int(v_id) > 100:
            continue

        if v_id not in video_groups:
            video_groups[v_id] = []
        video_groups[v_id].append(embeddings[idx])

    # Extract centroids from anchors
    K = len(anchors)
    centroids = [anchors[f"Anchor_{k}"]["centroid"] for k in range(K)]

    # 2. Build K x 2 Geometric Matrix for each Video
    logger.info("Constructing geometric profiles (K x 2 matrices) for each video...")
    video_ids = list(video_groups.keys())
    profiles = []

    for v_id in video_ids:
        vid_embeds = np.array(video_groups[v_id])
        matrix = np.zeros((K, 2))

        for k in range(K):
            # Calculate cosine similarity to the centroid
            sims = 1.0 - cdist(vid_embeds, [centroids[k]], metric="cosine").flatten()

            # Dimension 1: Relevance (Mean similarity)
            matrix[k, 0] = np.mean(sims)
            # Dimension 2: Density/Spread (Standard deviation)
            matrix[k, 1] = np.std(sims)

        profiles.append(matrix)

    # 3. Manifold Alignment (GPA)
    logger.info("Running Generalized Procrustes Analysis (GPA)...")
    consensus_matrix, aligned_profiles = generalized_procrustes(profiles)

    # 4. Polarization Detection (Dynamic N-Dimensional PCA)
    logger.info("Calculating residuals and running PCA to find Polarizing Videos...")
    flat_consensus = consensus_matrix.flatten()
    residuals = [aligned.flatten() - flat_consensus for aligned in aligned_profiles]

    n_pca_components = min(
        len(residuals), 5
    )  # Default up to 5 dimensions for macro view
    pca = PCA(n_components=n_pca_components, random_state=42)
    pca_projections = pca.fit_transform(residuals)

    # 5. Rank Videos by Deviation
    raw_scores = [np.linalg.norm(r) for r in residuals]
    min_score, max_score = min(raw_scores), max(raw_scores)

    results = []
    for i, v_id in enumerate(video_ids):
        raw = raw_scores[i]
        scaled_score = (
            ((raw - min_score) / (max_score - min_score)) * 100
            if max_score > min_score
            else 0.0
        )

        results.append(
            {
                "video_id": v_id,
                "raw_deviation": float(raw),
                "polarization_percent": float(scaled_score),
                "pca_projections": [
                    float(x) for x in pca_projections[i]
                ],  # Dynamically saves ALL axes
            }
        )

    results.sort(key=lambda x: x["polarization_percent"], reverse=True)

    # Save the output
    out_path = "Analysis/RAG_analysis/phase3_gpa_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Phase 3 Complete! Top Polarizing Videos:")
    for res in results[:5]:
        logger.info(
            f"  - Video {res['video_id']} (Normalised Deviation Score (%): {res["polarization_percent"]:.3f})"
        )

    logger.info(f"Results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_phase3_pipeline(
        micro_json_path="Analysis/RAG_analysis/phase1_micro_level.json",
        embeddings_path="Analysis/RAG_analysis/phase2_micro_embeddings.npy",
        anchors_path="Analysis/RAG_analysis/phase2_anchors.json",
    )
