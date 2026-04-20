import json
import logging
import numpy as np
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_phase4_pipeline(
    micro_json_path: str,
    embeddings_path: str,
    gpa_results_path: str,
    top_n_videos: int = None,
    n_pca_components: int = 2,  # <-- Easily change this to 1, 3, 5, etc.
):
    logger.info("Loading Phase 1, 2, and 3 outputs...")
    with open(micro_json_path, "r", encoding="utf-8") as f:
        micro_data = json.load(f)

    with open(gpa_results_path, "r", encoding="utf-8") as f:
        gpa_results = json.load(f)

    embeddings = np.load(embeddings_path)

    if top_n_videos:
        target_videos = [str(res["video_id"]) for res in gpa_results[:top_n_videos]]
    else:
        target_videos = [str(res["video_id"]) for res in gpa_results]

    logger.info(f"Targeting {len(target_videos)} videos for faction analysis...")
    factions_output = {}

    for v_id in target_videos:
        logger.info(f"--- Analyzing Debate for Video {v_id} ---")

        video_indices = [
            i for i, entry in enumerate(micro_data) if str(entry["video_id"]) == v_id
        ]

        # We need at least as many participants as components to run PCA
        if len(video_indices) <= n_pca_components:
            logger.warning(
                f"Video {v_id} has too few participants ({len(video_indices)}) for {n_pca_components} PCA components. Skipping."
            )
            continue

        vid_embeddings = embeddings[video_indices]
        vid_participants = [micro_data[i] for i in video_indices]

        # 3. Localized N-Dimensional PCA
        pca = PCA(n_components=n_pca_components, random_state=4242)
        projections = pca.fit_transform(vid_embeddings)
        variances = pca.explained_variance_ratio_ * 100

        axes_data = []

        # Loop through EVERY axis to find the distinct debates
        for c in range(n_pca_components):
            fac_pos = []
            fac_neg = []

            for i, coords in enumerate(projections):
                profile = {
                    "participant_id": vid_participants[i]["participant_id"],
                    "raw_text": vid_participants[i]["raw_text"],
                    "projection_score": float(
                        coords[c]
                    ),  # Score for THIS specific axis
                    "all_projections": [float(x) for x in coords],
                }

                if coords[c] > 0:
                    fac_pos.append(profile)
                else:
                    fac_neg.append(profile)

            # Sort by extremity
            fac_pos.sort(key=lambda x: x["projection_score"], reverse=True)
            fac_neg.sort(key=lambda x: x["projection_score"])

            axes_data.append(
                {
                    "axis_index": c + 1,  # e.g., PC1, PC2
                    "variance_explained_percent": float(variances[c]),
                    "faction_positive": fac_pos,
                    "faction_negative": fac_neg,
                }
            )

        factions_output[v_id] = {
            "all_variances_explained": [float(v) for v in variances],
            "axes": axes_data,
        }

        logger.info(
            f"Analyzed {n_pca_components} debate dimensions for {len(video_indices)} participants:"
        )
        for axis in axes_data:
            idx = axis["axis_index"]
            n_pos = len(axis["faction_positive"])
            n_neg = len(axis["faction_negative"])
            logger.info(
                f"  -> PC{idx}: Faction Positive ({n_pos}) vs. Faction Negative ({n_neg})"
            )

    # 5. Save the localized debates
    out_path = "Analysis/RAG_analysis/phase4_factions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(factions_output, f, indent=2)

    logger.info(f"Phase 4 Complete! Faction data saved to {out_path}")
    return factions_output


if __name__ == "__main__":
    run_phase4_pipeline(
        micro_json_path="Analysis/RAG_analysis/phase1_micro_level.json",
        embeddings_path="Analysis/RAG_analysis/phase2_micro_embeddings.npy",
        gpa_results_path="Analysis/RAG_analysis/phase3_gpa_results.json",
        top_n_videos=None,  # Change this if you want to inspect more/fewer videos
        n_pca_components=2,
    )
