import json
import logging
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_phase6_keyword_extraction(
    micro_json_path: str,
    embeddings_path: str,
    factions_path: str,
    model_name: str = "all-MiniLM-L6-v2",
):
    logger.info("Loading Phase 1, 2, and 4 outputs...")
    with open(micro_json_path, "r", encoding="utf-8") as f:
        micro_data = json.load(f)

    with open(factions_path, "r", encoding="utf-8") as f:
        factions = json.load(f)

    embeddings = np.load(embeddings_path)

    logger.info(f"Loading embedding model ({model_name}) to embed individual words...")
    model = SentenceTransformer(model_name, device="cpu")

    extracted_keywords = {}

    for v_id, faction_data in factions.items():
        logger.info(f"Extracting semantic axis keywords for Video {v_id}...")

        # 1. Isolate the document embeddings
        video_indices = [
            i for i, entry in enumerate(micro_data) if str(entry["video_id"]) == v_id
        ]
        if len(video_indices) < 4:
            continue

        vid_embeddings = embeddings[video_indices]

        # --- NEW DYNAMIC N-DIMENSIONAL LOGIC ---
        # Determine how many axes Phase 4 extracted for this video
        axes_data = faction_data.get("axes", [])
        n_pca_components = len(axes_data)

        if n_pca_components == 0:
            continue

        # Re-run PCA to get the actual axis vectors (components_)
        # Note: random_state MUST match Phase 4 (4242) to guarantee identical axes!
        pca = PCA(n_components=n_pca_components, random_state=4242)
        pca.fit(vid_embeddings)

        # 2. Gather all unique processed words used by participants for this video
        unique_words = set()
        for i in video_indices:
            tokens = micro_data[i].get("processed_tokens", [])
            for token in tokens:
                if len(token) > 2:
                    unique_words.add(token.lower())

        unique_words = list(unique_words)
        if not unique_words:
            continue

        # 3. Embed the individual words
        word_embeddings = model.encode(
            unique_words, show_progress_bar=False, convert_to_numpy=True
        )

        # 4. Extract Keywords for EVERY Axis
        axes_keywords = []
        for c in range(n_pca_components):
            debate_axis = pca.components_[c]

            # Dot product of word embeddings onto this specific PCA axis
            projections = np.dot(word_embeddings, debate_axis)
            sorted_indices = np.argsort(projections)

            axes_keywords.append(
                {
                    "axis_index": c + 1,
                    "faction_negative_keywords": [
                        unique_words[idx] for idx in sorted_indices[:5]
                    ],
                    "faction_positive_keywords": [
                        unique_words[idx] for idx in sorted_indices[::-1][:5]
                    ],
                }
            )

        extracted_keywords[v_id] = {"axes_keywords": axes_keywords}

    # Save the output
    out_path = "Analysis/RAG_analysis/phase6_axis_keywords.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extracted_keywords, f, indent=2)

    logger.info(f"Phase 6 Complete! Keywords saved to {out_path}")
    return extracted_keywords


if __name__ == "__main__":
    run_phase6_keyword_extraction(
        micro_json_path="Analysis/RAG_analysis/phase1_micro_level.json",
        embeddings_path="Analysis/RAG_analysis/phase2_micro_embeddings.npy",
        factions_path="Analysis/RAG_analysis/phase4_factions.json",
    )
