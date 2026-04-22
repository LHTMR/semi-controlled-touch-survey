import json
import logging
import os
from datetime import datetime

# Import the pipeline steps from our previous modules
from RAG_analysis_phase_1_ingestion import run_phase1_pipeline
from RAG_analysis_phase_2_embedding import run_phase2_pipeline
from RAG_analysis_phase_3_gpa import run_phase3_pipeline
from RAG_analysis_phase_4_factions import run_phase4_pipeline
from RAG_analysis_phase_5_reranker import run_phase5_pipeline
from RAG_analysis_phase_6_keywords import run_phase6_keyword_extraction


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_markdown_report(
    final_quotes_path: str, gpa_results_path: str, output_path: str
):
    """
    Translates the JSON math/quotes into a readable Markdown document.
    """
    logger.info("Generating final Markdown report...")

    with open(final_quotes_path, "r", encoding="utf-8") as f:
        quotes_data = json.load(f)

    with open(gpa_results_path, "r", encoding="utf-8") as f:
        gpa_data = {str(item["video_id"]): item for item in json.load(f)}

    try:
        with open(
            "Analysis/RAG_analysis/phase6_axis_keywords.json", "r", encoding="utf-8"
        ) as f:
            keywords_data = json.load(f)
    except FileNotFoundError:
        keywords_data = {}

    md_lines = []
    md_lines.append("# Geometric RAG Analysis: Semi-Controlled Touch Survey")
    md_lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    md_lines.append("---")

    # ==========================================
    # PART 1: MACRO CONSENSUS
    # ==========================================
    md_lines.append("## Part 1: The Global Semantic Anchors")
    md_lines.append(
        "The unsupervised model identified the following foundational categories of touch across the entire dataset. The quotes below are the mathematically 'purest' expressions of these categories, extracted via Cross-Encoder tournament.\n"
    )

    macro = quotes_data.get("macro_consensus_quotes", {})
    for anchor_name, data in macro.items():
        md_lines.append(f"### {anchor_name.replace('_', ' ').title()}")
        md_lines.append(f"> \"{data['best_representative_quote']}\"")
        md_lines.append(
            f"  >  — *Participant {data.get('participant_id', 'Unknown')} (Video {data['source_video_id']})*\n"
        )

    md_lines.append("---")

    # ==========================================
    # PART 2: MICRO DEBATES
    # ==========================================
    md_lines.append("## Part 2: Polarizing Touches & Faction Debates")
    md_lines.append(
        "The following videos deviated significantly from the global consensus. The participants who watched these videos formed distinct, opposing 'Schools of Thought'.\n"
    )

    micro = quotes_data.get("micro_debate_quotes", {})
    for v_id, data in micro.items():
        pol_score = gpa_data.get(v_id, {}).get("polarization_percent", 0.0)
        var_exp = data.get("variance_explained", 0.0)

        md_lines.append(f"### Debates on Video {v_id}")
        md_lines.append(f"**Relative Polarization:** {pol_score:.1f}%\n")

        vid_keywords = keywords_data.get(v_id, {}).get("axes_keywords", [])

        # Print a sub-header for every axis!
        for axis_data in data.get("axes", []):
            idx = axis_data["axis_index"]
            var_exp = axis_data["variance_explained"]
            md_lines.append(
                f"#### Primary Debate PC{idx} ({var_exp:.1f}% Structure Strength)"
            )

            # Find matching keywords
            kw = next((k for k in vid_keywords if k["axis_index"] == idx), {})
            kw_pos = ", ".join(kw.get("faction_positive_keywords", ["N/A"]))
            kw_neg = ", ".join(kw.get("faction_negative_keywords", ["N/A"]))

            fac_pos = axis_data.get("faction_positive_winner")
            if fac_pos:
                md_lines.append(f"##### Faction Positive: *{kw_pos}*")
                md_lines.append(f"> \"{fac_pos['quote']}\"")
                md_lines.append(f"  > — *Participant {fac_pos['participant_id']}*\n")

            fac_neg = axis_data.get("faction_negative_winner")
            if fac_neg:
                md_lines.append(f"##### Faction Negative: *{kw_neg}*")
                md_lines.append(f"> \"{fac_neg['quote']}\"")
                md_lines.append(f"  > — *Participant {fac_neg['participant_id']}*\n")

        md_lines.append("\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Report successfully saved to {output_path}")


def main_orchestrator(
    data_path: str = "Processed Data/touch_data_fixed.csv.txt",
    dict_path: str = "Analysis/All_words_by_frequency/word_grouping_dict.json",
    out_dir: str = "Analysis/RAG_analysis",
    run_all: bool = False,
):
    """
    Main execution pipeline.
    Set run_all=True to run from scratch, or False to just generate the report from existing JSON files.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Define all file paths
    p1_micro = f"{out_dir}/phase1_micro_level.json"
    p1_macro = f"{out_dir}/phase1_macro_level.json"
    p2_embeds = f"{out_dir}/phase2_micro_embeddings.npy"
    p2_anchors = f"{out_dir}/phase2_anchors.json"
    p3_gpa = f"{out_dir}/phase3_gpa_results.json"
    p4_factions = f"{out_dir}/phase4_factions.json"
    p5_quotes = f"{out_dir}/phase5_final_quotes.json"
    final_report = f"{out_dir}/FINAL_GEOMETRIC_REPORT.md"

    if run_all:
        logger.info("=== STARTING END-TO-END PIPELINE ===")
        # Phase 1
        micro_data, macro_data = run_phase1_pipeline(data_path, dict_path)
        with open(p1_micro, "w", encoding="utf-8") as f:
            json.dump(micro_data, f)
        with open(p1_macro, "w", encoding="utf-8") as f:
            json.dump(macro_data, f)

        # Phase 2
        run_phase2_pipeline(p1_micro, k_anchors=8)

        # Phase 3
        run_phase3_pipeline(p1_micro, p2_embeds, p2_anchors)

        # Phase 4
        run_phase4_pipeline(
            p1_micro, p2_embeds, p3_gpa, top_n_videos=None, n_pca_components=3
        )

        # Phase 5
        run_phase5_pipeline(p1_micro, p2_embeds, p2_anchors, p4_factions)

        # Phase 6
        run_phase6_keyword_extraction(
            p1_micro, p2_embeds, p4_factions, model_name="all-MiniLM-L6-v2"
        )

    # Last Phase: Report Generation
    generate_markdown_report(p5_quotes, p3_gpa, final_report)
    logger.info("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    # Because you already ran Phases 1-5, we can set run_all=False
    # to instantly generate the report from your saved files!
    main_orchestrator(data_path="Processed Data/touch_data_fixed.csv.txt", run_all=True)
