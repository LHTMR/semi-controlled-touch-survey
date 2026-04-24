import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from nltk.corpus import stopwords
from adjustText import adjust_text
import scipy.special as sc
from scipy.stats import betabinom, norm
import argparse
import json
import sys
import os

# Import your extractor script
import all_words_extractor as awe


def get_raw_wmw(A, B, C, D):
    """Calculates the raw natural log of Geometric WMW Odds."""
    term1_num = (2 * A * D) + (A * C) + (B * D)
    term1_den = (2 * B * C) + (A * C) + (B * D)
    term2_num = (2 * A * D) + (A * B) + (C * D)
    term2_den = (2 * B * C) + (A * B) + (C * D)

    if term1_den == 0 or term2_den == 0:
        return 0.0
    wmw_geo = np.sqrt((term1_num / term1_den) * (term2_num / term2_den))
    return np.log(wmw_geo)


def calc_hellinger_score(A, B, C, D):
    """Path 1: Signed Hellinger Distance between Prior and Posterior Beta."""
    # New way (calculates the true global proportions using the column totals)
    total_target = A + C
    total_other = B + D
    total_corpus = total_target + total_other

    alpha_0 = 2.0 * (total_target / total_corpus)
    beta_0 = 2.0 * (total_other / total_corpus)

    # Posterior parameters
    alpha_1 = alpha_0 + A
    beta_1 = beta_0 + B

    # Calculate Hellinger distance using log-beta functions to avoid underflow
    log_num = sc.betaln((alpha_1 + alpha_0) / 2.0, (beta_1 + beta_0) / 2.0)
    log_den = 0.5 * (sc.betaln(alpha_1, beta_1) + sc.betaln(alpha_0, beta_0))

    integral_term = min(
        1.0, np.exp(log_num - log_den)
    )  # Clamp to 1.0 to prevent floating point errors
    h_dist = np.sqrt(1.0 - integral_term)

    # Get direction from raw WMW
    raw_wmw = get_raw_wmw(A, B, C, D)
    sign = 1 if raw_wmw > 0 else -1 if raw_wmw < 0 else 0
    return sign * h_dist


def calc_theta_harmonic_f1(A, B, C, D):
    """
    Translates WMW Odds into Theta (Probability of Superiority),
    scales it to an absolute magnitude [0, 1], and applies a Harmonic Gate
    using the Hellinger Distance (Confidence).
    """
    # 1. Get raw WMW log-odds, then convert to raw odds
    log_odds = get_raw_wmw(A, B, C, D)
    odds = np.exp(log_odds)

    # 2. Convert Odds to Theta (Concordance Probability)
    theta = odds / (1.0 + odds)

    # 3. Scale Theta to an Absolute Effect Magnitude (0 = Neutral, 1 = Perfect)
    theta_magnitude = 2.0 * abs(theta - 0.5)

    # 4. Calculate Confidence (Unsigned Hellinger: 0 to 1)
    alpha_0 = 2.0 * (C / (C + D))
    beta_0 = 2.0 * (D / (C + D))
    alpha_1, beta_1 = alpha_0 + A, beta_0 + B

    log_num = sc.betaln((alpha_1 + alpha_0) / 2.0, (beta_1 + beta_0) / 2.0)
    log_den = 0.5 * (sc.betaln(alpha_1, beta_1) + sc.betaln(alpha_0, beta_0))
    integral_term = min(1.0, np.exp(log_num - log_den))
    confidence = np.sqrt(1.0 - integral_term)

    # 5. Harmonic Mean (F1-Score style)
    if theta_magnitude + confidence == 0.0:
        return 0.0

    # harmonic_score = 2 * (theta_magnitude * confidence) / (theta_magnitude + confidence)
    harmonic_score = 2 / (1 / theta_magnitude + 1 / confidence)

    # 6. Restore the direction (Positive for Video, Negative for Other)
    sign = 1 if theta > 0.5 else -1
    return sign * harmonic_score


def calc_hybrid_wmw_hellinger(A, B, C, D):
    """
    Hybrid Metric: Effect Size scaled by Information Gain.
    Shrinks the raw WMW log-odds toward zero if the evidence (frequency) is too weak
    to definitively separate the posterior from the prior.
    """
    # 1. Get the raw effect size and direction
    raw_wmw = get_raw_wmw(A, B, C, D)

    if raw_wmw == 0.0:
        return 0.0

    # 2. Get the weight of evidence (Unsigned Hellinger, bounded 0 to 1)
    alpha_0 = 2.0 * (C / (C + D))
    beta_0 = 2.0 * (D / (C + D))

    alpha_1 = alpha_0 + A
    beta_1 = beta_0 + B

    log_num = sc.betaln((alpha_1 + alpha_0) / 2.0, (beta_1 + beta_0) / 2.0)
    log_den = 0.5 * (sc.betaln(alpha_1, beta_1) + sc.betaln(alpha_0, beta_0))

    integral_term = min(1.0, np.exp(log_num - log_den))
    h_dist_unsigned = np.sqrt(1.0 - integral_term)

    # 3. Scale the effect by the confidence
    return raw_wmw * h_dist_unsigned


def calc_temperature_softmax_effect(A, B, C, D, gamma=20):
    """
    Applies a Temperature-Scaled Softmax.
    The WMW Log-Odds act as the base logit, and the Hellinger Distance
    acts as the inverse temperature.
    """
    raw_wmw = get_raw_wmw(A, B, C, D)
    if raw_wmw == 0.0:
        return 0.0

    # 1. Calculate Hellinger (0 to 1)
    alpha_0 = 2.0 * (C / (C + D))
    beta_0 = 2.0 * (D / (C + D))
    alpha_1, beta_1 = alpha_0 + A, beta_0 + B

    log_num = sc.betaln((alpha_1 + alpha_0) / 2.0, (beta_1 + beta_0) / 2.0)
    log_den = 0.5 * (sc.betaln(alpha_1, beta_1) + sc.betaln(alpha_0, beta_0))
    integral_term = min(1.0, np.exp(log_num - log_den))

    # This is our Inverse Temperature
    h_dist = np.sqrt(1.0 - integral_term)

    # 2. Cross-pollination: Logit * Inverse Temperature * Global Scale
    # (gamma allows you to tune how aggressively the temperature squashes the plot)
    z = raw_wmw * (h_dist * gamma)

    # 3. Apply the Softmax (Sigmoid) and recenter to [-1, 1] for the plot
    # 2 * sigmoid(z) - 1 is mathematically identical to tanh(z / 2)
    softmax_centered = np.tanh(z / 2.0)

    return softmax_centered


def calc_betabinom_surprise(A, B, C, D):
    """
    Calculates the 'Surprise Z-Score' of observing A occurrences in (A+C) trials,
    given the background probability derived from B occurrences in (B+D) trials.
    """
    # 1. Background evidence (Alpha and Beta for the Beta-Binomial)
    # B = successes in background, D = failures in background
    # Adding 1.0 provides Laplace smoothing
    a = B + 1.0
    b = D + 1.0

    # 2. Target Video properties
    n = A + C  # Total words in target video (number of trials)

    # Expected number of occurrences in the target video
    expected_p = a / (a + b)
    expected_A = n * expected_p

    # To prevent math domain errors on astronomical probabilities
    min_p_val = 1e-15

    if A > expected_A:
        # ENRICHED: Probability of observing A or MORE occurrences.
        # sf(k) is the survival function P(X > k). So P(X >= A) = sf(A - 1)
        p_val = betabinom.sf(A - 1, n, a, b)
        p_val = max(p_val, min_p_val)
        # Convert p-value to a positive Z-score
        z_score = norm.isf(p_val)

    elif A < expected_A:
        # DEPLETED: Probability of observing A or FEWER occurrences.
        p_val = betabinom.cdf(A, n, a, b)
        p_val = max(p_val, min_p_val)
        # Convert p-value to a negative Z-score
        z_score = -norm.isf(p_val)

    else:
        z_score = 0.0

    return z_score


def calc_betabinom_hellinger(A, B, C, D):
    """
    Calculates the discrete Hellinger Distance between the Expected Beta-Binomial PMF
    (based on background data) and the Observed Beta-Binomial PMF (based on target data).
    """
    n = A + C  # Total words in the target video

    # If the video has no words, there is no distance
    if n == 0:
        return 0.0

    # Create an array of all possible counts k from 0 to n
    k_vals = np.arange(n + 1)

    # 1. Expected PMF (Null model using background proportions)
    a_expected = B + 1.0
    b_expected = D + 1.0
    pmf_expected = betabinom.pmf(k_vals, n, a_expected, b_expected)

    # 2. Observed PMF (Alternative model using target proportions)
    a_observed = A + 1.0
    b_observed = C + 1.0
    pmf_observed = betabinom.pmf(k_vals, n, a_observed, b_observed)

    # 3. Calculate Discrete Hellinger Distance
    # H = sqrt( 1 - sum( sqrt(P[k] * Q[k]) ) )
    bhattacharyya_coeff = np.sum(np.sqrt(pmf_expected * pmf_observed))

    # Clamp to prevent tiny floating point errors from pushing it over 1.0
    bhattacharyya_coeff = min(1.0, max(0.0, bhattacharyya_coeff))

    h_dist = np.sqrt(1.0 - bhattacharyya_coeff)

    # 4. Apply Direction (Positive if enriched in Target, Negative if depleted)
    expected_mean = n * (a_expected / (a_expected + b_expected))
    observed_mean = n * (a_observed / (a_observed + b_observed))

    sign = 1 if observed_mean > expected_mean else -1

    return sign * h_dist


def calc_betabinom_kl(A, B, C, D):
    """
    Numerically Stable Kullback-Leibler (KL) Divergence using log-space.
    """
    n = A + C
    if n == 0:
        return 0.0

    k_vals = np.arange(n + 1)

    a_expected = B + 1.0
    b_expected = D + 1.0
    logpmf_expected = betabinom.logpmf(k_vals, n, a_expected, b_expected)

    a_observed = A + 1.0
    b_observed = C + 1.0
    logpmf_observed = betabinom.logpmf(k_vals, n, a_observed, b_observed)

    # KL(Obs || Exp) = sum( exp(log_obs) * (log_obs - log_exp) )
    # This prevents 0 * inf or 0 / 0 errors
    kl_div = np.sum(np.exp(logpmf_observed) * (logpmf_observed - logpmf_expected))

    # Apply Direction
    expected_mean = n * (a_expected / (a_expected + b_expected))
    observed_mean = n * (a_observed / (a_observed + b_observed))
    sign = 1 if observed_mean > expected_mean else -1

    return sign * kl_div


def calc_betabinom_js(A, B, C, D):
    """
    Numerically Stable Jensen-Shannon (JS) Distance using log-space.
    """
    n = A + C
    if n == 0:
        return 0.0

    k_vals = np.arange(n + 1)

    a_expected = B + 1.0
    b_expected = D + 1.0
    logpmf_expected = betabinom.logpmf(k_vals, n, a_expected, b_expected)

    a_observed = A + 1.0
    b_observed = C + 1.0
    logpmf_observed = betabinom.logpmf(k_vals, n, a_observed, b_observed)

    # 1. Calculate log(M) = log(0.5 * P + 0.5 * Q)
    # Using np.logaddexp ensures total numerical stability
    log_2 = np.log(2.0)
    logpmf_m = np.logaddexp(logpmf_observed - log_2, logpmf_expected - log_2)

    # 2. JS Divergence components
    kl_obs_m = np.sum(np.exp(logpmf_observed) * (logpmf_observed - logpmf_m))
    kl_exp_m = np.sum(np.exp(logpmf_expected) * (logpmf_expected - logpmf_m))

    js_div = 0.5 * kl_obs_m + 0.5 * kl_exp_m

    # 3. Convert to distance (clamp small float inaccuracies below 0)
    js_dist = np.sqrt(max(0.0, js_div))

    # 4. Apply Direction
    expected_mean = n * (a_expected / (a_expected + b_expected))
    observed_mean = n * (a_observed / (a_observed + b_observed))
    sign = 1 if observed_mean > expected_mean else -1

    return sign * js_dist


def load_transformation_dictionary(filepath):
    """Load transformation dictionary from JSON file."""
    print(f"Loading transformation dictionary from {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if it's the format with metadata and transformation_dictionary
        if isinstance(data, dict) and "transformation_dictionary" in data:
            # Format: {"metadata": {...}, "transformation_dictionary": {...}}
            transformation_dict = data["transformation_dictionary"]
        else:
            transformation_dict = data

        # Convert to the format we need: {main_word: [variants]}
        variant_to_main = {}
        for main_word, variants in transformation_dict.items():
            if isinstance(variants, list):
                # Format: {"main_word": ["variant1", "variant2", ...]}
                all_variants = [main_word] + variants
                for variant in all_variants:
                    variant_to_main[variant] = main_word
            elif isinstance(variants, dict) and "members" in variants:
                # Format: {"main_word": {"members": ["variant1", "variant2", ...], ...}}
                all_variants = [main_word] + variants["members"]
                for variant in all_variants:
                    variant_to_main[variant] = main_word
            else:
                # Unknown format, try to handle it
                variant_to_main[main_word] = main_word

        print(
            f"  Loaded transformation dictionary with {len(variant_to_main)} mappings"
        )
        return variant_to_main

    except Exception as e:
        print(f"Error loading transformation dictionary: {e}")
        return None


def calculate_all_metrics(
    df, target_videos, config, variant_to_main=None, text_columns=None
):
    print("Setting up advanced text processing...")

    # Use the same configuration as all_words_extractor
    awe_config = awe.DEFAULT_CONFIG.copy()

    # Load spaCy for negation binding
    nlp = awe.load_spacy_model(awe_config)
    if nlp is not None:
        awe_config["_spacy_model"] = nlp
    else:
        awe_config["use_spacy_negation"] = False

    stopwords_set = awe.get_stopwords(awe_config)

    print("Concatenating text columns...")
    if text_columns is None:
        text_columns = [
            "Sensory",
            "Emotional_self",
            "Emotional_touch",
            "Intention&Purpose",
            "Social_context",
        ]

    # Validate that all specified text columns exist in the dataframe
    missing_cols = [col for col in text_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Text columns not found in dataframe: {missing_cols}")

    print(f"Using text columns: {text_columns}")
    df["all_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

    print("Preprocessing text for each participant (this may take a moment)...")

    def process_and_filter(text):
        # 1. spaCy negation binding and tokenization - using the EXACT same function as all_words_extractor
        tokens = awe.preprocess_text(text, awe_config)
        # 2. Filter out stopwords and negative stopwords (e.g., "not_and")
        return [
            w
            for w in tokens
            if w not in stopwords_set
            and not awe.is_negative_stopword(w, stopwords_set, awe_config)
        ]

    df["tokens"] = df["all_text"].apply(process_and_filter)

    print("Grouping word variations...")

    if variant_to_main is not None:
        # Use the provided transformation dictionary
        print("Using provided transformation dictionary...")
    else:
        # Use awe to group typos and morphological variations (same as all_words_extractor)
        print("Generating word variation groups using awe...")
        # Flatten all tokens to calculate global frequencies for the grouping algorithm
        all_tokens_flat = [word for tokens in df["tokens"] for word in tokens]
        global_counts = Counter(all_tokens_flat)

        # Use awe to group typos and morphological variations
        groups = awe.group_word_variations(global_counts, awe_config)

        # Create a mapping dictionary to convert any variant back to its "Main Word"
        variant_to_main = {}
        for main_word, members in groups.items():
            for variant, _ in members:
                variant_to_main[variant] = main_word

    print("Applying groups and converting to presence/absence per participant...")
    # Map tokens to their main word and take a UNIQUE set per row (participant)
    df["unique_main_words"] = df["tokens"].apply(
        lambda tokens: set(variant_to_main.get(w, w) for w in tokens)
    )

    # Handle multiple target videos
    if isinstance(target_videos, (int, float)):
        target_videos = [int(target_videos)]

    # Separate into target video responses and other video responses
    target_mask = df["Touch No."].isin(target_videos)
    target_responses = df[target_mask]["unique_main_words"].tolist()
    other_responses = df[~target_mask]["unique_main_words"].tolist()

    # The total number of trials is now the total number of PARTICIPANTS
    total_target_participants = len(target_responses)
    total_other_participants = len(other_responses)

    # Count how many participants used each word
    target_doc_counts = Counter(
        word for response in target_responses for word in response
    )
    other_doc_counts = Counter(
        word for response in other_responses for word in response
    )

    all_main_words = set(target_doc_counts.keys()).union(set(other_doc_counts.keys()))

    print("Calculating geometric and probabilistic metrics...")
    results = []
    for word in all_main_words:
        # A = Participants who watched Target Video(s) and used the word
        A = target_doc_counts.get(word, 0)
        # B = Participants who watched Other Videos and used the word
        B = other_doc_counts.get(word, 0)

        # C = Participants who watched Target Video(s) and DID NOT use the word
        C = total_target_participants - A
        # D = Participants who watched Other Videos and DID NOT use the word
        D = total_other_participants - B

        results.append(
            {
                "Word": word,
                "Total_Freq": A + B,  # This now represents Total Participants!
                "Hellinger": calc_hellinger_score(A, B, C, D),
                "WMW": get_raw_wmw(A, B, C, D),
                "Hybrid_WMW_Hellinger": calc_hybrid_wmw_hellinger(A, B, C, D),
                "Theta_Harmonic_F1": calc_theta_harmonic_f1(A, B, C, D),
                "Temperature_Softmax_Effect": calc_temperature_softmax_effect(
                    A, B, C, D, gamma=config.get("temperature_gamma", 20)
                ),
                "Betabinom_Surprise": calc_betabinom_surprise(A, B, C, D),
                "Betabinom_Hellinger": calc_betabinom_hellinger(A, B, C, D),
                "Betabinom_KL": calc_betabinom_kl(A, B, C, D),
                "Betabinom_JS": calc_betabinom_js(A, B, C, D),
            }
        )

    return pd.DataFrame(results)


def plot_volcano(
    plot_df,
    score_column,
    title,
    target_videos,
    show_n_words=20,
    figsize=(10, 6),
    output_path=None,
    annotation_params=None,
):
    """
    Generate a volcano plot with improved annotation placement.

    Parameters:
    -----------
    plot_df : DataFrame
        Data containing words and scores
    score_column : str
        Column name for the score to plot
    title : str
        Plot title
    target_videos : list or int
        Target video number(s)
    show_n_words : int
        Number of top words to annotate on each side
    figsize : tuple
        Figure dimensions (width, height) in inches
    output_path : str or None
        If provided, save figure to this path instead of showing
    annotation_params : dict or None
        Custom parameters for adjust_text
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(
        x=score_column,
        y="Total_Freq",
        size="Total_Freq",
        sizes=(20, 400),
        alpha=0.6,
        data=plot_df,
        legend=False,
        color="purple",
    )

    texts = []
    top_pos = plot_df.sort_values(score_column, ascending=False).head(show_n_words)
    for _, row in top_pos.iterrows():
        texts.append(
            plt.text(
                row[score_column],
                row["Total_Freq"],
                row["Word"],
                fontsize=8,
                color="darkgreen",
            )
        )

    top_neg = plot_df.sort_values(score_column, ascending=True).head(show_n_words)
    for _, row in top_neg.iterrows():
        texts.append(
            plt.text(
                row[score_column],
                row["Total_Freq"],
                row["Word"],
                fontsize=8,
                color="darkred",
            )
        )

    plt.axvline(0, color="grey", linestyle="--")
    plt.title(title, fontsize=14)

    # Format target videos for display
    if isinstance(target_videos, list) and len(target_videos) > 1:
        target_str = f"VIDEOS {', '.join(map(str, target_videos))}"
    elif isinstance(target_videos, list):
        target_str = f"VIDEO {', '.join(map(str, target_videos))}"
    else:
        target_str = f"VIDEO {target_videos}"

    # Adjust the length of the strings for visual symmetry
    margin_left = ""
    margin_right = " " * 6
    if len(target_str) < len("OTHER videos"):
        margin_right += " " * (len("OTHER videos") - len(target_str))
    elif len(target_str) > len("OTHER videos"):
        margin_left += " " * (len(target_str) - len("OTHER videos"))

    plt.xlabel(
        f"{margin_left}← OTHER videos{str(" ")*5*int(figsize[0])}{target_str} →{margin_right}\n{score_column}",
        fontsize=12,
    )
    plt.ylabel("Number of Participants", fontsize=12)
    plt.yscale("log")

    # Default annotation parameters with improved overlap prevention
    default_params = {
        "arrowprops": dict(arrowstyle="-", color="gray", lw=0.5),
        "force_text": (0.2, 0.3),  # Increased force for better separation
        "force_points": (0.2, 0.3),
        "expand_text": (1.2, 1.3),  # Increased expansion
        "expand_points": (1.2, 1.3),
        "lim": 1000,  # Increased limit for more iterations
        "precision": 0.001,
        "only_move": {
            "points": "y",
            "text": "xy",
        },  # Allow text to move in both directions
        "avoid_self": True,  # Avoid overlapping with itself
        "avoid_points": True,  # Avoid overlapping with points
        "avoid_text": True,  # Avoid overlapping with other text
    }

    # Update with custom parameters if provided
    if annotation_params:
        default_params.update(annotation_params)

    # Use adjust_text to repel labels with improved overlap prevention
    adjust_text(texts, **default_params)

    plt.tight_layout()

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        # print(f"Figure saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Geometric Association Test with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single target video (default: only betabinom_hellinger plot)
  python geometric_association_test_modified.py --input Processed\\ Data/touch_data_fixed.csv.txt --target-video 12
  
  # Multiple target videos with all plots
  python geometric_association_test_modified.py --input Processed\\ Data/touch_data_fixed.csv.txt --target-video 12 13 14 --plots all
  
  # Specific plots only
  python geometric_association_test_modified.py --input Processed\\ Data/touch_data_fixed.csv.txt --target-video 12 --plots hellinger betabinom_hellinger
  
  # Save plots as PDF with custom dimensions
  python geometric_association_test_modified.py --input Processed\\ Data/touch_data_fixed.csv.txt --target-video 12 --plots betabinom_hellinger --plot-dir plots --fig-width 12 --fig-height 8
  
  # Save CSV results to directory
  python geometric_association_test_modified.py --input Processed\\ Data/touch_data_fixed.csv.txt --target-video 12 --results-dir results
  
  # With transformation dictionary
  python geometric_association_test_modified.py --input Processed\\ Data/touch_data_fixed.csv.txt --target-video 12 --transformation-dict grouping_dictionary.json
  
  # Custom parameters
  python geometric_association_test_modified.py --input Processed\\ Data/touch_data_fixed.csv.txt --target-video 12 --min-freq 5 --temperature-gamma 30 --show-words 15
  
  # Specify text columns
  python geometric_association_test_modified.py --input Processed\\ Data/touch_data_fixed.csv.txt --target-video 12 --text-columns Sensory Emotional_self Emotional_touch "Intention&Purpose" Social_context
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input data file path")
    parser.add_argument(
        "--target-video",
        "-t",
        nargs="+",
        type=int,
        required=True,
        help="Target video number(s) (one or more)",
    )
    parser.add_argument(
        "--text-columns",
        "-c",
        nargs="+",
        default=[
            "Sensory",
            "Emotional_self",
            "Emotional_touch",
            "Intention&Purpose",
            "Social_context",
            "Social_self",
            "Social_body",
            "Social_place",
        ],
        help='Text columns to analyze (default: Sensory Emotional_self Emotional_touch "Intention&Purpose" Social_context Social_self Social_body Social_place)',
    )
    parser.add_argument(
        "--min-freq",
        "-m",
        type=int,
        default=2,
        help="Minimum frequency threshold (default: 2)",
    )
    parser.add_argument(
        "--temperature-gamma",
        "-g",
        type=float,
        default=20.0,
        help="Temperature gamma parameter (default: 20.0)",
    )
    parser.add_argument(
        "--show-words",
        "-n",
        type=int,
        default=20,
        help="Number of words to show in plots (default: 20)",
    )
    parser.add_argument(
        "--transformation-dict",
        "-d",
        help="Path to transformation dictionary JSON file",
    )
    parser.add_argument(
        "--results-dir",
        "-o",
        help="Directory to save CSV results (creates directory if doesn't exist). CSV files will have names matching their plot counterparts.",
    )

    # New arguments for plot control
    parser.add_argument(
        "--plots",
        "-p",
        nargs="+",
        default=["betabinom_hellinger"],
        choices=[
            "all",
            "hellinger",
            "wmw",
            "temperature_softmax",
            "betabinom_surprise",
            "betabinom_hellinger",
            "betabinom_kl",
            "betabinom_js",
        ],
        help="Plots to generate (default: betabinom_hellinger only). Use 'all' for all plots.",
    )
    parser.add_argument(
        "--plot-dir",
        help="Directory to save plots as PDF (if not specified, plots are displayed)",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=10.0,
        help="Figure width in inches (default: 10)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=8.5,
        help="Figure height in inches (default: 8.5)",
    )
    parser.add_argument(
        "--annotation-force",
        type=float,
        default=0.8,
        help="Force parameter for annotation placement (default: 0.8)",
    )
    parser.add_argument(
        "--annotation-expand",
        type=float,
        default=2.0,
        help="Expand parameter for annotation placement (default: 2.0)",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Create config dictionary from arguments
    config = {
        "min_frequency_threshold": args.min_freq,
        "temperature_gamma": args.temperature_gamma,
        "show_n_words": args.show_words,
    }

    print("=" * 60)
    print("Geometric Association Test")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Target video(s): {args.target_video}")
    print(f"Text columns: {args.text_columns}")
    print(f"Minimum frequency threshold: {args.min_freq}")
    print(f"Temperature gamma: {args.temperature_gamma}")
    print(f"Show N words: {args.show_words}")
    print(f"Plots to generate: {args.plots}")
    print(f"Figure dimensions: {args.fig_width} x {args.fig_height} inches")
    if args.plot_dir:
        print(f"Plot directory: {args.plot_dir}")
    if args.results_dir:
        print(f"Results directory for CSV files: {args.results_dir}")
    if args.transformation_dict:
        print(f"Transformation dictionary: {args.transformation_dict}")
    print("=" * 60)

    # Load transformation dictionary if provided
    variant_to_main = None
    if args.transformation_dict:
        variant_to_main = load_transformation_dictionary(args.transformation_dict)
        if variant_to_main is None:
            print(
                "Warning: Could not load transformation dictionary, using automatic grouping"
            )

    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input, sep="|")
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Calculate metrics
    stats_df = calculate_all_metrics(
        df, args.target_video, config, variant_to_main, args.text_columns
    )

    # Apply frequency threshold
    plot_df = stats_df[stats_df["Total_Freq"] >= args.min_freq].copy()
    print(f"\nAfter filtering (freq >= {args.min_freq}): {len(plot_df)} words")

    # Save results if results directory specified
    if args.results_dir:
        # Create results directory if it doesn't exist
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"Created results directory: {args.results_dir}")

    # Determine which plots to generate
    plot_configs = []
    if "all" in args.plots:
        plot_configs = [
            ("Hellinger", "Hellinger Distance (Info Gain)"),
            ("WMW", "WMW Log Odds (Effect Size)"),
            ("Temperature_Softmax_Effect", "Temperature adjusted Softmax"),
            ("Betabinom_Surprise", "Standard Deviations of Surprise"),
            (
                "Betabinom_Hellinger",
                "Hellinger Distance between BetaBinomial Distributions",
            ),
            (
                "Betabinom_KL",
                "Kullback-Leibler Divergence (Bits of Information)",
            ),
            ("Betabinom_JS", "Jensen-Shannon Distance"),
        ]
    else:
        plot_mapping = {
            "hellinger": ("Hellinger", "Hellinger Distance (Info Gain)"),
            "wmw": ("WMW", "WMW Log Odds (Effect Size)"),
            "temperature_softmax": (
                "Temperature_Softmax_Effect",
                "Temperature adjusted Softmax",
            ),
            "betabinom_surprise": (
                "Betabinom_Surprise",
                "Standard Deviations of Surprise",
            ),
            "betabinom_hellinger": (
                "Betabinom_Hellinger",
                "Hellinger Distance between BetaBinomial Distributions",
            ),
            "betabinom_kl": (
                "Betabinom_KL",
                "Kullback-Leibler Divergence (Bits of Information)",
            ),
            "betabinom_js": ("Betabinom_JS", "Jensen-Shannon Distance"),
        }
        for plot_name in args.plots:
            if plot_name in plot_mapping:
                plot_configs.append(plot_mapping[plot_name])

    # Generate plots
    if plot_configs:
        print(f"\nGenerating {len(plot_configs)} plot(s)...")

        # Annotation parameters for improved placement
        annotation_params = {
            "force_text": (args.annotation_force, args.annotation_force * 1.5),
            "force_points": (args.annotation_force, args.annotation_force * 1.5),
            "expand_text": (args.annotation_expand, args.annotation_expand * 1.1),
            "expand_points": (args.annotation_expand, args.annotation_expand * 1.1),
        }

        for score_column, plot_title in plot_configs:
            title = f"{plot_title} - Video(s) {args.target_video}"

            # Create safe filename base
            safe_title = re.sub(r"[^\w\s-]", "", plot_title.lower())
            safe_title = re.sub(r"[-\s]+", "_", safe_title)
            filename_base = (
                f"{safe_title}_video_{'_'.join(map(str, args.target_video))}"
            )

            # Save CSV results if results directory specified
            if args.results_dir:
                csv_filename = f"{filename_base}.csv"
                csv_path = os.path.join(args.results_dir, csv_filename)
                plot_df[["Word", "Total_Freq", score_column]].sort_values(
                    by=score_column, ascending=False
                ).to_csv(csv_path, index=False)
                print(f"  CSV results saved to: {csv_path}")

            # Generate plot
            if args.plot_dir:
                # Create plot directory if it doesn't exist
                os.makedirs(args.plot_dir, exist_ok=True)

                pdf_filename = f"{filename_base}.pdf"
                output_path = os.path.join(args.plot_dir, pdf_filename)

                plot_volcano(
                    plot_df,
                    score_column,
                    title,
                    args.target_video,
                    args.show_words,
                    figsize=(args.fig_width, args.fig_height),
                    output_path=output_path,
                    annotation_params=annotation_params,
                )
                print(f"  Plot saved to: {output_path}")
            else:
                plot_volcano(
                    plot_df,
                    score_column,
                    title,
                    args.target_video,
                    args.show_words,
                    figsize=(args.fig_width, args.fig_height),
                    output_path=None,
                    annotation_params=annotation_params,
                )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
