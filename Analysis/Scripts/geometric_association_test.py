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

# Import your extractor script
import all_words_extractor as awe

MIN_FREQUENCY_THRESHOLD = 2
TEMPERATURE_GAMMA = 20
SHOW_N_WORDS = 20

TARGET_VIDEO = 12


def clean_and_tokenize(text):
    if pd.isna(text):
        return []
    words = re.findall(r"\b[a-z]{2,}\b", str(text).lower())
    stop_words = set(stopwords.words("english"))
    return [w for w in words if w not in stop_words]


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


def calc_temperature_softmax_effect(A, B, C, D, gamma=TEMPERATURE_GAMMA):
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


def calculate_all_metrics(df, target_video):
    print("Setting up advanced text processing...")
    config = awe.DEFAULT_CONFIG.copy()

    # Load spaCy for negation binding
    nlp = awe.load_spacy_model(config)
    if nlp is not None:
        config["_spacy_model"] = nlp
    else:
        config["use_spacy_negation"] = False

    stopwords_set = awe.get_stopwords(config)

    print("Concatenating text columns...")
    text_cols = [
        "Sensory",
        "Emotional_self",
        "Emotional_touch",
        "Intention&Purpose",
        "Social_context",
    ]
    df["all_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

    print("Preprocessing text for each participant (this may take a moment)...")

    def process_and_filter(text):
        # 1. spaCy negation binding and tokenization
        tokens = awe.preprocess_text(text, config)
        # 2. Filter out stopwords and negative stopwords (e.g., "not_and")
        return [
            w
            for w in tokens
            if w not in stopwords_set
            and not awe.is_negative_stopword(w, stopwords_set, config)
        ]

    df["tokens"] = df["all_text"].apply(process_and_filter)

    print("Grouping word variations...")
    # Flatten all tokens to calculate global frequencies for the grouping algorithm
    all_tokens_flat = [word for tokens in df["tokens"] for word in tokens]
    global_counts = Counter(all_tokens_flat)

    # Use awe to group typos and morphological variations
    groups = awe.group_word_variations(global_counts, config)

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

    # Separate into target video responses and other video responses
    target_responses = df[df["Touch No."] == target_video]["unique_main_words"].tolist()
    other_responses = df[df["Touch No."] != target_video]["unique_main_words"].tolist()

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
        # A = Participants who watched Video 12 and used the word
        A = target_doc_counts.get(word, 0)
        # B = Participants who watched Other Videos and used the word
        B = other_doc_counts.get(word, 0)

        # C = Participants who watched Video 12 and DID NOT use the word
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
                    A, B, C, D
                ),
                "Betabinom_Surprise": calc_betabinom_surprise(A, B, C, D),
                "Betabinom_Hellinger": calc_betabinom_hellinger(A, B, C, D),
            }
        )

    return pd.DataFrame(results)


def plot_volcano(plot_df, score_column, title, target_video):
    plt.figure(figsize=(10, 6))
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
    top_pos = plot_df.sort_values(score_column, ascending=False).head(SHOW_N_WORDS)
    for _, row in top_pos.iterrows():
        texts.append(
            plt.text(
                row[score_column],
                row["Total_Freq"],
                row["Word"],
                fontsize=10,
                weight="bold",
                color="darkgreen",
            )
        )

    top_neg = plot_df.sort_values(score_column, ascending=True).head(SHOW_N_WORDS)
    for _, row in top_neg.iterrows():
        texts.append(
            plt.text(
                row[score_column],
                row["Total_Freq"],
                row["Word"],
                fontsize=10,
                color="darkred",
            )
        )

    plt.axvline(0, color="grey", linestyle="--")
    plt.title(title, fontsize=14)
    plt.xlabel(
        f"← OTHER videos \\hfill VIDEO {target_video} →\n{score_column}", fontsize=12
    )
    plt.ylabel("Number of Participants", fontsize=12)
    plt.yscale("log")

    # Use adjust_text to repel labels
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    plt.tight_layout()
    plt.show()


# --- EXECUTION ---
df = pd.read_csv("Processed Data/touch_data_fixed.csv.txt", sep="|")
stats_df = calculate_all_metrics(df, TARGET_VIDEO)

# Optional: keep threshold low to see how the metrics handle the bottom tail natively
plot_df = stats_df[stats_df["Total_Freq"] >= MIN_FREQUENCY_THRESHOLD].copy()

# Generate the 3 plots sequentially
plot_volcano(
    plot_df,
    "Hellinger",
    f"Hellinger Distance (Info Gain) - Video {TARGET_VIDEO}",
    TARGET_VIDEO,
)
plot_volcano(
    plot_df,
    "WMW",
    f"WMW Log Odds (Effect Size) - Video {TARGET_VIDEO}",
    TARGET_VIDEO,
)
# plot_volcano(
#     plot_df,
#     "Hybrid_WMW_Hellinger",
#     f"WMW Log Odds weighted by Hellinger Distance (Confidence Weighting) - Video {TARGET_VIDEO}",
#     TARGET_VIDEO,
# )
# plot_volcano(
#     plot_df,
#     "Theta_Harmonic_F1",
#     f"Harmonic Mean of Hellinger Confidence and Absolute Effect Magnitude (from WMW Odds) - Video {TARGET_VIDEO}",
#     TARGET_VIDEO,
# )
plot_volcano(
    plot_df,
    "Temperature_Softmax_Effect",
    f"Temperature adjusted Softmax - Video {TARGET_VIDEO}",
    TARGET_VIDEO,
)
plot_volcano(
    plot_df,
    "Betabinom_Surprise",
    f"Standard Deviations of Surprise - Video {TARGET_VIDEO}",
    TARGET_VIDEO,
)
plot_volcano(
    plot_df,
    "Betabinom_Hellinger",
    f"Hellinger Distance between BetaBinomial Distributions - Video {TARGET_VIDEO}",
    TARGET_VIDEO,
)
