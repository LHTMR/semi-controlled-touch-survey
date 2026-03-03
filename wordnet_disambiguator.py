#!/usr/bin/env python3
"""
WordNet Disambiguator - Standalone Script

This script takes a list of words and asks the user to disambiguate each word
by selecting the corresponding definition from WordNet synsets.

Optionally extends synonyms with the closest words in WordNet using semantic
similarity measures (Wu-Palmer, path similarity, or an average of both).

Written by: Yohann OPOLKA, University of Borås, Sweden.
Last updated: 2026.2.12

IMPROVEMENTS:
- Robust interruption handling with automatic saving of partial results
- Enhanced resume functionality with validation
- Checkpointing for time-consuming closest words calculation
- Safe mode for maximum data protection

Usage:
    python wordnet_disambiguator.py --words "word1,word2,word3"
    python wordnet_disambiguator.py --input-file words.txt
    python wordnet_disambiguator.py (interactive mode)

    # With closest words extension (time-consuming):
    python wordnet_disambiguator.py --words "sensation" --extend-synonyms --num-closest 20
    python wordnet_disambiguator.py --words "apple" --extend-synonyms --similarity-method path

    # Safe mode with interruption protection:
    python wordnet_disambiguator.py --words "word1,word2" --safe-mode
    python wordnet_disambiguator.py --resume wordnet_selections_partial.json --force-resume

Output:
    Saves results to:
    - wordnet_selections.json (simple mapping)
    - wordnet_selections_synonyms.json (with synonyms and closest words)
    - wordnet_selections.csv (with definitions and synonyms)
    - wordnet_selections_extended.csv (with closest words, if --extend-synonyms used)
    - wordnet_selections.txt (synsets only, one per line)

    Partial/interrupted results saved to:
    - wordnet_selections_partial.json (during interruption)
    - closest_words_checkpoint.json (during closest words calculation)
"""

import nltk
from nltk.corpus import wordnet as wn
import argparse
import json
import csv
import sys
import os
import numpy as np
import signal
import atexit
import tempfile
import shutil
from datetime import datetime
from termcolor import colored
from tqdm import tqdm

# ============================================================================
# Global variables for interruption handling
# ============================================================================
INTERRUPTED = False
CURRENT_SELECTIONS = {}
CURRENT_OUTPUT_DIR = "."
SAFE_MODE = False
CHECKPOINT_INTERVAL = 1000  # For closest words calculation

# ============================================================================
# Signal and interruption handling
# ============================================================================


def signal_handler(signum, frame):
    """Handle interruption signals (Ctrl+C)."""
    global INTERRUPTED
    INTERRUPTED = True
    print(
        "\n\n"
        + colored("⚠️  Interruption requested. Saving current progress...", "yellow")
    )

    # Try to save current progress
    if CURRENT_SELECTIONS:
        try:
            save_partial_results(CURRENT_SELECTIONS, CURRENT_OUTPUT_DIR)
            print(
                colored(
                    "✓ Partial results saved to wordnet_selections_partial.json",
                    "green",
                )
            )
        except Exception as e:
            print(colored(f"✗ Could not save partial results: {e}", "red"))

    # Don't exit immediately - let the main loop handle the interruption
    # This allows for cleanup in disambiguate_words()


def save_partial_results(selections, output_dir):
    """Save partial results to a temporary file, then rename atomically."""
    partial_path = os.path.join(output_dir, "wordnet_selections_partial.json")

    # Create a temporary file first
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=output_dir)
    try:
        with os.fdopen(temp_fd, "w") as f:
            json.dump(selections, f, indent=2)

        # Atomic rename
        shutil.move(temp_path, partial_path)

        # Also create a timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            output_dir, f"wordnet_selections_backup_{timestamp}.json"
        )
        shutil.copy2(partial_path, backup_path)

    except Exception:
        # Clean up temp file if something went wrong
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# WordNet data download
# ============================================================================

# Download WordNet data if not already available
print("Checking WordNet availability...")
try:
    nltk.data.find("corpora/wordnet")
    print("✓ WordNet is available")
except LookupError:
    print("Downloading WordNet data...")
    try:
        nltk.download("wordnet")
        print("✓ WordNet downloaded successfully")
    except Exception as e:
        print(colored(f"✗ Error downloading WordNet: {e}", "red"))
        print("Please check your internet connection and try again.")
        sys.exit(1)

# Also download Open Multilingual WordNet for English if needed
# Currently not used? But exists if needed.
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    print("Downloading Open Multilingual WordNet for English...")
    try:
        nltk.download("omw-1.4")
        print("✓ Open Multilingual WordNet downloaded")
    except Exception as e:
        print(colored(f"Note: Could not download omw-1.4: {e}", "yellow"))
        print(
            "This may affect some WordNet operations but basic functionality should work."
        )

# Also download SentiWordNet if needed
# Currently not used? But exists if needed.
try:
    nltk.data.find("corpora/sentiwordnet")
except LookupError:
    print("Downloading SentiWordNet...")
    try:
        nltk.download("sentiwordnet")
        print("✓ SentiWordNet downloaded")
    except Exception as e:
        print(colored(f"Note: Could not download sentiwordnet: {e}", "yellow"))
        print(
            "This may affect some WordNet operations but basic functionality should work."
        )

# ============================================================================
# WordNet similarity and distance functions (adapted from wqs_word_test.py)
# ============================================================================


def synset_similarity(s1, s2, method="wup"):
    """
    Calculate similarity between two WordNet synsets.

    Args:
        s1: First synset
        s2: Second synset
        method: Similarity method ('wup', 'path', or 'both_average')

    Returns:
        Similarity score (float) or None if no path exists
    """
    if method == "wup":
        result = s1.wup_similarity(s2)
    elif method == "path":
        result = s1.path_similarity(s2)
    elif method == "both_average":
        path_sim = s1.path_similarity(s2)
        wup_sim = s1.wup_similarity(s2)
        if path_sim is not None and wup_sim is not None:
            result = np.mean([path_sim, wup_sim])
        elif path_sim is not None:
            result = path_sim
        elif wup_sim is not None:
            result = wup_sim
        else:
            result = None
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    return result


def synset_distance(s1, s2, method="wup"):
    """
    Calculate distance between two WordNet synsets (1 - similarity).

    Args:
        s1: First synset
        s2: Second synset
        method: Similarity method ('wup', 'path', or 'both_average')

    Returns:
        Distance score (float) or 1.0 if no similarity exists
    """
    similarity = synset_similarity(s1, s2, method=method)
    if similarity is None:
        return 1.0  # Maximum distance
    return 1.0 - similarity


def get_all_synsets(pos_filter=None):
    """
    Get all WordNet synsets, optionally filtered by part of speech.

    Args:
        pos_filter: List of POS tags to include (e.g., ['n', 'v', 'a', 'r'])
                   or None for all synsets

    Returns:
        List of synsets
    """
    if pos_filter is None:
        pos_filter = ["n", "v", "a", "r", "s"]  # 's' is satellite adjective

    all_synsets = []
    for pos in pos_filter:
        all_synsets.extend(list(wn.all_synsets(pos=pos)))

    return all_synsets


def find_closest_words(
    reference_synset,
    num_closest=10,
    similarity_method="wup",
    pos_filter=None,
    include_reference=True,
    show_progress=False,
    checkpoint_interval=1000,
    output_dir=".",
    word="",
):
    """
    Find the words closest to a reference synset in WordNet with checkpointing.

    Args:
        reference_synset: The reference synset (string name or synset object)
        num_closest: Number of closest words to return
        similarity_method: Similarity method ('wup', 'path', or 'both_average')
        pos_filter: List of POS tags to consider (None for all)
        include_reference: Whether to include the reference synset itself
        show_progress: Whether to show progress messages
        checkpoint_interval: Save checkpoint every N synsets
        output_dir: Directory to save checkpoint files
        word: The original word (for checkpoint naming)

    Returns:
        List of tuples (word, distance, synset_name) sorted by distance
    """
    global INTERRUPTED

    # Convert string to synset if needed
    if isinstance(reference_synset, str):
        reference_synset = wn.synset(reference_synset)

    ref_name = reference_synset.name()

    # Try to load from checkpoint
    checkpoint_path = os.path.join(output_dir, "closest_words_checkpoint.json")
    checkpoint_data = None

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            if (
                checkpoint_data.get("reference_synset") == ref_name
                and checkpoint_data.get("similarity_method") == similarity_method
                and checkpoint_data.get("pos_filter") == pos_filter
            ):

                distances = [
                    (item["word"], item["distance"], item["synset"])
                    for item in checkpoint_data.get("distances", [])
                ]
                processed_count = checkpoint_data.get("processed_count", 0)
                all_synsets = get_all_synsets(pos_filter)
                total_synsets = len(all_synsets)

                if show_progress:
                    print(
                        f"Loaded checkpoint: {processed_count}/{total_synsets} synsets processed"
                    )
                    print(f"Resuming from checkpoint...")

                # Start from where we left off
                start_index = processed_count

                # Filter out already processed synsets
                processed_synset_names = {item[2] for item in distances}
                all_synsets = [
                    s
                    for s in all_synsets[start_index:]
                    if s.name() not in processed_synset_names
                ]

            else:
                # Checkpoint doesn't match current parameters
                if show_progress:
                    print("Checkpoint parameters don't match. Starting fresh.")
                distances = []
                all_synsets = get_all_synsets(pos_filter)
                processed_count = 0
                total_synsets = len(all_synsets)
        except Exception as e:
            if show_progress:
                print(f"Could not load checkpoint: {e}. Starting fresh.")
            distances = []
            all_synsets = get_all_synsets(pos_filter)
            processed_count = 0
            total_synsets = len(all_synsets)
    else:
        distances = []
        all_synsets = get_all_synsets(pos_filter)
        processed_count = 0
        total_synsets = len(all_synsets)

    # Create progress bar if show_progress is True
    if show_progress:
        # Initialize tqdm progress bar
        pbar = tqdm(
            total=total_synsets,
            initial=processed_count,
            desc=f"Comparing to {ref_name[:20]:20s}",
            unit="synset",
            leave=False,  # Keep progress bar after completion
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    else:
        pbar = None

    # Calculate distances
    for i, synset in enumerate(all_synsets):
        # Check for interruption
        if INTERRUPTED:
            if show_progress:
                if pbar:
                    pbar.close()
                print("\nInterruption detected during closest words calculation.")
            # Save checkpoint before returning
            save_closest_words_checkpoint(
                ref_name,
                distances,
                processed_count + i,
                total_synsets,
                similarity_method,
                pos_filter,
                output_dir,
            )
            if show_progress:
                print(
                    f"Checkpoint saved. Processed {processed_count + i}/{total_synsets} synsets."
                )
            # Return partial results
            distances.sort(key=lambda x: x[1])
            return distances[:num_closest]

        # Skip the reference synset itself if not including it
        if not include_reference and synset == reference_synset:
            continue

        distance = synset_distance(reference_synset, synset, method=similarity_method)

        # Get the first lemma name as the representative word
        if synset.lemmas():
            word_name = synset.lemmas()[0].name().replace("_", " ")
            distances.append((word_name, distance, synset.name()))

        # Update progress bar
        if pbar:
            pbar.update(1)
            # Update description with current synset info
            if synset.lemmas():
                current_word = synset.lemmas()[0].name()
                pbar.set_description(f"Comparing: {current_word[:20]:20s}")

        # Save checkpoint periodically
        if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
            current_processed = processed_count + i + 1
            save_closest_words_checkpoint(
                ref_name,
                distances,
                current_processed,
                total_synsets,
                similarity_method,
                pos_filter,
                output_dir,
            )

    # Close progress bar if it exists
    if pbar:
        pbar.close()

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])

    # Clean up checkpoint file if we completed successfully
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            if show_progress:
                print("Checkpoint file cleaned up.")
        except Exception:
            pass

    # Return the closest N
    return distances[:num_closest]


def save_closest_words_checkpoint(
    reference_synset,
    distances,
    processed_count,
    total_synsets,
    similarity_method,
    pos_filter,
    output_dir,
):
    """Save checkpoint for closest words calculation."""
    checkpoint_path = os.path.join(output_dir, "closest_words_checkpoint.json")

    checkpoint_data = {
        "reference_synset": reference_synset,
        "similarity_method": similarity_method,
        "pos_filter": pos_filter,
        "processed_count": processed_count,
        "total_synsets": total_synsets,
        "distances": [{"word": w, "distance": d, "synset": s} for w, d, s in distances],
        "timestamp": datetime.now().isoformat(),
    }

    # Save to temporary file first, then rename atomically
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=output_dir)
    try:
        with os.fdopen(temp_fd, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        shutil.move(temp_path, checkpoint_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


# ============================================================================
# Main disambiguation functions
# ============================================================================


def parse_multiple_selections(selection_input, max_selection):
    """
    Parse multiple selection input (e.g., "1,3,5" or "1-3,5").

    Args:
        selection_input: User input string
        max_selection: Maximum valid selection number

    Returns:
        List of selected indices (1-based) or None if invalid
    """
    if not selection_input.strip():
        return []

    selection_input = selection_input.strip()
    selected_indices = []

    # Split by commas
    parts = [part.strip() for part in selection_input.split(",") if part.strip()]

    for part in parts:
        # Check for range (e.g., "1-3")
        if "-" in part:
            range_parts = part.split("-")
            if len(range_parts) != 2:
                return None  # Invalid range format

            try:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())

                # Ensure valid range
                if start < 1 or end > max_selection or start > end:
                    return None

                # Add all indices in range
                selected_indices.extend(range(start, end + 1))
            except ValueError:
                return None  # Not integers
        else:
            # Single number
            try:
                num = int(part)
                if num < 1 or num > max_selection:
                    return None
                selected_indices.append(num)
            except ValueError:
                return None  # Not an integer

    # Remove duplicates and sort
    selected_indices = sorted(set(selected_indices))
    return selected_indices


def disambiguate_words(
    words, previous_selections=None, output_dir=".", safe_mode=False
):
    """
    Disambiguate a list of words using WordNet synsets.

    Args:
        words: List of words to disambiguate
        previous_selections: Dictionary of previous selections {word: [synset_name1, synset_name2, ...]}
        output_dir: Directory to save partial results
        safe_mode: Whether to save after each word

    Returns:
        Dictionary of selections {word: [synset_name1, synset_name2, ...]}
    """
    global INTERRUPTED, CURRENT_SELECTIONS, CURRENT_OUTPUT_DIR

    if previous_selections is None:
        previous_selections = {}

    selections = {}
    CURRENT_SELECTIONS = selections
    CURRENT_OUTPUT_DIR = output_dir

    print(colored(f"\nStarting disambiguation of {len(words)} word(s)", "cyan"))
    if safe_mode:
        print(colored("Safe mode enabled: saving after each word", "green"))

    for i, word in enumerate(words):
        # Check for interruption
        if INTERRUPTED:
            print(
                colored(
                    "\n\n⚠️  Interruption detected. Saving partial results...", "yellow"
                )
            )
            if selections:
                try:
                    save_partial_results(selections, output_dir)
                    print(
                        colored(
                            f"✓ Saved {len(selections)} partial selections", "green"
                        )
                    )
                except Exception as e:
                    print(colored(f"✗ Could not save partial results: {e}", "red"))
            return selections

        print(colored(f"\n\n# ({i+1}/{len(words)})", "yellow"))
        print("Please select the most appropriate definition for the following word...")
        print(f"Word: {colored(word, 'cyan')}")
        print("(Press Ctrl+C to interrupt and save progress)")

        # Get all synsets for the word
        synsets = wn.synsets(word)

        if not synsets:
            print(colored(f"No synsets found for '{word}'. Skipping.", "red"))
            selections[word] = None

            # Save progress in safe mode
            if safe_mode and selections:
                try:
                    save_partial_results(selections, output_dir)
                except Exception:
                    pass
            continue

        print(f"\nSynsets definitions related to '{word}':")

        # Display all synsets with numbers
        ss_j = 1
        past_selections_indices = []

        for ss in synsets:
            print(f"\t{colored(ss_j, 'yellow')}. {ss.definition()} ({ss.name()})")

            # Check if this was previously selected (handle both old and new formats)
            prev_selections = previous_selections.get(word)
            if prev_selections:
                if isinstance(prev_selections, list):
                    # New format: list of synsets
                    if ss.name() in prev_selections:
                        past_selections_indices.append(ss_j)
                else:
                    # Old format: single synset
                    if ss.name() == prev_selections:
                        past_selections_indices.append(ss_j)
            ss_j += 1

        print("")

        # Get user input
        prompt = "Selection"
        if past_selections_indices:
            past_str = ",".join(str(idx) for idx in past_selections_indices)
            prompt += f" ({colored(past_str, 'green')})"
        prompt += " [comma-separated numbers, e.g., '1,3,5' or '1-3,5', or 'skip' to skip, 'stop' to save and exit]: "

        try:
            selection_input = input(prompt).strip()
        except KeyboardInterrupt:
            # Handle Ctrl+C during input
            INTERRUPTED = True
            continue  # Will be caught at top of loop

        if selection_input.upper() == "STOP":
            print(colored("Stopping early and saving progress...", "yellow"))
            if selections:
                try:
                    save_partial_results(selections, output_dir)
                    print(colored(f"✓ Saved {len(selections)} selections", "green"))
                except Exception as e:
                    print(colored(f"✗ Could not save results: {e}", "red"))
            return selections

        if selection_input.upper() == "SKIP":
            print(colored("Skipping this word.", "yellow"))
            selections[word] = None

            # Save progress in safe mode
            if safe_mode and selections:
                try:
                    save_partial_results(selections, output_dir)
                except Exception:
                    pass
            continue

        # Handle empty input (use previous selections if available)
        if not selection_input and past_selections_indices:
            selected_indices = past_selections_indices
        elif selection_input:
            # Parse multiple selections
            selected_indices = parse_multiple_selections(selection_input, len(synsets))
            if selected_indices is None:
                print(
                    colored(
                        f"Invalid input. Please enter comma-separated numbers between 1 and {len(synsets)}, or ranges like '1-3'.",
                        "red",
                    )
                )
                selected_indices = []
        else:
            selected_indices = []

        # Process selection
        if selected_indices:
            selected_synsets = []
            for idx in selected_indices:
                selected_synset = synsets[idx - 1]
                selected_synsets.append(selected_synset.name())

            selections[word] = selected_synsets
            synset_names = ", ".join(selected_synsets)
            print(f"--> {colored(synset_names, 'green')}")
        else:
            print(colored("No selection made. Skipping.", "yellow"))
            selections[word] = None

        # Save progress after each word in safe mode
        if safe_mode and selections:
            try:
                save_partial_results(selections, output_dir)
            except Exception as e:
                print(colored(f"Note: Could not save checkpoint: {e}", "yellow"))

    print(colored(f"\n✓ Completed disambiguation of {len(words)} word(s)", "green"))
    return selections


def save_results(
    selections,
    output_dir=".",
    extend_synonyms=False,
    num_closest=10,
    similarity_method="wup",
    pos_filter=None,
    show_progress=True,
    checkpoint_interval=1000,
    safe_mode=False,
):
    """
    Save disambiguation results to multiple file formats.

    Args:
        selections: Dictionary of {word: [synset_name1, synset_name2, ...]} or {word: synset_name}
        output_dir: Directory to save files
        extend_synonyms: Whether to find closest words in WordNet
        num_closest: Number of closest words to find (if extend_synonyms=True)
        similarity_method: Similarity method ('wup', 'path', or 'both_average')
        pos_filter: List of POS tags to consider for closest words
        show_progress: Whether to show progress messages
        checkpoint_interval: Checkpoint interval for closest words calculation
        safe_mode: Whether to use safe mode features
    """
    global INTERRUPTED

    # Save to JSON (simple mapping - backward compatible)
    json_path = os.path.join(output_dir, "wordnet_selections.json")

    # Use atomic write for safe mode
    if safe_mode:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=output_dir)
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(selections, f, indent=2)
            shutil.move(temp_path, json_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
    else:
        with open(json_path, "w") as f:
            json.dump(selections, f, indent=2)

    print(f"\nSimple mapping saved to {json_path}")

    # Prepare enhanced data with synonyms and optionally closest words
    enhanced_data = {}
    csv_rows = []
    csv_extended_rows = []  # For extended CSV with closest words

    # Create outer progress bar for processing words/synsets
    if show_progress and extend_synonyms:
        # Count total synsets to process
        total_synsets_to_process = 0
        for synset_value in selections.values():
            if synset_value:
                if isinstance(synset_value, list):
                    total_synsets_to_process += len(synset_value)
                else:
                    total_synsets_to_process += 1

        if total_synsets_to_process > 0:
            outer_pbar = tqdm(
                total=total_synsets_to_process,
                desc="Processing words",
                unit="synset",
                leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        else:
            outer_pbar = None
    else:
        outer_pbar = None

    synset_counter = 0
    for i, (word, synset_value) in enumerate(selections.items()):
        # Check for interruption
        if INTERRUPTED:
            # Close outer progress bar if it exists
            if outer_pbar:
                outer_pbar.close()
            print(
                colored(
                    "\nInterruption detected during save. Saving partial results...",
                    "yellow",
                )
            )
            # Save what we have so far
            if enhanced_data:
                enhanced_json_path = os.path.join(
                    output_dir, "wordnet_selections_synonyms_partial.json"
                )
                with open(enhanced_json_path, "w") as f:
                    json.dump(enhanced_data, f, indent=2)
                print(f"Partial enhanced data saved to {enhanced_json_path}")
            break

        # Handle both old (single synset) and new (list of synsets) formats
        if synset_value:
            if isinstance(synset_value, list):
                synset_names = synset_value
            else:
                # Old format: single synset
                synset_names = [synset_value]
        else:
            synset_names = []

        # Process each synset
        word_enhanced_data = []
        for synset_name in synset_names:
            try:
                synset = wn.synset(synset_name)
                definition = synset.definition()
                # Get synonyms from lemmas
                synonyms = [lemma.name() for lemma in synset.lemmas()]
                # Remove duplicates and the original word if present
                synonyms = list(dict.fromkeys(synonyms))  # Preserve order
                if word.lower() in [s.lower() for s in synonyms]:
                    synonyms = [s for s in synonyms if s.lower() != word.lower()]

                # Find closest words if requested
                closest_words_info = []
                closest_words = []
                if extend_synonyms:
                    # Update outer progress bar description
                    if outer_pbar:
                        outer_pbar.set_description(f"Processing: {word[:15]:15s}")

                    closest_words_info = find_closest_words(
                        synset_name,
                        num_closest=num_closest,
                        similarity_method=similarity_method,
                        pos_filter=pos_filter,
                        include_reference=False,
                        show_progress=show_progress,
                        checkpoint_interval=checkpoint_interval,
                        output_dir=output_dir,
                        word=word,
                    )

                    # Extract just the word names from the tuples
                    closest_words = [word_info[0] for word_info in closest_words_info]

                    # Update outer progress bar
                    if outer_pbar:
                        outer_pbar.update(1)
                        if closest_words_info:
                            closest_dist = closest_words_info[0][1]
                            outer_pbar.set_postfix_str(f"closest: {closest_dist:.3f}")

                # Enhanced JSON data for this synset
                synset_data = {
                    "synset": synset_name,
                    "definition": definition,
                    "synonyms": synonyms,
                    "closest_words": closest_words,
                    "closest_words_info": (
                        [
                            {"word": info[0], "distance": info[1], "synset": info[2]}
                            for info in closest_words_info
                        ]
                        if extend_synonyms
                        else []
                    ),
                }
                word_enhanced_data.append(synset_data)

                # CSV row (original format) - one row per synset
                csv_rows.append(
                    [
                        word,
                        synset_name,
                        definition,
                        ", ".join(synonyms) if synonyms else "N/A",
                    ]
                )

                # Extended CSV row with closest words - one row per synset
                if extend_synonyms:
                    csv_extended_rows.append(
                        [
                            word,
                            synset_name,
                            definition,
                            ", ".join(synonyms) if synonyms else "N/A",
                            ", ".join(closest_words) if closest_words else "N/A",
                            (
                                ", ".join(
                                    [
                                        f"{info[0]} ({info[1]:.3f})"
                                        for info in closest_words_info
                                    ]
                                )
                                if closest_words_info
                                else "N/A"
                            ),
                        ]
                    )

            except Exception as e:
                print(
                    colored(
                        f"Warning: Could not process synset {synset_name} for word '{word}': {e}",
                        "yellow",
                    )
                )
                definition = "N/A"
                synonyms = []
                closest_words = []
                closest_words_info = []

                # Add error synset data
                synset_data = {
                    "synset": synset_name,
                    "definition": definition,
                    "synonyms": synonyms,
                    "closest_words": closest_words,
                    "closest_words_info": [],
                }
                word_enhanced_data.append(synset_data)

                # CSV row for error case
                csv_rows.append(
                    [
                        word,
                        synset_name,
                        definition,
                        ", ".join(synonyms) if synonyms else "N/A",
                    ]
                )
                if extend_synonyms:
                    csv_extended_rows.append(
                        [word, synset_name, definition, "N/A", "N/A", "N/A"]
                    )

        # Store all synset data for this word
        if word_enhanced_data:
            enhanced_data[word] = word_enhanced_data
        else:
            # No synsets selected
            enhanced_data[word] = []
            csv_rows.append([word, "N/A", "N/A", "N/A"])
            if extend_synonyms:
                csv_extended_rows.append([word, "N/A", "N/A", "N/A", "N/A", "N/A"])

    # Close outer progress bar if it exists
    if outer_pbar:
        outer_pbar.close()

    # Save enhanced JSON
    enhanced_json_path = os.path.join(output_dir, "wordnet_selections_synonyms.json")

    if safe_mode and enhanced_data:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=output_dir)
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(enhanced_data, f, indent=2)
            shutil.move(temp_path, enhanced_json_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
    else:
        with open(enhanced_json_path, "w") as f:
            json.dump(enhanced_data, f, indent=2)

    print(f"Enhanced data with synonyms saved to {enhanced_json_path}")

    # Save to CSV (original format)
    csv_path = os.path.join(output_dir, "wordnet_selections.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "synset", "definition", "synonyms"])
        writer.writerows(csv_rows)

    print(f"CSV with synonyms saved to {csv_path}")

    # Save extended CSV if closest words were found
    if extend_synonyms and csv_extended_rows:
        csv_extended_path = os.path.join(output_dir, "wordnet_selections_extended.csv")
        with open(csv_extended_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "word",
                    "synset",
                    "definition",
                    "synonyms",
                    "closest_words",
                    "closest_words_with_distances",
                ]
            )
            writer.writerows(csv_extended_rows)

        print(f"Extended CSV with closest words saved to {csv_extended_path}")

    # Save to TXT (synsets only, one per line)
    txt_path = os.path.join(output_dir, "wordnet_selections.txt")
    with open(txt_path, "w") as f:
        for word, synset_value in selections.items():
            if synset_value:
                if isinstance(synset_value, list):
                    for synset_name in synset_value:
                        f.write(f"{synset_name}\n")
                else:
                    f.write(f"{synset_value}\n")

    print(f"Synsets only (TXT) saved to {txt_path}")


def load_previous_results(filepath, validate=True, force_resume=False):
    """
    Load previous disambiguation results from a file with validation.

    Args:
        filepath: Path to JSON or CSV file
        validate: Whether to validate that loaded synsets exist in WordNet
        force_resume: Continue even if validation fails

    Returns:
        Dictionary of previous selections {word: [synset_name1, synset_name2, ...]} or {word: synset_name}
    """
    if not os.path.exists(filepath):
        print(colored(f"File not found: {filepath}", "red"))
        return {}

    selections = {}
    invalid_entries = []

    try:
        if filepath.endswith(".json"):
            with open(filepath, "r") as f:
                data = json.load(f)

            # Handle multiple JSON formats
            for word, value in data.items():
                if isinstance(value, list):
                    # New format: list of synset data objects or synset names
                    synset_names = []
                    for item in value:
                        if isinstance(item, dict) and "synset" in item:
                            # List of enhanced data objects
                            synset_names.append(item["synset"])
                        elif isinstance(item, str):
                            # List of synset names
                            synset_names.append(item)
                    selections[word] = synset_names if synset_names else None
                elif isinstance(value, dict):
                    # Old enhanced format: {"synset": "...", "definition": "...", "synonyms": [...]}
                    if "synset" in value and value["synset"]:
                        selections[word] = value["synset"]
                else:
                    # Old simple format: just the synset name (or None)
                    selections[word] = value

        elif filepath.endswith(".csv"):
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row["word"]
                    synset_name = row.get("synset")
                    if synset_name and synset_name != "N/A":
                        # Handle multiple rows per word
                        if word in selections:
                            # Word already exists, add to list
                            if isinstance(selections[word], list):
                                selections[word].append(synset_name)
                            else:
                                # Convert from single to list
                                selections[word] = [selections[word], synset_name]
                        else:
                            selections[word] = synset_name
        else:
            print(colored(f"Unsupported file format: {filepath}", "red"))
            return {}

        print(f"Loaded {len(selections)} selections from {filepath}")

        # Validate synsets if requested
        if validate and selections:
            print("Validating loaded synsets...")
            valid_selections = {}

            for word, synset_value in list(selections.items()):
                if synset_value is None:
                    valid_selections[word] = None
                    continue

                # Handle both single synset and list of synsets
                if isinstance(synset_value, list):
                    valid_synsets = []
                    invalid_synsets = []

                    for synset_name in synset_value:
                        try:
                            # Try to load the synset to verify it exists
                            synset = wn.synset(synset_name)
                            valid_synsets.append(synset_name)
                        except Exception as e:
                            invalid_synsets.append((synset_name, str(e)))
                            print(
                                colored(
                                    f"  ✗ Invalid synset for '{word}': {synset_name}",
                                    "red",
                                )
                            )
                            print(f"     Error: {e}")

                    if valid_synsets:
                        valid_selections[word] = valid_synsets
                    else:
                        valid_selections[word] = None

                    if invalid_synsets and not force_resume:
                        # Handle invalid synsets in list
                        for synset_name, error in invalid_synsets:
                            print(
                                f"\nWhat should I do with synset '{synset_name}' for word '{word}'?"
                            )
                            print("  1. Skip this synset (remove from list)")
                            print("  2. Keep but mark as invalid")
                            print("  3. Try to find a similar valid synset")

                            choice = input("Choice [1/2/3, default=1]: ").strip()
                            if choice == "2":
                                # Keep as None (will be re-disambiguated)
                                pass
                            elif choice == "3":
                                # Try to find similar synsets
                                synsets = wn.synsets(word)
                                if synsets:
                                    print(
                                        f"Found {len(synsets)} possible synsets for '{word}':"
                                    )
                                    for j, ss in enumerate(
                                        synsets[:5], 1
                                    ):  # Show first 5
                                        print(f"  {j}. {ss.definition()} ({ss.name()})")

                                    sel = input(
                                        f"Select replacement [1-{len(synsets)}] or press Enter to skip: "
                                    ).strip()
                                    if (
                                        sel
                                        and sel.isdigit()
                                        and 1 <= int(sel) <= len(synsets)
                                    ):
                                        replacement = synsets[int(sel) - 1].name()
                                        if word in valid_selections and isinstance(
                                            valid_selections[word], list
                                        ):
                                            valid_selections[word].append(replacement)
                                        else:
                                            valid_selections[word] = [replacement]
                else:
                    # Single synset
                    try:
                        # Try to load the synset to verify it exists
                        synset = wn.synset(synset_value)
                        valid_selections[word] = synset_value
                    except Exception as e:
                        invalid_entries.append((word, synset_value, str(e)))
                        print(
                            colored(
                                f"  ✗ Invalid synset for '{word}': {synset_value}",
                                "red",
                            )
                        )
                        print(f"     Error: {e}")

                    if not force_resume:
                        # Ask user what to do
                        print(f"\nWhat should I do with '{word}'?")
                        print("  1. Skip this word (remove from resume list)")
                        print(
                            "  2. Keep but mark as invalid (will be re-disambiguated)"
                        )
                        print("  3. Try to find a similar valid synset")

                        choice = input("Choice [1/2/3, default=1]: ").strip()
                        if choice == "2":
                            valid_selections[word] = None  # Mark for re-disambiguation
                        elif choice == "3":
                            # Try to find similar synsets
                            synsets = wn.synsets(word)
                            if synsets:
                                print(
                                    f"Found {len(synsets)} possible synsets for '{word}':"
                                )
                                for j, ss in enumerate(synsets[:5], 1):  # Show first 5
                                    print(f"  {j}. {ss.definition()} ({ss.name()})")

                                sel = input(
                                    f"Select replacement [1-{len(synsets)}] or press Enter to skip: "
                                ).strip()
                                if (
                                    sel
                                    and sel.isdigit()
                                    and 1 <= int(sel) <= len(synsets)
                                ):
                                    valid_selections[word] = synsets[
                                        int(sel) - 1
                                    ].name()
                                    print(
                                        colored(
                                            f"  ✓ Updated to: {valid_selections[word]}",
                                            "green",
                                        )
                                    )
                                else:
                                    valid_selections[word] = None
                            else:
                                print(
                                    colored(
                                        f"  No synsets found for '{word}'. Skipping.",
                                        "yellow",
                                    )
                                )
                                valid_selections[word] = None
                        else:
                            # Skip (remove from list)
                            pass
                    else:
                        # Force resume: keep invalid entries as None
                        valid_selections[word] = None

            selections = valid_selections

            if invalid_entries:
                print(
                    colored(
                        f"\nFound {len(invalid_entries)} invalid synset(s)", "yellow"
                    )
                )
                if force_resume:
                    print(
                        "Force resume enabled - invalid entries marked for re-disambiguation"
                    )

                # Save invalid entries to a file for reference
                invalid_path = filepath.replace(".json", "_invalid.json").replace(
                    ".csv", "_invalid.json"
                )
                with open(invalid_path, "w") as f:
                    json.dump(
                        [
                            {"word": w, "synset": s, "error": e}
                            for w, s, e in invalid_entries
                        ],
                        f,
                        indent=2,
                    )
                print(f"Invalid entries saved to {invalid_path}")

        return selections

    except json.JSONDecodeError as e:
        print(colored(f"Error parsing JSON file {filepath}: {e}", "red"))

        # Try to recover partial data
        if force_resume:
            print("Attempting to recover partial data...")
            # Simple line-by-line recovery for JSON
            recovered = {}
            with open(filepath, "r") as f:
                content = f.read()
                # Try to extract word:synset pairs using regex
                import re

                pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'  # Simple key:value pattern
                matches = re.findall(pattern, content)
                for word, synset in matches:
                    if synset and synset != "null" and synset.lower() != "n/a":
                        recovered[word] = synset

            if recovered:
                print(f"Recovered {len(recovered)} entries from corrupted file")
                return recovered

        return {}

    except Exception as e:
        print(colored(f"Error loading file {filepath}: {e}", "red"))
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Disambiguate words using WordNet synsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --words "apple,orange,banana"
  %(prog)s --input-file my_words.txt
  %(prog)s --resume wordnet_selections.json
  %(prog)s (interactive mode)
  
  # With closest words extension (time-consuming):
  %(prog)s --words "sensation,feeling" --extend-synonyms --num-closest 20
  %(prog)s --words "apple" --extend-synonyms --similarity-method path --pos-filter n,v
  
  # Safe mode with interruption protection:
  %(prog)s --words "word1,word2" --safe-mode
  %(prog)s --resume wordnet_selections_partial.json --force-resume
  %(prog)s --words "complex" --extend-synonyms --safe-mode --checkpoint-interval 500
  
  # Resume with validation:
  %(prog)s --resume corrupted_results.json --force-resume --no-validate
        """,
    )

    parser.add_argument(
        "--words", type=str, help="Comma-separated list of words to disambiguate"
    )

    parser.add_argument(
        "--input-file", type=str, help="Text file containing words (one per line)"
    )

    parser.add_argument(
        "--resume", type=str, help="Resume from previous results file (JSON or CSV)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output files (default: current directory)",
    )

    # New arguments for extending synonyms with closest words
    parser.add_argument(
        "--extend-synonyms",
        action="store_true",
        help="Extend synonyms with closest words from WordNet (time-consuming)",
    )

    parser.add_argument(
        "--num-closest",
        type=int,
        default=10,
        help="Number of closest words to find (default: 10)",
    )

    parser.add_argument(
        "--similarity-method",
        type=str,
        default="wup",
        choices=["wup", "path", "both_average"],
        help="Similarity method for finding closest words (default: wup)",
    )

    parser.add_argument(
        "--pos-filter",
        type=str,
        default=None,
        help="Filter closest words by part of speech (comma-separated: n,v,a,r,s or None for all)",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Don't show progress messages when finding closest words",
    )

    # New arguments for interruption handling and robustness
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Enable safe mode: save after each word, atomic writes, validation",
    )

    parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Force resume even if validation fails (marks invalid entries for re-disambiguation)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Checkpoint interval for closest words calculation (default: 1000)",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation when resuming (faster but less safe)",
    )

    args = parser.parse_args()

    # Set global safe mode
    global SAFE_MODE
    SAFE_MODE = args.safe_mode

    if args.safe_mode:
        print(colored("\n🔒 SAFE MODE ENABLED", "green"))
        print("Features enabled:")
        print("  • Saving after each word")
        print("  • Atomic file writes")
        print("  • Automatic backup creation")
        print("  • Checkpointing for closest words calculation")
        print("  • Enhanced validation")
        print("  • Partial result saving on interruption")
        print()

    # Get list of words
    words = []

    if args.words:
        words = [w.strip() for w in args.words.split(",")]
    elif args.input_file:
        try:
            with open(args.input_file, "r") as f:
                words = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(colored(f"Error: Input file not found: {args.input_file}", "red"))
            return 1
    else:
        # Interactive mode
        print("Enter words to disambiguate (one per line, empty line to finish):")
        while True:
            word = input("> ").strip()
            if not word:
                break
            words.append(word)

    if not words:
        print(colored("No words provided. Exiting.", "yellow"))
        return 0

    print(f"\nDisambiguating {len(words)} word(s): {', '.join(words)}")

    # Load previous selections if resuming
    previous_selections = {}
    if args.resume:
        previous_selections = load_previous_results(
            args.resume, validate=not args.no_validate, force_resume=args.force_resume
        )
        if previous_selections:
            print(f"Loaded {len(previous_selections)} previous selections")

            # Check for partial/interrupted files
            partial_path = os.path.join(
                args.output_dir, "wordnet_selections_partial.json"
            )
            if os.path.exists(partial_path):
                print(
                    colored(
                        f"Note: Found partial results file: {partial_path}", "yellow"
                    )
                )
                print(
                    "You can use --resume with this file to continue from interruption."
                )

    # Disambiguate words
    selections = disambiguate_words(
        words, previous_selections, output_dir=args.output_dir, safe_mode=args.safe_mode
    )

    # Parse POS filter if provided
    pos_filter = None
    if args.pos_filter:
        pos_filter = [pos.strip() for pos in args.pos_filter.split(",")]
        print(f"Filtering closest words by POS: {pos_filter}")

    # Save results
    if selections:
        if args.extend_synonyms:
            print(
                colored(
                    "\n⚠️  WARNING: Extending synonyms with closest words is time-consuming!",
                    "yellow",
                )
            )
            print(
                "This may take several minutes depending on the number of words and WordNet size."
            )
            print("You can interrupt with Ctrl+C if it takes too long.\n")
            if args.safe_mode:
                print(
                    colored(
                        "Safe mode enabled: Checkpointing every {} synsets".format(
                            args.checkpoint_interval
                        ),
                        "green",
                    )
                )

        save_results(
            selections,
            args.output_dir,
            extend_synonyms=args.extend_synonyms,
            num_closest=args.num_closest,
            similarity_method=args.similarity_method,
            pos_filter=pos_filter,
            show_progress=not args.no_progress,
            checkpoint_interval=args.checkpoint_interval,
            safe_mode=args.safe_mode,
        )

        # Show summary
        print("\n" + "=" * 50)
        print("DISAMBIGUATION SUMMARY")
        print("=" * 50)
        for word, synset in selections.items():
            status = colored("✓", "green") if synset else colored("✗", "red")
            print(f"{status} {word}: {synset or 'No selection'}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # This might be caught by our signal handler, but just in case
        if not INTERRUPTED:
            print("\n\nInterrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(colored(f"\nUnexpected error: {e}", "red"))
        import traceback

        traceback.print_exc()
        sys.exit(1)
