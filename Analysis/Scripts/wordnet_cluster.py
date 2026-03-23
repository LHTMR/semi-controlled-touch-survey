#!/usr/bin/env python3
"""
WordNet Synset Clustering Script

This script performs Agglomerative Hierarchical Clustering (AHC) with complete linkage
on WordNet synsets based on their semantic dissimilarities.

The script reads disambiguated synsets from the output of wordnet_disambiguator.py
and clusters them based on WordNet similarity measures (Wu-Palmer, path similarity,
or an average of both).

By default, when processing enhanced JSON files with "closest_words_info" fields,
the script includes those additional synsets in the clustering. Use --no-closest-words
to exclude them.

Usage:
    python wordnet_cluster.py --input wordnet_selections.json --clusters 5
    python wordnet_cluster.py --input wordnet_selections_synonyms.json --clusters 3 --method path
    python wordnet_cluster.py --input wordnet_selections.txt --clusters 8 --output-dir ./clusters
    python wordnet_cluster.py --input wordnet_selections_synonyms.json --clusters 5 --no-closest-words

Output:
    - wordnet_clusters_summary.json: Overview of all clusters
    - wordnet_cluster_1.json, wordnet_cluster_1.txt, etc.: Per-cluster files
    - wordnet_clusters_dendrogram.pdf: Dendrogram visualization (if --plot-dendrogram)

Written by: Based on wordnet_disambiguator.py by Yohann OPOLKA
"""

import nltk
from nltk.corpus import wordnet as wn
import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime
from termcolor import colored
from tqdm import tqdm

# Try to import SciPy for hierarchical clustering
try:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# WordNet similarity and distance functions (from wordnet_disambiguator.py)
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


# ============================================================================
# Data loading functions
# ============================================================================


def get_synset_info(synset_name):
    """
    Get enhanced information for a synset from WordNet.

    Args:
        synset_name: WordNet synset name (e.g., "touch.v.01")

    Returns:
        dict: Enhanced synset information with definition and synonyms
    """
    try:
        synset = wn.synset(synset_name)
        definition = synset.definition()
        synonyms = [lemma.name() for lemma in synset.lemmas()]

        return {
            "synset": synset_name,
            "definition": definition,
            "synonyms": synonyms,
            "closest_words": [],  # Empty for derived synsets
            "closest_words_info": [],  # Empty for derived synsets
        }
    except Exception as e:
        print(
            colored(
                f"Warning: Could not get info for synset '{synset_name}': {e}", "yellow"
            )
        )
        return {
            "synset": synset_name,
            "definition": f"[Error loading definition: {e}]",
            "synonyms": [],
            "closest_words": [],
            "closest_words_info": [],
        }


def load_synsets_from_file(filepath, include_closest_words=True):
    """
    Load synsets from various file formats.

    Args:
        filepath: Path to input file (JSON, TXT, or CSV)
        include_closest_words: Whether to include synsets from closest_words_info (default: True)

    Returns:
        tuple: (synset_names, word_mapping, enhanced_data)
        - synset_names: List of synset names
        - word_mapping: Dict mapping synset_name -> original_word
        - enhanced_data: Full enhanced data if available, None otherwise
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    synset_names = []
    word_mapping = {}
    enhanced_data = None

    if filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)

        # Check if this is enhanced format or simple format
        if isinstance(list(data.values())[0], list) and len(list(data.values())[0]) > 0:
            if isinstance(list(data.values())[0][0], dict):
                # Enhanced format: {word: [{synset: "...", definition: "...", ...}, ...]}
                enhanced_data = data
                for word, synset_list in data.items():
                    for synset_data in synset_list:
                        if isinstance(synset_data, dict) and "synset" in synset_data:
                            # Add main synset
                            synset_name = synset_data["synset"]
                            if synset_name not in synset_names:  # Avoid duplicates
                                synset_names.append(synset_name)
                                word_mapping[synset_name] = word

                            # Add synsets from closest_words_info if requested
                            if (
                                include_closest_words
                                and "closest_words_info" in synset_data
                                and isinstance(synset_data["closest_words_info"], list)
                            ):
                                for closest_word_info in synset_data[
                                    "closest_words_info"
                                ]:
                                    if (
                                        isinstance(closest_word_info, dict)
                                        and "synset" in closest_word_info
                                    ):
                                        closest_synset = closest_word_info["synset"]
                                        if (
                                            closest_synset not in synset_names
                                        ):  # Avoid duplicates
                                            synset_names.append(closest_synset)
                                            # Map to original word, not the closest word
                                            word_mapping[closest_synset] = word

                                            # Create enhanced data entry for this closest word synset
                                            # if it doesn't already exist
                                            # Get synset info from WordNet
                                            closest_synset_info = get_synset_info(
                                                closest_synset
                                            )
                                            # Add to enhanced data
                                            if word not in enhanced_data:
                                                enhanced_data[word] = []
                                            # Check if this synset already exists for this word
                                            existing_synsets = [
                                                item.get("synset")
                                                for item in enhanced_data[word]
                                                if isinstance(item, dict)
                                            ]
                                            if closest_synset not in existing_synsets:
                                                enhanced_data[word].append(
                                                    closest_synset_info
                                                )
            else:
                # Simple format: {word: [synset1, synset2, ...]}
                for word, synset_list in data.items():
                    for synset_name in synset_list:
                        if synset_name:  # Skip None values
                            if synset_name not in synset_names:  # Avoid duplicates
                                synset_names.append(synset_name)
                                word_mapping[synset_name] = word
        else:
            # Old format: {word: single_synset}
            for word, synset_name in data.items():
                if synset_name:  # Skip None values
                    if synset_name not in synset_names:  # Avoid duplicates
                        synset_names.append(synset_name)
                        word_mapping[synset_name] = word

    elif filepath.endswith(".txt"):
        # Simple text file: one synset per line
        with open(filepath, "r") as f:
            for line in f:
                synset_name = line.strip()
                if synset_name:
                    if synset_name not in synset_names:  # Avoid duplicates
                        synset_names.append(synset_name)
                        # For text files, we don't have word mapping
                        word_mapping[synset_name] = synset_name.split(".")[0]

    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .json or .txt")

    print(f"Loaded {len(synset_names)} synsets from {filepath}")
    if include_closest_words and enhanced_data:
        print(f"  (including synsets from closest_words_info)")

    return (
        synset_names,
        word_mapping,
        enhanced_data,
    )


# ============================================================================
# Distance matrix calculation
# ============================================================================


def calculate_distance_matrix(synset_names, method="wup", show_progress=True):
    """
    Calculate pairwise distance matrix for all synsets.

    Args:
        synset_names: List of synset names
        method: Similarity method ('wup', 'path', or 'both_average')
        show_progress: Whether to show progress bar

    Returns:
        numpy.ndarray: N×N distance matrix
    """
    n = len(synset_names)
    print(f"Calculating distance matrix for {n} synsets...")

    # Load all synset objects
    synsets = []
    invalid_synsets = []
    for i, name in enumerate(synset_names):
        try:
            synsets.append(wn.synset(name))
        except Exception as e:
            print(colored(f"Warning: Invalid synset '{name}': {e}", "yellow"))
            invalid_synsets.append(i)

    # Remove invalid synsets
    if invalid_synsets:
        print(f"Removing {len(invalid_synsets)} invalid synsets")
        valid_indices = [i for i in range(n) if i not in invalid_synsets]
        synset_names = [synset_names[i] for i in valid_indices]
        synsets = [synsets[i] for i in valid_indices]
        n = len(synsets)

    # Initialize distance matrix
    dist_matrix = np.zeros((n, n))

    # Calculate distances
    if show_progress:
        pbar = tqdm(total=n * (n - 1) // 2, desc="Calculating distances", unit="pair")

    for i in range(n):
        for j in range(i + 1, n):
            distance = synset_distance(synsets[i], synsets[j], method=method)
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance  # Symmetric

            if show_progress:
                pbar.update(1)

    if show_progress:
        pbar.close()

    return dist_matrix, synset_names


# ============================================================================
# Clustering functions
# ============================================================================


def perform_hierarchical_clustering(dist_matrix, n_clusters=5, method="complete"):
    """
    Perform hierarchical clustering using SciPy.

    Args:
        dist_matrix: N×N distance matrix
        n_clusters: Number of clusters to form
        method: Linkage method ('complete', 'average', 'single', etc.)

    Returns:
        numpy.ndarray: Cluster assignments for each synset
        float: Cut height used
        numpy.ndarray: Linkage matrix
        bool: Whether exact number of clusters was achieved
    """

    # Convert to condensed distance matrix (upper triangular)
    condensed_dist = squareform(dist_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=method)

    n_points = dist_matrix.shape[0]

    # First, try to get exactly n_clusters using fcluster with maxclust criterion
    clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    actual_n_clusters = len(np.unique(clusters))
    exact_clusters_achieved = actual_n_clusters == n_clusters

    # Debug output
    if not exact_clusters_achieved:
        print(
            f"Note: fcluster with criterion='maxclust' returned {actual_n_clusters} clusters instead of {n_clusters}"
        )
        print(f"  This can happen when there are tied distances in the linkage matrix.")
        print(
            f"  Will attempt to find the correct cut height for {n_clusters} clusters."
        )

    # Calculate the correct cut height for n_clusters
    # The linkage matrix has n-1 rows where n is the number of data points
    # Each row represents a merge: [cluster_i, cluster_j, distance, cluster_size]
    # To get k clusters, we need to cut after the (n - k)th merge (0-based indexing)
    # The cut height should be just above the height of that merge

    if n_clusters == 1:
        # If we want 1 cluster, cut height is just above the last merge
        cut_height = linkage_matrix[-1, 2] + 0.001
    elif n_clusters == n_points:
        # If we want each point as its own cluster, cut height is just below the first merge
        cut_height = max(0.0, linkage_matrix[0, 2] - 0.001)
    else:
        # For k clusters, we cut after the (n - k)th merge (0-based index: n - k - 1)
        # But we need to be careful: if multiple merges happen at the same height,
        # we might need to adjust the cut height

        # Get the merge index for k clusters (0-based)
        merge_index = n_points - n_clusters - 1

        if merge_index < 0:
            # This happens when n_clusters > n_points
            # Should not happen as we check this earlier, but handle it
            merge_index = 0
            cut_height = max(0.0, linkage_matrix[0, 2] - 0.001)
        elif merge_index >= len(linkage_matrix):
            # This happens when n_clusters < 1
            merge_index = len(linkage_matrix) - 1
            cut_height = linkage_matrix[-1, 2] + 0.001
        else:
            # The height at which we have k clusters is the height of merge (n-k)
            # We want to cut just above this height
            cut_height = linkage_matrix[merge_index, 2] + 1e-10

    # Verify that this cut height gives the correct number of clusters
    verify_clusters = fcluster(linkage_matrix, cut_height, criterion="distance")
    n_verify_clusters = len(np.unique(verify_clusters))

    # If we don't get the right number of clusters, find the exact height
    if n_verify_clusters != n_clusters:
        # Use a more robust method to find the correct cut height
        cut_height, exact_found = _find_exact_cut_height_for_clusters(
            linkage_matrix, n_clusters
        )
        if exact_found:
            # Re-cluster with the correct cut height
            clusters = fcluster(linkage_matrix, cut_height, criterion="distance")
            exact_clusters_achieved = True
        else:
            # Couldn't find exact height, use the best we found
            clusters = fcluster(linkage_matrix, cut_height, criterion="distance")
            exact_clusters_achieved = False
    else:
        # We got the right number with the theoretical cut height
        clusters = verify_clusters
        exact_clusters_achieved = True

    # Double-check that we have the right number of clusters
    final_n_clusters = len(np.unique(clusters))
    if not exact_clusters_achieved:
        print(
            f"Warning: Could not get exactly {n_clusters} clusters due to tied distances."
        )
        print(
            f"  Using {final_n_clusters} clusters instead (cut height: {cut_height:.4f})."
        )
        print(f"  This is the closest possible given the dendrogram structure.")

    return clusters, cut_height, linkage_matrix, exact_clusters_achieved


def _find_exact_cut_height_for_clusters(linkage_matrix, n_clusters):
    """
    Find the exact cut height that produces exactly n_clusters.
    Uses a binary search approach on the linkage heights.

    Args:
        linkage_matrix: SciPy linkage matrix
        n_clusters: Desired number of clusters

    Returns:
        tuple: (cut_height, exact_found)
        - cut_height: Best cut height found (exact if possible, otherwise closest)
        - exact_found: Boolean indicating if exact height was found
    """
    n_points = len(linkage_matrix) + 1

    # Handle edge cases
    if n_clusters <= 1:
        return linkage_matrix[-1, 2] + 1e-10, True
    if n_clusters >= n_points:
        return max(0.0, linkage_matrix[0, 2] - 1e-10), True

    # Get all unique heights in the linkage matrix
    heights = linkage_matrix[:, 2]
    unique_heights = np.unique(heights)

    # If we have fewer unique heights than (n_points - n_clusters),
    # we might have tied distances. We need to handle this carefully.

    # First, try the theoretical cut height
    # For k clusters, cut after merge (n - k) which is at index (n - k - 1)
    merge_idx = n_points - n_clusters - 1

    if merge_idx < 0:
        low_height = 0.0
    else:
        low_height = linkage_matrix[merge_idx, 2]

    if merge_idx + 1 >= len(linkage_matrix):
        high_height = low_height + 1.0  # Some reasonable upper bound
    else:
        high_height = linkage_matrix[merge_idx + 1, 2]

    # Binary search for the exact cut height
    epsilon = 1e-10
    max_iterations = 100

    for i in range(max_iterations):
        # Try a height in the middle
        test_height = (low_height + high_height) / 2.0

        # Get clusters at this height
        test_clusters = fcluster(linkage_matrix, test_height, criterion="distance")
        n_test_clusters = len(np.unique(test_clusters))

        if n_test_clusters == n_clusters:
            # Found exact height
            return test_height, True
        elif n_test_clusters > n_clusters:
            # Too many clusters, need higher cut (more merging)
            low_height = test_height
        else:
            # Too few clusters, need lower cut (less merging)
            high_height = test_height

        # Check for convergence
        if (high_height - low_height) < epsilon:
            break

    # If binary search didn't find exact height, try scanning unique heights
    # This handles cases with tied distances
    sorted_heights = np.sort(unique_heights)

    # Add small offsets to test around each height
    test_heights = []
    for h in sorted_heights:
        test_heights.extend([h - 1e-10, h, h + 1e-10])

    # Also test midpoints between consecutive heights
    for i in range(len(sorted_heights) - 1):
        test_heights.append((sorted_heights[i] + sorted_heights[i + 1]) / 2.0)

    test_heights = np.unique(test_heights)
    test_heights = test_heights[test_heights >= 0]  # Remove negative heights

    best_height = None
    best_diff = float("inf")

    for height in test_heights:
        test_clusters = fcluster(linkage_matrix, height, criterion="distance")
        n_test_clusters = len(np.unique(test_clusters))

        if n_test_clusters == n_clusters:
            return height, True

        diff = abs(n_test_clusters - n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_height = height

    # Return the best we found
    if best_height is not None:
        return best_height, False  # Not exact, but best we could do

    # Fallback: use the theoretical height
    if merge_idx >= 0 and merge_idx < len(linkage_matrix):
        return linkage_matrix[merge_idx, 2] + 1e-10, False
    else:
        return linkage_matrix[-1, 2] + 1e-10, False


# ============================================================================
# Output generation
# ============================================================================


def save_cluster_results(
    synset_names,
    word_mapping,
    enhanced_data,
    clusters,
    cut_height,
    output_dir=".",
    n_clusters=None,
    exact_clusters_achieved=True,
):
    """
    Save clustering results to files.

    Args:
        synset_names: List of synset names
        word_mapping: Dict mapping synset_name -> original_word
        enhanced_data: Enhanced data if available
        clusters: Cluster assignments for each synset
        cut_height: Height at which dendrogram was cut
        output_dir: Directory to save output files
        n_clusters: Number of clusters requested (for reference)
        exact_clusters_achieved: Whether exact number of clusters was achieved

    Returns:
        tuple: (summary_dict, actual_output_dir_used)
        - summary_dict: Summary of clustering results
        - actual_output_dir_used: Directory where files were actually saved
    """
    # Convert to absolute path for reliability
    original_output_dir = os.path.abspath(output_dir)
    actual_output_dir = original_output_dir

    # Debug: print output directory
    # print(f"Debug: Saving results to directory: {original_output_dir}")

    # Create output directory with robust error handling
    directory_created = False
    max_attempts = 10
    
    for attempt in range(max_attempts):
        try:
            # Check if path exists and is not a directory
            if os.path.exists(actual_output_dir):
                if os.path.isdir(actual_output_dir):
                    # Directory already exists, we can use it
                    directory_created = True
                    break
                else:
                    # A file exists with this name, try alternative
                    if attempt == 0:
                        print(f"Warning: '{actual_output_dir}' exists but is not a directory")
                    if attempt < max_attempts - 1:
                        actual_output_dir = f"{original_output_dir}_{attempt + 1}"
                        continue
                    else:
                        # Too many attempts, fall back to current directory
                        print(f"Error: Could not find available directory name after {max_attempts} attempts")
                        actual_output_dir = os.path.abspath(".")
                        print(f"Falling back to current directory: {actual_output_dir}")
                        os.makedirs(actual_output_dir, exist_ok=True)
                        directory_created = True
                        break
            
            # Path doesn't exist, try to create it
            os.makedirs(actual_output_dir, exist_ok=True)
            directory_created = True
            break
            
        except Exception as e:
            if attempt < max_attempts - 1:
                # Try alternative name
                actual_output_dir = f"{original_output_dir}_{attempt + 1}"
                print(f"Warning: Error creating directory '{original_output_dir}': {e}")
                print(f"Trying alternative: '{actual_output_dir}'")
            else:
                # Last attempt failed, fall back to current directory
                print(f"Error creating directory '{actual_output_dir}': {e}")
                actual_output_dir = os.path.abspath(".")
                print(f"Falling back to current directory: {actual_output_dir}")
                try:
                    os.makedirs(actual_output_dir, exist_ok=True)
                    directory_created = True
                except Exception as fallback_error:
                    print(f"Critical error: Cannot even create in current directory: {fallback_error}")
                    raise fallback_error
                break
    
    if not directory_created:
        # Should not happen, but just in case
        actual_output_dir = os.path.abspath(".")
        print(f"Emergency fallback to current directory: {actual_output_dir}")
        os.makedirs(actual_output_dir, exist_ok=True)

    # Get unique clusters
    unique_clusters = np.unique(clusters)
    actual_n_clusters = len(unique_clusters)

    # Use actual number of clusters, not requested number
    if n_clusters is None:
        n_clusters = actual_n_clusters

    # Create summary data
    summary = {
        "n_synsets": len(synset_names),
        "n_clusters_requested": n_clusters,
        "n_clusters_actual": actual_n_clusters,
        "exact_clusters_achieved": exact_clusters_achieved,
        "cut_height": float(cut_height),
        "clusters": [],
    }

    # For each cluster
    for cluster_id in unique_clusters:
        # Get synsets in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_synsets = [synset_names[i] for i in cluster_indices]

        # Get words in this cluster
        cluster_words = [word_mapping[synset] for synset in cluster_synsets]

        # Calculate cluster statistics
        cluster_size = len(cluster_synsets)

        # Save per-cluster files
        cluster_dir = os.path.join(actual_output_dir, f"cluster_{cluster_id}")
        try:
            os.makedirs(cluster_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating cluster directory {cluster_dir}: {e}")
            # Skip this cluster if we can't create its directory
            continue

        # Save simple JSON
        simple_json = {}
        for synset, word in zip(cluster_synsets, cluster_words):
            if word not in simple_json:
                simple_json[word] = []
            simple_json[word].append(synset)

        simple_path = os.path.join(cluster_dir, f"wordnet_cluster_{cluster_id}.json")
        try:
            with open(simple_path, "w") as f:
                json.dump(simple_json, f, indent=2)
        except Exception as e:
            print(f"Error saving simple JSON to {simple_path}: {e}")

        # Save text file
        txt_path = os.path.join(cluster_dir, f"wordnet_cluster_{cluster_id}.txt")
        try:
            with open(txt_path, "w") as f:
                for synset in cluster_synsets:
                    f.write(f"{synset}\n")
        except Exception as e:
            print(f"Error saving text file to {txt_path}: {e}")

        # Save enhanced data if available
        if enhanced_data:
            enhanced_cluster_data = {}
            for synset in cluster_synsets:
                # Find this synset in enhanced data
                for word, synset_list in enhanced_data.items():
                    for synset_data in synset_list:
                        if (
                            isinstance(synset_data, dict)
                            and synset_data.get("synset") == synset
                        ):
                            if word not in enhanced_cluster_data:
                                enhanced_cluster_data[word] = []
                            enhanced_cluster_data[word].append(synset_data)

            enhanced_path = os.path.join(
                cluster_dir, f"wordnet_cluster_{cluster_id}_enhanced.json"
            )
            try:
                with open(enhanced_path, "w") as f:
                    json.dump(enhanced_cluster_data, f, indent=2)
            except Exception as e:
                print(f"Error saving enhanced JSON to {enhanced_path}: {e}")

        # Add to summary
        cluster_summary = {
            "cluster_id": int(cluster_id),
            "size": cluster_size,
            "synsets": cluster_synsets,
            "words": list(set(cluster_words)),  # Unique words
            "word_count": len(set(cluster_words)),
        }
        summary["clusters"].append(cluster_summary)

    # Save summary
    summary_path = os.path.join(actual_output_dir, "wordnet_clusters_summary.json")
    # print(f"Debug: Saving summary to: {summary_path}")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        # print(f"Debug: Summary saved successfully")
    except Exception as e:
        print(f"Error saving summary to {summary_path}: {e}")
        # Try to save in current directory as fallback
        fallback_path = os.path.abspath("wordnet_clusters_summary.json")
        print(f"Falling back to: {fallback_path}")
        with open(fallback_path, "w") as f:
            json.dump(summary, f, indent=2)
        summary_path = fallback_path

    print(f"\nClustering results saved to {actual_output_dir}/")
    print(f"  - Summary: {summary_path}")
    print(f"  - {len(unique_clusters)} cluster directories created")

    return summary, actual_output_dir


def plot_dendrogram(linkage_matrix, synset_names, cut_height, output_dir="."):
    """
    Plot and save dendrogram visualization.

    Args:
        linkage_matrix: SciPy linkage matrix
        synset_names: List of synset names (for labels)
        cut_height: Cut height for clustering
        output_dir: Directory to save plot
    """

    try:
        import matplotlib.pyplot as plt
        import matplotlib

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot dendrogram
        dendrogram(
            linkage_matrix,
            labels=synset_names,
            orientation="right",
            color_threshold=cut_height,
            above_threshold_color="gray",
        )

        # Add cut line
        plt.axvline(x=cut_height, color="red", linestyle="--", alpha=0.7, linewidth=2)

        # Calculate number of clusters at this cut height for verification
        from scipy.cluster.hierarchy import fcluster

        clusters_at_height = fcluster(linkage_matrix, cut_height, criterion="distance")
        n_clusters_at_height = len(np.unique(clusters_at_height))

        # Add informative text
        plt.text(
            cut_height + 0.01,
            0.5,
            f"Cut height: {cut_height:.3f}\nClusters: {n_clusters_at_height}",
            transform=plt.gca().get_xaxis_transform(),
            color="red",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="red"
            ),
        )

        # Customize plot
        plt.title(
            f"WordNet Synset Clustering Dendrogram (Complete Linkage)\nCut height for {n_clusters_at_height} clusters: {cut_height:.4f}"
        )
        plt.xlabel("Distance")
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, "wordnet_clusters_dendrogram.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  - Dendrogram: {plot_path}")

    except ImportError:
        print(
            colored(
                "Warning: matplotlib not available for dendrogram plotting", "yellow"
            )
        )
    except Exception as e:
        print(colored(f"Warning: Could not plot dendrogram: {e}", "yellow"))


# ============================================================================
# Main function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Cluster WordNet synsets using hierarchical clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input wordnet_selections.json --clusters 5
  %(prog)s --input wordnet_selections_synonyms.json --clusters 3 --method path
  %(prog)s --input wordnet_selections.txt --clusters 8 --output-dir ./clusters
  %(prog)s --input wordnet_selections.json --clusters 5 --plot-dendrogram
  %(prog)s --input wordnet_selections.json --clusters 5 --no-progress
  %(prog)s --input wordnet_selections_synonyms.json --clusters 5 --no-closest-words
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (JSON or TXT from wordnet_disambiguator.py)",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of clusters to form (default: 5)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="wup",
        choices=["wup", "path", "both_average"],
        help="Similarity method for distance calculation (default: wup)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./wordnet_clusters",
        help="Output directory for cluster results (default: ./wordnet_clusters)",
    )

    parser.add_argument(
        "--plot-dendrogram",
        action="store_true",
        help="Plot and save dendrogram visualization (requires matplotlib)",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars for distance calculation",
    )

    parser.add_argument(
        "--linkage",
        type=str,
        default="complete",
        choices=["complete", "average", "single", "ward"],
        help="Linkage method for hierarchical clustering (default: complete)",
    )

    parser.add_argument(
        "--no-closest-words",
        action="store_true",
        help="Exclude synsets from closest_words_info (default: include them)",
    )

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input):
        print(colored(f"Error: Input file not found: {args.input}", "red"))
        return 1

    # Check number of clusters
    if args.clusters < 1:
        print(colored("Error: Number of clusters must be at least 1", "red"))
        return 1

    # Download WordNet data if needed
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
            return 1

    # Load synsets
    try:
        synset_names, word_mapping, enhanced_data = load_synsets_from_file(
            args.input,
            include_closest_words=not args.no_closest_words,  # Default True, False if --no-closest-words
        )
    except Exception as e:
        print(colored(f"Error loading input file: {e}", "red"))
        return 1

    if len(synset_names) == 0:
        print(colored("Error: No valid synsets found in input file", "red"))
        return 1

    print(f"Total synsets loaded: {len(synset_names)}")

    # Check for duplicates
    from collections import Counter

    synset_counts = Counter(synset_names)
    duplicates = {synset: count for synset, count in synset_counts.items() if count > 1}
    if duplicates:
        print(colored(f"Warning: Found {len(duplicates)} duplicate synsets:", "yellow"))
        for synset, count in list(duplicates.items())[:5]:  # Show first 5 duplicates
            print(colored(f"  {synset}: {count} occurrences", "yellow"))
        if len(duplicates) > 5:
            print(colored(f"  ... and {len(duplicates) - 5} more duplicates", "yellow"))

    if enhanced_data:
        print(f"Enhanced data structure: {type(enhanced_data)}")
        # Count main synsets vs closest words synsets
        main_synsets = 0
        closest_synsets = 0
        for word, synset_list in enhanced_data.items():
            for synset_data in synset_list:
                if isinstance(synset_data, dict) and "synset" in synset_data:
                    main_synsets += 1
                    if "closest_words_info" in synset_data and isinstance(
                        synset_data["closest_words_info"], list
                    ):
                        closest_synsets += len(synset_data["closest_words_info"])
        print(f"  Main synsets: {main_synsets}")
        print(f"  Closest words synsets: {closest_synsets}")
        print(f"  Total expected: {main_synsets + closest_synsets}")

    if args.clusters > len(synset_names):
        print(
            colored(
                f"Warning: Number of clusters ({args.clusters}) exceeds number of synsets ({len(synset_names)}). Using {len(synset_names)} clusters instead.",
                "yellow",
            )
        )
        args.clusters = len(synset_names)

    # Calculate distance matrix
    try:
        print(f"Calculating distance matrix for {len(synset_names)} synsets...")
        dist_matrix, synset_names = calculate_distance_matrix(
            synset_names, method=args.method, show_progress=not args.no_progress
        )
        print(f"Distance matrix shape: {dist_matrix.shape}")
        print(f"Number of synsets after distance calculation: {len(synset_names)}")
    except Exception as e:
        print(colored(f"Error calculating distance matrix: {e}", "red"))
        return 1

    # Check if SciPy is available before performing clustering
    if not SCIPY_AVAILABLE:
        print(colored("\nERROR: SciPy is required for hierarchical clustering.", "red"))
        print(
            colored(
                "\nPlease install SciPy using one of the following methods:", "yellow"
            )
        )
        print("\n1. Using pip:")
        print("   pip install scipy")
        print("\n2. Using conda:")
        print("   conda install scipy")
        print("\n3. For system-wide installation (Linux):")
        print("   sudo apt-get install python3-scipy  # Debian/Ubuntu")
        print("   sudo yum install python3-scipy      # RHEL/CentOS")
        print("   sudo dnf install python3-scipy      # Fedora")
        print("\nAfter installation, run the script again.")
        return 1

    # Perform clustering
    print(f"\nPerforming hierarchical clustering with {args.clusters} clusters...")
    print(f"Distance matrix has {dist_matrix.shape[0]} points")
    try:
        clusters, cut_height, linkage_matrix, exact_clusters_achieved = (
            perform_hierarchical_clustering(
                dist_matrix, n_clusters=args.clusters, method=args.linkage
            )
        )

        # Debug: check cluster assignments
        unique_clusters = np.unique(clusters)
        actual_n_clusters = len(unique_clusters)

        if exact_clusters_achieved:
            print(f"✓ Clustering completed with exactly {actual_n_clusters} clusters.")
        else:
            print(
                f"✓ Clustering completed with {actual_n_clusters} clusters (requested: {args.clusters})."
            )

        print(f"  Cut height: {cut_height:.4f}")
        print(f"  Number of clusters found: {actual_n_clusters}")
        print(f"  Cluster IDs: {sorted(unique_clusters)}")

        # Show cluster sizes
        cluster_sizes = {}
        for cluster_id in unique_clusters:
            size = np.sum(clusters == cluster_id)
            cluster_sizes[cluster_id] = size

        print(f"  Cluster sizes:")
        for cluster_id, size in sorted(cluster_sizes.items()):
            print(f"    Cluster {cluster_id}: {size} synsets")

    except Exception as e:
        print(colored(f"Error during clustering: {e}", "red"))
        import traceback

        traceback.print_exc()
        return 1

    # Save results
    try:
        summary, actual_output_dir = save_cluster_results(
            synset_names,
            word_mapping,
            enhanced_data,
            clusters,
            cut_height,
            args.output_dir,
            args.clusters,
            exact_clusters_achieved,
        )
    except Exception as e:
        print(colored(f"Error saving results: {e}", "red"))
        return 1

    # Plot dendrogram if requested
    if args.plot_dendrogram and linkage_matrix is not None:
        plot_dendrogram(linkage_matrix, synset_names, cut_height, actual_output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"Total synsets: {summary['n_synsets']}")
    print(f"Clusters requested: {summary['n_clusters_requested']}")
    print(f"Clusters actual: {summary['n_clusters_actual']}")
    if not summary["exact_clusters_achieved"]:
        print(
            colored(
                f"Note: Got {summary['n_clusters_actual']} clusters instead of {summary['n_clusters_requested']} due to tied distances in the dendrogram",
                "yellow",
            )
        )
        print(
            colored(
                f"  When multiple merges happen at the same height, exact cluster counts may not be possible.",
                "yellow",
            )
        )
    print(f"Cut height: {summary['cut_height']:.4f}")
    print("\nCluster sizes:")
    for cluster in summary["clusters"]:
        print(
            f"  Cluster {cluster['cluster_id']}: {cluster['size']} synsets, {cluster['word_count']} unique words"
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(colored(f"\nUnexpected error: {e}", "red"))
        import traceback

        traceback.print_exc()
        sys.exit(1)
