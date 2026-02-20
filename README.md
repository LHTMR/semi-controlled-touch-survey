
# Semi controlled touch survey

## File Tree
Structure of the dataset, i.e., where files are located.

```
.
├── .git
├── .gitattributes
├── .gitignore
├── Emojigrid_high_res.png
├── Figures
│   ├── 9set_1_code_map.svg
│   ├── 9set_2_code_map.svg
│   ├── 9set_3_code_map.svg
│   ├── 9set_4_code_map.svg
│   ├── 9set_5_code_map.svg
│   ├── 9set_6_code_map.svg
│   ├── Appropriateness
│   │   ├── Appropriate_comparison.pdf
│   │   ├── Appropriateness.pdf
│   │   └── Innapropriate_comparison.pdf
│   ├── Emotional_self_word-freq.pdf
│   ├── Emotional_touch_word-freq.pdf
│   ├── Intention&Purpose_word-freq.pdf
│   ├── Sensory_word-freq.pdf
│   ├── Social_body_word-freq.pdf
│   ├── Social_context_word-freq.pdf
│   ├── Social_place_word-freq.pdf
│   ├── Social_self_word-freq.pdf
│   ├── Valence_Arousal
│   │   ├── Arousal_comparison.pdf
│   │   ├── Valence_Arousal_coordinates.pdf
│   │   └── Valence_comparison.pdf
│   ├── Word_frequencies
│   │   ├── Emotional_self_word-freq.pdf
│   │   ├── Emotional_touch_word-freq.pdf
│   │   ├── Intention&Purpose_word-freq.pdf
│   │   ├── Sensory_word-freq.pdf
│   │   ├── Social_body_word-freq.pdf
│   │   ├── Social_context_word-freq.pdf
│   │   ├── Social_place_word-freq.pdf
│   │   └── Social_self_word-freq.pdf
│   ├── affect_map.svg
│   ├── emotional_map.svg
│   ├── person_map.svg
│   ├── sensory_map.svg
│   └── top9_code_map.svg
├── LICENSE
├── Processed Data
│   ├── Emotional_self_word-freq-plot-data.txt
│   ├── Emotional_self_word-freq.txt
│   ├── Emotional_touch_word-freq-plot-data.txt
│   ├── Emotional_touch_word-freq.txt
│   ├── Intention&Purpose_word-freq-plot-data.txt
│   ├── Intention&Purpose_word-freq.txt
│   ├── Sensory_word-freq-plot-data.txt
│   ├── Sensory_word-freq.txt
│   ├── Social_body_word-freq-plot-data.txt
│   ├── Social_body_word-freq.txt
│   ├── Social_context_word-freq-plot-data.txt
│   ├── Social_context_word-freq.txt
│   ├── Social_place_word-freq-plot-data.txt
│   ├── Social_place_word-freq.txt
│   ├── Social_self_word-freq-plot-data.txt
│   ├── Social_self_word-freq.txt
│   ├── descriptor_map_data.tsv
│   ├── descriptor_map_data.xlsx
│   └── touch_data.txt
├── Qualitative_Flavia
│   ├── clean_maxqda_codes.R
│   ├── code_maps.R
│   ├── coded_export_data_processing.R
│   ├── coded_preprocessing_quality_control.R
│   ├── maxqda_data_preprocessing.R
│   └── survey_data_preprocessing.R
├── README.md
├── analysis_IASATposter.R
├── descriptor_maps.R
├── semi-controlled-touch-survey.Rproj
└── wordnet_disambiguator.py
```

## Dataset Content

Here is a brief overview of the different documents contained in this folder.

### analysis_IASATposter.R

This R script is for analyzing data from a social touch survey. It performs data cleaning, processing, and visualization.

*   **Libraries Used:** It uses several R libraries, including `dplyr` for data manipulation, `readxl` for reading Excel files, `tidyr` for tidying data, `ggplot2` for plotting, `Hmisc` for statistical functions, and `quanteda` for text analysis.
*   **Input:** It reads raw survey data from an Excel file (`.xlsx`) located in a specific directory.
*   **Data Processing:**
    *   It separates respondent data and touch data.
    *   It cleans and reshapes the touch data, creating new variables like "Valence" and "Arousal".
    *   It adds descriptive labels to the touch data based on touch number, speed, force, etc.
    *   It saves the processed touch data as a text file (`touch_data.txt`).
*   **Analysis and Visualization:**
    *   It calculates and plots the number of responses per touch and per respondent.
    *   It creates boxplots to visualize the distribution of "Valence" and "Arousal" for different touch types.
    *   It performs direct comparisons of "Valence" and "Arousal" based on speed and force.
    *   It calculates and plots confidence intervals for "Valence" and "Arousal".
    *   It performs word frequency analysis on the text data from the survey.
    *   It analyzes the "Appropriateness" of the touch, creating bar plots to show the distribution of responses.
*   **Output:** The script generates several outputs:
    *   A processed data file: `touch_data.txt`.
    *   Several plots are generated and saved as PDF files in the `Figures/` directory.
    *   Word frequency data is saved as text files.

### .git, .gitattributes, .gitignore

These files indicate that this dataset is now managed as a Git repository. The `.git` directory contains the version control history, `.gitattributes` specifies attributes for Git operations, and `.gitignore` specifies which files and directories should be ignored by Git.

### Qualitative_Flavia/

This directory contains R scripts for qualitative data analysis and preprocessing, originally developed by Flavia. These scripts handle MAXQDA data export processing, code mapping, and survey data preprocessing.

#### clean_maxqda_codes.R

This R script is used for cleaning and exploring manually coded data that was exported from the qualitative data analysis software MAXQDA. The main goal of this script is to link the coded segments from MAXQDA with the survey data.

*   **Libraries Used:** It uses `readxl` for reading Excel files, `readr` for reading text files, and `tidyverse` for data manipulation.
*   **Input:**
    *   It reads an Excel file (`.xlsx`) containing coded segments exported from MAXQDA.
    *   It reads the `touch_data.txt` file, which is the processed survey data.
*   **Data Processing and Exploration:**
    *   It attempts to find the coded segments from the MAXQDA data within the `touch_data`.
    *   It extracts and compares variable names and codes from the MAXQDA data to identify inconsistencies or matches.
    *   It filters and views the MAXQDA data based on specific codes (e.g., "Annoy / Tease") to inspect the coded segments.
    *   It tries to identify segments that were coded but not associated with a specific video.
    *   It checks for duplicate coded segments.
*   **Output:** The script is primarily for interactive exploration and cleaning. It does not appear to save any new data files. The main output is the exploration and debugging of the coded data, likely done within an interactive R session.

#### coded_export_data_processing.R

This is a comprehensive R script designed to clean, re-format, and process a coded export file from a qualitative data analysis software. The script's main purpose is to transform the raw exported data into a structured format suitable for further analysis.

*   **Libraries Used:** It uses `readxl` for reading Excel files, `openxlsx` for writing Excel files, `readr` for reading and writing text files, and `tidyverse` for extensive data manipulation.
*   **Input:**
    *   An Excel file (`.xlsx`) which is a coded export from a coding software.
*   **Data Processing:**
    *   **Column Selection and Renaming:** It starts by selecting a subset of important columns and renaming them for clarity.
    *   **Data Filtering:** It removes rows related to demographics and keeps only rows containing specific keywords like "RELATIONAL", "Autocode", or "Video".
    *   **Video ID Extraction:** A significant portion of the script is dedicated to extracting a `VideoID` for each row. It uses a custom function to find the video number from different columns (`Code` and `Other`) based on a list of `video_questions`. It also includes logic to infer missing `VideoID`s from surrounding rows.
    *   **Duplicate Handling:** The script identifies and removes duplicate rows based on combinations of `PID`, `VideoID`, `Segment`, and `Code`.
    *   **Question Extraction:** It extracts the specific question associated with each coded segment. This is a complex process that involves parsing strings from the `Code` and `Other` columns. For rows with multiple questions, it splits them into separate rows.
    *   **Handling Missing Data:** The script identifies rows with missing question information and attempts to resolve them by looking at the "Video" cell which contains all answers in an ordered manner. It also reads a manually created file to fill in missing questions.
    *   **Final Structuring:** The script restructures the data to have one row per coded segment, with columns for `PID`, `VideoID`, `Segment`, `Code`, and `Question`.
*   **Output:**
    *   The script generates several intermediate and final CSV files for debugging and for the final processed data, with `output_preprocessed_codes_ilona07_05_24.csv` as the likely main output file.

#### coded_preprocessing_quality_control.R

This R script is for quality control and debugging of the preprocessed coded data. It aims to identify and address issues like missing data and duplicate rows in the output of the previous processing script.

*   **Libraries Used:** It uses `tidyverse` for data manipulation and `readxl` for reading Excel files.
*   **Input:**
    *   `output_preprocessed_codes_ilona19_04.csv`: The preprocessed coded data from the previous script.
    *   `attempt1_allRecoded.xlsx`: The original coded export file from MAXQDA.
    *   `Copy of missing_Q_situationalwhat60_wQuestionColumn_ADDED.xlsx` and `Copy of missing_Q_remaining38_wQuestionColumn_ADDED.xlsx`: Excel files that contain manually added question information for rows where the question was missing.
*   **Quality Control and Debugging:**
    *   **Missing Question Data:**
        *   It identifies rows with missing `Question` information in the preprocessed data.
        *   It writes out the rows with missing questions to separate CSV files (`missing_Q_situationalwhat60.csv` and `missing_Q_remaining38.csv`) for manual inspection.
        *   It reads in the manually corrected files to solve the missing data issue.
    *   **Duplicate Rows:**
        *   It identifies duplicate rows and writes them to a CSV file (`duplicate_rows.csv`).
        *   It provides a "Solution for duplicates" section that categorizes the reasons for duplicates and suggests how to handle them.
*   **Output:** The script generates several CSV files that highlight issues in the preprocessed data, which are intended for manual inspection and correction:
    *   `missing_Q_situationalwhat60.csv`
    *   `missing_Q_remaining38.csv`
    *   `duplicate_rows.csv`

#### code_maps.R

This R script is for visualizing the frequency of different codes from the coded data. It creates "code maps" that show the distribution of codes across different touch types, speeds, and forces.

*   **Libraries Used:** It uses `tidyverse` for data manipulation, `svglite` for saving plots as SVG files, `readxl` for reading Excel files, and `ggdark` for creating plots with a dark theme.
*   **Input:**
    *   An Excel file (`.xlsx`) that contains code frequencies per video, likely `Frequencies_perVideo.xlsx`.
*   **Data Processing:**
    *   It reads the code frequency data from the Excel file.
    *   It separates the `Video` column into `Contact`, `Direction`, `Speed (cm/s)`, and `Force`.
    *   It cleans and formats these new columns.
    *   It creates a `Type` column by combining `Direction` and `Contact`.
    *   It groups the data by `Code` and the new variables to summarize the frequency of each code.
*   **Visualization:**
    *   It creates a custom dark theme for the plots.
    *   It ranks the codes by their overall frequency and creates a bar plot to show the frequency of each code.
    *   The main visualization is a set of facet plots, where each facet represents a code. Within each facet, it plots the `Type` of touch against the `Speed (cm/s)`, and the size and color of the points represent the frequency and force.
    *   It saves the generated plot as an SVG file in the `Figures/` directory.
*   **Output:**
    *   The script generates a plot of the code maps and saves it as an SVG file in the `Figures/` directory.

### descriptor_maps.R

This R script is for creating "descriptor maps," which are visualizations of word frequencies associated with different types of touch. It groups words into higher-level "descriptors" and then into broader "Groups" (e.g., "emotion", "person", "sensory").

*   **Libraries Used:** It uses `tidyverse` for data manipulation, `svglite` for saving plots as SVG files, and `ggdark` for creating plots with a dark theme.
*   **Input:**
    *   It reads multiple `_word-freq-plot-data.txt` files from the `Processed Data/` directory. These files are the output of the word frequency analysis in `analysis_IASATposter.R`.
*   **Data Processing:**
    *   It reads and combines all the word frequency data files.
    *   **Descriptor Mapping:** It maps the stemmed words to more meaningful `descriptor`s (e.g., "affect" becomes "affection").
    *   **Group Mapping:** It then maps these `descriptor`s to broader `Group`s (e.g., "emotion", "body", "place").
    *   It summarizes the frequency of each `descriptor` and `Group`.
    *   It saves the processed descriptor map data to a file.
*   **Visualization:**
    *   It creates facet plots for different groups of descriptors (e.g., `sensory_descriptors`, `person_descriptors`).
    *   Each facet represents a descriptor, and it plots the `Type` of touch against the `Speed (cm/s)`. The size of the points represents the frequency, and the `Force` is represented by the fill color.
*   **Output:**
    *   `Processed Data/descriptor_map_data.tsv`: A tab-separated file containing the processed descriptor map data.
    *   Several SVG files in the `Figures/` directory, each showing a map for a different group of descriptors (e.g., `sensory_map.svg`, `person_map.svg`).

### Emojigrid_high_res.png

This is a high-resolution image file named `Emojigrid_high_res.png`. This is the "Emojigrid", a tool used in the survey for participants to rate their emotional state. The Emojigrid is a graphical tool that allows users to report their feelings in terms of valence (pleasure-displeasure) and arousal (activation-deactivation) by selecting a coordinate on the emoji grid.

### Figures/

This directory contains various plots and visualizations generated by the R scripts. The files are in SVG and PDF formats.

*   **Code Maps (`.svg`):**
    *   `9set_1_code_map.svg` to `9set_6_code_map.svg`: These are the code map visualizations created by `code_maps.R`.
    *   `affect_map.svg`, `emotional_map.svg`, `person_map.svg`, `sensory_map.svg`, `top9_code_map.svg`: These are the descriptor maps created by `descriptor_maps.R`.

*   **Word Frequency Plots (`.pdf`):**
    *   `Emotional_self_word-freq.pdf`, `Emotional_touch_word-freq.pdf`, `Intention&Purpose_word-freq.pdf`, `Sensory_word-freq.pdf`, `Social_body_word-freq.pdf`, `Social_context_word-freq.pdf`, `Social_place_word-freq.pdf`, `Social_self_word-freq.pdf`: These are the word frequency plots generated by `analysis_IASATposter.R`.

*   **Subdirectories:**
    *   **`Appropriateness/`:** Contains plots related to the appropriateness of touch.
    *   **`Valence_Arousal/`:** Contains plots related to valence and arousal ratings.
    *   **`Word_frequencies/`:** This directory contains duplicates of the word frequency plots.

### LICENSE

This file contains the MIT License for the software and associated documentation files in this dataset. The copyright holder is S McIntyre (2024). The MIT License is a permissive free software license that allows for the free use, copying, modification, and distribution of the software, with the condition that the original copyright and permission notices are included in all copies. The software is provided "as is" without any warranty.

#### maxqda_data_preprocessing.R

This R script is for preprocessing and exploring data exported from MAXQDA. It appears to be an initial exploration of the data structure, with a focus on understanding the relationship between codes, questions, and video IDs.

*   **Libraries Used:** It uses `readxl` for reading Excel files, `readr` for reading text files, and `tidyverse` for data manipulation.
*   **Input:**
    *   `MAXQDA 24 Coded Segments_NewProject.xlsx`: An Excel file containing coded segments exported from MAXQDA.
*   **Data Processing and Exploration:**
    *   **Code and Variable Name Comparison:** It extracts code names from both the column headers and the `Code` column and compares them to find differences.
    *   **Video Question Identification:** It identifies the unique video questions from the column names.
    *   **Video ID Extraction:** It creates a `VideoID` column by identifying the start of each video's data.
    *   **Participant Data Subsetting:** It includes an example of how to filter the data for a specific participant and select relevant columns.
    *   **Code Exploration:** It demonstrates how to find which specific codes were applied for a given question and participant.
*   **Output:** The script is primarily for interactive exploration and does not save any new data files.

### Processed Data/

This directory contains various data files that are the result of the preprocessing and analysis scripts.

*   **`descriptor_map_data.tsv` and `descriptor_map_data.xlsx`:** These files contain the processed descriptor map data, as generated by `descriptor_maps.R`.

*   **Word Frequency Data (`.txt`):**
    *   `*_word-freq.txt`: These files contain the word frequency data for each question, generated by `analysis_IASATposter.R`.
    *   `*_word-freq-plot-data.txt`: These files contain the data used to generate the word frequency plots, also from `analysis_IASATposter.R`.

*   **`touch_data.txt`:** This is the main processed data file for the touch survey data, created by `analysis_IASATposter.R`.

### semi-controlled-touch-survey.Rproj

This is an RStudio project file. It is used by the RStudio IDE to manage project-specific settings. When a user opens this `.Rproj` file in RStudio, the IDE will automatically set the working directory to the directory containing this file and load the project's settings. This makes it easier to work on the project and ensures that the code runs in the correct context. The file contains settings related to workspace management, code indexing, text encoding, and document weaving.

#### survey_data_preprocessing.R

This R script is designed to clean and re-format the original survey data file. The primary goal is to transform the wide-format survey data into a long-format data frame, which is more suitable for analysis and for import into qualitative data analysis software like MAXQDA.

*   **Libraries Used:** It uses `readxl` for reading Excel files, `openxlsx` for writing Excel files, `readr` for reading CSV files, and `tidyverse` for data manipulation.
*   **Input:**
    *   `SocialTouchProlific_February26_2024_original.xlsx`: The original survey data file, where each row represents a participant and columns contain demographic information and responses to questions for multiple videos.
    *   `video_info.csv`: A CSV file containing metadata for each video shown in the survey, such as the intensity of the touch, speed, body part used, and movement type.
*   **Data Processing:**
    *   **Data Reshaping:** The script iterates through each participant's data (each row) and transforms it from a wide format to a long format. The original format has one row per participant, with answers to video questions in columns prefixed with the video ID. The script creates a new data frame where each row represents a single video watched by a single participant.
    *   **Data Cleaning:** It performs data cleaning by checking each response. If a response is a single character (e.g., "a" or "1"), it is considered an invalid or low-effort response and is replaced with `NA`. If a participant-video combination has too many (`>4`) `NA` values, a warning is printed.
    *   **Merging with Video Info:** For each new row in the long-format data, the script merges the participant's survey data with the corresponding video's metadata from `video_info.csv`. This adds contextual information about the touch stimuli to each response.
    *   **Unique Key Generation:** A unique key is created for each row by concatenating the participant's ID and the video ID, ensuring that every participant-video entry is unique.
*   **Output:**
    *   `output_ilona15_03_24.xlsx`: An Excel file containing the preprocessed survey data in a long format, with clearly defined columns. This file is ready to be imported into MAXQDA for qualitative coding and analysis.

### wordnet_disambiguator.py

This Python script is a WordNet disambiguation tool that helps resolve word sense ambiguity by allowing users to select the correct WordNet synset definition for each word in a list. It can also extend synonyms with the closest words in WordNet using semantic similarity measures.

*   **Libraries Used:** It uses `nltk` for WordNet access and natural language processing, along with standard Python libraries for argument parsing, JSON, CSV, and file operations.
*   **Features:**
    *   **Word Sense Disambiguation:** Presents WordNet synsets and definitions for each word, allowing users to select the appropriate sense.
    *   **Synonym Extension:** Optionally extends synonyms with the closest words in WordNet using semantic similarity measures (Wu-Palmer, path similarity, or average of both).
    *   **Robust Interruption Handling:** Automatically saves partial results and supports resume functionality with validation.
    *   **Checkpointing:** Saves progress during time-consuming closest words calculations.
    *   **Safe Mode:** Provides maximum data protection with automatic backup of partial results.
*   **Input:**
    *   Can accept words via command-line arguments (`--words` flag) or from an input file (`--input-file`).
    *   Supports interactive mode when no arguments are provided.
*   **Data Processing:**
    *   For each word, retrieves all WordNet synsets (different senses/meanings).
    *   Presents definitions and examples to the user for selection.
    *   Optionally calculates semantic similarity to find closest words in WordNet.
    *   Handles interruptions gracefully with automatic saving of partial results.
*   **Output:**
    *   `wordnet_selections.json`: Simple mapping of words to selected synsets.
    *   `wordnet_selections_synonyms.json`: Extended mapping with synonyms and closest words.
    *   `wordnet_selections.csv`: CSV file with definitions and synonyms.
    *   `wordnet_selections_extended.csv`: CSV file with closest words (if `--extend-synonyms` used).
    *   `wordnet_selections.txt`: Simple list of synsets, one per line.
    *   Partial/interrupted results saved to `wordnet_selections_partial.json` and `closest_words_checkpoint.json`.
*   **Usage Examples:**
    *   `python wordnet_disambiguator.py --words "word1,word2,word3"`
    *   `python wordnet_disambiguator.py --input-file words.txt`
    *   `python wordnet_disambiguator.py --words "sensation" --extend-synonyms --num-closest 20`
    *   `python wordnet_disambiguator.py --words "apple" --extend-synonyms --similarity-method path`
    *   `python wordnet_disambiguator.py --words "word1,word2" --safe-mode`
    *   `python wordnet_disambiguator.py --resume wordnet_selections_partial.json --force-resume`
