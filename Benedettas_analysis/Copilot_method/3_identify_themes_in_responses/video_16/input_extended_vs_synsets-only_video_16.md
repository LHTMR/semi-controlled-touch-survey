# Step 3 video 16 
Using the scripts "step3_one_video_extended" and "step3_one_video_synset_only" for extended and synset-only respectively, I gave the inputs reported below. 
The input files were taken from the "results" folder, which contains the files genrated by running "wordnet_multiple_definiton.py". 
The extended script used synsets and closest words, while the synset-only script used synsets alone.


# Extended input for themes 
python step3_one_video_extended.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Affection closeness" --out-detailed "step3_detailed_affection.csv" --out-summary "step3_summary_affection.csv"

python step3_one_video_extended.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes “Child-related" --out-detailed "step3_detailed_child.csv" --out-summary "step3_summary_child.csv"

python step3_one_video_extended.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Leisure at home" --out-detailed "step3_detailed_leisure.csv" --out-summary "step3_summary_leisure.csv"

python step3_one_video_extended.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes " medical examination therapy" --out-detailed "step3_detailed_medical.csv" --out-summary "step3_summary_medical.csv"

python step3_one_video_extended.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes " Pain injury soreness" --out-detailed "step3_detailed_pain.csv" --out-summary "step3_summary_pain.csv"

python step3_one_video_extended.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes " Playful teasing joking" --out-detailed "step3_detailed_playful.csv" --out-summary "step3_summary_playful.csv"

python step3_one_video_extended.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "strager boundary concern" --out-detailed "step3_detailed_stranger.csv" --out-summary "step3_summary_stranger.csv"

python step3_one_video_extended.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes " Work school context" --out-detailed "step3_detailed_work.csv" --out-summary "step3_summary_work.csv"


# Example output for one theme of the extended 
Saved detailed results to: step3_detailed_affection.csv
Saved summary counts to: step3_summary_affection.csv

=== STEP 3 SUMMARY (one video) ===
   Affection closeness: 43 responses flagged

# Synsets-only input for themes 

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Affection closeness" --out-detailed "step3_detailed_affection.csv" --out-summary "step3_summary_affection.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Work school context" --out-detailed "step3_detailed_work.csv" --out-summary "step3_summary_work.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "strager boundary concern" --out-detailed "step3_detailed_stranger.csv" --out-summary "step3_summary_stranger.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Playful teasing joking" --out-detailed "step3_detailed_playful.csv" --out-summary "step3_summary_playful.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Pain injury soreness" --out-detailed "step3_detailed_pain.csv" --out-summary "pain.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "medical examination therapy" --out-detailed "step3_detailed_medical.csv" --out-summary "step3_summary_medical.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Child-related" --out-detailed "step3_detailed_child.csv" --out-summary "step3_summary_child.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Comfort and emotional support" --out-detailed "step3_detailed_comfort.csv" --out-summary "step3_summary_comfort.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" ----only-themes “Leisure at home" --out-detailed "step3_detailed_home.csv" --out-summary "step3_summary_home.csv" --no-closest

python step3_one_video_synset_only.py --results-root "Results" --excel-file "touch_data_video_16_social_context.xlsx" --only-themes "Getting attention_communication" --out-detailed "step3_detailed_attention.csv" --out-summary "step3_summary_attention.csv" --no-closest


# Example of one output for one theme for synsets-only
 
=== STEP 3 SUMMARY (one video) ===
   Affection closeness: 32 responses flagged

