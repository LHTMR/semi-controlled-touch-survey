import pandas as pd

# loads the grouped word freqeuncy csv
df = pd.read_csv('word_frequencies_grouped.csv')

# keeo only the main word of each group
df_filtered = df[df['is_main'] == True]

# tests different cutoff values
for cutoff in [3, 5, 10]:
    df_cutoff = df_filtered[df_filtered['total_group_count'] >= cutoff]
    print(f"Cutoff >= {cutoff}: {len(df_cutoff)} words remaining")

# apply chosen cutoff
df_cutoff = df_filtered[df_filtered['total_group_count'] >= 20]

# saves filtered words to a txt file, one word per line
df_cutoff['group_key'].to_csv(
    'words_filtered_main_only.txt', index=False, header=False)

# also save the filtered rows to a CSV file
df_cutoff.to_csv('words_filtered_main_only.csv', index=False)

print(f"Done! {len(df_cutoff)} words saved to words_filtered_main_only.txt")
print(f"Done! {len(df_cutoff)} words saved to words_filtered_main_only.csv")
