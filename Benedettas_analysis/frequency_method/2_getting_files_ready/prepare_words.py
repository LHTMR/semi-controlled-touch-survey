import pandas as pd

# loads the grouped word freqeuncy csv
df = pd.read_csv('word_frequencies_grouped.csv')


# tests different cutoff values
for cutoff in [3, 5, 10, 12]:
    df_cutoff = df[df['total_group_count'] >= cutoff]
    print(f"Cutoff >= {cutoff}: {len(df_cutoff)} words remaining")

# apply chosen cutoff
df_cutoff = df[df['total_group_count'] >= 20]

# saves filtered words to a txt file, one word per line
df_cutoff.to_csv(  # also with . you can create a function opertaing on what came before it, in this case the df_cutoff
    'words_filtered.csv', index=False)

print(f"Done! {len(df_cutoff)} words saved to words_filtered.txt")

