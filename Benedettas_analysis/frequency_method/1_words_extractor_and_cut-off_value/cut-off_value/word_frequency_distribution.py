import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('word_frequencies_grouped.csv')

# Filter to only include rows where is_main is True
df_filtered = df[df['is_main'] == True]

# Print the filtered data to see which groups have which counts
print(df_filtered[['group_key', 'total_group_count']])

# Create a histogram of total_group_count
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['total_group_count'],
         bins=df_filtered.shape[0], edgecolor='black')
plt.xlabel('Total Group Count')
plt.ylabel('Frequency')
plt.title('Distribution of Total Group Counts for Main Words')
plt.xlim(df_filtered['total_group_count'].min() - 10,
         df_filtered['total_group_count'].max() + 10)

# Save the plot to a file
plt.savefig('distribution_plot.png')

# Display the plot in a window
plt.show()


# Bar chart of top 30 words
df_top = df_filtered.sort_values(
    'total_group_count', ascending=False).head(df_filtered.shape[0])

plt.figure(figsize=(12, 8))
plt.barh(df_top['group_key'], df_top['total_group_count'], edgecolor='black')
plt.xlabel(' Frequency ')
plt.ylabel(' Word ')
plt.title(' Word Frequency ')
plt.gca().invert_yaxis()

# The vertical line (axvline) cuts on the x-axis, which is frequency. The horizontal line (axhline) cuts on the y-axis, which is words (rank). So:
# blue vertical = frequency cutoff (x = 50)
# red horizontal = word rank cutoff (top 30)
freq_cutoff = 20
rank_cutoff = 30
plt.axvline(x=freq_cutoff, color='blue', linestyle='--',
            linewidth=1.5, label=f'Frequency cutoff ({freq_cutoff})')
plt.axhline(y=rank_cutoff - 0.5, color='red', linestyle='--',
            linewidth=1.5, label=f'Rank cutoff (top {rank_cutoff})')
plt.legend()

plt.tight_layout()
plt.savefig('top_words_plot.png')
plt.show()
