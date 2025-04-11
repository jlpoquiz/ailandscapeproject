import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/original/published_papers/no_duplicates.csv")

# Set the style for a nicer plot
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Group by publication year and count the papers
collapsed_data = data.groupby(['Publication Year']).size().reset_index(name='Number of Papers')

# Remove outlier year 279
collapsed_data = collapsed_data[collapsed_data['Publication Year'] != 279]
collapsed_data = collapsed_data[collapsed_data['Publication Year'] != 2025]

# OPTION 1: Make 2024 visually distinct with hatching pattern and different color
ax = sns.barplot(
    x='Publication Year', 
    y='Number of Papers', 
    data=collapsed_data,
    palette="viridis"
)

# Find the index of year 2024 in the plot
if 2024 in collapsed_data['Publication Year'].values:
    idx_2024 = collapsed_data['Publication Year'].tolist().index(2024)
    # Change the hatch pattern and color for 2024
    ax.patches[idx_2024].set_hatch('///')
    ax.patches[idx_2024].set_edgecolor('red')
    ax.patches[idx_2024].set_linewidth(1.5)

# Customize the plot
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Add a note about 2024 data
if 2024 in collapsed_data['Publication Year'].values:
    plt.figtext(0.5, 0.01, '* 2024 data represents only a half-year', 
                ha='center', fontsize=10, style='italic')

# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Enhance the plot with value labels on top of each bar
for i, v in enumerate(collapsed_data['Number of Papers']):
    ax.text(i, v + 5, str(v), ha='center', fontsize=9)

plt.savefig("/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/original/published_papers/papers_per_year.png", dpi=300)

# # OPTION 2: Project 2024 data to full year (uncomment to use this approach instead)
# '''
# plt.figure(figsize=(12, 6))

# # Copy the data for manipulation
# adjusted_data = collapsed_data.copy()

# # If 2024 exists, create projected value
# if 2024 in adjusted_data['Publication Year'].values:
#     # Get the 2024 paper count
#     papers_2024 = adjusted_data.loc[adjusted_data['Publication Year'] == 2024, 'Number of Papers'].values[0]
    
#     # Project to full year (multiply by 2 for half-year data)
#     adjusted_data.loc[adjusted_data['Publication Year'] == 2024, 'Number of Papers'] = papers_2024 * 2

# # Create the bar plot with the adjusted data
# ax = sns.barplot(
#     x='Publication Year', 
#     y='Number of Papers', 
#     data=adjusted_data,
#     palette="viridis"
# )

# # Find the index of year 2024 in the plot
# if 2024 in adjusted_data['Publication Year'].values:
#     idx_2024 = adjusted_data['Publication Year'].tolist().index(2024)
#     # Change the pattern for 2024 to indicate projection
#     ax.patches[idx_2024].set_hatch('xxx')
#     ax.patches[idx_2024].set_edgecolor('red')
#     ax.patches[idx_2024].set_alpha(0.8)  # Slightly transparent to indicate projection

# # Customize the plot
# plt.title('Number of Research Papers Published Per Year (2024 Projected)', fontsize=16)
# plt.xlabel('Year', fontsize=12)
# plt.ylabel('Number of Papers', fontsize=12)
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Add a note about 2024 data
# if 2024 in adjusted_data['Publication Year'].values:
#     plt.figtext(0.5, 0.01, '* 2024 bar shows projected full-year value based on January-June data', 
#                 ha='center', fontsize=10, style='italic')

# # Add a grid for better readability
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Enhance the plot with value labels on top of each bar
# for i, v in enumerate(adjusted_data['Number of Papers']):
#     label = f"{int(v)}" if adjusted_data['Publication Year'].iloc[i] != 2024 else f"{int(v)} (proj.)"
#     ax.text(i, v + 5, label, ha='center', fontsize=9)

# plt.show()
# '''