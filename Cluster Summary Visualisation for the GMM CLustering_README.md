# Create the README.md content
readme_content = """
# Customer Clustering and Visualization

This project performs customer clustering and visualizes the distribution of customers across various flag types and the average ticket sizes across clusters. The analysis is done using PySpark for data processing and Pandas, Seaborn, and Matplotlib for data visualization.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [License](#license)

## Introduction

This project aims to analyze customer data by grouping them into clusters and visualizing key metrics such as the distribution of specific flag types and average ticket sizes within each cluster. This helps in understanding customer behavior and identifying potential areas for improvement.

## Dependencies

- Python 3.x
- Pandas
- Seaborn
- Matplotlib
- PySpark

You can install the required Python libraries using:

\`\`\`bash
pip install pandas seaborn matplotlib pyspark
\`\`\`

## Usage

1. Ensure you have all dependencies installed.
2. Load your customer data into a DataFrame named \`df\`.
3. Run the provided script to perform clustering and generate the visualizations.

## Code Explanation

### Step 1: Import Libraries

\`\`\`python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import avg, col, count, when
from pyspark.sql import functions as F
\`\`\`

### Step 2: Extract Relevant Columns

- Extract columns containing 'flag'.
- Extract columns containing 'tkt_size' but not containing 'flag'.

\`\`\`python
flag_columns = [col for col in df.columns if 'flag' in col]
tkt_size_columns = [col for col in df.columns if 'tkt_size' in col and 'flag' not in col]
\`\`\`

### Step 3: Calculate Average Ticket Sizes by Cluster

\`\`\`python
avg_tkt_sizes = clusters.groupBy("prediction").agg(
    *[F.round(avg(c), 0).alias(f"{c}_avg__tkt_size") for c in tkt_size_columns]
).orderBy(col("prediction")).toPandas()
\`\`\`

### Step 4: Calculate High Occurrence Counts in Flag Columns by Cluster

\`\`\`python
cluster_counts = clusters.groupBy("prediction").agg(
    *[count(when(col(c) == 3, c)).alias(f"{c}_high_count") for c in flag_columns]
).orderBy(col("prediction")).toPandas()
\`\`\`

### Step 5: Normalize Data

- Normalize the counts and average ticket sizes within each row (cluster).

\`\`\`python
heatmap_data_normalized = cluster_counts.set_index('prediction')[
    [f"{c}_high_count" for c in flag_columns]
].apply(lambda x: x / x.sum(), axis=1)

avg_tkt_sizes_normalized = avg_tkt_sizes.set_index('prediction')[
    [f"{c}_avg__tkt_size" for c in tkt_size_columns]
].apply(lambda x: x, axis=1)

heatmap_data_normalized = heatmap_data_normalized.T
avg_tkt_sizes_normalized = avg_tkt_sizes_normalized.T
\`\`\`

### Step 6: Plot Heatmaps

- Plot heatmaps for customer distribution across flag types and average ticket sizes.

\`\`\`python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 16))

sns.heatmap(heatmap_data_normalized, annot=True, cmap="YlGnBu", fmt='.1%', cbar_kws={'format': '%.0f%%'}, ax=ax1, vmax=0.25)
ax1.set_title('Relative Distribution of Customers across Flag Types by Cluster')
ax1.set_ylabel('Flag Columns')
ax1.set_xlabel('Cluster Prediction')
ax1.set_yticklabels(flag_columns, rotation=0, ha="right")
ax1.xaxis.set_ticks_position("top")

sns.heatmap(avg_tkt_sizes_normalized, annot=True, cmap="YlOrRd", cbar=True, ax=ax2, fmt='.0f', vmax=25000)
ax2.set_title('Average Ticket Sizes by Cluster')
ax2.set_xlabel('Cluster Prediction')
ax2.set_ylabel('Ticket Size Columns')
ax2.set_yticklabels(tkt_size_columns, rotation=0, ha="right")

plt.tight_layout()
plt.show()
\`\`\`

## Results

The script generates two heatmaps:
1. Relative Distribution of Customers across Flag Types by Cluster
2. Average Ticket Sizes by Cluster

These visualizations help in understanding the customer distribution and average ticket sizes within each cluster.

## License

This project is licensed under the MIT License.
"""

# Save the content to a README.md file
with open("/mnt/data/README.md", "w") as file:
    file.write(readme_content)
