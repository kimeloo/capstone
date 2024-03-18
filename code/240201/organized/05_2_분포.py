import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file and specify the column to analyze
pwd = '~/Documents/Coding/capstone/data/240201'
df1 = pd.read_csv(pwd+'/03_htnderv_s1_train.csv')
df2 = pd.read_csv(pwd+'/04_htnderv_s1_all.csv')
# Define column names
column_names = ['height', 'weight', 'weight20', 'age_s1', 'gender', 'race']

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Flatten the axes for easier iteration
axes = axes.flatten()

# Plot each column's distribution
for i, column in enumerate(column_names):
    # Plot histogram for each column in df1
    axes[i].hist(df1[column], alpha=0.5, label='df1', color='blue', bins=20)
    # Plot histogram for each column in df2
    axes[i].hist(df2[column], alpha=0.5, label='df2', color='red', bins=20)
    axes[i].set_title(f'Distribution of {column} in df1 and df2')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.tight_layout()
plt.show()
