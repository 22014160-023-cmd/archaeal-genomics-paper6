#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal

df = pd.read_csv("Master_Data.csv")
print(f"Loaded {len(df)} genomes")

features = ['gc_content', 'gene_count', 'te_count', 'genome_size_mb', 'avg_gene_len', 'gc3']
print("\nKruskal-Wallis Tests:")
for col in features:
    groups = [df[df['group']==g][col].dropna().values for g in df['group'].unique()]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        h, p = kruskal(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {col}: H={h:.3f}, p={p:.4f} {sig}")

def wheatsheaf_index(X, labels, target):
    dist = squareform(pdist(X))
    t_idx = [i for i, g in enumerate(labels) if g == target]
    o_idx = [i for i, g in enumerate(labels) if g != target]
    if len(t_idx) < 3:
        return np.nan
    within = np.mean([dist[i][j] for i in t_idx for j in t_idx if i < j])
    between = np.mean([dist[i][j] for i in t_idx for j in o_idx])
    return within / between if between > 0 else np.nan

print("\nWheatsheaf Index:")
feature_cols = ['gc_content', 'gene_count', 'te_count', 'genome_size_mb']
X = StandardScaler().fit_transform(df[feature_cols].values)
for group in df['group'].unique():
    ws = wheatsheaf_index(X, df['group'].values, group)
    if not np.isnan(ws):
        status = "STRONG" if ws < 0.8 else "MODERATE" if ws < 1.0 else "DIVERGENCE"
        print(f"  {group}: WS={ws:.3f} ({status})")
print("\nDone!")
