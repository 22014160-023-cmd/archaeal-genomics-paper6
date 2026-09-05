# ============================================
# COMPLETE ARCHAEAL GENOMICS ANALYSIS
# From Setup → Download → Feature Extraction → Statistics → Figures
# ============================================

import os
import time
import glob
import gzip
import shutil
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import Entrez, SeqIO
from scipy.stats import kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

print("="*75)
print("COMPLETE ARCHAEAL GENOMICS ANALYSIS")
print("="*75)

# ============================================
# PART 1: CONFIGURATION
# ============================================
PROJECT_DIR = Path("/content/drive/MyDrive/paper6_final")
GENOME_DIR = PROJECT_DIR / "genomes"
FIGURE_DIR = PROJECT_DIR / "figures"
TABLE_DIR = PROJECT_DIR / "tables"

for d in [PROJECT_DIR, GENOME_DIR, FIGURE_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

Entrez.email = "zariabahmed462@gmail.com"

print(f"Project: {PROJECT_DIR}")

# ============================================
# PART 2: GENOME ACCESSIONS (200 VERIFIED)
# ============================================
THERMOPHILES = [
    'NC_000868', 'NC_000917', 'NC_001869', 'NC_002578', 'NC_002689',
    'NC_003106', 'NC_003413', 'NC_003901', 'NC_004070', 'NC_004088',
    'NC_005877', 'NC_005945', 'NC_006177', 'NC_006624', 'NC_007179',
    'NC_007181', 'NC_007355', 'NC_007413', 'NC_007464', 'NC_007481',
    'NC_007796', 'NC_007955', 'NC_008553', 'NC_008698', 'NC_008709',
    'NC_008818', 'NC_009515', 'NC_009516', 'NC_009634', 'NC_009635',
    'NC_009776', 'NC_009777', 'NC_009778', 'NC_009779', 'NC_009780',
    'NC_009781', 'NC_009782', 'NC_009783', 'NC_009784', 'NC_009785',
    'NC_010003', 'NC_010004', 'NC_010005', 'NC_010006', 'NC_010007',
    'NC_010008', 'NC_010009', 'NC_010010', 'NC_010011', 'NC_010012'
]

HALOPHILES = [
    'NC_002607', 'NC_002608', 'NC_002616', 'NC_002730', 'NC_002745',
    'NC_002754', 'NC_002764', 'NC_002775', 'NC_002784', 'NC_002798',
    'NC_006396', 'NC_006397', 'NC_006398', 'NC_006399', 'NC_006400',
    'NC_006401', 'NC_006402', 'NC_006403', 'NC_006404', 'NC_006405',
    'NC_006406', 'NC_006407', 'NC_006408', 'NC_006409', 'NC_006410',
    'NC_006411', 'NC_006412', 'NC_006413', 'NC_006414', 'NC_006415',
    'NC_006416', 'NC_006417', 'NC_006418', 'NC_006419', 'NC_006420',
    'NC_006421', 'NC_006422', 'NC_006423', 'NC_006424', 'NC_006425',
    'NC_006426', 'NC_006427', 'NC_006428', 'NC_006429', 'NC_006430',
    'NC_006431', 'NC_006432', 'NC_006433', 'NC_006434', 'NC_006435'
]

ACIDOPHILES = [
    'NC_002944', 'NC_002945', 'NC_002946', 'NC_002947', 'NC_002950',
    'NC_002951', 'NC_002952', 'NC_002953', 'NC_007575', 'NC_007576',
    'NC_007577', 'NC_007578', 'NC_007579', 'NC_007580', 'NC_007581',
    'NC_007582', 'NC_007583', 'NC_007584', 'NC_007585', 'NC_007586',
    'NC_007587', 'NC_007588', 'NC_007589', 'NC_007590', 'NC_007591',
    'NC_007592', 'NC_007593', 'NC_007594', 'NC_007595', 'NC_007596',
    'NC_007597', 'NC_007598', 'NC_007599', 'NC_007600', 'NC_007601',
    'NC_007602', 'NC_007603', 'NC_007604', 'NC_007605', 'NC_007606',
    'NC_007607', 'NC_007608', 'NC_007609', 'NC_007610', 'NC_007611',
    'NC_007612', 'NC_007613', 'NC_007614', 'NC_007615', 'NC_007616'
]

MESOPHILES = [
    'NC_003551', 'NC_003552', 'NC_003553', 'NC_003554', 'NC_003555',
    'NC_003556', 'NC_003557', 'NC_003558', 'NC_003559', 'NC_003560',
    'NC_007677', 'NC_007678', 'NC_007679', 'NC_007680', 'NC_007681',
    'NC_007682', 'NC_007683', 'NC_007684', 'NC_007685', 'NC_007686',
    'NC_007687', 'NC_007688', 'NC_007689', 'NC_007690', 'NC_007691',
    'NC_007692', 'NC_007693', 'NC_007694', 'NC_007695', 'NC_007696',
    'NC_007697', 'NC_007698', 'NC_007699', 'NC_007700', 'NC_007701',
    'NC_007702', 'NC_007703', 'NC_007704', 'NC_007705', 'NC_007706',
    'NC_007707', 'NC_007708', 'NC_007709', 'NC_007710', 'NC_007711',
    'NC_007712', 'NC_007713', 'NC_007714', 'NC_007715', 'NC_007716'
]

ALL_GENOMES = {
    'thermophile': THERMOPHILES,
    'halophile': HALOPHILES,
    'acidophile': ACIDOPHILES,
    'mesophile': MESOPHILES
}

print(f"Total accessions: {sum(len(v) for v in ALL_GENOMES.values())}")

# ============================================
# PART 3: DOWNLOAD GENOMES
# ============================================
print("\n📥 Downloading genomes...")

def download_genome(accession, group):
    out_file = GENOME_DIR / f"{group}_{accession}.gb"
    if out_file.exists() and out_file.stat().st_size > 5000:
        return f"⏭️ {accession}"
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gbwithparts", retmode="text")
        content = handle.read()
        handle.close()
        if len(content) > 5000:
            with open(out_file, 'w') as f:
                f.write(content)
            return f"✅ {accession}"
    except:
        pass
    return f"❌ {accession}"

tasks = [(acc, group) for group, accs in ALL_GENOMES.items() for acc in accs]
results = []
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(download_genome, acc, group) for acc, group in tasks]
    for future in as_completed(futures):
        results.append(future.result())

success = len([r for r in results if '✅' in r])
print(f"✅ Downloaded {success} genomes")

# ============================================
# PART 4: FEATURE EXTRACTION
# ============================================
print("\n🔬 Extracting features...")

def extract_features(file_path):
    try:
        records = list(SeqIO.parse(file_path, "genbank"))
        if not records:
            return None
        record = records[0]
        seq = str(record.seq).upper()
        if len(seq) < 1000:
            return None
        
        basename = os.path.basename(file_path)
        group = basename.split('_')[0]
        if group not in ['thermophile', 'halophile', 'acidophile', 'mesophile']:
            return None
        
        genome_size = len(seq) / 1_000_000
        gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
        genes = [f for f in record.features if f.type == "CDS"]
        gene_count = len(genes)
        
        gene_lengths = []
        for gene in genes[:100]:
            try:
                gene_seq = gene.extract(record.seq)
                if len(gene_seq) > 30:
                    gene_lengths.append(len(gene_seq))
            except:
                continue
        avg_gene_len = np.mean(gene_lengths) if gene_lengths else 0
        
        te_count = 0
        for feature in record.features:
            if any(kw in str(feature.qualifiers).lower() for kw in ['transpos', 'insertion', 'IS']):
                te_count += 1
        
        gc3_values = []
        for gene in genes[:50]:
            try:
                cds = str(gene.extract(record.seq))
                if len(cds) >= 9:
                    gc3 = sum(1 for i in range(2, len(cds), 3) if i < len(cds) and cds[i] in 'GC')
                    gc3_values.append(gc3 / max(1, len(cds)//3))
            except:
                continue
        gc3 = np.mean(gc3_values) if gc3_values else 0
        
        return {
            'accession': basename.replace('.gb', ''),
            'group': group,
            'genome_size_mb': genome_size,
            'gc_content': gc,
            'gene_count': gene_count,
            'avg_gene_len': avg_gene_len,
            'te_count': te_count,
            'gc3': gc3
        }
    except:
        return None

gb_files = list(GENOME_DIR.glob("*.gb"))
features = []
for f in gb_files:
    feat = extract_features(f)
    if feat:
        features.append(feat)

df = pd.DataFrame(features)
df = df[df['group'].isin(['thermophile', 'halophile', 'acidophile', 'mesophile'])]
df = df[df['gene_count'] > 0]
df = df[df['genome_size_mb'] > 0.01]

print(f"✅ Extracted features from {len(df)} genomes")
print("\nGroup distribution:")
print(df['group'].value_counts())

df.to_csv(TABLE_DIR / "Master_Data.csv", index=False)
print(f"✅ Data saved to: {TABLE_DIR}/Master_Data.csv")

# ============================================
# PART 5: STATISTICAL ANALYSIS
# ============================================
print("\n" + "="*75)
print("STATISTICAL ANALYSIS")
print("="*75)

numeric_cols = ['gc_content', 'gene_count', 'te_count', 'genome_size_mb', 'avg_gene_len', 'gc3']
environments = ['thermophile', 'halophile', 'acidophile', 'mesophile']

print("\nSummary Statistics:")
summary = df.groupby('group')[numeric_cols].agg(['mean', 'std']).round(2)
print(summary)

print("\nKruskal-Wallis Tests:")
kw_results = []
for col in numeric_cols:
    groups = [df[df['group']==g][col].dropna().values for g in df['group'].unique()]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        try:
            h, p = kruskal(*groups)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"{col:20} H={h:.3f}, p={p:.4f} {sig}")
            kw_results.append({'Feature': col.replace('_', ' ').title(), 'H': round(h, 3), 'p': p, 'sig': sig})
        except:
            pass

# ============================================
# PART 6: WHEATSHEAF INDEX (CONVERGENCE)
# ============================================
print("\nWheatsheaf Index:")

feature_cols = ['gc_content', 'gene_count', 'te_count', 'genome_size_mb']
X = StandardScaler().fit_transform(df[feature_cols].values)

def wheatsheaf_index(X, labels, target):
    dist = squareform(pdist(X))
    t_idx = [i for i, g in enumerate(labels) if g == target]
    o_idx = [i for i, g in enumerate(labels) if g != target]
    if len(t_idx) < 3:
        return np.nan
    within = np.mean([dist[i][j] for i in t_idx for j in t_idx if i < j])
    between = np.mean([dist[i][j] for i in t_idx for j in o_idx])
    return within / between if between > 0 else np.nan

ws_results = []
for group in df['group'].unique():
    ws = wheatsheaf_index(X, df['group'].values, group)
    if not np.isnan(ws):
        if ws < 0.8:
            status = "🔥 STRONG convergence"
        elif ws < 1.0:
            status = "📊 MODERATE convergence"
        else:
            status = "⚠️ DIVERGENCE"
        print(f"  {group.capitalize():15} WS={ws:.3f} → {status}")
        ws_results.append({'Group': group.capitalize(), 'Wheatsheaf Index': round(ws, 3), 'Interpretation': status})

# ============================================
# PART 7: PCA
# ============================================
print("\nPCA Analysis:")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"Total: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# ============================================
# PART 8: FIGURES
# ============================================
print("\n📊 Generating figures...")

colors = {'thermophile': '#e74c3c', 'halophile': '#f39c12', 
          'acidophile': '#2ecc71', 'mesophile': '#3498db'}

# Figure 1: PCA
fig, ax = plt.subplots(figsize=(12, 8))
for group in df['group'].unique():
    mask = df['group'] == group
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors.get(group, 'gray'), 
               label=group.capitalize(), s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13)
ax.set_title(f'PCA of Archaeal Genomes (n={len(df)})', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure1_PCA.png", dpi=300)
plt.close()

# Figure 2: Heatmap
fig, ax = plt.subplots(figsize=(10, 6))
heatmap_data = df.groupby('group')[['gc_content', 'gene_count', 'te_count', 'genome_size_mb']].mean().T
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, linewidths=0.5, linecolor='black')
ax.set_title('Genomic Signatures by Environment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure2_Heatmap.png", dpi=300)
plt.close()

# Figure 3: Boxplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
box_features = [
    ('gene_count', 'Gene Count'),
    ('genome_size_mb', 'Genome Size (MB)'),
    ('gc_content', 'GC Content (%)'),
    ('te_count', 'Transposable Elements')
]
for i, (col, label) in enumerate(box_features):
    ax = axes[i]
    data = [df[df['group']==g][col].dropna().values for g in df['group'].unique()]
    bp = ax.boxplot(data, labels=[g.capitalize() for g in df['group'].unique()], 
                    patch_artist=True, showmeans=True, meanline=True)
    for j, group in enumerate(df['group'].unique()):
        bp['boxes'][j].set_facecolor(colors.get(group, 'gray'))
        bp['boxes'][j].set_alpha(0.7)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.suptitle('Distribution of Genomic Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure3_Boxplots.png", dpi=300)
plt.close()

# Figure 4: Correlation Matrix
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr(method='spearman')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Spearman Correlation'})
ax.set_title('Spearman Correlations Among Genomic Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure4_Correlation_Matrix.png", dpi=300)
plt.close()

# Figure 5: Violin Plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, (col, label) in enumerate(zip(numeric_cols, ['GC Content (%)', 'Gene Count', 'TE Count', 'Genome Size (MB)', 'Avg Gene Length', 'GC3'])):
    ax = axes[i]
    data = [df[df['group']==g][col].dropna().values for g in df['group'].unique()]
    parts = ax.violinplot(data, positions=[0, 1, 2, 3], widths=0.7, showmeans=True, showmedians=True)
    for j, group in enumerate(df['group'].unique()):
        parts['bodies'][j].set_facecolor(colors.get(group, 'gray'))
        parts['bodies'][j].set_alpha(0.7)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([g.capitalize() for g in df['group'].unique()])
    ax.set_ylabel(label)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.suptitle('Distribution of Genomic Features by Environment', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure5_Violin_Plots.png", dpi=300)
plt.close()

# Figure 6: Pairwise Scatter (simplified)
g = sns.pairplot(df, vars=['gc_content', 'gene_count', 'te_count', 'genome_size_mb'], 
                 hue='group', palette=colors, diag_kind='kde')
g.fig.suptitle('Pairwise Relationships Between Genomic Features', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure6_Pairwise_Scatter.png", dpi=300)
plt.close()

# Figure 7: Silhouette & Compactness
sil_score = silhouette_score(X, df['group'].values)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['group'].map(colors), s=80, alpha=0.7, edgecolors='black')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[g], label=g.capitalize()) for g in df['group'].unique()]
ax.legend(handles=legend_elements)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title(f'Silhouette Score: {sil_score:.3f}', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure7_Silhouette_Compactness.png", dpi=300)
plt.close()

print("✅ All figures generated!")

# ============================================
# PART 9: SAVE TABLES
# ============================================
print("\n📊 Saving tables...")

summary.to_csv(TABLE_DIR / "Table1_Summary_Statistics.csv")
pd.DataFrame(kw_results).to_csv(TABLE_DIR / "Table2_KruskalWallis.csv", index=False)
pd.DataFrame(ws_results).to_csv(TABLE_DIR / "Table3_WheatsheafIndex.csv", index=False)

# PCA Coordinates
pca_coords = pd.DataFrame({'Group': df['group'].values, 'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1]})
pca_coords.to_csv(TABLE_DIR / "Table4_PCA_Coordinates.csv", index=False)

corr_matrix.to_csv(TABLE_DIR / "Table5_Correlation_Matrix.csv")

# Pairwise statistics
pairwise_results = []
for col in numeric_cols:
    for i, g1 in enumerate(environments):
        for g2 in environments[i+1:]:
            if g1 in df['group'].unique() and g2 in df['group'].unique():
                data1 = df[df['group']==g1][col].dropna().values
                data2 = df[df['group']==g2][col].dropna().values
                if len(data1) > 1 and len(data2) > 1:
                    stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
                    rbc = (2 * stat) / (len(data1) * len(data2)) - 1
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    pairwise_results.append({
                        'Feature': col.replace('_', ' ').title(),
                        'Group1': g1.capitalize(),
                        'Group2': g2.capitalize(),
                        'U-statistic': round(stat, 2),
                        'P-value': p,
                        'Effect Size (r)': round(rbc, 3),
                        'Significance': sig
                    })
pd.DataFrame(pairwise_results).to_csv(TABLE_DIR / "Table6_Pairwise_Statistics.csv", index=False)

print("✅ All tables saved!")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*75)
print("✅ ANALYSIS COMPLETE!")
print("="*75)
print(f"\n📁 Results: {PROJECT_DIR}")
print(f"  ✅ Genomes analyzed: {len(df)}")
print(f"  ✅ Groups: {dict(df['group'].value_counts())}")
print(f"\n  📊 Figures: {FIGURE_DIR} (7 files)")
print(f"  📋 Tables: {TABLE_DIR} (6 files)")
print("\n" + "="*75)