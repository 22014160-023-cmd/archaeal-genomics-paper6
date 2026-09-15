"""
================================================================
Archaeal genomics analysis pipeline
Paper: Genomic Convergence and Divergence in Archaeal Extremophiles
Authors: Nadia Zeeshan, Isma Abid, Zariab Ahmed
Department of Biochemistry and Biotechnology
University of Gujrat, Pakistan
================================================================

Run order:
  1. Download + extract features  →  archaeal_data/archaeal_genomes_features.csv
  2. Statistics + 20 figures      →  figures/Figure01..20*.png
  3. Preview + zip figures
  4. Generate tables              →  tables/Table1..8*.{csv,tex}
"""

# ============================================================
# SECTION 1 — IMPORTS & GLOBAL CONFIG
# ============================================================
import os
import time
import zipfile
import warnings
from getpass import getpass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import seaborn as sns

from scipy import stats
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE

from Bio import Entrez, SeqIO
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------- Output directories ----------
OUTDIR = "archaeal_data"
FIGDIR = "figures"
TABDIR = "tables"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

# ---------- Plot style ----------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ---------- Consistent palette ----------
COLORS = {
    'thermophile': '#C0392B',
    'halophile':   '#E67E22',
    'acidophile':  '#16A085',
    'mesophile':   '#2C3E50',
}
GROUPS = ['thermophile', 'halophile', 'acidophile', 'mesophile']
GROUP_LABELS = [g.capitalize() for g in GROUPS]

FEATURES = ['gc_content', 'gene_count', 'te_count',
            'genome_size_mb', 'avg_gene_length', 'gc3']
FEATURE_LABELS = {
    'gc_content':      'GC content (%)',
    'gene_count':      'Gene count',
    'te_count':        'TE count',
    'genome_size_mb':  'Genome size (MB)',
    'avg_gene_length': 'Avg gene length (bp)',
    'gc3':             'GC3',
}


# ============================================================
# SECTION 2 — DOWNLOAD GENOMES FROM NCBI + EXTRACT FEATURES
# ============================================================
Entrez.email = "nadia.zeeshan@uog.edu.pk"
Entrez.tool  = "archaeal_genomics_paper6"

THERMOPHILE_GENERA = [
    "Pyrococcus","Thermococcus","Sulfolobus","Pyrobaculum","Thermoplasma",
    "Methanocaldococcus","Methanothermococcus","Methanopyrus","Ignicoccus",
    "Nanoarchaeum","Acidilobus","Aeropyrum","Desulfurococcus","Hyperthermus",
    "Staphylothermus","Pyrodictium","Pyrolobus","Thermofilum","Thermoproteus",
    "Vulcanisaeta","Caldivirga","Archaeoglobus","Ferroglobus","Geoglobus",
    "Methanotorris","Methanococcus","Palaeococcus",
]
HALOPHILE_GENERA = [
    "Halobacterium","Haloarcula","Halococcus","Haloferax","Halorubrum",
    "Haloterrigena","Natronomonas","Natrialba","Halobiforma","Halostagnicola",
    "Halorhabdus","Halalkalicoccus","Halopiger","Halosimplex","Halovivax",
    "Natronococcus","Natronorubrum","Halogeometricum","Haladaptatus",
]
ACIDOPHILE_GENERA = [
    "Ferroplasma","Picrophilus","Acidianus","Metallosphaera",
    "Stygiolobus","Sulfurisphaera",
]
MESOPHILE_GENERA = [
    "Methanobrevibacter","Methanosphaera","Methanobacterium","Methanoregula",
    "Methanocorpusculum","Methanoculleus","Methanosaeta","Methanosarcina",
    "Methanomassiliicoccus","Nitrosopumilus","Nitrososphaera","Cenarchaeum",
]


def classify_environment(organism):
    genus = organism.split()[0] if organism else ""
    for lst, label in [(THERMOPHILE_GENERA, "thermophile"),
                       (HALOPHILE_GENERA,   "halophile"),
                       (ACIDOPHILE_GENERA,  "acidophile"),
                       (MESOPHILE_GENERA,   "mesophile")]:
        for g in lst:
            if genus == g or genus.startswith(g):
                return label
    return None


def search_archaeal_genomes(retmax=400):
    query = ('archaea[Organism] AND "complete genome"[Assembly Level] '
             'AND "latest refseq"[Filter]')
    print("Querying NCBI Assembly database...")
    handle = Entrez.esearch(db="assembly", term=query, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    print(f"Found {len(ids)} assembly records")
    return ids


def fetch_assembly_summary(ids, chunk=100):
    rows = []
    for i in range(0, len(ids), chunk):
        sub = ids[i:i+chunk]
        handle = Entrez.esummary(db="assembly", id=",".join(sub))
        try:
            summary = Entrez.read(handle)
        except Exception as e:
            print(f"  chunk {i} failed: {e}")
            continue
        finally:
            handle.close()
        docs = summary["DocumentSummarySet"]["DocumentSummary"]
        for d in docs:
            rows.append({
                "assembly_id":   d.get("AssemblyAccession", ""),
                "organism":      d.get("Organism", ""),
                "assembly_name": d.get("AssemblyName", ""),
                "assembly_level":d.get("AssemblyStatus", ""),
                "ftp_path":      d.get("FtpPath_RefSeq", "") or d.get("FtpPath_GenBank", ""),
                "species_taxid": d.get("SpeciesTaxid", ""),
                "submitter":     d.get("SubmitterOrganization", ""),
                "biosample":     d.get("BioSampleAccn", ""),
            })
        time.sleep(0.4)
    return pd.DataFrame(rows)


def download_gbff(ftp_path, outdir):
    if not ftp_path:
        return None
    base = ftp_path.replace("ftp://", "https://")
    asm  = os.path.basename(ftp_path)
    url  = f"{base}/{asm}_genomic.gbff.gz"
    out  = os.path.join(outdir, f"{asm}.gbff.gz")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return out
    try:
        import urllib.request
        urllib.request.urlretrieve(url, out)
        return out
    except Exception as e:
        print(f"  download failed for {asm}: {e}")
        return None


def extract_features(gbff_path):
    import gzip
    opener = gzip.open if gbff_path.endswith(".gz") else open
    features = {"genome_size_mb": None, "gc_content": None,
                "gene_count": 0, "te_count": 0,
                "avg_gene_length": None, "gc3": None}
    try:
        with opener(gbff_path, "rt", errors="ignore") as fh:
            record = next(SeqIO.parse(fh, "genbank"))
    except Exception as e:
        print(f"  parse failed: {e}")
        return None
    seq = str(record.seq).upper()
    n   = len(seq)
    if n == 0:
        return None
    features["genome_size_mb"] = n / 1e6
    features["gc_content"] = (seq.count("G") + seq.count("C")) / n * 100

    cds_lengths = []
    te_count    = 0
    third_pos   = []
    te_keywords = ("transposase", "insertion sequence", "is element",
                   "integrase", "transposon")

    for feat in record.features:
        if feat.type == "CDS":
            features["gene_count"] += 1
            try:
                start = int(feat.location.start)
                end   = int(feat.location.end)
                cds_lengths.append(end - start)
                cds = seq[start:end]
                L = (len(cds)//3)*3
                cds = cds[:L]
                for i in range(2, L, 3):
                    third_pos.append(cds[i])
            except Exception:
                continue
        qual_text = " ".join(
            str(v).lower()
            for vals in feat.qualifiers.values()
            for v in (vals if isinstance(vals, list) else [vals])
        )
        if any(k in qual_text for k in te_keywords):
            te_count += 1

    features["te_count"] = te_count
    if cds_lengths:
        features["avg_gene_length"] = float(np.mean(cds_lengths))
    if third_pos:
        features["gc3"] = (third_pos.count("G") + third_pos.count("C")) / len(third_pos)
    return features


def run_pipeline(max_genomes=300):
    ids = search_archaeal_genomes(retmax=max_genomes)
    if not ids:
        return pd.DataFrame()
    meta = fetch_assembly_summary(ids)
    meta["environment"] = meta["organism"].apply(classify_environment)
    meta = meta.dropna(subset=["environment"]).reset_index(drop=True)
    print(f"After classification: {len(meta)} genomes")
    print(meta["environment"].value_counts())

    rows = []
    for _, r in tqdm(meta.iterrows(), total=len(meta)):
        gbff = download_gbff(r["ftp_path"], OUTDIR)
        if not gbff:
            continue
        feats = extract_features(gbff)
        if not feats:
            continue
        rows.append({"assembly_id": r["assembly_id"],
                     "organism":    r["organism"],
                     "environment": r["environment"],
                     **feats})
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["genome_size_mb","gc_content","gene_count",
                           "avg_gene_length","gc3"]).reset_index(drop=True)
    df = df[(df["genome_size_mb"] > 0.01) & (df["gene_count"] > 0)]
    return df


print("=" * 60)
print("STEP 1 — DOWNLOAD + FEATURE EXTRACTION")
print("=" * 60)
df_raw = run_pipeline(max_genomes=300)
csv_path = os.path.join(OUTDIR, "archaeal_genomes_features.csv")
df_raw.to_csv(csv_path, index=False)
print(f"Saved raw dataset: {csv_path} ({len(df_raw)} genomes)")


# ============================================================
# SECTION 3 — REBALANCE + STATISTICS + 20 FIGURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 — REBALANCE + STATISTICS + FIGURES")
print("=" * 60)

np.random.seed(42)
CAP = 30
samples = []
for g in GROUPS:
    sub = df_raw[df_raw['environment'] == g]
    if len(sub) > CAP:
        sub = sub.sample(CAP, random_state=42)
    samples.append(sub)
df = pd.concat(samples, ignore_index=True)

print(f"Rebalanced dataset: {len(df)} genomes (cap = {CAP} per group)")
print(df['environment'].value_counts())

FEATURES = ['gc_content','gene_count','te_count',
            'genome_size_mb','avg_gene_length','gc3']
FEATURE_LABELS_LIST = [FEATURE_LABELS[f] for f in FEATURES]

X = df[FEATURES].values
X_std = StandardScaler().fit_transform(X)
labels = df['environment'].values

# ---------- Kruskal-Wallis ----------
print("\nKRUSKAL-WALLIS")
kw_results = {}
for f, lab in zip(FEATURES, FEATURE_LABELS_LIST):
    groups = [df[df['environment']==g][f].values for g in GROUPS]
    H, p = stats.kruskal(*groups)
    kw_results[f] = (H, p)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  {lab:<22} H = {H:8.3f}  p = {p:.4g}  {sig}")

# ---------- Spearman correlations ----------
corr = df[FEATURES].corr('spearman')
corr.columns = FEATURE_LABELS_LIST
corr.index   = FEATURE_LABELS_LIST
print("\nSPEARMAN CORRELATIONS")
print(corr.round(3))

# ---------- Wheatsheaf Index ----------
def wheatsheaf(Xs, labs, target):
    within  = Xs[labs == target]
    between = Xs[labs != target]
    if len(within) < 2 or len(between) < 2:
        return np.nan
    return pdist(within,'euclidean').mean() / pdist(between,'euclidean').mean()

WS = {g: wheatsheaf(X_std, labels, g) for g in GROUPS}
print("\nWHEATSHEAF INDEX")
for g in GROUPS:
    print(f"  {g:<14} WS = {WS[g]:.3f}")

# ---------- Silhouette ----------
sil_global = silhouette_score(X_std, labels)
sil_per_sample = silhouette_samples(X_std, labels)
print(f"\nSilhouette: {sil_global:.3f}")

# ---------- PCA ----------
pca = PCA(n_components=min(6, len(FEATURES)))
pcs = pca.fit_transform(X_std)
var = pca.explained_variance_ratio_ * 100
print(f"PCA: PC1 = {var[0]:.1f}%, PC2 = {var[1]:.1f}%, total = {var[0]+var[1]:.1f}%")

# ---------- Save stats summary ----------
with open(os.path.join(OUTDIR, "stats_summary.txt"), "w") as fh:
    fh.write("KRUSKAL-WALLIS\n")
    for f,(H,p) in kw_results.items():
        fh.write(f"{f}\tH={H:.4f}\tp={p:.4g}\n")
    fh.write("\nWHEATSHEAF\n")
    for g,v in WS.items():
        fh.write(f"{g}\t{v:.4f}\n")
    fh.write(f"\nSILHOUETTE\t{sil_global:.4f}\n")
    fh.write(f"PCA\tPC1={var[0]:.2f}\tPC2={var[1]:.2f}\n")

# ---------- Helper: save a figure ----------
figure_files = []
def save_fig(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    figure_files.append(name)
    print(f"  Saved {name}")

def add_ellipse(ax, x, y, color, n_std=2.0, alpha=0.15):
    if len(x) < 3: return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2*n_std*np.sqrt(vals)
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)),
                         width=w, height=h, angle=angle,
                         facecolor=color, alpha=alpha,
                         edgecolor=color, lw=1))


# ---------- VIEW 1 — Dataset & Distribution ----------
print("\nVIEW 1 — Dataset & Distribution")

# Figure 1 — group sizes
fig, ax = plt.subplots(figsize=(6,4))
sizes = [(df['environment']==g).sum() for g in GROUPS]
bars = ax.bar(GROUP_LABELS, sizes, color=[COLORS[g] for g in GROUPS],
              edgecolor='black', linewidth=0.8)
for b, s in zip(bars, sizes):
    ax.text(b.get_x()+b.get_width()/2, s+0.5, str(s), ha='center', fontsize=10)
ax.set_ylabel('Number of genomes')
ax.set_title(f'Dataset composition (n = {len(df)})')
plt.tight_layout(); save_fig(fig, "Figure01_group_sizes.png")

# Figure 2 — genome size histograms
fig, axes = plt.subplots(2,2, figsize=(9,7))
for i, g in enumerate(GROUPS):
    ax = axes.flat[i]
    sub = df[df['environment']==g]['genome_size_mb']
    ax.hist(sub, bins=12, color=COLORS[g], edgecolor='white', alpha=0.9)
    ax.axvline(sub.mean(), color='black', linestyle='--', lw=1.2,
               label=f'mean = {sub.mean():.2f}')
    ax.set_title(f'{g.capitalize()} (n={len(sub)})')
    ax.set_xlabel('Genome size (MB)'); ax.set_ylabel('Count')
    ax.legend(fontsize=8)
plt.suptitle('Genome size distribution per group', y=1.00, fontweight='bold')
plt.tight_layout(); save_fig(fig, "Figure02_genome_size_hist.png")

# Figure 3 — gene count histograms
fig, axes = plt.subplots(2,2, figsize=(9,7))
for i, g in enumerate(GROUPS):
    ax = axes.flat[i]
    sub = df[df['environment']==g]['gene_count']
    ax.hist(sub, bins=12, color=COLORS[g], edgecolor='white', alpha=0.9)
    ax.axvline(sub.mean(), color='black', linestyle='--', lw=1.2,
               label=f'mean = {sub.mean():.0f}')
    ax.set_title(f'{g.capitalize()} (n={len(sub)})')
    ax.set_xlabel('Gene count'); ax.set_ylabel('Count')
    ax.legend(fontsize=8)
plt.suptitle('Gene count distribution per group', y=1.00, fontweight='bold')
plt.tight_layout(); save_fig(fig, "Figure03_gene_count_hist.png")

# Figure 4 — GC content histograms
fig, axes = plt.subplots(2,2, figsize=(9,7))
for i, g in enumerate(GROUPS):
    ax = axes.flat[i]
    sub = df[df['environment']==g]['gc_content']
    ax.hist(sub, bins=12, color=COLORS[g], edgecolor='white', alpha=0.9)
    ax.axvline(sub.mean(), color='black', linestyle='--', lw=1.2,
               label=f'mean = {sub.mean():.1f}%')
    ax.set_title(f'{g.capitalize()} (n={len(sub)})')
    ax.set_xlabel('GC content (%)'); ax.set_ylabel('Count')
    ax.legend(fontsize=8)
plt.suptitle('GC content distribution per group', y=1.00, fontweight='bold')
plt.tight_layout(); save_fig(fig, "Figure04_gc_content_hist.png")


# ---------- VIEW 2 — Comparative Genomic Features ----------
print("\nVIEW 2 — Comparative Genomic Features")

# Figure 5 — boxplots
fig, axes = plt.subplots(2,2, figsize=(10,7))
for i, (f, lab) in enumerate(zip(FEATURES[:4], FEATURE_LABELS_LIST[:4])):
    ax = axes.flat[i]
    sns.boxplot(data=df, x='environment', y=f, order=GROUPS,
                palette=[COLORS[g] for g in GROUPS], ax=ax, showmeans=True,
                meanprops={'marker':'D','markerfacecolor':'white',
                           'markeredgecolor':'black','markersize':6})
    ax.set_title(lab); ax.set_xlabel(''); ax.set_xticklabels(GROUP_LABELS)
    ax.grid(axis='y', alpha=0.25)
plt.suptitle('Boxplots of genomic features', y=1.00, fontweight='bold')
plt.tight_layout(); save_fig(fig, "Figure05_boxplots.png")

# Figure 6 — violins
fig, axes = plt.subplots(2,2, figsize=(10,7))
for i, (f, lab) in enumerate(zip(FEATURES[:4], FEATURE_LABELS_LIST[:4])):
    ax = axes.flat[i]
    sns.violinplot(data=df, x='environment', y=f, order=GROUPS,
                   palette=[COLORS[g] for g in GROUPS], ax=ax, inner='quartile')
    ax.set_title(lab); ax.set_xlabel(''); ax.set_xticklabels(GROUP_LABELS)
    ax.grid(axis='y', alpha=0.25)
plt.suptitle('Violin plots of genomic features', y=1.00, fontweight='bold')
plt.tight_layout(); save_fig(fig, "Figure06_violins.png")

# Figure 7 — ridge plot (gene count)
fig, ax = plt.subplots(figsize=(8,5))
y_positions = np.arange(len(GROUPS))
for i, g in enumerate(GROUPS):
    vals = df[df['environment']==g]['gene_count'].values
    if len(vals) < 3: continue
    kde = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 200)
    ys = kde(xs); ys = ys / ys.max() * 0.8
    ax.fill_between(xs, i, i+ys, color=COLORS[g], alpha=0.7,
                    edgecolor='black', linewidth=0.6)
ax.set_yticks(y_positions); ax.set_yticklabels(GROUP_LABELS)
ax.set_xlabel('Gene count')
ax.set_title('Ridge plot — gene count per group')
plt.tight_layout(); save_fig(fig, "Figure07_ridge_gene_count.png")

# Figure 8 — strip plots
fig, axes = plt.subplots(2,3, figsize=(12,7))
for i, (f, lab) in enumerate(zip(FEATURES, FEATURE_LABELS_LIST)):
    ax = axes.flat[i]
    sns.stripplot(data=df, x='environment', y=f, order=GROUPS,
                  palette=[COLORS[g] for g in GROUPS], ax=ax,
                  jitter=0.25, size=4, alpha=0.7,
                  edgecolor='black', linewidth=0.3)
    ax.set_title(lab); ax.set_xlabel('')
    ax.set_xticklabels(GROUP_LABELS, rotation=20)
    ax.grid(axis='y', alpha=0.25)
plt.suptitle('All genomes — strip plots', y=1.00, fontweight='bold')
plt.tight_layout(); save_fig(fig, "Figure08_strip_all.png")


# ---------- VIEW 3 — Multivariate Structure ----------
print("\nVIEW 3 — Multivariate Structure")

# Figure 9 — PCA with ellipses
fig, ax = plt.subplots(figsize=(8,6))
for g in GROUPS:
    m = labels == g
    ax.scatter(pcs[m,0], pcs[m,1], c=COLORS[g], label=g.capitalize(),
               s=55, alpha=0.8, edgecolors='black', linewidths=0.5)
    add_ellipse(ax, pcs[m,0], pcs[m,1], COLORS[g], n_std=2.0)
ax.set_xlabel(f'PC1 ({var[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var[1]:.1f}%)')
ax.set_title(f'PCA of archaeal genomes (n={len(df)})')
ax.legend(loc='upper right'); ax.grid(alpha=0.25)
plt.tight_layout(); save_fig(fig, "Figure09_pca_ellipses.png")

# Figure 10 — PCA with loadings
fig, ax = plt.subplots(figsize=(8,6))
for g in GROUPS:
    m = labels == g
    ax.scatter(pcs[m,0], pcs[m,1], c=COLORS[g], label=g.capitalize(),
               s=45, alpha=0.7, edgecolors='black', linewidths=0.4)
scale = 1.4 * np.abs(pcs[:,:2]).max()
for i, lab in enumerate(FEATURE_LABELS_LIST):
    x = pca.components_[0,i] * scale
    y = pca.components_[1,i] * scale
    ax.annotate('', xy=(x,y), xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.text(x*1.08, y*1.08, lab, fontsize=8, color='dimgray')
ax.set_xlabel(f'PC1 ({var[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var[1]:.1f}%)')
ax.set_title('PCA with feature loadings')
ax.legend(loc='upper right'); ax.grid(alpha=0.25)
plt.tight_layout(); save_fig(fig, "Figure10_pca_loadings.png")

# Figure 11 — t-SNE
tsne = TSNE(n_components=2, perplexity=min(20, len(df)-1),
            random_state=42, init='pca', learning_rate='auto')
emb = tsne.fit_transform(X_std)
fig, ax = plt.subplots(figsize=(8,6))
for g in GROUPS:
    m = labels == g
    ax.scatter(emb[m,0], emb[m,1], c=COLORS[g], label=g.capitalize(),
               s=55, alpha=0.8, edgecolors='black', linewidths=0.5)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title(f't-SNE embedding (n={len(df)})')
ax.legend(loc='upper right'); ax.grid(alpha=0.25)
plt.tight_layout(); save_fig(fig, "Figure11_tsne.png")

# Figure 12 — dendrogram
fig, ax = plt.subplots(figsize=(10,4))
Z = linkage(X_std, method='ward')
dendrogram(Z, ax=ax, no_labels=True, color_threshold=0)
ax.set_title('Hierarchical clustering of genomes (Ward linkage)')
ax.set_ylabel('Distance')
plt.tight_layout(); save_fig(fig, "Figure12_dendrogram.png")


# ---------- VIEW 4 — Correlations & Relationships ----------
print("\nVIEW 4 — Correlations & Relationships")

# Figure 13 — correlation heatmap
fig, ax = plt.subplots(figsize=(8,6.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, ax=ax,
            cbar_kws={'label':'Spearman correlation'})
ax.set_title('Spearman correlations among genomic features')
plt.tight_layout(); save_fig(fig, "Figure13_correlation_heatmap.png")

# Figure 14 — pairwise scatter (corner)
plot_df = df[['environment'] + FEATURES[:4]].copy()
plot_df['environment'] = pd.Categorical(plot_df['environment'],
                                        categories=GROUPS, ordered=True)
g_obj = sns.pairplot(plot_df, hue='environment', hue_order=GROUPS,
                     palette=[COLORS[g] for g in GROUPS],
                     diag_kind='kde', corner=False, height=1.8,
                     plot_kws={'alpha':0.75,'s':25,
                               'edgecolor':'black','linewidth':0.3})
g_obj.fig.suptitle('Pairwise scatter matrix', y=1.01, fontweight='bold')
plt.savefig(os.path.join(FIGDIR, "Figure14_pairwise_scatter.png"),
            dpi=300, bbox_inches='tight')
plt.close(g_obj.fig)
figure_files.append("Figure14_pairwise_scatter.png")
print("  Saved Figure14_pairwise_scatter.png")

# Figure 15 — gene count vs genome size
fig, ax = plt.subplots(figsize=(7,5))
for g in GROUPS:
    m = df['environment']==g
    ax.scatter(df.loc[m,'gene_count'], df.loc[m,'genome_size_mb'],
               c=COLORS[g], label=g.capitalize(), s=45, alpha=0.8,
               edgecolors='black', linewidths=0.4)
    if m.sum() >= 3:
        z = np.polyfit(df.loc[m,'gene_count'], df.loc[m,'genome_size_mb'], 1)
        xs = np.linspace(df.loc[m,'gene_count'].min(),
                         df.loc[m,'gene_count'].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color=COLORS[g], lw=1.5, alpha=0.8)
ax.set_xlabel('Gene count'); ax.set_ylabel('Genome size (MB)')
ax.set_title('Gene count vs. genome size')
ax.legend(loc='lower right'); ax.grid(alpha=0.25)
plt.tight_layout(); save_fig(fig, "Figure15_gene_vs_size.png")

# Figure 16 — GC vs GC3
fig, ax = plt.subplots(figsize=(7,5))
for g in GROUPS:
    m = df['environment']==g
    ax.scatter(df.loc[m,'gc_content'], df.loc[m,'gc3'],
               c=COLORS[g], label=g.capitalize(), s=45, alpha=0.8,
               edgecolors='black', linewidths=0.4)
r, _ = stats.spearmanr(df['gc_content'], df['gc3'])
ax.set_xlabel('GC content (%)'); ax.set_ylabel('GC3')
ax.set_title(f'GC content vs. GC3  (Spearman r = {r:.2f})')
ax.legend(loc='lower right'); ax.grid(alpha=0.25)
plt.tight_layout(); save_fig(fig, "Figure16_gc_vs_gc3.png")


# ---------- VIEW 5 — Convergence & Environmental Signal ----------
print("\nVIEW 5 — Convergence & Environmental Signal")

# Figure 17 — Wheatsheaf Index
fig, ax = plt.subplots(figsize=(7,5))
names = [g.capitalize() for g in GROUPS]
vals  = [WS[g] for g in GROUPS]
bars = ax.bar(names, vals, color=[COLORS[g] for g in GROUPS],
              edgecolor='black', linewidth=0.8)
ax.axhline(1.0, color='red', linestyle='--', lw=1.5,
           label='Divergence threshold (1.0)')
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}',
            ha='center', fontsize=10)
ax.set_ylabel('Wheatsheaf Index')
ax.set_title('Wheatsheaf Index by group')
ax.set_ylim(0, max(1.2, max(vals)*1.15))
ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.25)
plt.tight_layout(); save_fig(fig, "Figure17_wheatsheaf.png")

# Figure 18 — silhouette plot
fig, ax = plt.subplots(figsize=(7,5))
y_lower = 10
for g in GROUPS:
    m = labels == g
    sv = np.sort(sil_per_sample[m])
    size = sv.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sv,
                     facecolor=COLORS[g], edgecolor=COLORS[g], alpha=0.8)
    ax.text(-0.05, y_lower + size/2, g.capitalize(), fontsize=9)
    y_lower = y_upper + 10
ax.axvline(sil_global, color='red', linestyle='--', lw=1.5,
           label=f'Global = {sil_global:.3f}')
ax.set_xlabel('Silhouette coefficient'); ax.set_yticks([])
ax.set_title('Silhouette plot per group')
ax.legend(loc='lower right')
plt.tight_layout(); save_fig(fig, "Figure18_silhouette.png")

# Figure 19 — distance to centroid
fig, ax = plt.subplots(figsize=(7,5))
dist_data = []
for g in GROUPS:
    m = labels == g
    centroid = X_std[m].mean(axis=0)
    d = np.linalg.norm(X_std[m] - centroid, axis=1)
    for v in d:
        dist_data.append({'group': g.capitalize(), 'distance': v})
dist_df = pd.DataFrame(dist_data)
sns.boxplot(data=dist_df, x='group', y='distance',
            palette=[COLORS[g] for g in GROUPS], ax=ax)
ax.set_xlabel(''); ax.set_ylabel('Euclidean distance to centroid')
ax.set_title('Within-group dispersion')
ax.grid(axis='y', alpha=0.25)
plt.tight_layout(); save_fig(fig, "Figure19_centroid_distance.png")

# Figure 20 — radar chart
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, polar=True)
z = StandardScaler().fit_transform(df[FEATURES])
zdf = pd.DataFrame(z, columns=FEATURES); zdf['environment'] = df['environment']
n_feat = len(FEATURES)
angles = np.linspace(0, 2*np.pi, n_feat, endpoint=False).tolist()
angles += angles[:1]
for g in GROUPS:
    means = zdf[zdf['environment']==g][FEATURES].mean().values.tolist()
    means += means[:1]
    ax.plot(angles, means, color=COLORS[g], lw=2, label=g.capitalize())
    ax.fill(angles, means, color=COLORS[g], alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(FEATURE_LABELS_LIST, fontsize=9)
ax.set_title('Mean standardized feature profile per group', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout(); save_fig(fig, "Figure20_radar.png")

print(f"\nAll {len(figure_files)} figures saved to {FIGDIR}/")


# ============================================================
# SECTION 4 — PREVIEW GRID + ZIP FIGURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 — PREVIEW + ZIP FIGURES")
print("=" * 60)

png_files = sorted([os.path.join(FIGDIR, f) for f in os.listdir(FIGDIR)
                    if f.startswith("Figure") and f.endswith(".png")])
fig, axes = plt.subplots(5, 4, figsize=(20, 24))
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i < len(png_files):
        img = mpimg.imread(png_files[i])
        ax.imshow(img)
        ax.set_title(os.path.basename(png_files[i]).replace('.png',''),
                     fontsize=10, fontweight='bold')
    ax.axis('off')
plt.suptitle("Archaeal genomics — 20 figures (5 views × 4 panels)",
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
preview_path = "figures_preview_grid.png"
plt.savefig(preview_path, dpi=80, bbox_inches='tight')
plt.close()
print(f"Preview grid saved: {preview_path}")

zip_figs = "archaeal_figures.zip"
with zipfile.ZipFile(zip_figs, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in png_files:
        zf.write(f, arcname=os.path.basename(f))
    for extra in ["archaeal_genomes_features.csv", "stats_summary.txt"]:
        p = os.path.join(OUTDIR, extra)
        if os.path.exists(p):
            zf.write(p, arcname=extra)
    if os.path.exists(preview_path):
        zf.write(preview_path, arcname=os.path.basename(preview_path))
print(f"Created {zip_figs}")


# ============================================================
# SECTION 5 — GENERATE ALL MANUSCRIPT TABLES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 — TABLES")
print("=" * 60)

def save_table(tdf, name, caption, label, float_fmt="%.3f"):
    tdf.to_csv(os.path.join(TABDIR, f"{name}.csv"))
    latex = tdf.to_latex(
        caption=caption, label=f"tab:{label}",
        float_format=float_fmt, escape=False,
        column_format="l" + "c"*(tdf.shape[1]-1),
    )
    with open(os.path.join(TABDIR, f"{name}.tex"), "w") as fh:
        fh.write(latex)
    print(f"  Saved {name}.csv / .tex")

# --- Table 1 — Dataset composition ---
rows = []
for g in GROUPS:
    sub = df[df['environment'] == g]
    rows.append({'Group': g.capitalize(),
                 'n': len(sub),
                 'Percent': f"{100*len(sub)/len(df):.1f}%"})
t1 = pd.DataFrame(rows)
t1.loc[len(t1)] = {'Group': 'Total', 'n': len(df), 'Percent': '100.0%'}
save_table(t1, "Table1_dataset_composition",
           "Dataset composition by environmental class.",
           "dataset_composition", float_fmt="%d")

# --- Table 2 — Summary statistics ---
summary_rows = []
for g in GROUPS:
    sub = df[df['environment'] == g]
    row = {'Group': g.capitalize(), 'n': len(sub)}
    for f in FEATURES:
        m = sub[f].mean(); s = sub[f].std()
        if f == 'gc3':
            row[FEATURE_LABELS[f]] = f"{m:.3f} ± {s:.3f}"
        elif f in ('genome_size_mb',):
            row[FEATURE_LABELS[f]] = f"{m:.2f} ± {s:.2f}"
        else:
            row[FEATURE_LABELS[f]] = f"{m:.0f} ± {s:.0f}"
    summary_rows.append(row)
t2 = pd.DataFrame(summary_rows)
save_table(t2, "Table2_summary_statistics",
           "Summary statistics (mean ± SD) per environmental group.",
           "summary_statistics", float_fmt="%s")

# --- Table 3 — Kruskal-Wallis ---
kw_rows = []
for f in FEATURES:
    H, p = kw_results[f]
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    kw_rows.append({'Feature': FEATURE_LABELS[f],
                    'H statistic': f"{H:.3f}",
                    'p value': f"{p:.3g}",
                    'Significance': sig})
t3 = pd.DataFrame(kw_rows)
save_table(t3, "Table3_kruskal_wallis",
           "Kruskal-Wallis tests across environmental groups.",
           "kruskal_wallis", float_fmt="%s")

# --- Table 4 — Mann-Whitney U (pairwise) ---
mw_rows = []
for f in FEATURES:
    for i, g1 in enumerate(GROUPS):
        for g2 in GROUPS[i+1:]:
            a = df[df['environment']==g1][f].values
            b = df[df['environment']==g2][f].values
            if len(a) < 2 or len(b) < 2: continue
            U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            n1, n2 = len(a), len(b)
            r = 2*U/(n1*n2) - 1
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            mw_rows.append({'Feature': FEATURE_LABELS[f],
                            'Comparison': f"{g1.capitalize()} vs {g2.capitalize()}",
                            'U': f"{U:.1f}", 'p': f"{p:.3g}",
                            'r': f"{r:+.3f}", 'Sig': sig})
t4 = pd.DataFrame(mw_rows)
save_table(t4, "Table4_mann_whitney",
           "Pairwise Mann-Whitney U tests.",
           "mann_whitney", float_fmt="%s")

# --- Table 5 — Correlations ---
corr_rounded = corr.round(3)
corr_rounded.insert(0, 'Feature', corr_rounded.index)
save_table(corr_rounded, "Table5_correlations",
           "Spearman rank correlations among genomic features.",
           "correlations", float_fmt="%.3f")

# --- Table 6 — Wheatsheaf Index & silhouette ---
ws_rows = []
for g in GROUPS:
    mask = labels == g
    ws_rows.append({'Group': g.capitalize(),
                    'n': int(mask.sum()),
                    'Wheatsheaf Index': f"{WS[g]:.3f}",
                    'Mean silhouette': f"{sil_per_sample[mask].mean():.3f}",
                    'Interpretation': 'Convergence' if WS[g] < 1 else 'Divergence'})
ws_rows.append({'Group': 'Global', 'n': len(df),
                'Wheatsheaf Index': '—',
                'Mean silhouette': f"{sil_global:.3f}",
                'Interpretation': 'Overall separation'})
t6 = pd.DataFrame(ws_rows)
save_table(t6, "Table6_wheatsheaf_silhouette",
           "Wheatsheaf Index (WS) and mean silhouette per group.",
           "wheatsheaf_silhouette", float_fmt="%s")

# --- Table 7 — PCA loadings ---
loadings = pd.DataFrame(
    pca.components_.T,
    index=[FEATURE_LABELS[f] for f in FEATURES],
    columns=['PC1','PC2','PC3']
).round(3)
loadings.insert(0, 'Feature', loadings.index)
loadings.loc[len(loadings)] = ['Variance explained (%)',
                                f"{var[0]:.1f}",
                                f"{var[1]:.1f}",
                                f"{var[2]:.1f}"]
save_table(loadings, "Table7_pca_loadings",
           "PCA loadings for the top three components.",
           "pca_loadings", float_fmt="%.3f")

# --- Table 8 — Full genome list ---
t8 = df[['assembly_id','organism','environment',
         'gc_content','gene_count','te_count','genome_size_mb',
         'avg_gene_length','gc3']].copy()
t8.columns = ['Assembly ID','Organism','Environment',
              'GC content (%)','Gene count','TE count',
              'Genome size (MB)','Avg gene length (bp)','GC3']
t8 = t8.sort_values(['Environment','Organism']).reset_index(drop=True)
t8.to_csv(os.path.join(TABDIR, "Table8_genome_list.csv"), index=False)
print("  Saved Table8_genome_list.csv")

# Zip tables
zip_tables = "archaeal_tables.zip"
with zipfile.ZipFile(zip_tables, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fn in os.listdir(TABDIR):
        zf.write(os.path.join(TABDIR, fn), arcname=fn)
print(f"Created {zip_tables}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Data:    {OUTDIR}/")
print(f"  Figures: {FIGDIR}/  ({len(figure_files)} figures)")
print(f"  Tables:  {TABDIR}/  (8 tables)")