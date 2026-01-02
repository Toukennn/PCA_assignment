# -*- coding: utf-8 -*-
### Team Members (Alphabetical Order):
# 1. Khatibi, Amir Reza (s360765);"""

StudentIDs = [360765]

import numpy as np
import pandas as pd
from IPython.display import display

var_entertainment_feat_types = ['Interests', 'Movies', 'Music']
var_personal_feat_types = ['Finance', 'Phobias']
fixed_feat_types = ['Personality', 'Health']

label_types = ['Demographic']

variables_by_type = {
    'Demographics': ['Age', 'Height', 'Weight', 'Number of siblings',
                     'Gender', 'Hand', 'Education', 'Only child', 'Home Town Type',
                     'Home Type'],
    'Finance': ['Finances', 'Shopping centres', 'Branded clothing',
                'Entertainment spending', 'Spending on looks',
                'Spending on gadgets', 'Spending on healthy eating'],
    'Health': ['Smoking', 'Alcohol', 'Healthy eating'],
    'Interests': ['History', 'Psychology', 'Politics', 'Mathematics',
                  'Physics', 'Internet', 'PC', 'Economy Management',
                  'Biology', 'Chemistry', 'Reading', 'Geography',
                  'Foreign languages', 'Medicine', 'Law', 'Cars',
                  'Art exhibitions', 'Religion', 'Countryside, outdoors',
                  'Dancing', 'Musical instruments', 'Writing', 'Passive sport',
                  'Active sport', 'Gardening', 'Celebrities', 'Shopping',
                  'Science and technology', 'Theatre', 'Fun with friends',
                  'Adrenaline sports', 'Pets'],
    'Movies': ['Movies', 'Horror', 'Thriller', 'Comedy', 'Romantic',
               'Sci-fi', 'War', 'Fantasy/Fairy tales', 'Animated',
               'Documentary', 'Western', 'Action'],
    'Music': ['Music', 'Slow songs or fast songs', 'Dance', 'Folk',
              'Country', 'Classical music', 'Musical', 'Pop', 'Rock',
              'Metal or Hardrock', 'Punk', 'Hiphop, Rap', 'Reggae, Ska',
              'Swing, Jazz', 'Rock n roll', 'Alternative', 'Latino',
              'Techno, Trance', 'Opera'],
    'Personality': ['Daily events', 'Prioritising workload',
                    'Writing notes', 'Workaholism', 'Thinking ahead',
                    'Final judgement', 'Reliability', 'Keeping promises',
                    'Loss of interest', 'Friends versus money', 'Funniness',
                    'Fake', 'Criminal damage', 'Decision making', 'Elections',
                    'Self-criticism', 'Judgment calls', 'Hypochondria',
                    'Empathy', 'Eating to survive', 'Giving',
                    'Compassion to animals', 'Borrowed stuff',
                    'Loneliness', 'Cheating in school', 'Health',
                    'Changing the past', 'God', 'Dreams', 'Charity',
                    'Number of friends', 'Punctuality', 'Lying', 'Waiting',
                    'New environment', 'Mood swings', 'Appearence and gestures',
                    'Socializing', 'Achievements', 'Responding to a serious letter',
                    'Children', 'Assertiveness', 'Getting angry',
                    'Knowing the right people', 'Public speaking',
                    'Unpopularity', 'Life struggles', 'Happiness in life',
                    'Energy levels', 'Small - big dogs', 'Personality',
                    'Finding lost valuables', 'Getting up', 'Interests or hobbies',
                    "Parents' advice", 'Questionnaires or polls', 'Internet usage'],
    'Phobias': ['Flying', 'Storm', 'Darkness', 'Heights', 'Spiders', 'Snakes',
                'Rats', 'Ageing', 'Dangerous dogs', 'Fear of public speaking']
}

labels = variables_by_type['Demographics']
features_all = []
for tt in variables_by_type.keys():
    if tt != 'Demographics':
        features_all += variables_by_type[tt]

def which_features(*StudentIDs):
    random_seed = min(StudentIDs)
    np.random.seed(random_seed)
    features_ = np.random.choice(features_all, int((2 * len(features_all)) / 3), replace=False).tolist()
    features = []
    features_by_type = {tt: [] for tt in variables_by_type.keys() if tt != 'Demographics'}
    for tt in variables_by_type.keys():
        ft_list = variables_by_type[tt]
        for ii in range(len(ft_list)):
            if ft_list[ii] in features_:
                features.append(ft_list[ii])
                features_by_type[tt].append(ft_list[ii])

    return features, features_by_type

features, features_by_type = which_features(*StudentIDs)

print(f'*** THESE ARE THE {len(features)} SELECTED FEATURES (SEE VARIABLE features):')
for ff in features:
    print(f'{ff}')
print('*************************************')
print('')
print('*** SELECTED FEATURES BY TYPES (SEE VARIABLE features_by_type):')
for tt in features_by_type.keys():
    print(f'{tt}: {features_by_type[tt]}')
    print('')
print('*************************************')
print('')
print('*** THESE ARE THE LABELS (SEE VARIABLE labels):')
for ll in labels:
    print(f'{ll}')
print('*************************************')

def which_rows(df, frac, *StudentIDs):
    random_seed = min(StudentIDs)
    df_ = df.sample(frac=frac, random_state=random_seed)
    return df_

responses_hw = pd.read_csv('responses_hw.csv', index_col=0)
responses = which_rows(responses_hw, 0.75, *StudentIDs)
responses = responses.loc[:, features + labels]

responses_ft = responses.loc[:, features]
responses_lb = responses.loc[:, labels]

print('')
print('*** THIS IS YOUR PERSONAL DATASET (features AND labels TOGETHER, SEE VARIABLE responses)')
display(responses)
print('')
print('*** THIS IS YOUR PERSONAL DATASET (features, SEE VARIABLE responses_ft)')
display(responses_ft)
print('')
print('*** THIS IS YOUR PERSONAL DATASET (labels, SEE VARIABLE responses_lb)')
display(responses_lb)

random_seed = min(StudentIDs)
np.random.seed(random_seed)

your_scaler = np.random.choice(['StandardScaler', 'MinMaxScaler'])

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

responses_ft_enc = responses_ft.copy()

ordinal_maps = {
    "Smoking": {"never smoked": 0, "tried smoking": 1, "former smoker": 2, "current smoker": 3},
    "Alcohol": {"never": 0, "social drinker": 1, "drink a lot": 2},
    "Punctuality": {"early": 0, "on time": 1, "late": 2},
    "Lying": {"never": 0, "only to avoid hurting someone": 1, "sometimes": 2, "everytime it suits me": 3},
    "Internet usage": {
        "no time at all": 0,
        "less than an hour a day": 1,
        "few hours a day": 2,
        "most of the day": 3,
    },
    "Gender": {"female": 0, "male": 1},
    "Hand": {"left": 0, "right": 1},
    "Only child": {"no": 0, "yes": 1},
    "Home Town Type": {"village": 0, "city": 1},
    "Home Type": {"block of flats": 0, "house/bungalow": 1},
    "Education": {
        "currently a primary school pupil": 0,
        "primary school": 1,
        "secondary school": 2,
        "college/bachelor degree": 3,
        "masters degree": 4,
        "doctorate degree": 5,
    },
}

for col in responses_ft_enc.columns:
  if col in ordinal_maps:
    responses_ft_enc[col] = responses_ft_enc[col].map(ordinal_maps[col]).astype(float)

if your_scaler == "StandardScaler":
  scaler = StandardScaler()
elif your_scaler == "MinMaxScaler":
  scaler = MinMaxScaler()
else:
  raise ValueError(f"Unexpected scaler: {your_scaler}")

responses_ft_pp = pd.DataFrame(
    scaler.fit_transform(responses_ft_enc),
    index = responses_ft_enc.index,
    columns = responses_ft_enc.columns
)

display(responses_ft_enc.head())
display(responses_ft_pp.head())
print("Scaler used:", your_scaler)

# 1) checking if categorical encoding has introduced any NaNs
na_enc = responses_ft_enc.isna().sum()
if na_enc.sum() > 0:
  display(na_enc[na_enc > 0].sort_values(ascending=False))
  raise ValueError("Some encoded features contain NaNs!")

# 2) comparing scaling effects!
summary = pd.DataFrame({
    "enc_mean": responses_ft_enc.mean(),
    "enc_std": responses_ft_enc.std(ddof=1),
    "pp_mean": responses_ft_pp.mean(),
    "pp_std": responses_ft_pp.std(ddof=1)
})

display(summary.head(10))

var_enc = responses_ft_enc.var(ddof=1)
var_pp = responses_ft_pp.var(ddof=1)

plt.figure()
plt.hist(var_enc.values, bins=30)
plt.xlabel("Variance")
plt.ylabel("Count")
plt.title("Feature variances - encoded dataset")
plt.grid(True)
plt.show()

plt.figure()
plt.hist(var_pp.values, bins=30)
plt.xlabel("Variance")
plt.ylabel("Count")
plt.title(f"Feature variances - preprocessed dataset ({your_scaler})")
plt.grid(True)
plt.show()

top_k = 15
top_enc = var_enc.sort_values(ascending=False).head(top_k)
top_pp = var_pp.sort_values(ascending=False).head(top_k)

plt.figure()
plt.bar(np.arange(top_k), top_enc.values)
plt.xticks(np.arange(top_k))
plt.ylabel("Variance")
plt.title(f"Top-{top_k} variances (encoded)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.bar(np.arange(top_k), top_pp.values)
plt.xticks(np.arange(top_k), top_pp.index, rotation=90)
plt.ylabel("Variance")
plt.title(f"Top-{top_k} variances (preprocessed - {your_scaler})")
plt.tight_layout()
plt.show()

print("Variance summary (encoded):")
display(var_enc.describe())

print("Variance summary (preprocessed):")
display(var_pp.describe())

def pca_numpy(df):
  X = df.to_numpy(dtype=float)
  Xc = X - X.mean(axis=0, keepdims=True)
  C = (Xc.T @ Xc) / (Xc.shape[0]-1)
  eigvals, eigvecs = np.linalg.eigh(C)
  idx = np.argsort(eigvals)[::-1]
  eigvals = eigvals[idx]
  eigvecs = eigvecs[:, idx]
  exp_ratio = eigvals / eigvals.sum()
  return eigvals, eigvecs, exp_ratio

eigvals_enc, eigvecs_enc, exp_ratio_enc = pca_numpy(responses_ft_enc)
eigvals_pp, eigvecs_pp, exp_ratio_pp = pca_numpy(responses_ft_pp)

print("Top 10 explained variance ratios (ENC):", np.round(exp_ratio_enc[:10], 4))
print("Top 10 explained variance ratios (PP): ", np.round(exp_ratio_pp[:10], 4))

cum_enc = np.cumsum(exp_ratio_enc)
cum_pp = np.cumsum(exp_ratio_pp)

plt.figure()
plt.plot(np.arange(1, len(cum_enc)+1), cum_enc, marker="o", linewidth=1, markersize=3)
plt.plot(np.arange(1, len(cum_pp)+1), cum_pp, marker="o", linewidth=1, markersize=3)
plt.axhline(0.33, linewidth=1)
plt.xlabel("Number of PCs")
plt.ylabel("Cumulative explained variance")
plt.legend(["Encoded", "Preprocessed", "33% threshold"])
plt.grid(True)
plt.show()

columns_hw = pd.read_csv('columns_hw.csv')

X = responses_ft_pp.to_numpy(dtype=float)
pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
m = int(np.searchsorted(cum, 0.33) + 1)

print("m (minimum PCs to reach 33% variance):", m)
print("Preserved variance:", cum[m-1])

m = min(m ,5)

# explained variance barplot
plt.figure()
plt.bar(np.arange(1, len(cum)+1), 100*pca.explained_variance_ratio_)
plt.axvline(m, linewidth=1)
plt.xlabel("PC index")
plt.ylabel("Explained variance (%)")
plt.title("Explained variance per PC")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(np.arange(1, len(cum) + 1), cum, marker='o', markersize=3, linewidth=1)
plt.axhline(0.33, linewidth=1)
plt.axvline(m, linewidth=1)
plt.xlabel("Number of PCs")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance")
plt.tight_layout()
plt.show()

pca = PCA(n_components=m).fit(X)

feature_names = list(responses_ft_pp.columns)

try:
    columns_dicts = columns_hw.set_index('short')['original'].to_dict()
except Exception:
    columns_dicts = {}

loadings = pd.DataFrame(
    pca.components_.T,  # (n_features, m)
    index=feature_names,
    columns=[f"PC{i+1}" for i in range(m)]
)

def top_pos_neg(loadings_df, pc, top_k=5):
    v = loadings_df[pc]
    pos = v.sort_values(ascending=False).head(top_k)
    neg = v.sort_values(ascending=True).head(top_k)
    return pos, neg

def pretty_name(f):
    return columns_dicts.get(f, f)

TOP_K = 5
print("\nTop positive/negative features per PC (based on loadings):")
print("-" * 70)
for pc in loadings.columns:
    pos, neg = top_pos_neg(loadings, pc, top_k=TOP_K)
    print(f"\n{pc}")
    print("  Positive features:")
    for f, val in pos.items():
        print(f"    {pretty_name(f)} ({f}): {val:+.3f}")
    print("  Negative features:")
    for f, val in neg.items():
        print(f"    {pretty_name(f)} ({f}): {val:+.3f}")

for pc in loadings.columns:
    v = loadings[pc]
    plt.figure(figsize=(14, 4))
    plt.bar(np.arange(len(v)), v.values)
    plt.xticks(np.arange(len(v)), [pretty_name(f) for f in v.index], rotation=90, fontsize=7)
    plt.ylabel("Loading")
    plt.title(f"{pc}: loadings for all features")
    plt.tight_layout()
    plt.show()

responses_ft_pca = pca.transform(X)
responses_ft_pca = pd.DataFrame(
    responses_ft_pca,
    index=responses_ft_pp.index,
    columns=[f"PC{i+1}" for i in range(m)]
)

display(responses_ft_pca.head())

pc_names = ['Latin organized', 'Fashion enthusiast', 'Humanities lover',
            'Suspected ADHD', 'Cautious alpha man']

X_scaled_df = responses_ft_pp

# Fit PCA with 5 components
pca5 = PCA(n_components=5, random_state=0)
scores5 = pca5.fit_transform(X_scaled_df)

# Explained variance (per PC):
evr = pca5.explained_variance_ratio_
cum_evr = np.cumsum(evr)

labels = [f"PC{i+1}\n({pc_names[i]})" for i in range(5)]

plt.figure(figsize=(10,4))
plt.bar(np.arange(1, 6), evr * 100)
plt.xticks(np.arange(1, 6), labels, rotation=0)
plt.ylabel("Explained variance (%)")
plt.title("Explained variance per PC (scaled data, 5 PCs)")
plt.tight_layout()
plt.show()

# Cumulative explained variance
plt.figure(figsize=(8,4))
plt.plot(np.arange(1, 6), cum_evr, marker='o')
plt.xticks(np.arange(1, 6), labels, rotation=0)
plt.ylim(0, 1.05)
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance (scaled data, 5 PCs)")
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

scores_df = pd.DataFrame(
    scores5,
    columns=[f"PC{i+1}" for i in range(5)],
    index=X_scaled_df.index
)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(scores_df["PC1"], scores_df["PC2"], scores_df["PC3"], s=20)

ax.set_xlabel(f"PC1: {pc_names[0]}")
ax.set_ylabel(f"PC2: {pc_names[1]}")
ax.set_zlabel(f"PC3: {pc_names[2]}")
ax.set_title("3D PCA score plot (scaled data)")

plt.tight_layout()
plt.show()

def analyze_kmeans(df, title, random_seed=0):
    # Reduce dimensions using PCA (limit to 5 PCs or fewer)
    pca = PCA(n_components=min(df.shape[1], 5), random_state=random_seed)
    pca_result = pca.fit_transform(df)

    # Evaluate silhouette scores for k = 3 to 10
    silhouette_scores = []
    for k in range(3, 11):
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
        cluster_labels = kmeans.fit_predict(pca_result)
        score = silhouette_score(pca_result, cluster_labels)
        silhouette_scores.append(score)

    # Determine the best k based on silhouette score
    best_k = int(np.argmax(silhouette_scores) + 3)
    print(f"\nBest number of clusters (k) for {title}: {best_k}")
    print(f"Silhouette Score for {title}: {silhouette_scores[best_k - 3]:.4f}")

    # Fit k-Means using the best k
    kmeans = KMeans(n_clusters=best_k, random_state=random_seed, n_init="auto")
    cluster_labels = kmeans.fit_predict(pca_result)

    # Plot the results with centroids
    m = min(pca_result.shape[1], 3)  # Limit plotting to 2D or 3D
    if m == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    s=200, c='red', marker='*', label='Centroids')
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.title(f'Score Graph with Centroids for {title}')
        plt.legend(); plt.grid(True)
        plt.show()

    elif m == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                   c=cluster_labels, cmap='viridis', alpha=0.7)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
                   s=250, c='red', marker='*', label='Centroids')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
        plt.title(f'Score Graph with Centroids for {title}')
        plt.legend()
        plt.show()

    # Display the centroids
    print(f"\nCluster Centroids for {title} (in PCA space):")
    for i, centroid in enumerate(kmeans.cluster_centers_):
        print(f"Centroid {i+1}: {np.round(centroid, 4)}")

    return kmeans, pca, pca_result, cluster_labels

kmeans_mm, pca_mm, pca_result_mm, labels_mm = analyze_kmeans(
    responses_ft_pp, "MinMax Scaled Data (responses_ft_pp)", random_seed=random_seed
)

## Function used to plot the characteristics of each centroid:

def project_centroids(kmeans, pca_model, original_df, title,
                                  top_print=8, top_plot=25):

    # centroids in PCA-score space
    centroids_pca = kmeans.cluster_centers_

    # back-project to feature space (scaled/encoded feature domain)
    centroids_feat = pca_model.inverse_transform(centroids_pca)
    centroid_df = pd.DataFrame(centroids_feat, columns=original_df.columns)

    print(f"\n==================== {title} ====================")

    for i, centroid in centroid_df.iterrows():
        print(f"\nCentroid {i+1}:")

        top_features = centroid.sort_values(ascending=False).head(top_print)
        bottom_features = centroid.sort_values(ascending=True).head(top_print)

        print("Positively related features (highest reconstructed values):")
        for feature, value in top_features.items():
            print(f"    {feature}: {value:.4f}")

        print("\nNegatively related features (lowest reconstructed values):")
        for feature, value in bottom_features.items():
            print(f"    {feature}: {value:.4f}")

        # Plot: use top |value| features to keep it readable
        top_abs = centroid.reindex(centroid.abs().sort_values(ascending=False).head(top_plot).index)

        plt.figure(figsize=(12, 4))
        plt.bar(np.arange(len(top_abs)), top_abs.values)
        plt.xticks(np.arange(len(top_abs)), top_abs.index, rotation=90)
        plt.title(f"{title} — Centroid {i+1}: top {top_plot} features (by |value|)")
        plt.ylabel("Reconstructed feature value")
        plt.tight_layout()
        plt.show()

    return centroid_df

centroids_mm_df = project_centroids(
    kmeans=kmeans_mm,
    pca_model=pca_mm,            # PCA object returned by analyze_kmeans
    original_df=responses_ft_pp,
    title="MinMax Scaled Data (responses_ft_pp)"
)

## Here I have decided to analyze each centroid based on their PCs and check which PCs characterize that centorid:

centroid_df = pd.DataFrame(
    kmeans_mm.cluster_centers_,
    columns=pc_names

)

# Pretty print with rounding
print("Centroids in PCA space (PC coordinates):\n")
centroid_df.index = range(1, len(centroid_df) + 1)
display(centroid_df.round(2))
print()

for i, row in centroid_df.iterrows():
    plt.figure(figsize=(6, 4))

    plt.bar(pc_names, row.values)
    plt.axhline(0, color='black', linewidth=0.8)

    plt.title(f'Centroid {i+1} — PC profile')
    plt.ylabel('Centroid coordinate')
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.show()

#### Write the code for the visualizations cited in item 2 above:

eval_df = responses_hw.loc[responses_ft_pp.index].copy()
eval_df["cluster"] = labels_mm

def find_label(keyword):
    mask = columns_hw["original"].str.lower().fillna("")
    hits = columns_hw[mask.str.contains(keyword.lower(), regex=False)]
    return hits.iloc[0]["short"] if len(hits) > 0 else None

label_map = {
    "Gender": find_label("gender"),
    "Age": find_label("age"),
    "Education": find_label("education"),
    "Smoking": find_label("smoking"),
    "Online time": find_label("online")
}

# Keep only labels that exist
label_map = {k: v for k, v in label_map.items() if v in eval_df.columns}

print("Selected external labels:")
for k, v in label_map.items():
    print(f"  {k} → {v}")

def plot_categorical(df, col, title):
    # Whole dataset
    df[col].value_counts().plot(kind="bar", figsize=(6,3))
    plt.title(f"{title} — overall distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Per cluster
    pd.crosstab(df["cluster"], df[col], normalize="index").plot(
        kind="bar", stacked=True, figsize=(7,4)
    )

    plt.title(f"{title} — distribution by cluster")
    ax = plt.gca()
    ax.set_xticklabels([1,2,3,4])
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()

def plot_numeric(df, col, title):
    # Whole dataset
    plt.figure(figsize=(6,3))
    plt.hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=20)
    plt.title(f"{title} — overall")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.show()

    # By cluster
    plt.figure(figsize=(6,3))
    df.boxplot(column=col, by="cluster")
    plt.title(f"{title} — by cluster")
    ax = plt.gca()
    ax.set_xticklabels([1,2,3,4])
    plt.suptitle("")
    plt.tight_layout()
    plt.show()

for label_name, col in label_map.items():
  values = eval_df[col]
  numeric_ratio = pd.to_numeric(values, errors="coerce").notna().mean()

  print(f"\n--- {label_name} ---")
  if numeric_ratio > 0.7:
      plot_numeric(eval_df, col, label_name)
  else:
      plot_categorical(eval_df, col, label_name)


import matplotlib.patches as mpatches

def plot_scoregraphs_by_cluster_colored_by_label(
    scores, clusters, label_values, label_name="Label", pc_names=None
):
    clusters = np.asarray(clusters)
    label_series = pd.Series(label_values)

    xlab = "PC1" if pc_names is None else f"PC1: {pc_names[0]}"
    ylab = "PC2" if pc_names is None else f"PC2: {pc_names[1]}"

    numeric = pd.to_numeric(label_series, errors="coerce")
    is_numeric = (numeric.notna().mean() > 0.7) and (label_series.nunique(dropna=True) > 10)

    # Global score graph
    plt.figure(figsize=(7, 5))

    if is_numeric:
        sc = plt.scatter(scores[:, 0], scores[:, 1], c=numeric, s=20, alpha=0.75)
        cbar = plt.colorbar(sc)
        cbar.set_label(label_name)
    else:
        cats = label_series.astype("category")
        codes = cats.cat.codes
        sc = plt.scatter(scores[:, 0], scores[:, 1], c=codes, s=20, alpha=0.75)

        handles = []
        for i, cat in enumerate(cats.cat.categories):
            color = sc.cmap(sc.norm(i))
            handles.append(mpatches.Patch(color=color, label=str(cat)))
        plt.legend(handles=handles, title=label_name, loc="upper right",
                   fontsize=8, title_fontsize=9)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(f"Score graph colored by {label_name}")
    plt.tight_layout()
    plt.show()

    # Cluster-separated graphs
    for c in np.unique(clusters):
        mask = clusters == c
        plt.figure(figsize=(7, 5))

        if is_numeric:
            sc = plt.scatter(scores[mask, 0], scores[mask, 1],
                             c=numeric[mask], s=25, alpha=0.8)
            cbar = plt.colorbar(sc)
            cbar.set_label(label_name)
        else:
            cats = label_series.astype("category")
            codes = cats.cat.codes
            sc = plt.scatter(scores[mask, 0], scores[mask, 1],
                             c=codes[mask], s=25, alpha=0.8)

            handles = []
            for i, cat in enumerate(cats.cat.categories):
                color = sc.cmap(sc.norm(i))
                handles.append(mpatches.Patch(color=color, label=str(cat)))
            plt.legend(handles=handles, title=label_name, loc="upper right",
                       fontsize=8, title_fontsize=9)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(f"Cluster {c+1} — colored by {label_name}")
        plt.tight_layout()
        plt.show()

TARGET_LABELS = ["Gender", "Age", "Education", "Smoking", "Online Time"]

def map_labels_to_columns(columns_hw, target_labels):
    mapping = {}
    originals = columns_hw["original"].str.lower().fillna("")

    for label in target_labels:
        hits = columns_hw[originals.str.contains(label.lower(), regex=False)]
        if len(hits) > 0:
            mapping[label] = hits.iloc[0]["short"]

    return mapping


label_column_map = map_labels_to_columns(columns_hw, TARGET_LABELS)

# PCA scores and cluster labels (already computed)
scores = pca_result_mm
clusters = labels_mm

for label_name, col in label_column_map.items():
    label_values = responses_hw.loc[responses_ft_pp.index, col]

    plot_scoregraphs_by_cluster_colored_by_label(
        scores=scores,
        clusters=clusters,
        label_values=label_values,
        label_name=label_name,
        pc_names=['Latin organized', 'Fashion enthusiast']  # PC1 / PC2
    )

def internal_silhouette_evaluation(X, labels, title="Silhouette analysis"):

    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))

    # 1) Overall silhouette score
    overall = silhouette_score(X, labels)
    print(f"{title}")
    print("-" * 60)
    print(f"Overall silhouette score: {overall:.4f}")

    # 2) Per-sample and per-cluster silhouette
    sample_sil = silhouette_samples(X, labels)

    per_cluster_avg = []
    for c in sorted(np.unique(labels)):
        avg_c = sample_sil[labels == c].mean()
        per_cluster_avg.append((c, avg_c))

    per_cluster_df = pd.DataFrame(per_cluster_avg, columns=["cluster", "avg_silhouette"])
    # Display clusters as 1..K for readability
    per_cluster_df["cluster_display"] = per_cluster_df["cluster"] + 1
    display(per_cluster_df[["cluster_display", "avg_silhouette"]])

    # 3) Silhouette plot (cluster-wise + overall reference)
    fig, ax = plt.subplots(figsize=(8, 5))

    y_lower = 10
    for c in sorted(np.unique(labels)):
        sil_vals = sample_sil[labels == c]
        sil_vals.sort()

        size_c = sil_vals.shape[0]
        y_upper = y_lower + size_c

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            sil_vals,
            alpha=0.8
        )

        ax.text(-0.05, y_lower + 0.5 * size_c, f"Cluster {c+1}")
        y_lower = y_upper + 10

    ax.axvline(overall, linestyle="--", linewidth=1)
    ax.set_title(f"{title} (overall = {overall:.3f})")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Samples (grouped by cluster)")
    ax.set_yticks([])  # cleaner look

    # silhouette scores are in [-1, 1], but set reasonable bounds
    ax.set_xlim([-0.2, 1.0])

    plt.tight_layout()
    plt.show()

    return overall, per_cluster_df

overall_sil, per_cluster_sil_df = internal_silhouette_evaluation(
    X=pca_result_mm,
    labels=labels_mm,
    title="Exercise 6 — Internal evaluation (MinMax / responses_ft_pp)"
)