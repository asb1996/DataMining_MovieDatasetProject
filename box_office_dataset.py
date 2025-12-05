import pandas as pd
import numpy as np
import re
import os
import kagglehub
import ast


#datasets
path1 = kagglehub.dataset_download("aditya126/movies-box-office-dataset-2000-2024")
path2 = kagglehub.dataset_download("stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset")
path3 = kagglehub.dataset_download("utkarshx27/movies-dataset")

print("Path 1:", path1)
print("Path 2:", path2)
print("Path 3:", path3)

print(os.listdir(path1))
print(os.listdir(path2))
print(os.listdir(path3))


# Load CSVs

# Box office data
box = pd.read_csv(os.path.join(path1, "enhanced_box_office_data(2000-2024)u.csv"))

# Rotten Tomatoes MOVIES file (this is the one with ratings + genres)
rt_movies = pd.read_csv(os.path.join(path2, "rotten_tomatoes_movies.csv"))

# Budget / revenue dataset (utkarshx27)
budget = pd.read_csv(os.path.join(path3, "movie_dataset.csv"))

print(box.columns)
print(rt_movies.columns)
print(budget.columns)



# Prep budget dataset for cast/lead actor info


# Parse release_date to get year
budget["release_date_parsed"] = pd.to_datetime(budget["release_date"], errors="coerce")
budget["year"] = budget["release_date_parsed"].dt.year



#Clean / rename box office data
# Keep only 2015–2024
box = box[(box["Year"] >= 2015) & (box["Year"] <= 2024)].copy()

# Clean money columns: strip '$' and ',' and cast to float
for col in ["$Worldwide", "$Domestic", "$Foreign"]:
    box[col] = (
        box[col]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

# Rename to nicer names
box = box.rename(
    columns={
        "Release Group": "title",
        "Year": "year",
        "$Worldwide": "worldwide_gross",
        "$Domestic": "domestic_gross",
        "$Foreign": "international_gross",
        "Genres": "genres_box",
    }
)

# title cleaning for joins
def clean_title(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

box["title_clean"] = box["title"].apply(clean_title)

# Clean / filter RT movies
rt_movies["movie_title_clean"] = rt_movies["movie_title"].apply(clean_title)


# year from original_release_date 
rt_movies["original_release_date_parsed"] = pd.to_datetime(
    rt_movies["original_release_date"], errors="coerce"
)
rt_movies["year"] = rt_movies["original_release_date_parsed"].dt.year

# Filter to same year range
rt_movies = rt_movies[(rt_movies["year"] >= 2015) & (rt_movies["year"] <= 2024)].copy()

# using RT genres 
rt_small = rt_movies[
    [
        "movie_title_clean",
        "year",
        "tomatometer_rating",
        "tomatometer_count",
        "genres",              # RT genres
        "original_release_date_parsed",
        "directors",
        "runtime",
    ]
].drop_duplicates(subset=["movie_title_clean", "year"])


# Merge box office + RT

df = box.merge(
    rt_small,
    left_on=["title_clean", "year"],
    right_on=["movie_title_clean", "year"],
    how="inner",
)

print("Merged shape (box + RT):", df.shape)
print(df[["title", "year", "worldwide_gross", "tomatometer_rating"]].head())

# Merge in BUDGETS (utkarshx27 movie_dataset.csv)
# budget columns include: 'budget', 'revenue', 'release_date', 'title', 'runtime', 'director', ...
# 1) Clean numeric money columns (budget, revenue)
for col in ["budget", "revenue"]:
    budget[col] = (
        budget[col]
        .astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

#Parse release_date -> year
budget["release_date_parsed"] = pd.to_datetime(budget["release_date"], errors="coerce")
budget["year"] = budget["release_date_parsed"].dt.year

# Filter to same 2015–2024 range 
budget = budget[(budget["year"] >= 2015) & (budget["year"] <= 2024)].copy()

#Clean title
budget["title_clean"] = budget["title"].apply(clean_title)

#Keeping only relevant columns for merge
budget_small = budget[
    [
        "title_clean",
        "year",
        "budget",   # production budget
        "revenue",  # TMDb worldwide revenue (backup)
    ]
].drop_duplicates(subset=["title_clean", "year"])

# Merge into df (left join to keep our box+RT rows)
df = df.merge(
    budget_small,
    on=["title_clean", "year"],
    how="left",
    suffixes=("", "_tmdb"),
)

print("Merged shape (with budgets):", df.shape)
print(df[["title", "year", "budget", "revenue"]].head())

budget["title_clean"] = budget["title"].apply(clean_title)

# Extract lead actor from 'cast' column
def extract_lead_actor(cast_str):
   
    if not isinstance(cast_str, str) or cast_str.strip() == "":
        return np.nan

    try:
        cast_list = ast.literal_eval(cast_str)
        if isinstance(cast_list, list) and len(cast_list) > 0:
            # Sort by 'order' if present; otherwise just use the first entry
            try:
                cast_list = sorted(cast_list, key=lambda x: x.get("order", 0))
            except Exception:
                pass
            first = cast_list[0]
            name = first.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    except Exception:
        pass

    
    parts = [p.strip() for p in cast_str.split("|") if p.strip()]
    if len(parts) > 0:
        return parts[0]
    return np.nan

budget["lead_actor"] = budget["cast"].apply(extract_lead_actor)

# Keep only the columns we need for merging into df
lead_small = budget[["title_clean", "year", "lead_actor"]].dropna().drop_duplicates(
    subset=["title_clean", "year"]
)


# Merge lead actor info
if "title_clean" not in df.columns:
    df["title_clean"] = df["title"].apply(clean_title)

df = df.merge(
    lead_small,
    on=["title_clean", "year"],
    how="left"
)

print("Merged shape (with lead_actor):", df.shape)
print(df[["title", "year", "lead_actor"]].head())


# Computing production_budget, profit, ROI
# Use utkarsh 'budget' as production_budget
df["production_budget"] = df["budget"]

# If Box Office Mojo worldwide_gross is missing, fall back to TMDb revenue
# Profit = worldwide - production_budget
df["worldwide_gross_final"] = np.where(
    df["worldwide_gross"].notna(),
    df["worldwide_gross"],
    df["revenue"]
)


df["worldwide_gross"] = df["worldwide_gross_final"]

df["profit_proxy"] = df["worldwide_gross"] - df["production_budget"]

# Avoid divide-by-zero for ROI
denom = df["production_budget"].replace(0, np.nan)
df["roi_proxy"] = df["profit_proxy"] / denom


# Profit = worldwide - production_budget
df["profit_proxy"] = df["worldwide_gross_final"] - df["production_budget"]

# Avoid divide-by-zero for ROI
denom = df["production_budget"].replace(0, np.nan)
df["roi_proxy"] = df["profit_proxy"] / denom

# Oscars flag for lead actor
oscars_path = kagglehub.dataset_download("unanimad/the-oscar-award")
oscars_file = os.path.join(oscars_path, "the_oscar_award.csv")
oscars = pd.read_csv(oscars_file)

# acting categories only
acting_cats = [
    "ACTOR IN A LEADING ROLE",
    "ACTOR IN A SUPPORTING ROLE",
    "ACTRESS IN A LEADING ROLE",
    "ACTRESS IN A SUPPORTING ROLE",
]

oscars["category_up"] = oscars["category"].astype(str).str.upper()
oscars["name_clean"] = oscars["name"].astype(str).str.strip().str.lower()

winner_names = set(
    oscars.loc[
        (oscars["winner"] == True) & (oscars["category_up"].isin(acting_cats)),
        "name_clean",
    ]
)

# Map Oscars to lead actor
df["lead_actor_clean"] = df["lead_actor"].astype(str).str.strip().str.lower()
df["lead_actor_has_oscar"] = df["lead_actor_clean"].isin(winner_names).astype(int)



# Actor tier (A/B/C), trending, and star power score
actor_df = df.dropna(subset=["lead_actor"]).copy()

actor_stats = actor_df.groupby("lead_actor").agg(
    n_movies=("title", "nunique"),
    total_worldwide=("worldwide_gross", "sum"),
    first_year=("year", "min"),
    has_oscar=("lead_actor_has_oscar", "max"),  # 0 or 1
)

# percentile rank by worldwide box office
actor_stats["gross_rank_pct"] = actor_stats["total_worldwide"].rank(pct=True)

def map_tier(pct):
    if pct >= 0.90:
        return "A"
    elif pct >= 0.70:
        return "B"
    else:
        return "C"

actor_stats["lead_actor_tier"] = actor_stats["gross_rank_pct"].apply(map_tier)

def map_status(row):
    if row["n_movies"] <= 2 and row["first_year"] >= 2018:
        return "trending"
    else:
        return "well_known"

actor_stats["lead_actor_status"] = actor_stats.apply(map_status, axis=1)

# star power score (0–100) 
# combine: box office percentile + Oscar bonus
# score = 100 * (0.7 * gross_rank_pct + 0.3 * has_oscar)
actor_stats["star_power_score"] = 100.0 * (
    0.7 * actor_stats["gross_rank_pct"] + 0.3 * actor_stats["has_oscar"]
)

# Map back to df
tier_map = actor_stats["lead_actor_tier"].to_dict()
status_map = actor_stats["lead_actor_status"].to_dict()
star_map = actor_stats["star_power_score"].to_dict()

df["lead_actor_tier"] = df["lead_actor"].map(tier_map)
df["lead_actor_status"] = df["lead_actor"].map(status_map)
df["star_power_score"] = df["lead_actor"].map(star_map)




# Flags: animated / live-action / franchise / sequel
genre_col = "genres"  # RT genres

def is_animated(genres):
    if not isinstance(genres, str):
        return 0
    return int("animation" in genres.lower())

df["is_animated"] = df[genre_col].apply(is_animated)
df["is_live_action"] = 1 - df["is_animated"]

franchise_keywords = [
    "star wars",
    "avengers",
    "jurassic",
    "fast & furious",
    "fast and furious",
    "harry potter",
    "marvel",
    "mission: impossible",
    "transformers",
    "spider-man",
    "spider man",
    "batman",
    "superman",
    "x-men",
    "x men",
]

def is_franchise_title(title):
    t = title.lower()
    return any(k in t for k in franchise_keywords)

sequel_pattern = re.compile(
    r"\b(2|3|4|5|6|7|8|9|10|ii|iii|iv|v|vi|vii|viii|ix|x|part\s+\d+)\b",
    flags=re.IGNORECASE,
)

def is_sequel_title(title):
    return bool(sequel_pattern.search(title))

df["is_franchise"] = df["title"].apply(is_franchise_title).astype(int)
df["is_sequel"] = df["title"].apply(is_sequel_title).astype(int)
df["is_standalone"] = ((df["is_franchise"] == 0) & (df["is_sequel"] == 0)).astype(int)




# final columns 
final_cols = [
    "title",
    "year",
    "original_release_date_parsed",
    "domestic_gross",
    "international_gross",
    "worldwide_gross",
    "production_budget",
    "profit_proxy",
    "roi_proxy",
    "tomatometer_rating",
    "tomatometer_count",
    "genres",
    "runtime",
    "directors",
    "lead_actor",
    "lead_actor_has_oscar",
    "lead_actor_tier",
    "lead_actor_status",
    "star_power_score",        
    "is_animated",
    "is_live_action",
    "is_franchise",
    "is_sequel",
    "is_standalone",
]



final_df = df[final_cols].copy()

final_df.to_csv("box_office_success_2015_2024.csv", index=False)

print("Saved box_office_success_2015_2024.csv with shape:", final_df.shape)
