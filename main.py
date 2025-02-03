from fastapi import FastAPI, Body, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import List, Dict
from jellyfish import jaro_winkler_similarity, levenshtein_distance
from collections import defaultdict

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post("/refine/trim/")
async def trim_columns(columns: List[str] = Form(...)):
    """Trim whitespace from specified columns"""
    try:
        df = pd.read_csv("cleaned_data.csv")
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        df.to_csv("cleaned_data.csv", index=False)
        return {"message": f"Successfully trimmed columns: {', '.join(columns)}"}
    except Exception as e:
        return {"error": f"Error during trimming: {e}"}

@app.post("/refine/fill_blanks/")
async def fill_blanks(columns: List[str] = Form(...)):
    """Fill blank values in specified columns"""
    try:
        df = pd.read_csv("cleaned_data.csv")
        for col in columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].mean())
        df.to_csv("cleaned_data.csv", index=False)
        return {"message": f"Successfully filled blanks in columns: {', '.join(columns)}"}
    except Exception as e:
        return {"error": f"Error during filling blanks: {e}"}

@app.post("/refine/remove_duplicates/")
async def remove_duplicates(columns: List[str] = Form(...)):
    """Remove duplicate rows based on specified columns"""
    try:
        df = pd.read_csv("cleaned_data.csv")
        initial_rows = len(df)
        df = df.drop_duplicates(subset=columns)
        removed_rows = initial_rows - len(df)
        df.to_csv("cleaned_data.csv", index=False)
        return {
            "message": f"Removed {removed_rows} duplicate rows based on columns: {', '.join(columns)}"
        }
    except Exception as e:
        return {"error": f"Error during duplicate removal: {e}"}

@app.post("/refine/cluster/")
async def get_clusters(column: str = Form(...), method: str = Form(...)):
    """Get clusters of similar values in a column using specified method"""
    try:
        df = pd.read_csv("cleaned_data.csv")
        if column not in df.columns:
            return {"error": "Column not found in dataset"}

        values = df[column].dropna().unique()
        clusters = []

        if method == "fingerprint":
            # Simple fingerprint clustering (lowercase, sort chars)
            value_dict = defaultdict(list)
            for val in values:
                key = ''.join(sorted(str(val).lower().replace(' ', '')))
                value_dict[key].append(str(val))
            clusters = [v for v in value_dict.values() if len(v) > 1]

        elif method == "ngram":
            # N-gram similarity clustering
            threshold = 0.8
            processed_clusters = set()
            
            for i, val1 in enumerate(values):
                if str(val1) in processed_clusters:
                    continue
                    
                current_cluster = [str(val1)]
                processed_clusters.add(str(val1))
                
                for val2 in values[i+1:]:
                    if str(val2) not in processed_clusters:
                        similarity = jaro_winkler_similarity(str(val1), str(val2))
                        if similarity > threshold:
                            current_cluster.append(str(val2))
                            processed_clusters.add(str(val2))
                            
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)

        return {
            "clusters": [
                {
                    "values": cluster,
                    "size": len(cluster),
                    "count": sum(df[column].isin(cluster)),
                }
                for cluster in clusters
            ]
        }
    except Exception as e:
        return {"error": f"Error during clustering: {e}"}

@app.post("/refine/merge_clusters/")
async def merge_clusters(
    column: str = Body(...),
    clusters: List[List[str]] = Body(...),
    new_values: List[str] = Body(...)
):
    """Merge clusters of values into new values"""
    try:
        df = pd.read_csv("cleaned_data.csv")
        if column not in df.columns:
            return {"error": "Column not found in dataset"}
        if len(clusters) != len(new_values):  # Crucial check!
            return {"error": "Number of clusters and new values must match"}
        
        for cluster, new_value in zip(clusters, new_values):
            df.loc[df[column].isin(cluster), column] = new_value

        df.to_csv("cleaned_data.csv", index=False)
        return {"message": f"Successfully merged clusters in column {column}"}
    except Exception as e:
        return {"error": f"Error during cluster merging: {e}"}

@app.get("/")
def index():
    return {"name": "First Data"}

@app.post("/randomsample/")
async def generate_samples(samples: int = Form(default=1), ratio: float = Form(default=0.75)):
    try:
        # Load the cleaned dataset
        population = pd.read_csv("cleaned_data.csv")
    except FileNotFoundError:
        return {"error": "Cleaned data file not found. Please clean the data first."}
    except Exception as e:
        return {"error": f"Error loading cleaned data: {e}"}
    
    # Generate train and test samples
    train = population.sample(frac=ratio, random_state=42)
    train.to_csv('1_train.csv', encoding='utf-8', index=False)
    remaining_population = population.loc[~population.index.isin(train.index), :]
    remaining_samples = samples - 1

    for i in range(1, remaining_samples + 1):
        test = remaining_population.sample(frac=i / remaining_samples, random_state=42)
        test.to_csv(f'{i}_test.csv', encoding='utf-8', index=False)
        remaining_population = remaining_population.loc[~remaining_population.index.isin(test.index), :]

    return {"message": f'{samples} sample(s) created using cleaned data'}

@app.post("/sklearnsample/")
async def create_upload_file(response: str = Form(...), algo: str = Form(...), stratify_col: str = Form(default='')):
    try:
        # Load the cleaned dataset
        population = pd.read_csv("cleaned_data.csv")
    except FileNotFoundError:
        return {"error": "Cleaned data file not found. Please clean the data first."}
    except Exception as e:
        return {"error": f"Error loading cleaned data: {e}"}

    try:
        y = population[[response]]
        predictors = list(population.columns)
        predictors.remove(response)
        x = population[predictors]

        if algo == "simple":
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=x[stratify_col])

        x_train.to_csv('x_train.csv', encoding='utf-8', index=False)
        x_test.to_csv('x_test.csv', encoding='utf-8', index=False)
        y_train.to_csv('y_train.csv', encoding='utf-8', index=False)
        y_test.to_csv('y_test.csv', encoding='utf-8', index=False)

        return {"message": "Samples created: x_train, x_test, y_train, y_test using cleaned data"}
    except Exception as e:
        return {"error": f"Error during sampling: {e}"}
    
@app.post("/cleandata/")
async def clean_data(
    csv_file: UploadFile = File(...),
    merge_columns: str = Form(default=""),
    filler: str = Form(default=","),
):
    try:
        # Load the CSV file into a DataFrame
        data = pd.read_csv(csv_file.file, encoding="ISO-8859-1")
    except Exception as e:
        return {"error": f"Error loading file: {e}"}

    # Remove duplicate rows while keeping only one
    data = data.drop_duplicates()

    missing_cols = data.columns[data.isnull().any()]
    cleaned_data = data.copy()

    for col in missing_cols:
        if cleaned_data[col].dtype == np.number:  # Check if column is numerical
            if cleaned_data[col].skew() > 1:  # Outlier detection using skewness
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
            else:
                cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
        else:
            # Handle mixed-type columns
            if cleaned_data[col].dtype == object:
                try:
                    # Try converting column to numeric, coercing errors to NaN
                    numeric_col = pd.to_numeric(cleaned_data[col], errors="coerce")
                    if numeric_col.notna().sum() > 0:  # Check if numeric data exists
                        if numeric_col.skew() > 1:
                            cleaned_data[col] = numeric_col.fillna(numeric_col.median())
                        else:
                            cleaned_data[col] = numeric_col.fillna(numeric_col.mean())
                    else:  # No numeric data, fill with mode
                        cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
                except Exception:
                    # If conversion fails, fallback to filling with mode
                    cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
            else:
                # For purely categorical columns, fill with mode
                cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)

    # Merge columns if provided
    if merge_columns:
        columns_to_merge = merge_columns.split(",")
        if len(columns_to_merge) == 2 and all(col in cleaned_data.columns for col in columns_to_merge):
            cleaned_data[columns_to_merge[0] + "_merged"] = (
                cleaned_data[columns_to_merge[0]].astype(str)
                + filler
                + cleaned_data[columns_to_merge[1]].astype(str)
            )
        else:
            return {"error": "Invalid column names for merging."}

    # Convert numeric columns to integers if possible
    for col in cleaned_data.select_dtypes(include=[np.number]).columns:
        cleaned_data[col] = cleaned_data[col].astype(int)

    # Save the cleaned data
    cleaned_file_path = "cleaned_data.csv"
    cleaned_data.to_csv(cleaned_file_path, index=False, encoding="utf-8")

    # Return the column names of the cleaned data
    column_names = cleaned_data.columns.tolist()

    return {
        "message": "Data cleaned successfully with duplicates removed and integers where applicable",
        "cleaned_file": cleaned_file_path,
        "columns": column_names,  # Return the cleaned data columns
    }


@app.post("/getcolumns/")  # Changed to POST to be consistent with other endpoints that might receive files
async def get_columns(): # Removed csv_file argument
    try:
        data = pd.read_csv("cleaned_data.csv", encoding='utf-8') # Read cleaned_data.csv
        return {"columns": list(data.columns)}
    except FileNotFoundError:
        return {"error": "Cleaned data file not found. Please clean the data first."}
    except Exception as e:
        return {"error": f"Error loading cleaned data: {e}"}

@app.get("/test/")
async def test_endpoint():
    return {"message": "Test endpoint is working!"}
