from fastapi import FastAPI, Body, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from jellyfish import jaro_winkler_similarity, levenshtein_distance
from collections import defaultdict
from fastapi.responses import FileResponse
import os
import re
from datetime import datetime
import subprocess
import git
import json

DATASET_FILE = "datasets.json"
MODEL_FILE = "models.json"


app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

def load_models():
    """Load model names and versions from file."""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    return {}

def save_models(data):
    """Save model names and versions to file."""
    with open(MODEL_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_datasets():
    """Load dataset names and versions from file."""
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r") as f:
            return json.load(f)
    return {}

def save_datasets(data):
    """Save dataset names and versions to file."""
    with open(DATASET_FILE, "w") as f:
        json.dump(data, f, indent=4)

class VersionCreate(BaseModel):
    file: str
    message: str

class VersionCheckout(BaseModel):
    version: str

def ensure_git_and_dvc():
    """Ensure Git and DVC are properly initialized"""
    try:
        # Check if we're in a Git repo
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True)
            # Configure Git (required for first commit)
            subprocess.run(['git', 'config', 'user.email', "dataversion@example.com"], check=True)
            subprocess.run(['git', 'config', 'user.name', "Data Version Control"], check=True)
            
        # Check if DVC is initialized
        if not os.path.exists('.dvc'):
            subprocess.run(['dvc', 'init'], check=True)
            subprocess.run(['git', 'add', '.dvc'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Initialize DVC'], check=True)
            
        return True
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error initializing repository: {str(e)}")

@app.post("/model/commit/")
async def create_model_version(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    message: str = Form(...),
):
    try:
        ensure_git_and_dvc()
        models = load_models()

        if model_name not in models:
            models[model_name] = []

        # Save the uploaded file
        file_path = f"models/{file.filename}"
        os.makedirs("models", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Check if the file is tracked by Git
        git_tracked = subprocess.run(
            ['git', 'ls-files', '--error-unmatch', file_path],
            capture_output=True,
            text=True,
        ).returncode == 0

        if git_tracked:
            # Stop Git from tracking the file
            subprocess.run(['git', 'rm', '--cached', file_path], check=True)
            subprocess.run(['git', 'commit', '-m', f"stop tracking {file_path}"], check=True)

        # Add file to DVC
        subprocess.run(['dvc', 'add', file_path], check=True)

        # Generate version tag
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f'model_v_{timestamp}'

        # Stage all files in Git
        subprocess.run(['git', 'add', '-A'], check=True)

        # Check if there are changes before committing
        status_output = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()

        if status_output:
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'tag', version_tag], check=True)  # Add Git tag
        else:
            print("No changes to commit.")

        # Store model name and version
        models[model_name].append(version_tag)
        save_models(models)

        return {"message": "Model version created successfully", "model": model_name, "version": version_tag}
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error in version control: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
       
@app.get("/model/versions/")
async def get_model_versions():
    """Get all models and their version history"""
    try:
        ensure_git_and_dvc()
        models = load_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/checkout/")
async def checkout_model_version(version: VersionCheckout):
    """Checkout a specific version of the model"""
    try:
        ensure_git_and_dvc()

        # Load models.json and datasets.json before checkout
        models_before = load_models()
        datasets_before = load_datasets()  # Load datasets.json
        print("Models before checkout:", models_before)
        print("Datasets before checkout:", datasets_before)

        # Get a list of existing Git tags
        git_tags = subprocess.run(['git', 'tag'], capture_output=True, text=True).stdout.splitlines()

        if version.version not in git_tags:
            raise HTTPException(status_code=404, detail=f"Version {version.version} not found. Available versions: {git_tags}")

        # Reset and checkout to avoid conflicts
        subprocess.run(['git', 'reset', '--hard'], check=True)
        subprocess.run(['git', 'checkout', '-f', version.version], check=True)

        # Pull the corresponding data from DVC
        try:
            subprocess.run(['dvc', 'pull'], check=True)
        except subprocess.CalledProcessError:
            print("DVC pull failed, possibly due to no remote storage.")

        # Restore models.json and datasets.json after checkout
        save_models(models_before)
        save_datasets(datasets_before)  # Restore datasets.json
        print("Models after checkout:", load_models())
        print("Datasets after checkout:", load_datasets())

        return {"message": f"Successfully checked out version {version.version}"}
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error checking out version: {str(e)}")
    
@app.post("/model/delete/")
async def delete_model_version(version: VersionCheckout):
    """Delete a specific version of a model."""
    try:
        ensure_git_and_dvc()

        # Load models.json
        models = load_models()

        # Find and remove the version from models.json
        version_found = False
        for model_name, version_list in models.items():
            if version.version in version_list:
                version_list.remove(version.version)
                version_found = True
                break

        if not version_found:
            raise HTTPException(status_code=404, detail=f"Version {version.version} not found in models.json")

        # Save the updated models.json
        save_models(models)

        # Delete the Git tag
        subprocess.run(['git', 'tag', '-d', version.version], check=True)

        return {"message": f"Version {version.version} deleted successfully"}
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error deleting version: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dvc/file_data/")
async def get_file_headers(file: str):
    """Fetch the headers of the file for preview."""
    try:
        df = pd.read_csv(file, nrows=1)  # Read only first row
        return {"headers": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
@app.post("/dvc/init/")
async def initialize_dvc():
    """Initialize DVC in the project"""
    try:
        ensure_git_and_dvc()
        return {"message": "Git and DVC initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dvc/commit/")
async def create_version(version: VersionCreate, dataset_name: str):
    """Create a new version under a given dataset name"""
    try:
        ensure_git_and_dvc()
        datasets = load_datasets()

        if dataset_name not in datasets:
            datasets[dataset_name] = []

        # Generate version tag
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f'v_{timestamp}'

        # Add file to DVC
        subprocess.run(['dvc', 'add', version.file], check=True)

        # Stage all files in Git
        subprocess.run(['git', 'add', '-A'], check=True)

        # Check if there are changes before committing
        status_output = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()

        if status_output:
            subprocess.run(['git', 'commit', '-m', version.message], check=True)
            subprocess.run(['git', 'tag', version_tag], check=True)  # Add Git tag
        else:
            print("No changes to commit.")

        # Store dataset name and version
        datasets[dataset_name].append(version_tag)
        save_datasets(datasets)

        return {"message": "Version created successfully", "dataset": dataset_name, "version": version_tag}
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error in version control: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dvc/versions/")
async def get_versions():
    """Get all datasets and their version history"""
    try:
        ensure_git_and_dvc()
        datasets = load_datasets()
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dvc/checkout/")
async def checkout_version(version: VersionCheckout):
    try:
        ensure_git_and_dvc()

        # Load datasets.json before checkout
        datasets_before = load_datasets()
        print("Datasets before checkout:", datasets_before)

        # Get a list of existing Git tags
        git_tags = subprocess.run(['git', 'tag'], capture_output=True, text=True).stdout.splitlines()

        if version.version not in git_tags:
            raise HTTPException(status_code=404, detail=f"Version {version.version} not found. Available versions: {git_tags}")

        # Reset and checkout to avoid conflicts
        subprocess.run(['git', 'reset', '--hard'], check=True)
        subprocess.run(['git', 'checkout', '-f', version.version], check=True)

        # Pull the corresponding data from DVC
        try:
            subprocess.run(['dvc', 'pull'], check=True)
        except subprocess.CalledProcessError:
            print("DVC pull failed, possibly due to no remote storage.")

        # Restore datasets.json after checkout
        save_datasets(datasets_before)
        print("Datasets after checkout:", load_datasets())

        return {"message": f"Successfully checked out version {version.version}"}
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error checking out version: {str(e)}")

@app.get("/dvc/status/")
async def get_dvc_status():
    """Get current DVC status"""
    try:
        ensure_git_and_dvc()
        result = subprocess.run(['dvc', 'status'], capture_output=True, text=True, check=True)
        return {"status": result.stdout.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dvc/delete/")
async def delete_version(version: VersionCheckout):
    try:
        ensure_git_and_dvc()

        # Load datasets.json
        datasets = load_datasets()

        # Find and remove the version from datasets.json
        version_found = False
        for dataset_name, version_list in datasets.items():
            if version.version in version_list:
                version_list.remove(version.version)
                version_found = True
                break

        if not version_found:
            raise HTTPException(status_code=404, detail=f"Version {version.version} not found in datasets.json")

        # Save the updated datasets.json
        save_datasets(datasets)

        # Delete the Git tag
        subprocess.run(['git', 'tag', '-d', version.version], check=True)

        return {"message": f"Version {version.version} deleted successfully"}
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error deleting version: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SplitColumnsRequest(BaseModel):
    column: str
    separator: str
    max_columns: int

class JoinColumnsRequest(BaseModel):
    columns: List[str]
    separator: str

@app.post("/refine/split_columns/")
async def split_columns(request: SplitColumnsRequest):
    """Split a single column into multiple columns based on a separator"""
    try:
        df = pd.read_csv("cleaned_data.csv")
        
        if request.column not in df.columns:
            return {"error": f"Column {request.column} not found in dataset"}
        
        # Split the column into separate columns
        split_df = df[request.column].str.split(
            request.separator, 
            n=request.max_columns-1, 
            expand=True
        )
        
        # Name the new columns
        for i in range(len(split_df.columns)):
            new_col_name = f"{request.column}_{i+1}"
            df[new_col_name] = split_df[i]
            
        # Remove original column
        df = df.drop(columns=[request.column])
        
        # Save the modified dataset
        df.to_csv("cleaned_data.csv", index=False)
        
        return {
            "message": f"Successfully split column {request.column} into {len(split_df.columns)} columns"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/refine/join_columns/")
async def join_columns(request: JoinColumnsRequest):
    """Join multiple columns into a single column using a separator"""
    try:
        df = pd.read_csv("cleaned_data.csv")
        
        # Verify all columns exist
        missing_cols = [col for col in request.columns if col not in df.columns]
        if missing_cols:
            return {"error": f"Columns not found: {', '.join(missing_cols)}"}
        
        # Create new column name based on joined column names
        new_column_name = "_".join(request.columns) + "_joined"
        
        # Join the columns
        df[new_column_name] = df[request.columns].astype(str).agg(
            request.separator.join, axis=1
        )
        
        # Remove original columns
        df = df.drop(columns=request.columns)
        
        # Save the modified dataset
        df.to_csv("cleaned_data.csv", index=False)
        
        return {
            "message": f"Successfully joined columns into {new_column_name}"
        }
    except Exception as e:
        return {"error": str(e)}

# Create a Pydantic model to validate input data
class SplitCellsRequest(BaseModel):
    column: str
    separator: str
    use_regex: Optional[bool] = False

@app.post("/refine/split_cells/")
async def split_cells(request: SplitCellsRequest):
    try:
        df = pd.read_csv("cleaned_data.csv")

        if request.column not in df.columns:
            return {"error": f"Column {request.column} not found in dataset"}

        # Split using regular expression or standard separator
        if request.use_regex:
            df[request.column] = df[request.column].apply(
                lambda x: re.split(request.separator, str(x)) if isinstance(x, str) else [x]
            )
        else:
            df[request.column] = df[request.column].apply(
                lambda x: str(x).split(request.separator) if isinstance(x, str) else [x]
            )

        # Exploding lists into separate rows
        df_exploded = df.explode(request.column, ignore_index=True)
        df_exploded.to_csv("cleaned_data.csv", index=False)

        return {"message": f"Successfully split values in column {request.column}"}

    except Exception as e:
        return {"error": str(e)}


@app.get("/download-cleaned-data/")
async def download_cleaned_data():
    """Endpoint to download the cleaned data CSV file"""
    file_path = "cleaned_data.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Cleaned data file not found")
    
    return FileResponse(
        path=file_path,
        filename="cleaned_data.csv",
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"}
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

class CaseTransformRequest(BaseModel):
    column: str
    case_type: str


@app.post("/refine/case_transform/")
async def case_transform(request: CaseTransformRequest):
    """Transform the case of text in a specified column"""
    try:
        df = pd.read_csv("cleaned_data.csv")
        if request.column not in df.columns:
            return {"error": "Column not found in dataset"}
        
        # Normalize the case type to lowercase
        case_type = request.case_type.lower()

        if case_type == "upper":
            df[request.column] = df[request.column].str.upper()
        elif case_type == "lower":
            df[request.column] = df[request.column].str.lower()
        elif case_type == "title":
            df[request.column] = df[request.column].str.title()
        else:
            return {"error": f"Invalid case type specified: {case_type}. Valid options are 'upper', 'lower', or 'title'."}
        
        df.to_csv("cleaned_data.csv", index=False)
        return {"message": f"Successfully transformed the case of column {request.column} to {case_type}"}
    except Exception as e:
        return {"error": f"Error during case transformation: {e}"}

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

    y = population[[response]]
    predictors = list(population.columns)
    predictors.remove(response)
    x = population[predictors]
    if algo == "simple":
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, stratify=x[stratify_col])
    x_train.to_csv('x_train.csv', encoding='utf-8')
    x_test.to_csv('x_test.csv', encoding='utf-8')

    y_train.to_csv('y_train.csv', encoding='utf-8')
    y_test.to_csv('y_test.csv', encoding='utf-8')
    return {"message": " samples created: x_train, x_test, y_train, y_test"}

    
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

@app.get("/getcolumns/")
async def get_columns_get():
    """GET endpoint for retrieving columns and preview data"""
    try:
        data = pd.read_csv("cleaned_data.csv", encoding='utf-8')
        
        # Convert DataFrame to dict while handling NaN values
        preview_data = data.head(10).where(pd.notna(data.head(10)), None).to_dict('records')
        
        return {
            "columns": list(data.columns),
            "preview": preview_data
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Cleaned data file not found. Please clean the data first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cleaned data: {str(e)}")

@app.post("/getcolumns/")
async def get_columns_post():
    """POST endpoint for retrieving columns and preview data"""
    try:
        data = pd.read_csv("cleaned_data.csv", encoding='utf-8')
        # Return first 5 rows of data for preview
        preview_data = data.head().to_dict('records')
        return {
            "columns": list(data.columns),
            "preview": preview_data
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Cleaned data file not found. Please clean the data first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cleaned data: {str(e)}")

@app.get("/test/")
async def test_endpoint():
    return {"message": "Test endpoint is working!"}
