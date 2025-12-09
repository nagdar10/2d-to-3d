# 2D to 3D Converter

This project is a Python implementation of an algorithm to convert a single 2D image into a 3D image by generating a depth map.

## Dependencies

- OpenCV

## Usage

## Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application using Python. You will be prompted to select an image file.

```bash
python src/main.py
```

### Command Line Arguments

You can customize the clustering algorithm and its parameters using command-line arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--algo` | Clustering algorithm to use. Options: `dbscan`, `kmeans`. | `dbscan` |
| `--k` | Number of clusters (only for K-Means). | `3` |
| `--eps-factor` | Epsilon factor for DBSCAN (proportion of image size). | `0.02` |

### Examples

**Run with default settings (DBSCAN):**
```bash
python src/main.py
```

**Run with K-Means clustering (3 clusters):**
```bash
python src/main.py --algo kmeans --k 3
```

**Run with DBSCAN and a custom epsilon factor:**
```bash
python src/main.py --algo dbscan --eps-factor 0.05
```


Run ssh agent:

- `eval "$(ssh-agent -s)"`
- `ssh-add ~/.ssh/id_ed25519_2` - or any other ssh that has access to nagdar10 github
