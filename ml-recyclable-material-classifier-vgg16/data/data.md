# Data

This project uses an image dataset for binary material classification:

- `O`: organic material
- `R`: recyclable material

The dataset contains 1,200 JPG images, split into:

| Split | O | R | Total |
|---|---:|---:|---:|
| train | 500 | 500 | 1,000 |
| test | 100 | 100 | 200 |

The `train` split is further divided into training (800 images) and validation (200 images) subsets at run time, via a 20% validation split.

## Source

`scripts/01_data_pipeline.py` downloads the archive from IBM Skills Network cloud storage:

```text
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip
```

## Local Layout

Keep local data under:

```text
data/raw/
```

The archive extracts to:

```text
data/raw/o-vs-r-split/
├── train/
│   ├── O/
│   └── R/
└── test/
    ├── O/
    └── R/
```

The `data/raw/` folder is ignored by Git so the repository stays lightweight.
