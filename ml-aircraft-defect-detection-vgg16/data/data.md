# Data

This project uses an image dataset for binary aircraft damage classification:

- `crack`: crack damage
- `dent`: dent damage

The dataset contains 446 JPG images (640x640), pre-split into:

| Split | crack | dent | Total |
|---|---:|---:|---:|
| train | 150 | 150 | 300 |
| valid | 48 | 48 | 96 |
| test | 25 | 25 | 50 |

## Source

Originally published as the [Aircraft Damage Detection](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) dataset on Roboflow (CC BY 4.0), re-hosted by IBM Skills Network. `scripts/01_data_pipeline.py` downloads the archive from:

```text
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar
```

## Local Layout

Keep local data under:

```text
data/raw/
```

The archive extracts to:

```text
data/raw/aircraft_damage_dataset_v1/
├── train/
│   ├── crack/
│   └── dent/
├── valid/
│   ├── crack/
│   └── dent/
└── test/
    ├── crack/
    └── dent/
```

The `data/raw/` folder is ignored by Git so the repository stays lightweight.
