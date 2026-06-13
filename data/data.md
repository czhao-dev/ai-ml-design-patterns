# Data

This project uses a satellite image dataset for binary land classification:

- `class_0_non_agri`: non-agricultural land
- `class_1_agri`: agricultural land

The dataset contains 6,000 JPG tiles, split evenly between the two classes.

## Source

The data pipeline downloads the archive from IBM Skills Network cloud storage:

```text
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
```

One earlier data-loading workflow references this alternate copy:

```text
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar
```

## Local Layout

Keep local data under:

```text
data/raw/
```

The archive currently lives locally at:

```text
data/raw/images_dataSAT.tar
```

The `data/raw/` folder is ignored by Git so the repository stays lightweight.
