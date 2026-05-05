# FMA → GTZAN Genre Mapping

This document records the rationale for mapping FMA top-level genres to the GTZAN
genre vocabulary used by the CNN music classifier.

## Background

The [Free Music Archive (FMA) small subset](https://github.com/mdeff/fma) contains
8,000 tracks spread evenly across 8 top-level genres (1,000 tracks each).  The GTZAN
dataset — the original training source — uses 10 genre labels:

> blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock

Because the classifier's label vocabulary is fixed to GTZAN genre names (derived from
directory names at ingest time), FMA tracks must be placed into matching directories.
Only FMA genres that map to an existing GTZAN class are included; the rest are
discarded.

## Mapping table

| FMA genre | GTZAN directory | Confidence | Rationale |
|-----------|-----------------|------------|-----------|
| Hip-Hop | `hiphop` | **Strong** | Direct genre equivalence. |
| Pop | `pop` | **Strong** | Direct genre equivalence. |
| Rock | `rock` | **Strong** | Direct genre equivalence. |
| Folk | `country` | **Moderate** | Both are acoustic, roots-based genres with significant stylistic overlap. Folk is not identical to country, but it is the closest available GTZAN class. |
| Electronic | `disco` | **Weak** | Disco is the only rhythmic/dance-oriented GTZAN class. Electronic spans a much broader range of styles; this mapping will introduce some label noise. |

## Excluded FMA genres

| FMA genre | Reason for exclusion |
|-----------|----------------------|
| Experimental | No GTZAN equivalent; highly heterogeneous style. |
| International | Umbrella label covering many unrelated world-music styles; no single GTZAN genre is a reasonable proxy. |
| Instrumental | Cross-genre label (an instrumental track could be jazz, rock, classical, etc.); mapping would be arbitrary. |

Note: Blues, Classical, Country, Disco, Jazz, Metal, and Reggae do not appear as
top-level genres in the FMA small subset, so those GTZAN classes receive no new data
from this source.

## Practical implications

- The **Folk→country** and **Electronic→disco** mappings will add label noise to those
  GTZAN classes. If per-class accuracy on `country` or `disco` degrades after adding
  FMA data, revisit or exclude those mapped genres.
- Tracks listed in `fma_metadata/not_found.pickle` (audio key) are skipped during
  preparation regardless of genre.
- Sampling is stratified: 200 tracks per mapped genre (1,000 total) drawn with a fixed
  random seed for reproducibility.

## Preparation script

```
scripts/prepare_fma_subset.py --help
```

The script reads `fma_metadata/tracks.csv`, applies this mapping, samples tracks, and
copies MP3 files into `training_data/fma_subset/{gtzan_genre}/`. The output directory
is a drop-in replacement for `training_data/gtzan_dataset/` from the pipeline's
perspective.
