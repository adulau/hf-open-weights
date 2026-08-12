# Hugging Face open-weight model catalogue

`hf_open_weights.py` builds a local catalogue of Hugging Face model repositories
that contain recognizable weight files. It records license information,
training-related Model Card sections, and datasets declared or linked by the
model author. It also collects organization, explicitly declared country,
language, and tag metadata and produces aggregate statistics.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U huggingface_hub
```

Optionally:

```bash
export HF_TOKEN="hf_..."
```

## Run

Default policy:

```bash
python hf_open_weights.py
```

This creates:

- `hf-open-weights.sqlite`
- `hf-open-weights.csv`
- `hf-open-weights.jsonl`
- `hf-open-weights-stats.json`

Test on a smaller sample:

```bash
python hf_open_weights.py --limit 500
```

Start with the models that have the most Hugging Face likes (also commonly
called stars):

```bash
python hf_open_weights.py --sort most-starred --limit 500
```

The default, `--sort last-modified`, starts with the most recently modified
models. `--since` can be combined with either ordering; with `most-starred`, the
crawler checks the timestamp of every enumerated model rather than stopping at
the first old model.

Strict open-source/open-content licenses only:

```bash
python hf_open_weights.py --policy strict
```

Every public repository exposing recognizable model-weight files, regardless of
license classification:

```bash
python hf_open_weights.py --policy public-weights
```

Scan recently changed models:

```bash
python hf_open_weights.py --since 2026-08-01T00:00:00Z
```

## License classes

The script deliberately does **not** claim Hugging Face has an authoritative
`open_weight` flag.

It classifies declared Model Card licenses into:

- `open-source`: conventional open-source/open-content licenses;
- `open-weight`: model licenses commonly described as open-weight/source-available,
  including OpenRAIL, Llama, Gemma, and similar families;
- `restricted`: clearly research-only, non-commercial, or otherwise limited;
- `unknown`: missing, custom, or not safely auto-classifiable.

The default `--policy open-weight` includes the first two classes.

Always inspect the actual model repository and license before redistribution or
production use.

## Dataset provenance

The output contains:

- `datasets_declared`: structured `datasets:` Model Card metadata;
- `datasets_linked`: Hugging Face dataset URLs found in the card text;
- `datasets_all`: union of the two.

A missing dataset means “not discovered/documented by this script”, not “the
model was trained without a dataset”.

## Publisher metadata and statistics

The CSV, JSONL, and SQLite outputs include:

- `downloads`, `likes`, and `followers`: the per-model engagement counters
  returned by the Hugging Face Hub (a counter is null when the Hub does not
  provide it);
- `organization`: the Model Card's custom `organization`/`organisation` value,
  falling back to the repository namespace;
- `organization_source`: either `model-card` or `repository-namespace`, so the
  fallback is not mistaken for a verified legal organization;
- `countries`: values explicitly declared under custom `country` or `countries`
  Model Card keys;
- `languages` and `tags`: normalized structured Model Card values.

Country is deliberately **not inferred** from a person's name, organization,
language, or free-form card text. Missing country data remains missing.

`hf-open-weights-stats.json` contains model counts grouped by organization,
country, language, license, pipeline, library, training-information status, and
gating, plus total downloads, likes, and followers. Multi-country and
multi-language models are counted once in each declared group. Pass
`--stats ""` to disable it.

## Training provenance

`training_text` contains Markdown sections whose headings indicate training,
pretraining, post-training, fine-tuning, hyperparameters, optimization, or
training data. This is deterministic section extraction, not an LLM-generated
summary.

`training_info_status` is one of:

- `training-text-and-datasets`
- `training-text-only`
- `datasets-only`
- `not-documented`

## Scheduling

For a daily update, for example:

```cron
15 3 * * * cd /srv/hf-catalogue && .venv/bin/python hf_open_weights.py --since "$(date -u -d '2 days ago' +\%FT\%TZ)" >>crawl.log 2>&1
```

Using a small overlap (two days above) makes the update tolerant of scheduling
gaps; SQLite uses `model_id` as the primary key and upserts records.
