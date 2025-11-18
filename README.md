# Automated Contextual Metadata Annotation with Grounded LLMs

This repository provides a framework for performing experiments with Large Language Models (LLMs) to annotate publications with metadata. It is designed to work with OpenRouter and supports the use of tools. The framework allows for structured output handling to prepare for a fully automated metadata annotation process. It is particularly useful for annotating scientific publications with contextual metadata, such as entities and relationships, based on their content. The framework is built using Python, utilizing the uv package for task management.

## 1. Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone this repository.
3. Copy `default.env` to `.env` and set the variables. See [Application Configuration](#application-configuration).

### 1.1 Application Configuration

The `.env` file will contain secrets (like API keys) that should not be shared
and is thus not part of the repository. For performing OpenRouter experiments,
you need to enter your API key there.

### 1.2 Experiment Configuration

You can find the settings for single experiments in the `configs` folder. The
settings are described in the corresponding `Config` classes in the code or in
the example configurations.

## 2. Publication Replication

To replicate the experiments from the publication, run the following command in the terminal
after the setup:

```sh
uv run experiment configs/pilot/review_simplified.yml
```

The results will be written in a new folder with a timestamp in `experiments`. You can use the
[Show tool](#4-show) to inspect the conversations and statistics. The following command without an `index` will list a table of conversations in the experiment, or, with an `index`, the corresponding conversation:

```sh
uv run show -e experiments/[timestamp]-pilot-review-simplified.yml conversation [index]
```

The raw results of all our experiments for the publication are stored in `saved_experiments/pilot/review-simplified` and the rendered conversations, statistics and plots can be found in `rendered_experiments/pilot/review-simplified`.
