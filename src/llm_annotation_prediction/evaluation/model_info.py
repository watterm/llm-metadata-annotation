# Model information for NephGen-9 models at the time of the experiments.
# Each entry is a dict with cost, context, and output limit info for use with pandas.
# Cost is defined in $ per million tokens and given in ranges.

model_info = {
    "google/gemini-2.0-flash-001": {
        "input_cost_min": 0.1,
        "input_cost_max": 0.1,
        "output_cost_min": 0.4,
        "output_cost_max": 0.4,
        "context": 1_000_000,
        "output_limit": 8_000,
    },
    "openai/gpt-4o": {
        "input_cost_min": 2.5,
        "input_cost_max": 2.5,
        "output_cost_min": 10.0,
        "output_cost_max": 10.0,
        "context": 128_000,
        "output_limit": 16_000,
    },
    "meta-llama/llama-4-maverick": {
        "input_cost_min": 0.18,
        "input_cost_max": 0.63,
        "output_cost_min": 0.6,
        "output_cost_max": 1.8,
        "context": 1_050_000,
        "output_limit": 1_050_000,
    },
    "mistral/ministral-8b": {
        "input_cost_min": 0.1,
        "input_cost_max": 0.1,
        "output_cost_min": 0.1,
        "output_cost_max": 0.1,
        "context": 131_000,
        "output_limit": 131_000,
    },
    "mistralai/mistral-large-2411": {
        "input_cost_min": 2.0,
        "input_cost_max": 2.0,
        "output_cost_min": 6.0,
        "output_cost_max": 6.0,
        "context": 131_000,
        "output_limit": 131_000,
    },
    "openai/o4-mini": {
        "input_cost_min": 1.1,
        "input_cost_max": 1.1,
        "output_cost_min": 4.4,
        "output_cost_max": 4.4,
        "context": 200_000,
        "output_limit": 100_000,
    },
    "mistralai/mistral-small-3.1-24b-instruct": {
        "input_cost_min": 0.1,
        "input_cost_max": 0.1,
        "output_cost_min": 0.3,
        "output_cost_max": 0.3,
        "context": 33_000,
        "output_limit": 33_000,
    },
    "openai/gpt-4o-mini": {
        "input_cost_min": 0.15,
        "input_cost_max": 0.15,
        "output_cost_min": 0.6,
        "output_cost_max": 0.6,
        "context": 128_000,
        "output_limit": 16_000,
    },
    "openai/gpt-4.1-nano": {
        "input_cost_min": 0.1,
        "input_cost_max": 0.1,
        "output_cost_min": 0.4,
        "output_cost_max": 0.4,
        "context": 1_050_000,
        "output_limit": 33_000,
    },
    "openai/gpt-4.1-mini": {
        "input_cost_min": 0.4,
        "input_cost_max": 0.4,
        "output_cost_min": 1.6,
        "output_cost_max": 1.6,
        "context": 1_050_000,
        "output_limit": 33_000,
    },
    "openai/gpt-4.1": {
        "input_cost_min": 2.0,
        "input_cost_max": 2.0,
        "output_cost_min": 8.0,
        "output_cost_max": 8.0,
        "context": 1_050_000,
        "output_limit": 33_000,
    },
    "google/gemini-2.5-flash-preview": {
        "input_cost_min": 0.15,
        "input_cost_max": 0.15,
        "output_cost_min": 0.6,
        "output_cost_max": 0.6,
        "context": 1_050_000,
        "output_limit": 66_000,
    },
    "mistralai/mistral-medium-3": {
        "input_cost_min": 0.4,
        "input_cost_max": 0.4,
        "output_cost_min": 2.0,
        "output_cost_max": 2.0,
        "context": 131_000,
        "output_limit": 131_000,
    },
    "google/gemini-2.5-pro-preview": {
        "input_cost_min": 1.25,
        "input_cost_max": 2.5,
        "output_cost_min": 10.0,
        "output_cost_max": 15.0,
        "context": 1_000_000,
        "output_limit": 66_000,
    },
}
