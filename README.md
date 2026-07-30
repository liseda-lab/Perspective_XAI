# Perspective_XAI

**Shaping Scientific Explanations to Expert Perspectives with Persona-Conditioned Reinforcement Learning**

This repository accompanies our submission to *Nature Machine Intelligence*. It contains the code, configuration files, persona materials, and evaluation data needed to reproduce the perspective explanation framework described in the paper.

> **Faster engineering implementation**: a maintained PyTorch implementation of the underlying path-search machinery is at [liseda-lab/REx_PyTorch](https://github.com/liseda-lab/REx_PyTorch). It is significantly faster than this TensorFlow version, supports both OpenAI and local LLMs (e.g. Qwen3.5-9B for training, Qwen3-4B for scoring), and **extends the approach to additional knowledge graphs (Oregano DR/DTI, PrimeKG)** beyond the Hetionet datasets used in this paper. It also adds a decoupled rerank pipeline for handling slow datasets that the in-loop scorer here cannot handle efficiently.

## Overview

Most explainable AI methods treat explanation quality as a fixed property of model outputs. We argue instead that explanation quality is **conditional on the user's epistemic perspective** — the characteristic way an expert prioritizes mechanisms, weighs uncertainty, and constructs explanatory narratives. The paper contributes:

1. **Agentic personas** — compact representations of recurring epistemic perspectives, derived from clustered expert feedback rather than from demographics or roles.
2. **A reinforcement learning framework over knowledge graphs** that uses persona-aligned rewards to steer explanation generation, replacing direct human feedback at training time.
3. **Empirical evidence (n = 22 evaluation study)** that perspective-conditioned explanations are preferred over general-purpose ones, improve perceived relevance and validity, and match or exceed state-of-the-art predictive performance.

Experiments in the paper use the [Hetionet](https://het.io) knowledge graph for both drug repurposing (DR) and drug–target interaction (DTI) tasks. For experiments on additional graphs (Oregano DR/DTI, PrimeKG), see the PyTorch implementation linked above.

## Repository structure

```
Perspective_XAI/
├── perspective_approach/    # The reinforcement learning approach (training + test)
│   ├── code/
│   │   ├── data/            # Knowledge graph and batch loading
│   │   ├── model/           # Agent, environment, trainer
│   │   ├── options.py       # CLI flags
│   │   └── test_beam_parser.py
│   ├── configs/             # Per-dataset training/test configs
│   ├── datasets/            # Hetionet drug repurposing and drug-target slices
│   ├── personas/            # Persona narrative files used as prompts at training time
│   ├── Dockerfile
│   └── run.sh
├── personas/                # Persona synthesis pipeline (Phase I of the paper)
│   ├── agentic_personas/    # Persona generation scripts
│   ├── clustering/          # Expert feedback embeddings, clustering analyses, results
│   ├── final_personas/      # Final persona narratives (Elena, Leo)
│   └── verbalization/       # GPT-4o-mini path verbalizers (Appendix B.3)
└── evaluation/              # User study materials and analyses
    ├── credibility_check/         # Persona credibility check materials
    ├── persona_creation_study/    # Persona creation study (n = 11) materials
    └── persona_evaluation_study/  # Persona evaluation study (n = 22) materials
```

## Running the approach

### Prerequisites

- Docker (for the provided container) **or** UV / Python 3.10+ for a local install.
- An `OPENAI_API_KEY` in the environment when `agentic_ai_enabled=1` in your config (LLM scoring uses GPT-4o-mini by default).

### With Docker

```sh
docker build -t perspective-image perspective_approach
docker run --gpus all -d --name perspective_space \
    -v $(pwd)/perspective_approach:/perspective_approach \
    perspective-image tail -f /dev/null
docker exec -it perspective_space bash
```

### Run a config

From inside the container (or from a local UV environment):

```sh
cd perspective_approach
uv run bash run.sh configs/{dataset}
```

where `{dataset}` is e.g. `hetionet_dr` or `hetionet_dt`.

## CLI flags

`--agentic_ai_enabled` (`0` / `1`, default `0`): turn persona-shaped scoring on. Requires `--persona_path` to point at a persona narrative file (e.g. `personas/elena.txt`).

`--persona_path` (path): persona narrative file used as the prompt prefix for the LLM judge during training and test.

`--no_llm_rerank` (`0` / `1`, default `0`): skip per-batch LLM scoring during `test()` so the test loop runs at neutral pace and the paths JSON is written with `agentic_score = 0.0`. Useful when you want HITS@k / MRR metrics quickly without paying per-batch GPT-4o-mini calls (training is unaffected). Defaults to off — existing configs reproduce the original in-loop behavior.

## Datasets

We use the [Hetionet](https://het.io) heterogeneous biomedical knowledge graph for both DR and DTI tasks. Dataset slices are bundled under `perspective_approach/datasets/`.

Each dataset directory contains:

```
dataset/
├── graph.txt                          # All triples except dev/test
├── dev.txt, test.txt, train.txt
├── clustered_IC_classes_edgeType.json # Edge IC scores per cluster
└── vocab/{entity_vocab.json, relation_vocab.json}
```

`graph.txt` may be split into `graph_part*.txt` due to GitHub file size limits; concatenate with:

```sh
cat graph_part*.txt > graph.txt
```

## Authors

- **Susana Nunes**
- **Tiago Guerreiro**
- **Catia Pesquita**

LASIGE, Faculdade de Ciências da Universidade de Lisboa.

## Contact

For comments or assistance, please contact **scnunes@ciencias.ulisboa.pt**.

## Acknowledgments

Built on the [REx](https://github.com/liseda-lab/REx) reinforcement learning framework for path-based explanation generation.
