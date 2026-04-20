# contrastive_decoding_llm_judge

Code for the paper ["Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge"](https://arxiv.org/abs/2510.18196).

## Installation

```bash
uv sync
```

`llama_eval.py` uses Flash Attention 2, which requires a separate install:

```bash
uv pip install flash-attn --no-build-isolation
```

## Usage

### Evaluate with contrastive decoding (SummEval)

```bash
python src/llama_eval.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --assistant_model meta-llama/Llama-3.2-3B-Instruct \
  --prompt_fp prompts/summeval/coh_detailed_range.txt \
  --save_fp results/llama31_8b_coh_contrastive.json \
  --summeval_fp data/summeval.json \
  --decoding contrastive \
  --contrastive_lamb 1.0 \
  --contrastive_asst_temperature 1.0
```

### Evaluate with greedy decoding

```bash
python src/llama_eval.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt_fp prompts/summeval/coh_detailed_range.txt \
  --save_fp results/llama31_8b_coh_greedy.json \
  --summeval_fp data/summeval.json \
  --decoding greedy
```

### Meta-evaluate results (correlation with human judgments)

```bash
python src/meta_eval_summeval.py \
  --input_fp results/llama31_8b_coh_contrastive.json \
  --dimension coherence \
  --ignore_score 1 \
  --max_score 5
```

### BigGen benchmark evaluation

```bash
python src/biggen_eval_contrastive.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --assistant_model meta-llama/Llama-3.2-3B-Instruct \
  --decoding contrastive
```

## Prompts

Prompt templates for SummEval are in `prompts/summeval/`.

## Related Work

If you use the SummEval dataset or G-Eval prompts, please also cite:

```bibtex
@article{fabbri-etal-2021-summeval,
    title = "{S}umm{E}val: Re-evaluating Summarization Evaluation",
    author = "Fabbri, Alexander R.  and
      Kry{\'s}ci{\'n}ski, Wojciech  and
      McCann, Bryan  and
      Xiong, Caiming  and
      Socher, Richard  and
      Radev, Dragomir",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.tacl-1.24/",
    doi = "10.1162/tacl_a_00373",
    pages = "391--409",
}

@inproceedings{liu-etal-2023-g,
    title = "{G}-Eval: {NLG} Evaluation using Gpt-4 with Better Human Alignment",
    author = "Liu, Yang  and
      Iter, Dan  and
      Xu, Yichong  and
      Wang, Shuohang  and
      Xu, Ruochen  and
      Zhu, Chenguang",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.153/",
    doi = "10.18653/v1/2023.emnlp-main.153",
    pages = "2511--2522",
}
```

## Citation

```bibtex
@misc{fujinuma2026contrastivedecodingmitigatesscore,
      title={Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge}, 
      author={Yoshinari Fujinuma},
      year={2026},
      eprint={2510.18196},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.18196}, 
}
```
