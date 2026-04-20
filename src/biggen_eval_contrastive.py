#!/usr/bin/env python3
"""
BiGGen-Bench evaluation with contrastive decoding support.
Adapted from infer_llada_judge_biggen.py to use contrastive decoding instead of LLaDA.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import argparse
import numpy as np
import re
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Tuple

from contrastive_decoding import ContrastiveLlamaForCausalLM, ContrastiveQwen2ForCausalLM


def parse_result_score(generated_text):
    """
    Parse score from text in format '[RESULT] X' where X is a number
    If regex fails, try to extract the last number from the text
    Returns the parsed score as float, or None if not found
    """
    # First try the [RESULT] X format
    match = re.search(r'\[RESULT\]\s*(\d+(?:\.\d+)?)', generated_text)
    if match:
        return float(match.group(1))

    match = re.search(r'\s*(\d+(?:\.\d+)?)\s*out of', generated_text)
    if match:
        return float(match.group(1))

    # Fallback: get the last number in the text
    numbers = re.findall(r'\d+(?:\.\d+)?', generated_text)
    if numbers:
        return float(numbers[-1])

    return None


def load_biggen_data_from_hf(dataset_id, subset, split):
    """Load BiGGen-Bench or Feedback-Collection data from HuggingFace datasets"""
    print(f"Loading dataset {dataset_id} (subset: {subset}, split: {split})...")

    # Handle subset parameter - if it's "default" or None, don't pass it
    if subset == "default" or subset is None:
        dataset = load_dataset(dataset_id, split=split)
    else:
        dataset = load_dataset(dataset_id, subset, split=split)

    print(f"Dataset fields: {dataset.column_names if hasattr(dataset, 'column_names') else list(dataset[0].keys())}")

    # Print first example to see structure
    if len(dataset) > 0:
        print(f"First example keys: {list(dataset[0].keys())}")
        print(f"First example sample: {str(dataset[0])[:500]}...")

    # Detect dataset format
    is_feedback_collection = 'orig_instruction' in dataset[0] if len(dataset) > 0 else False

    # Convert to list of dictionaries
    data = []
    for example in dataset:
        if is_feedback_collection:
            # Feedback-Collection dataset format with 'orig_' prefix
            rubric = {}
            if 'orig_criteria' in example:
                rubric['criteria'] = example['orig_criteria']
            if 'orig_score1_description' in example:
                rubric['score1'] = example['orig_score1_description']
                rubric['score2'] = example.get('orig_score2_description', '')
                rubric['score3'] = example.get('orig_score3_description', '')
                rubric['score4'] = example.get('orig_score4_description', '')
                rubric['score5'] = example.get('orig_score5_description', '')

            data.append({
                'source': example.get('orig_instruction', ''),
                'reference': example.get('orig_reference_answer', ''),
                'system_output': example.get('orig_response', ''),
                'scores': {
                    'overall': float(example.get('orig_score', 0.0))
                },
                'custom_rubric': rubric if rubric else None,
                'capability': example.get('capability', ''),
                'task': example.get('task', ''),
            })
        else:
            # BiGGen-Bench format
            rubric = {}
            if 'score_rubric' in example and example['score_rubric']:
                score_rubric = example['score_rubric']
                if 'criteria' in score_rubric:
                    rubric['criteria'] = score_rubric['criteria']
                if 'score1_description' in score_rubric:
                    rubric['score1'] = score_rubric['score1_description']
                    rubric['score2'] = score_rubric.get('score2_description', '')
                    rubric['score3'] = score_rubric.get('score3_description', '')
                    rubric['score4'] = score_rubric.get('score4_description', '')
                    rubric['score5'] = score_rubric.get('score5_description', '')

            data.append({
                'source': example.get('instruction', example.get('input', example.get('prompt', ''))),
                'reference': example.get('reference', example.get('reference_answer', example.get('gold_answer', ''))),
                'system_output': example.get('output', example.get('response', example.get('answer', example.get('generated_text', '')))),
                'scores': {
                    'overall': float(example.get('score', example.get('human_score', example.get('rating', 0.0))))
                },
                'custom_rubric': rubric if rubric else None,
                'capability': example.get('capability', ''),
                'task': example.get('task', ''),
            })

    return data


def create_judge_prompt_template(orig_instruction, orig_response, orig_reference_answer, orig_criteria,
                                score1_text, score2_text, score3_text, score4_text, score5_text,
                                min_score=1, max_score=5, custom_prompt_suffix=None):
    """Create evaluation prompt using the BiGGen-Bench judge template format"""

    # Calculate score labels based on the range
    score_labels = [min_score + i for i in range(5)]

    prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of {max_score}, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between {min_score} and {max_score}. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between {min_score} and {max_score})"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score {max_score}):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score {score_labels[0]}: {score1_text}
Score {score_labels[1]}: {score2_text}
Score {score_labels[2]}: {score3_text}
Score {score_labels[3]}: {score4_text}
Score {score_labels[4]}: {score5_text}

###Feedback:
"""

    # Append custom prompt suffix if provided
    if custom_prompt_suffix:
        prompt = prompt.rstrip() + " " + custom_prompt_suffix

    return prompt


def get_default_score_rubrics():
    """Get default score rubric descriptions"""
    return {
        'criteria': "Overall Quality: The overall quality of the response considering accuracy, completeness, relevance, and helpfulness.",
        'score1': "Very poor - The response is largely incorrect, irrelevant, or unhelpful",
        'score2': "Poor - The response has significant issues but shows some understanding",
        'score3': "Moderate - The response is acceptable but has notable limitations",
        'score4': "Good - The response is strong with only minor issues",
        'score5': "Excellent - The response is comprehensive, accurate, and highly helpful"
    }


class BiGGenEvaluator:
    """Evaluates models on BiGGen-Bench using contrastive decoding."""

    def __init__(
        self,
        model_name: str,
        assistant_model_name: str = None,
        decoding: str = 'greedy',
        max_new_tokens: int = 256,
        min_score: float = None,
        max_score: float = None,
        custom_prompt_suffix: str = None
    ):
        self.model_name = model_name
        self.assistant_model_name = assistant_model_name
        self.decoding = decoding
        self.max_new_tokens = max_new_tokens
        self.min_score = min_score
        self.max_score = max_score
        self.custom_prompt_suffix = custom_prompt_suffix

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize the main model and assistant model if needed."""
        if self.decoding == 'contrastive':
            if not self.assistant_model_name:
                raise ValueError("Assistant model required for contrastive decoding")

            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                self.assistant_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
            self.assistant_tokenizer = AutoTokenizer.from_pretrained(self.assistant_model_name)

            if "Qwen" in self.model_name:
                self.model = ContrastiveQwen2ForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
                # Handle vocab size issues for Qwen models
                self.model.config.vocab_size = len(self.tokenizer)
                if hasattr(self.model.config, 'get_text_config'):
                    self.model.config.get_text_config().vocab_size = len(self.tokenizer)
                self.model.resize_token_embeddings(len(self.tokenizer))

                self.assistant_model.config.vocab_size = len(self.assistant_tokenizer)
                if hasattr(self.assistant_model.config, 'get_text_config'):
                    self.assistant_model.config.get_text_config().vocab_size = len(self.assistant_tokenizer)
                self.assistant_model.resize_token_embeddings(len(self.assistant_tokenizer))
            else:
                self.model = ContrastiveLlamaForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            if self.decoding == 'contrastive':
                self.assistant_model = self.assistant_model.to('cuda')

    def _generate_responses(self, batch_prompts: List[str], **kwargs) -> Tuple[torch.Tensor, List, torch.Tensor]:
        """Generate responses using the specified decoding strategy."""
        # Tokenize batch
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Generate based on decoding strategy
        if self.decoding == 'contrastive':
            outputs = self.model.generate_contrastive(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                assistant_model=self.assistant_model,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )
        elif self.decoding == 'greedy':
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )
        else:  # sampling
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )

        return outputs.sequences, outputs.scores, inputs.input_ids

    def evaluate_batch(
        self,
        batch: List[Dict],
        **kwargs
    ) -> List[Dict]:
        """Evaluate a batch of BiGGen-Bench instances."""
        batch_prompts = []

        # Prepare prompts for the batch
        for instance in batch:
            instruction = instance['source']
            response = instance['system_output']
            reference = instance['reference']

            # Use custom rubric if available, otherwise use default
            if instance.get('custom_rubric') and instance['custom_rubric'].get('criteria'):
                rubric = instance['custom_rubric']
                criteria = rubric['criteria']
                score_descriptions = [
                    rubric.get('score1', ''),
                    rubric.get('score2', ''),
                    rubric.get('score3', ''),
                    rubric.get('score4', ''),
                    rubric.get('score5', '')
                ]
            else:
                default_rubric = get_default_score_rubrics()
                criteria = default_rubric['criteria']
                score_descriptions = [
                    default_rubric['score1'],
                    default_rubric['score2'],
                    default_rubric['score3'],
                    default_rubric['score4'],
                    default_rubric['score5']
                ]

            # Create prompt using judge template with appropriate score range
            min_score = self.min_score if self.min_score is not None else 1
            max_score = self.max_score if self.max_score is not None else 5
            prompt = create_judge_prompt_template(
                instruction, response, reference, criteria, *score_descriptions,
                min_score=min_score, max_score=max_score,
                custom_prompt_suffix=self.custom_prompt_suffix
            )

            # Format using chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(formatted_prompt)

        # Generate responses
        sequences, logits, input_ids = self._generate_responses(batch_prompts, **kwargs)

        # Process results
        results = []
        for idx, instance in enumerate(batch):
            # Decode response (find actual input length excluding padding)
            input_length = input_ids[idx].shape[0]
            generated_text = self.tokenizer.decode(
                sequences[idx][input_length:],
                skip_special_tokens=True
            ).strip()

            # Parse score
            parsed_score = parse_result_score(generated_text)

            # Clamp predicted score to range if specified
            if parsed_score is not None and (self.min_score is not None or self.max_score is not None):
                original_parsed_score = parsed_score
                if self.min_score is not None and parsed_score < self.min_score:
                    parsed_score = self.min_score
                if self.max_score is not None and parsed_score > self.max_score:
                    parsed_score = self.max_score

            original_score = instance['scores']['overall']

            # Create result
            result = instance.copy()
            result['model_response'] = generated_text
            result['parsed_score'] = parsed_score
            result['score_difference'] = abs(parsed_score - original_score) if parsed_score is not None else None
            results.append(result)

        return results

    def evaluate_dataset(
        self,
        data: List[Dict],
        batch_size: int = 8,
        **kwargs
    ) -> Tuple[List[Dict], Dict]:
        """
        Evaluate entire BiGGen-Bench dataset.

        Returns:
            Tuple of (results list, summary statistics dict)
        """
        print(f"Evaluating {len(data)} BiGGen-Bench instances")
        all_results = []

        # Initialize tracking variables
        total_examples = 0
        parsed_scores = 0
        exact_matches = 0
        score_differences = []
        predicted_scores = []
        original_scores = []

        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data[batch_start:batch_end]

            print(f"\nProcessing batch {batch_start//batch_size + 1}: examples {batch_start+1}-{batch_end}")

            # Evaluate batch
            batch_results = self.evaluate_batch(batch, **kwargs)
            all_results.extend(batch_results)

            # Update statistics
            for result in batch_results:
                total_examples += 1
                if result['parsed_score'] is not None:
                    parsed_scores += 1
                    predicted_scores.append(result['parsed_score'])
                    original_scores.append(result['scores']['overall'])
                    score_differences.append(result['score_difference'])

                    if result['score_difference'] < 0.01:
                        exact_matches += 1

                    print(f"  Example: Original={result['scores']['overall']:.1f}, "
                          f"Predicted={result['parsed_score']:.1f}, "
                          f"Diff={result['score_difference']:.3f}")

        # Compute summary statistics
        summary = {
            'total_examples': total_examples,
            'successfully_parsed': parsed_scores,
            'parse_rate': parsed_scores / total_examples if total_examples > 0 else 0,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_matches / parsed_scores if parsed_scores > 0 else 0,
        }

        if score_differences:
            summary['mean_absolute_difference'] = float(np.mean(score_differences))
            summary['median_absolute_difference'] = float(np.median(score_differences))
            summary['min_difference'] = float(np.min(score_differences))
            summary['max_difference'] = float(np.max(score_differences))

        # Compute correlations
        if len(predicted_scores) >= 2:
            pearson_corr, pearson_p = pearsonr(predicted_scores, original_scores)
            spearman_corr, spearman_p = spearmanr(predicted_scores, original_scores)
            kendall_corr, kendall_p = kendalltau(predicted_scores, original_scores)

            summary['pearson_correlation'] = float(pearson_corr)
            summary['pearson_p_value'] = float(pearson_p)
            summary['spearman_correlation'] = float(spearman_corr)
            summary['spearman_p_value'] = float(spearman_p)
            summary['kendall_correlation'] = float(kendall_corr)
            summary['kendall_p_value'] = float(kendall_p)

        return all_results, summary


def main():
    parser = argparse.ArgumentParser(
        description='BiGGen-Bench evaluation with contrastive decoding'
    )

    # Data parameters
    parser.add_argument('--hf_dataset', type=str, default='prometheus-eval/BiGGen-Bench-Results',
                       help='HuggingFace dataset ID')
    parser.add_argument('--hf_subset', type=str, default='default',
                       help='HuggingFace dataset subset/configuration')
    parser.add_argument('--hf_split', type=str, default='human_eval',
                       help='Dataset split to use (default: human_eval)')
    parser.add_argument('--num_examples', type=int, default=-1,
                       help='Number of examples to evaluate (-1 for all)')
    parser.add_argument('--save_fp', type=str, default='results/biggen_results.json',
                       help='Path to save results')
    parser.add_argument('--min_score', type=float, default=None,
                       help='Minimum score to include (inclusive)')
    parser.add_argument('--max_score', type=float, default=None,
                       help='Maximum score to include (inclusive)')

    # Model parameters
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Main model for evaluation')
    parser.add_argument('--assistant_model', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                       help='Assistant model for contrastive decoding')

    # Generation parameters
    parser.add_argument('--decoding', choices=['contrastive', 'greedy', 'sampling'],
                       default='greedy', help='Decoding strategy to use')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='Maximum number of tokens to generate')

    # Contrastive decoding parameters
    parser.add_argument('--contrastive_lamb', type=float, default=1.0,
                       help='Lambda parameter for contrastive decoding')
    parser.add_argument('--contrastive_asst_temperature', type=float, default=1.0,
                       help='Temperature for assistant model in contrastive decoding')

    # Prompt customization
    parser.add_argument('--custom_prompt_suffix', type=str, default=None,
                       help='Custom text to append at the end of the prompt (e.g., "What is the score? Provide only rating and no other text. [RESULT] ")')

    args = parser.parse_args()

    # Load BiGGen-Bench data
    biggen_data = load_biggen_data_from_hf(args.hf_dataset, args.hf_subset, args.hf_split)
    print(f"Loaded {len(biggen_data)} examples from {args.hf_dataset}")

    # Shift gold scores if range is specified
    # Original scores are 1-5, shift them based on min_score
    if args.min_score is not None:
        shift_amount = args.min_score - 1
        for example in biggen_data:
            original_score = example['scores']['overall']
            shifted_score = original_score + shift_amount
            example['scores']['overall'] = shifted_score
            example['original_score'] = original_score

        print(f"Shifted all gold scores by {shift_amount:+.0f} to range [{args.min_score}, {args.max_score}]")
        print(f"Original range [1, 5] → New range [{1 + shift_amount}, {5 + shift_amount}]")

    # Limit number of examples if specified
    if args.num_examples > 0:
        biggen_data = biggen_data[:args.num_examples]
        print(f"Using first {args.num_examples} examples")

    # Initialize evaluator
    evaluator = BiGGenEvaluator(
        model_name=args.model,
        assistant_model_name=args.assistant_model if args.decoding == 'contrastive' else None,
        decoding=args.decoding,
        max_new_tokens=args.max_new_tokens,
        min_score=args.min_score,
        max_score=args.max_score,
        custom_prompt_suffix=args.custom_prompt_suffix
    )

    # Set contrastive parameters if using contrastive decoding
    if args.decoding == 'contrastive':
        evaluator.model.config.contrastive_lamb = args.contrastive_lamb
        evaluator.model.config.contrastive_asst_temperature = args.contrastive_asst_temperature

    # Evaluate
    results, summary = evaluator.evaluate_dataset(
        biggen_data,
        batch_size=args.batch_size
    )

    # Save results
    output_data = {
        'config': vars(args),
        'summary': summary,
        'results': results
    }

    import os
    os.makedirs(os.path.dirname(args.save_fp) if os.path.dirname(args.save_fp) else '.', exist_ok=True)
    with open(args.save_fp, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.save_fp}")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Decoding: {args.decoding}")
    if args.decoding == 'contrastive':
        print(f"Assistant Model: {args.assistant_model}")
        print(f"Lambda: {args.contrastive_lamb}")
        print(f"Temperature: {args.contrastive_asst_temperature}")
    print(f"\nTotal examples: {summary['total_examples']}")
    print(f"Successfully parsed: {summary['successfully_parsed']}/{summary['total_examples']} "
          f"({summary['parse_rate']*100:.1f}%)")
    print(f"Exact matches: {summary['exact_matches']}/{summary['successfully_parsed']} "
          f"({summary['exact_match_rate']*100:.1f}%)")

    if 'mean_absolute_difference' in summary:
        print(f"\nScore Differences:")
        print(f"  Mean: {summary['mean_absolute_difference']:.3f}")
        print(f"  Median: {summary['median_absolute_difference']:.3f}")
        print(f"  Range: [{summary['min_difference']:.3f}, {summary['max_difference']:.3f}]")

    if 'pearson_correlation' in summary:
        print(f"\nCorrelations:")
        print(f"  Pearson: {summary['pearson_correlation']:.4f} (p={summary['pearson_p_value']:.4f})")
        print(f"  Spearman: {summary['spearman_correlation']:.4f} (p={summary['spearman_p_value']:.4f})")
        print(f"  Kendall: {summary['kendall_correlation']:.4f} (p={summary['kendall_p_value']:.4f})")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
