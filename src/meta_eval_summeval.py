from collections import defaultdict
from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau, wilcoxon
import json
import re
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm


def bootstrap_correlation_ci(scores1, scores2, n_bootstrap=1000, alpha=0.05, corr_func=spearmanr):
    """
    Compute bootstrap confidence intervals for correlation

    Args:
        scores1: First set of scores
        scores2: Second set of scores
        n_bootstrap: Number of bootstrap iterations (default: 10000)
        alpha: Significance level (default: 0.05 for 95% CI)
        corr_func: Correlation function to use (default: spearmanr)

    Returns:
        tuple: (lower_bound, upper_bound) of confidence interval
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    n = len(scores1)
    bootstrap_corrs = []

    for _ in tqdm(range(n_bootstrap), desc=f"Bootstrap {corr_func.__name__}", leave=False):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        sample1 = scores1[indices]
        sample2 = scores2[indices]

        corr, _ = corr_func(sample1, sample2)
        bootstrap_corrs.append(corr)

    # Calculate confidence intervals
    lower = np.percentile(bootstrap_corrs, 100 * alpha/2)
    upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha/2))

    return lower, upper


def calculate_correlation(pred_score, human_score, result, compute_ci=True, n_bootstrap=100, alpha=0.05):
    assert len(pred_score) == len(human_score)

    if (len(result) == 0):
        result = {
            'pearson': 0, 'spearman': 0, 'kendalltau': 0,
            'pearson_list': [], 'spearman_list': [], 'kendalltau_list': [],
            'pearson_pval_list': [], 'spearman_pval_list': [], 'kendalltau_pval_list': []
        }

    # Compute correlations
    pearson_r, pearson_p = pearsonr(pred_score, human_score)
    spearman_r, spearman_p = spearmanr(pred_score, human_score)
    kendalltau_r, kendalltau_p = kendalltau(pred_score, human_score)

    result['pearson'] += pearson_r
    result['spearman'] += spearman_r
    result['kendalltau'] += kendalltau_r

    # Store individual document correlations for SE calculation
    result['pearson_list'].append(pearson_r)
    result['spearman_list'].append(spearman_r)
    result['kendalltau_list'].append(kendalltau_r)

    # Store p-values
    result['pearson_pval_list'].append(pearson_p)
    result['spearman_pval_list'].append(spearman_p)
    result['kendalltau_pval_list'].append(kendalltau_p)

    return result


def print_correlations(result, n):
    if (n == 0):
        n = 1

    # Calculate average correlations
    pearson_avg = result['pearson'] / n
    spearman_avg = result['spearman'] / n
    kendalltau_avg = result['kendalltau'] / n

    # Calculate average p-values
    pearson_pval_avg = np.mean(result.get('pearson_pval_list', [0])) if 'pearson_pval_list' in result else 0
    spearman_pval_avg = np.mean(result.get('spearman_pval_list', [0])) if 'spearman_pval_list' in result else 0
    kendalltau_pval_avg = np.mean(result.get('kendalltau_pval_list', [0])) if 'kendalltau_pval_list' in result else 0

    # Calculate SE-based confidence intervals if we have individual correlations
    has_list = 'pearson_list' in result and len(result['pearson_list']) > 0

    if has_list and len(result['pearson_list']) > 1:
        # Calculate standard error and 95% CI for each metric
        # CI = mean ± 1.96 * SE, where SE = std / sqrt(n)
        z_score = 1.96  # for 95% CI

        pearson_se = np.std(result['pearson_list'], ddof=1) / np.sqrt(len(result['pearson_list']))
        spearman_se = np.std(result['spearman_list'], ddof=1) / np.sqrt(len(result['spearman_list']))
        kendalltau_se = np.std(result['kendalltau_list'], ddof=1) / np.sqrt(len(result['kendalltau_list']))

        pearson_margin = z_score * pearson_se
        spearman_margin = z_score * spearman_se
        kendalltau_margin = z_score * kendalltau_se

        # Print table with correlations in +/- format and p-values
        table = PrettyTable(['Metric', 'Correlation ± 95% CI', 'p-value'])
        table.add_row(['Pearson', f'{round(pearson_avg, 3)} ± {round(pearson_margin, 3)}', f'{pearson_pval_avg:.4e}'])
        table.add_row(['Spearman', f'{round(spearman_avg, 3)} ± {round(spearman_margin, 3)}', f'{spearman_pval_avg:.4e}'])
        table.add_row(['Kendall', f'{round(kendalltau_avg, 3)} ± {round(kendalltau_margin, 3)}', f'{kendalltau_pval_avg:.4e}'])
    else:
        # Print table without CIs (backward compatibility)
        table = PrettyTable(['Metric', 'Correlation', 'p-value'])
        table.add_row(['Pearson', round(pearson_avg, 4), f'{pearson_pval_avg:.4e}'])
        table.add_row(['Spearman', round(spearman_avg, 4), f'{spearman_pval_avg:.4e}'])
        table.add_row(['Kendall', round(kendalltau_avg, 4), f'{kendalltau_pval_avg:.4e}'])

    print(table)


def print_pooled_correlations(result):
    """Print pooled correlations with bootstrap CIs"""
    # Calculate margin of error (half-width of CI)
    pearson_margin = (result['pearson_ci'][1] - result['pearson_ci'][0]) / 2
    spearman_margin = (result['spearman_ci'][1] - result['spearman_ci'][0]) / 2
    kendalltau_margin = (result['kendalltau_ci'][1] - result['kendalltau_ci'][0]) / 2

    # Print table with correlations in +/- format and p-values
    table = PrettyTable(['Metric', 'Correlation ± 95% CI', 'p-value'])
    table.add_row(['Pearson', f'{round(result["pearson"], 3)} ± {round(pearson_margin, 3)}', f'{result["pearson_pval"]:.4e}'])
    table.add_row(['Spearman', f'{round(result["spearman"], 3)} ± {round(spearman_margin, 3)}', f'{result["spearman_pval"]:.4e}'])
    table.add_row(['Kendall', f'{round(result["kendalltau"], 3)} ± {round(kendalltau_margin, 3)}', f'{result["kendalltau_pval"]:.4e}'])

    print(table)


def compare_correlations_wilcoxon(correlations1, correlations2, method1_name="Method 1", method2_name="Method 2"):
    """
    Compare two sets of paired document-level correlations using Wilcoxon Signed-Ranks Test

    Args:
        correlations1: Dict with keys 'pearson', 'spearman', 'kendalltau' containing lists
        correlations2: Dict with keys 'pearson', 'spearman', 'kendalltau' containing lists
        method1_name: Name of first method
        method2_name: Name of second method

    Returns:
        Dict with test results for each metric
    """
    print(f"\n=== Wilcoxon Signed-Ranks Test: {method1_name} vs {method2_name} ===")

    results = {}
    table = PrettyTable(['Metric', f'{method1_name} Mean', f'{method2_name} Mean', 'Difference', 'p-value', 'Significant'])

    for metric in ['pearson', 'spearman', 'kendalltau']:
        corrs1 = np.array(correlations1[f'{metric}_list'])
        corrs2 = np.array(correlations2[f'{metric}_list'])

        # Perform Wilcoxon test
        statistic, p_value = wilcoxon(corrs1, corrs2, alternative='two-sided')

        mean1 = np.mean(corrs1)
        mean2 = np.mean(corrs2)
        diff = mean2 - mean1

        # Determine significance
        significant = "✓" if p_value < 0.05 else "✗"

        metric_name = metric.capitalize() if metric != 'kendalltau' else 'Kendall'
        table.add_row([
            metric_name,
            round(mean1, 3),
            round(mean2, 3),
            f"{diff:+.3f}",
            f"{p_value:.4f}",
            significant
        ])

        results[metric] = {
            'statistic': statistic,
            'p_value': p_value,
            'mean1': mean1,
            'mean2': mean2,
            'difference': diff
        }

    print(table)
    return results


def parse_output(output, ignore_score, counts, max_score, non_numerical_count, clamping_count):
    output = output.lower()
    output = output.replace("coherence", "")
    output = output.replace("relevance", "")
    output = output.replace("score", "")
    output = output.replace(": ", "")
    output = output.replace("- ", "")
    output = output.replace("answer", "")
    matched = re.search(r"^ ?([+-]?[\d\.]+)", output)
    #print(output, matched)
    is_non_numerical = False
    if (matched):
        try:
            score = float(matched.group(1))
        except:
            score = ignore_score
            is_non_numerical = True
    else:
        score = ignore_score
        is_non_numerical = True

    # Count non-numerical outputs
    if is_non_numerical:
        non_numerical_count[0] += 1

    # Track clamping
    original_score = score
    if score < ignore_score:
        score = ignore_score
        clamping_count[0] += 1
    if score > max_score:
        score = max_score
        clamping_count[1] += 1

    if int(score) in [0, 1, 2, 3, 4, 5, 6, 7]:
        counts[int(score)] += 1
    else:
        counts[ignore_score] += 1
    #print(f"output: {output}")
    #print(f"score: {score}")
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fp', type=str, default='results/gpt4_rel_detailed.json')
    parser.add_argument('--dimension', type=str, default='relevance')
    parser.add_argument('--ignore_score', type=float, default=0)
    parser.add_argument('--max_score', type=float, default=4)
    parser.add_argument('--skip_per_doc', action='store_true', help='Skip per-document average correlation output')
    args = parser.parse_args()

    print(f"Processing file: {args.input_fp}")
    jobj = json.load(open(args.input_fp))
    pred_scores, human_scores = {}, {}

    #counts = defaultdict(int)
    counts = {}
    for i in range(int(args.ignore_score), int(args.max_score) + 1):
        counts[i] = 0

    # Counters for non-numerical outputs and clamping
    non_numerical_count = [0]  # Using list to make it mutable in function
    clamping_count = [0, 0]  # [below_min, above_max]
    total_outputs = 0

    print("Calculating correlation for G-Eval")
    for item in jobj:
        doc_id = item["doc_id"]
        if (doc_id not in pred_scores):
            pred_scores[doc_id] = []
            human_scores[doc_id] = []

        all_responses = item["all_responses"]
        total_outputs += len(all_responses)
        #import ipdb; ipdb.set_trace(context=20)
        all_scores = [parse_output(x, args.ignore_score, counts, args.max_score, non_numerical_count, clamping_count) for x in all_responses]
        score = sum(all_scores) / len(all_scores)

        pred_scores[doc_id].append(score)
        human_scores[doc_id].append(item['scores'][args.dimension])

    print('len(pred_scores): {}'.format(len(pred_scores)))
    print('len(human_scores): {}'.format(len(human_scores)))

    results = {
        'pearson': 0, 'spearman': 0, 'kendalltau': 0,
        'pearson_list': [], 'spearman_list': [], 'kendalltau_list': [],
        'pearson_pval_list': [], 'spearman_pval_list': [], 'kendalltau_pval_list': []
    }
    d_ctr = 0

    # Collect all scores for pooled correlation
    all_pred_scores = []
    all_human_scores = []
    all_pred_scores_pool = []
    all_human_scores_pool = []

    for doc_id in pred_scores:
        pred_scores_doc = pred_scores[doc_id]
        human_scores_doc = human_scores[doc_id]
        all_pred_scores_pool.extend(pred_scores_doc)
        all_human_scores_pool.extend(human_scores_doc)

        if (len(set(human_scores_doc)) <= 1) or (len(set(pred_scores_doc)) <= 1):
            continue

        results = calculate_correlation(pred_scores_doc, human_scores_doc, results)
        d_ctr += 1

        # Collect for pooled analysis
        all_pred_scores.extend(pred_scores_doc)
        all_human_scores.extend(human_scores_doc)

    # Compute pooled correlation across all data points
    all_pred_scores = np.array(all_pred_scores_pool)
    all_human_scores = np.array(all_human_scores_pool)

    print(f"\nPooled data points: {len(all_pred_scores)}")

    if len(all_pred_scores) < 2:
        print("WARNING: Not enough data points for pooled correlation (need at least 2)")
        pooled_results = None
    else:
        pearson_r, pearson_p = pearsonr(all_pred_scores, all_human_scores)
        spearman_r, spearman_p = spearmanr(all_pred_scores, all_human_scores)
        kendalltau_r, kendalltau_p = kendalltau(all_pred_scores, all_human_scores)

        pooled_results = {
            'pearson': pearson_r,
            'spearman': spearman_r,
            'kendalltau': kendalltau_r,
            'pearson_pval': pearson_p,
            'spearman_pval': spearman_p,
            'kendalltau_pval': kendalltau_p,
        }

        # Compute bootstrap CIs for pooled correlations
        pooled_results['pearson_ci'] = bootstrap_correlation_ci(all_pred_scores, all_human_scores, corr_func=pearsonr)
        pooled_results['spearman_ci'] = bootstrap_correlation_ci(all_pred_scores, all_human_scores, corr_func=spearmanr)
        pooled_results['kendalltau_ci'] = bootstrap_correlation_ci(all_pred_scores, all_human_scores, corr_func=kendalltau)

    if not args.skip_per_doc:
        print("\n=== Per-Document Average Correlation ===")
        print_correlations(results, n=d_ctr)

    if pooled_results is not None:
        print("\n=== Pooled Correlation (All Data Points) ===")
        print_pooled_correlations(pooled_results)
    else:
        print("\n=== Pooled Correlation (All Data Points) ===")
        print("Skipped: Not enough data points (all documents had constant scores)")
    print(f"counts={dict(sorted(counts.items()))}")
    print(f"\n=== Output Statistics ===")
    print(f"Total outputs: {total_outputs}")
    print(f"Non-numerical outputs: {non_numerical_count[0]} ({100 * non_numerical_count[0] / total_outputs:.2f}%)")
    print(f"Clamped below minimum: {clamping_count[0]} ({100 * clamping_count[0] / total_outputs:.2f}%)")
    print(f"Clamped above maximum: {clamping_count[1]} ({100 * clamping_count[1] / total_outputs:.2f}%)")
    print(f"Total clamped: {sum(clamping_count)} ({100 * sum(clamping_count) / total_outputs:.2f}%)")
    
    # Create a bar plot of the counts
    #plt.figure(figsize=(10, 6))
    #counts_dict = dict(sorted(counts.items()))
    #print(counts_dict)
    #plt.bar(counts_dict.keys(), counts_dict.values())
    #plt.xlabel('Scores')
    #plt.ylabel('Frequency')
    #plt.title(f'Distribution of Parsed Scores - {os.path.basename(args.input_fp)}')
    #plt.grid(True, alpha=0.3)
    ## Extract filename from input_fp path
    #plot_filename = f'score_distribution_{args.input_fp.split("/")[-1].replace(".json", "")}.png'
    #plt.savefig(plot_filename)
    #plt.close()
