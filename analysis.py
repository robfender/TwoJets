#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Analysis Module

This module consolidates functionality from:
- population_analysis.py
- spin_analysis.py
- uncertainty_analysis.py
- threshold_analysis.py

It provides functions for population distribution analysis, spin correlation analysis,
and uncertainty/threshold analysis using Monte Carlo simulations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
from tqdm import tqdm
import os
import sys


# ------------------------------------------------------------------------------
# Population Analysis Functions
# ------------------------------------------------------------------------------

def perform_population_analysis(all_jets_list, categories):
    """
    Perform statistical analyses on jet populations
    
    Parameters:
    all_jets_list: List of jet data
    categories: Dictionary of categorized data
    
    Returns:
    Dictionary with analysis results
    """
    print("Extracting data for population analysis...")

    # Extract categories
    bh = categories['bh']
    nsnourf = categories['nsnourf']
    prec = categories['prec']
    fixednourf = categories['fixednourf']

    # Extract beta*gamma values for statistical tests
    bh_betagammas = [jet[7] for jet in bh]
    ns_betagammas = [jet[7] for jet in nsnourf]
    prec_betagammas = [jet[7] for jet in prec]
    fixed_betagammas = [jet[7] for jet in fixednourf]

    # Run statistical tests
    print("\nRunning statistical tests...")

    # 1. All BH vs NS
    test1 = run_statistical_tests(
        bh_betagammas,
        ns_betagammas,
        "All BH vs NS actual speeds (with no URFs)",
        dataset1_name="Black Holes",
        dataset2_name="Neutron Stars"
    )

    # 2. Precessing vs Fixed axis
    test2 = run_statistical_tests(
        prec_betagammas,
        fixed_betagammas,
        "Precessing vs definitely fixed axis actual speeds with no URFs",
        dataset1_name="Precessing Jets",
        dataset2_name="Fixed Axis Jets"
    )

    # 3. Placeholder test (currently comparing BH to itself)
    test3 = run_statistical_tests(
        bh_betagammas,
        bh_betagammas,
        "Placeholder for new test",
        dataset1_name="Sample 1",
        dataset2_name="Sample 2"
    )

    # Create a summary of all statistical test results
    print(f"\n{'=' * 80}")
    print("SUMMARY OF STATISTICAL TESTS")
    print(f"{'=' * 80}")

    summary_df = pd.DataFrame({
        'Comparison': [
            'All BH vs NS',
            'Precessing vs Fixed',
            'Placeholder'
        ],
        'K-S p-value': [
            test1['ks_pval'],
            test2['ks_pval'],
            test3['ks_pval']
        ],
        'K-S Result': [
            test1['ks_result'],
            test2['ks_result'],
            test3['ks_result']
        ],
        'A-D p-value': [
            test1['ad_pval'],
            test2['ad_pval'],
            test3['ad_pval']
        ],
        'A-D Result': [
            test1['ad_result'],
            test2['ad_result'],
            test3['ad_result']
        ],
    })

    print(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))

    # Save results to CSV for later reference
    summary_df.to_csv('population_statistical_results.csv', index=False)
    print("\nResults saved to 'population_statistical_results.csv'")

    # Compile results into a dictionary
    results = {
        'populations': {
            'bh_betagammas': bh_betagammas,
            'ns_betagammas': ns_betagammas,
            'slowbh_betagammas': slowbh_betagammas,
            'prec_betagammas': prec_betagammas,
            'fixed_betagammas': fixed_betagammas,
            'nonprec_betagammas': nonprec_betagammas
        },
        'statistical_tests': {
            'bh_vs_ns': test1,
            'slowbh_vs_ns': test2,
            'prec_vs_fixed': test3
        }
    }

    return results


def run_statistical_tests(data1, data2, label, alpha=0.05, dataset1_name=None, dataset2_name=None):
    """
    Run K-S and A-D tests on two datasets and present results in a clear format
    
    Parameters:
    data1, data2: Lists of data to compare
    label: Description of the comparison being made
    alpha: Significance level (default 0.05)
    dataset1_name: Name of the first dataset (default None, uses "Dataset 1")
    dataset2_name: Name of the second dataset (default None, uses "Dataset 2")
    
    Returns:
    Dictionary with test results
    """
    print(f"\n{'=' * 80}")
    print(f"STATISTICAL COMPARISON: {label}")
    print(f"{'=' * 80}")

    # Set default names if not provided
    dataset1_name = dataset1_name or "Dataset 1"
    dataset2_name = dataset2_name or "Dataset 2"

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(data1, data2)

    # Anderson-Darling test
    # Note: scipy's implementation differs slightly from Mathematica's
    # Using the k-sample Anderson-Darling test
    ad_stat, ad_crit, ad_sig = stats.anderson_ksamp([data1, data2])
    # Convert significance level to p-value (approximate)
    ad_pval = ad_sig

    # Create a nice table for test statistics
    ks_result = "rejected" if ks_pval < alpha else "not rejected"
    ad_result = "rejected" if ad_pval < alpha else "not rejected"

    # DataFrame for nice tabular display
    results_df = pd.DataFrame({
        'Test': ['Kolmogorov-Smirnov', 'Anderson-Darling'],
        'Statistic': [ks_stat, ad_stat],
        'P-Value': [ks_pval, ad_pval],
        'Conclusion': [
            f"The null hypothesis is {ks_result} at the {alpha * 100:.0f}% significance level",
            f"The null hypothesis is {ad_result} at the {alpha * 100:.0f}% significance level"
        ]
    })

    # Display results
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))

    # Additional information
    print("\nSample sizes:")
    print(f"{dataset1_name}: {len(data1)} observations")
    print(f"{dataset2_name}: {len(data2)} observations")

    # Summary statistics
    print("\nSummary Statistics (for beta*gamma):")
    stats_df = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        dataset1_name: [
            np.mean(data1),
            np.median(data1),
            np.std(data1),
            np.min(data1),
            np.max(data1)
        ],
        dataset2_name: [
            np.mean(data2),
            np.median(data2),
            np.std(data2),
            np.min(data2),
            np.max(data2)
        ]
    })

    print(tabulate(stats_df, headers='keys', tablefmt='grid', showindex=False))

    return {
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'ad_stat': ad_stat,
        'ad_pval': ad_pval,
        'ks_result': ks_result,
        'ad_result': ad_result
    }


# ------------------------------------------------------------------------------
# Spin Analysis Functions 
# ------------------------------------------------------------------------------

def perform_spin_analysis(all_jets_list, categories, alpha=0.05):
    """
    Analyze correlations between black hole spin measurements and βγ
    
    Parameters:
    all_jets_list: List of jet data
    categories: Dictionary of categorized data
    alpha: Significance level for statistical tests (default 0.05)
    
    Returns:
    Dictionary with spin correlation results
    """
    print(f"\n{'=' * 80}")
    print("SPIN CORRELATION ANALYSIS")
    print(f"{'=' * 80}")

    # Extract paired data
    rr = categories['rr']  # Regular measurements (Reflection)
    rrl = categories['rrl']  # Lower speed limits (Reflection)
    cc = categories['cc']  # Regular measurements (Continuum)
    ccl = categories['ccl']  # Lower speed limits (Continuum) 
    qq = categories['qq']  # Regular measurements (QPO)
    qql = categories['qql']  # Lower speed limits (QPO)

    # Analyze each spin measurement method
    reflection_stats = analyze_spin_correlation(rr, rrl, "Reflection Spin vs. βγ", alpha)
    continuum_stats = analyze_spin_correlation(cc, ccl, "Continuum Spin vs. βγ", alpha)
    qpo_stats = analyze_spin_correlation(qq, qql, "QPO Spin vs. βγ", alpha)

    # Create a summary of all correlation results
    print(f"\n{'=' * 80}")
    print("SUMMARY OF SPIN CORRELATION ANALYSES")
    print(f"{'=' * 80}")

    # Prepare summary data
    correlation_summary = []
    header = ['Spin Method', 'N (measurements)', 'N (limits)', 'Pearson r',
              'p-value', 'Spearman rho', 'p-value', "Kendall's tau", 'p-value',
              'Significant', 'Limits Consistent', 'Limits Tested', 'Limits Inconsistent']

    # Add data if available
    if reflection_stats:
        correlation_summary.append([
            'Reflection', len(rr), len(rrl),
            reflection_stats.get('pearson_r', 'N/A'),
            reflection_stats.get('pearson_p', 'N/A'),
            reflection_stats.get('spearman_r', 'N/A'),
            reflection_stats.get('spearman_p', 'N/A'),
            reflection_stats.get('kendall_tau', 'N/A'),
            reflection_stats.get('kendall_p', 'N/A'),
            'Yes' if reflection_stats.get('is_significant', False) else 'No',
            'Yes' if reflection_stats.get('limits_consistent', True) else 'No',
            reflection_stats.get('limits_tested', 0),
            reflection_stats.get('limits_inconsistent', 0)
        ])

    if continuum_stats:
        correlation_summary.append([
            'Continuum', len(cc), len(ccl),
            continuum_stats.get('pearson_r', 'N/A'),
            continuum_stats.get('pearson_p', 'N/A'),
            continuum_stats.get('spearman_r', 'N/A'),
            continuum_stats.get('spearman_p', 'N/A'),
            continuum_stats.get('kendall_tau', 'N/A'),
            continuum_stats.get('kendall_p', 'N/A'),
            'Yes' if continuum_stats.get('is_significant', False) else 'No',
            'Yes' if continuum_stats.get('limits_consistent', True) else 'No',
            continuum_stats.get('limits_tested', 0),
            continuum_stats.get('limits_inconsistent', 0)
        ])

    if qpo_stats:
        correlation_summary.append([
            'QPO', len(qq), len(qql),
            qpo_stats.get('pearson_r', 'N/A'),
            qpo_stats.get('pearson_p', 'N/A'),
            qpo_stats.get('spearman_r', 'N/A'),
            qpo_stats.get('spearman_p', 'N/A'),
            qpo_stats.get('kendall_tau', 'N/A'),
            qpo_stats.get('kendall_p', 'N/A'),
            'Yes' if qpo_stats.get('is_significant', False) else 'No',
            'Yes' if qpo_stats.get('limits_consistent', True) else 'No',
            qpo_stats.get('limits_tested', 0),
            qpo_stats.get('limits_inconsistent', 0)
        ])

    # Display summary if we have data
    if correlation_summary:
        summary_df = pd.DataFrame(correlation_summary, columns=header)
        print(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.4f'))

        # Save results to CSV
        summary_df.to_csv('spin_correlation_results.csv', index=False)
        print("\nResults saved to 'spin_correlation_results.csv'")
    else:
        print("Insufficient data for correlation summary.")

    # Perform censored data analysis if we have limits
    if len(rrl) > 0 or len(ccl) > 0 or len(qql) > 0:
        print("\nPerforming censored data analysis to account for lower limits...")
        censored_results = perform_censored_analysis(rr, rrl, cc, ccl, qq, qql)
    else:
        censored_results = None

    # Return combined results
    return {
        'reflection': reflection_stats,
        'continuum': continuum_stats,
        'qpo': qpo_stats,
        'censored_analysis': censored_results
    }


def analyze_spin_correlation(data_points, data_limits, label, alpha=0.05):
    """
    Analyze correlation between BH spin and βγ, ensuring consistency with lower limits
    
    Parameters:
    data_points: List of (spin, betagamma) pairs for regular measurements
    data_limits: List of (spin, betagamma) pairs for lower limits
    label: Description of the spin measurement method
    alpha: Significance level (default 0.05)
    
    Returns:
    Dictionary with correlation statistics or None if insufficient data
    """
    print(f"\n{'=' * 80}")
    print(f"CORRELATION ANALYSIS: {label}")
    print(f"{'=' * 80}")

    # Report both regular measurements and limits
    print(f"Regular measurements: {len(data_points)}")
    print(f"Lower limits: {len(data_limits)}")

    # Extract data for regular points (not limits)
    x_vals = [x for x, y in data_points]
    y_vals = [y for x, y in data_points]

    # Extract limit data
    x_limits = [x for x, y in data_limits]
    y_limits = [y for x, y in data_limits]

    # Don't include limits in basic correlation calculations
    if len(data_points) > 1:
        # Pearson correlation (parametric)
        pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)

        # Spearman rank correlation (non-parametric)
        spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)

        # Kendall's Tau (non-parametric)
        kendall_tau, kendall_p = stats.kendalltau(x_vals, y_vals)

        # Determine if any correlation is significant based on p-values
        correlation_is_significant = (pearson_p < alpha) or (spearman_p < alpha) or (kendall_p < alpha)

        # Create a nice table for correlation statistics
        results_df = pd.DataFrame({
            'Test': ['Pearson r', 'Spearman rho', "Kendall's tau"],
            'Coefficient': [pearson_r, spearman_r, kendall_tau],
            'P-Value': [pearson_p, spearman_p, kendall_p],
            'Conclusion': [
                f"{'Significant' if pearson_p < alpha else 'Not significant'} at {alpha * 100:.0f}% level",
                f"{'Significant' if spearman_p < alpha else 'Not significant'} at {alpha * 100:.0f}% level",
                f"{'Significant' if kendall_p < alpha else 'Not significant'} at {alpha * 100:.0f}% level"
            ]
        })

        # Display results
        print("\nCorrelation statistics (using only measurements, not limits):")
        print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))

        # Linear regression and limit consistency check
        limits_consistent = True
        inconsistent_limits = 0

        if len(x_vals) > 2:  # Need at least 3 points for meaningful regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
            print("\nLinear Regression:")
            print(f"Formula: βγ = {slope:.4f} × Spin + {intercept:.4f}")
            print(f"Standard Error: {std_err:.4f}")
            print(f"R-squared: {r_value ** 2:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(
                f"Regression is {'statistically significant' if p_value < alpha else 'not statistically significant'} at {alpha * 100:.0f}% level")

            # Check if limits are consistent with regression
            if data_limits:
                print("\nConsistency check for lower limits:")
                for i, (x_limit, y_limit) in enumerate(data_limits):
                    expected_y = slope * x_limit + intercept
                    consistent = y_limit >= expected_y
                    if not consistent:
                        inconsistent_limits += 1
                    print(
                        f"  Limit {i + 1}: Spin ≥ {x_limit:.2f}, βγ = {y_limit:.2f}, Predicted βγ = {expected_y:.2f}, {'Consistent' if consistent else 'Inconsistent'}")

                # If any limits are inconsistent, mark the regression as inconsistent
                if inconsistent_limits > 0:
                    limits_consistent = False
                    print(
                        f"\nWARNING: {inconsistent_limits} of {len(data_limits)} lower limits are inconsistent with the regression.")
                    print("This suggests the fit does not properly account for all available data.")
                else:
                    print("\nAll lower limits are consistent with the regression line.")

            # Determine final significance, accounting for both p-value and limit consistency
            regression_is_significant = p_value < alpha

            # Only consider the correlation significant if both the correlation/regression 
            # is significant AND all limits are consistent
            is_significant = (correlation_is_significant or regression_is_significant) and limits_consistent

            # Add a clear explanation of the final determination
            if is_significant:
                if not limits_consistent:
                    print("\nFatal error: This should never happen.")
                else:
                    print(
                        "\nFinal determination: Correlation is statistically significant and consistent with all lower limits.")
            else:
                if correlation_is_significant or regression_is_significant:
                    print("\nFinal determination: Despite statistical significance in the direct measurements,")
                    print("the correlation is rejected due to inconsistency with lower limits.")
                else:
                    print("\nFinal determination: Correlation is not statistically significant.")

            return {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'kendall_tau': kendall_tau,
                'kendall_p': kendall_p,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'std_err': std_err,
                'p_value': p_value,
                'limits_consistent': limits_consistent,
                'limits_tested': len(data_limits),
                'limits_inconsistent': inconsistent_limits,
                'is_significant': is_significant  # Now accounts for limit consistency
            }
        else:
            # If we don't have enough points for regression but still have correlation results
            is_significant = correlation_is_significant and limits_consistent
            return {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'kendall_tau': kendall_tau,
                'kendall_p': kendall_p,
                'limits_consistent': limits_consistent,
                'limits_tested': len(data_limits),
                'limits_inconsistent': inconsistent_limits,
                'is_significant': is_significant
            }
    else:
        if len(data_points) == 0:
            print("No direct measurements available for correlation analysis.")
        else:
            print("Only one measurement available - insufficient for correlation analysis.")

        if data_limits:
            print(f"\nHave {len(data_limits)} lower limits, but cannot perform standard correlation analysis.")

    return None


def perform_censored_analysis(rr, rrl, cc, ccl, qq, qql):
    """
    A simplified version of censored data analysis that doesn't require lifelines
    
    Parameters:
    rr, cc, qq: Regular measurement data for each method
    rrl, ccl, qql: Lower limit data for each method
    
    Returns:
    Dictionary with basic analysis results
    """
    results = {}

    # Only attempt analysis if we have sufficient data
    methods = [
        ('Reflection', rr, rrl),
        ('Continuum', cc, ccl),
        ('QPO', qq, qql)
    ]

    for name, measurements, limits in methods:
        if len(measurements) + len(limits) >= 3:  # Need at least 3 total data points
            print(f"\nBasic analysis for {name} spin limits:")

            # Display basic information about limits
            if limits:
                limit_bg_values = [bg for _, bg in limits]
                print(f"Number of lower limits: {len(limits)}")
                print(f"Range of βγ values for limits: {min(limit_bg_values):.3f} - {max(limit_bg_values):.3f}")

            results[name] = {
                'n_measurements': len(measurements),
                'n_limits': len(limits)
            }

    return results


# ------------------------------------------------------------------------------
# Uncertainty Analysis Functions
# ------------------------------------------------------------------------------

def run_population_uncertainty(all_jets_list, categories, num_iterations=1000,
                               angle_uncertainty=10, distance_uncertainty=0.1,
                               test_method="ks"):
    """
    Run Monte Carlo simulations to assess how angle and distance uncertainties 
    affect population distribution statistical conclusions
    
    Parameters:
    all_jets_list: Original jet data
    categories: Dictionary of categorized data
    num_iterations: Number of Monte Carlo iterations
    angle_uncertainty: Uncertainty in degrees (±) around the original inclination angle
    distance_uncertainty: Fractional uncertainty (±) in distance, e.g., 0.1 = 10%
    test_method: Statistical test to use - "ks" (Kolmogorov-Smirnov), 
                "ad" (Anderson-Darling), or "average" (average of both)
    
    Returns:
    Dictionary with analysis results
    """
    print(f"\n{'=' * 80}")
    print(f"POPULATION UNCERTAINTY ANALYSIS: Inclination Angle Variations (±{angle_uncertainty}°) and")
    print(f"                                 Distance Variations (±{distance_uncertainty * 100:.0f}%)")
    print(f"{'=' * 80}")
    print(f"Running {num_iterations} Monte Carlo iterations using {test_method.upper()} test method...")

    # Validate test method
    valid_methods = ["ks", "ad", "average"]
    if test_method not in valid_methods:
        raise ValueError(f"Invalid test_method '{test_method}'. Must be one of {valid_methods}")

    # Track statistical test results across all iterations
    bh_vs_ns_results = {'rejected': 0, 'not rejected': 0}
    prec_vs_fixed_results = {'rejected': 0, 'not rejected': 0}
    placeholder_results = {'rejected': 0, 'not rejected': 0}

    # Store p-values from all iterations
    bh_vs_ns_pvals = []
    prec_vs_fixed_pvals = []
    placeholder_pvals = []

    # Original lists for reference
    bh = categories['bh']
    nsnourf = categories['nsnourf']
    prec = categories['prec']
    fixednourf = categories['fixednourf']

    # For tracking how changes affect βγ distributions
    original_values = {
        'bh_betagammas': [jet[7] for jet in bh],
        'ns_betagammas': [jet[7] for jet in nsnourf],
        'prec_betagammas': [jet[7] for jet in prec],
        'fixed_betagammas': [jet[7] for jet in fixednourf]
    }

    # Track perturbed values to analyze distribution changes
    all_perturbed = {
        'bh_betagammas': [],
        'ns_betagammas': [],
        'prec_betagammas': [],
        'fixed_betagammas': []
    }

    # Run Monte Carlo iterations
    for i in tqdm(range(num_iterations)):
        # Create a deep copy of the original jets list
        perturbed_jets = []
        for jet in all_jets_list:
            # Create a new jet entry with same values
            new_jet = jet.copy()

            # 1. Perturb inclination angle
            original_angle = jet[4]
            # Generate uniform random value within uncertainty bounds
            perturbed_angle = original_angle + np.random.uniform(-angle_uncertainty, angle_uncertainty)
            # Ensure angle is within physical limits (0-90 degrees)
            perturbed_angle = max(0, min(90, perturbed_angle))
            new_jet[4] = perturbed_angle

            # 2. Perturb distance
            original_distance = jet[2]
            # Generate uniform random value within fractional uncertainty
            perturbed_distance = original_distance * (
                        1 + np.random.uniform(-distance_uncertainty, distance_uncertainty))
            # Ensure distance is positive
            perturbed_distance = max(0.1, perturbed_distance)  # Minimum distance of 0.1 kpc
            new_jet[2] = perturbed_distance

            # 3. Recalculate apparent speed (depends on distance)
            new_jet[3] = (new_jet[1] / 173.0) * perturbed_distance

            # 4. Recalculate derived values based on perturbed parameters
            if new_jet[3] > 0:
                theta_rad = np.radians(new_jet[4])
                beta = new_jet[3] / (np.sin(theta_rad) + new_jet[3] * np.cos(theta_rad))
                new_jet[5] = min(beta, 0.90)  # Clip beta to 0.90
            else:
                new_jet[5] = 0

            # Recalculate gamma
            if new_jet[5] > 0:
                new_jet[6] = 1 / np.sqrt(1 - new_jet[5] ** 2)
            else:
                new_jet[6] = 1

            # Recalculate beta*gamma
            new_jet[7] = new_jet[5] * new_jet[6]

            perturbed_jets.append(new_jet)

        # Create the perturbed categories
        perturbed_bh = [perturbed_jets[all_jets_list.index(jet)] for jet in bh]
        perturbed_nsnourf = [perturbed_jets[all_jets_list.index(jet)] for jet in nsnourf]
        perturbed_prec = [perturbed_jets[all_jets_list.index(jet)] for jet in prec]
        perturbed_fixednourf = [perturbed_jets[all_jets_list.index(jet)] for jet in fixednourf]

        # Extract beta*gamma values for statistical tests
        bh_betagammas = [jet[7] for jet in perturbed_bh]
        ns_betagammas = [jet[7] for jet in perturbed_nsnourf]
        prec_betagammas = [jet[7] for jet in perturbed_prec]
        fixed_betagammas = [jet[7] for jet in perturbed_fixednourf]

        # Store perturbed values for later analysis
        all_perturbed['bh_betagammas'].extend(bh_betagammas)
        all_perturbed['ns_betagammas'].extend(ns_betagammas)
        all_perturbed['prec_betagammas'].extend(prec_betagammas)
        all_perturbed['fixed_betagammas'].extend(fixed_betagammas)

        # Run statistical tests
        # 1. All BH vs NS
        result1 = run_statistical_test(bh_betagammas, ns_betagammas, test_method)
        bh_vs_ns_results[result1] += 1
        bh_vs_ns_pvals.append(result1 == "not rejected")  # 1 if not rejected, 0 if rejected

        # 2. Precessing vs Fixed axis
        result2 = run_statistical_test(prec_betagammas, fixed_betagammas, test_method)
        prec_vs_fixed_results[result2] += 1
        prec_vs_fixed_pvals.append(result2 == "not rejected")  # 1 if not rejected, 0 if rejected

        # 3. Placeholder test (using BH vs itself just as a placeholder)
        result3 = run_statistical_test(bh_betagammas, bh_betagammas, test_method)
        placeholder_results[result3] += 1
        placeholder_pvals.append(result3 == "not rejected")  # 1 if not rejected, 0 if rejected


    # Calculate percentages
    bh_vs_ns_pct = (bh_vs_ns_results['rejected'] / num_iterations) * 100
    prec_vs_fixed_pct = (prec_vs_fixed_results['rejected'] / num_iterations) * 100
    placeholder_pct = (placeholder_results['rejected'] / num_iterations) * 100

    # Print summary results
    print(f"\n{'=' * 80}")
    print(f"POPULATION UNCERTAINTY ANALYSIS RESULTS (using {test_method.upper()} test)")
    print(f"{'=' * 80}")
    
    dist_results_df = pd.DataFrame({
        'Comparison': [
            'All BH vs NS',
            'Precessing vs Fixed',
            'Placeholder'
        ],
        'Null Hypothesis Rejected (%)': [
            f"{bh_vs_ns_pct:.1f}%",
            f"{prec_vs_fixed_pct:.1f}%",
            f"{placeholder_pct:.1f}%"
        ],
        'Null Hypothesis Not Rejected (%)': [
            f"{100 - bh_vs_ns_pct:.1f}%",
            f"{100 - prec_vs_fixed_pct:.1f}%",
            f"{100 - placeholder_pct:.1f}%"
        ],
        'Mean non-rejection rate': [
            f"{np.mean(bh_vs_ns_pvals):.4f}" if bh_vs_ns_pvals else "N/A",
            f"{np.mean(prec_vs_fixed_pvals):.4f}" if prec_vs_fixed_pvals else "N/A",
            f"{np.mean(placeholder_pvals):.4f}" if placeholder_pvals else "N/A"
        ]
    })
    
    print(tabulate(dist_results_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Save results to CSV
    dist_results_df.to_csv('population_uncertainty_results.csv', index=False)
    print("\nResults saved to 'population_uncertainty_results.csv'")
    
    # Return results without creating visualizations
    return {
        'bh_vs_ns_results': bh_vs_ns_results,
        'prec_vs_fixed_results': prec_vs_fixed_results,
        'placeholder_results': placeholder_results,
        'bh_vs_ns_pvals': bh_vs_ns_pvals,
        'prec_vs_fixed_pvals': prec_vs_fixed_pvals,
        'placeholder_pvals': placeholder_pvals,
        'original_values': original_values,
        'perturbed_values': all_perturbed,
        'test_method': test_method
    }


def run_statistical_test(data1, data2, test_method="ks", significance=0.05):
    """
    Run statistical test and return whether null hypothesis is rejected

    Parameters:
    data1, data2: Data samples to compare
    test_method: "ks" for Kolmogorov-Smirnov, "ad" for Anderson-Darling,
                 or "average" for average of both
    significance: Significance level (default 0.05)

    Returns:
    "rejected" or "not rejected"
    """
    if test_method == "ks":
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(data1, data2)
        return "rejected" if ks_pval < significance else "not rejected"

    elif test_method == "ad":
        # Anderson-Darling test
        # Note: scipy's implementation returns significance level, not p-value
        ad_stat, _, ad_sig = stats.anderson_ksamp([data1, data2])
        return "rejected" if ad_sig < significance else "not rejected"

    elif test_method == "average":
        # Run both tests and average the results
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(data1, data2)
        ks_reject = ks_pval < significance

        # Anderson-Darling test
        ad_stat, _, ad_sig = stats.anderson_ksamp([data1, data2])
        ad_reject = ad_sig < significance * 100

        # Average result - reject if both or if average probability > 0.5
        # If exactly one test rejects, we use the more conservative "not rejected"
        both_reject = ks_reject and ad_reject
        return "rejected" if both_reject else "not rejected"


def run_spin_uncertainty(all_jets_list, categories, num_iterations=1000,
                         angle_uncertainty=10, distance_uncertainty=0.1):
    """
    Run Monte Carlo simulations to assess how angle and distance uncertainties
    affect spin correlation analyses, with proper handling of measurements vs. lower limits

    Parameters:
    all_jets_list: Original jet data
    categories: Dictionary of categorized data
    num_iterations: Number of Monte Carlo iterations
    angle_uncertainty: Uncertainty in degrees (±) around the original inclination angle
    distance_uncertainty: Fractional uncertainty (±) in distance, e.g., 0.1 = 10%

    Returns:
    Dictionary with analysis results
    """
    print(f"\n{'=' * 80}")
    print(f"SPIN CORRELATION UNCERTAINTY ANALYSIS: Inclination Angle Variations (±{angle_uncertainty}°) and")
    print(f"                                       Distance Variations (±{distance_uncertainty * 100:.0f}%)")
    print(f"{'=' * 80}")
    print(f"Running {num_iterations} Monte Carlo iterations...")

    # Track correlation results for spin tests
    reflection_corr_results = {'significant': 0, 'not significant': 0}
    continuum_corr_results = {'significant': 0, 'not significant': 0}
    qpo_corr_results = {'significant': 0, 'not significant': 0}

    # Store correlation coefficients and p-values
    reflection_pearson_r = []
    reflection_pearson_p = []
    reflection_slopes = []

    continuum_pearson_r = []
    continuum_pearson_p = []
    continuum_slopes = []

    qpo_pearson_r = []
    qpo_pearson_p = []
    qpo_slopes = []

    # Track consistency of lower limits with regression
    reflection_limit_consistency = []
    continuum_limit_consistency = []
    qpo_limit_consistency = []

    # Get jets with spin measurements from categories
    # Use existing categories from data_preparation.py
    reflections = categories['reflections']
    reflectionslimits = categories['reflectionslimits']
    continuums = categories['continuums']
    continuumslimits = categories['continuumslimits']
    qpos = categories['qpos']
    qposlimits = categories['qposlimits']

    # Original paired data
    rr_original = categories['rr']  # [(spin, βγ), ...]
    rrl_original = categories['rrl']
    cc_original = categories['cc']
    ccl_original = categories['ccl']
    qq_original = categories['qq']
    qql_original = categories['qql']

    # Run Monte Carlo iterations
    for i in tqdm(range(num_iterations)):
        # Create a deep copy of the original jets list
        perturbed_jets = []
        for jet in all_jets_list:
            # Create a new jet entry with same values
            new_jet = jet.copy()

            # 1. Perturb inclination angle
            original_angle = jet[4]
            # Generate uniform random value within uncertainty bounds
            perturbed_angle = original_angle + np.random.uniform(-angle_uncertainty, angle_uncertainty)
            # Ensure angle is within physical limits (0-90 degrees)
            perturbed_angle = max(0, min(90, perturbed_angle))
            new_jet[4] = perturbed_angle

            # 2. Perturb distance
            original_distance = jet[2]
            # Generate uniform random value within fractional uncertainty
            perturbed_distance = original_distance * (
                        1 + np.random.uniform(-distance_uncertainty, distance_uncertainty))
            # Ensure distance is positive
            perturbed_distance = max(0.1, perturbed_distance)  # Minimum distance of 0.1 kpc
            new_jet[2] = perturbed_distance

            # 3. Recalculate apparent speed (depends on distance)
            new_jet[3] = (new_jet[1] / 173.0) * perturbed_distance

            # 4. Recalculate derived values based on perturbed parameters
            if new_jet[3] > 0:
                theta_rad = np.radians(new_jet[4])
                beta = new_jet[3] / (np.sin(theta_rad) + new_jet[3] * np.cos(theta_rad))
                new_jet[5] = min(beta, 0.90)  # Clip beta to 0.90
            else:
                new_jet[5] = 0

            # Recalculate gamma
            if new_jet[5] > 0:
                new_jet[6] = 1 / np.sqrt(1 - new_jet[5] ** 2)
            else:
                new_jet[6] = 1

            # Recalculate beta*gamma
            new_jet[7] = new_jet[5] * new_jet[6]

            # Keep the same limit flag (">F", "F", or "S") since we're perturbing
            # the measurements but not changing whether it's a limit or not

            perturbed_jets.append(new_jet)

        # Extract perturbed values for each category
        # For reflection
        perturbed_reflections = [perturbed_jets[all_jets_list.index(jet)] for jet in reflections]
        perturbed_reflectionslimits = [perturbed_jets[all_jets_list.index(jet)] for jet in reflectionslimits]

        # For continuum
        perturbed_continuums = [perturbed_jets[all_jets_list.index(jet)] for jet in continuums]
        perturbed_continuumslimits = [perturbed_jets[all_jets_list.index(jet)] for jet in continuumslimits]

        # For QPO
        perturbed_qpos = [perturbed_jets[all_jets_list.index(jet)] for jet in qpos]
        perturbed_qposlimits = [perturbed_jets[all_jets_list.index(jet)] for jet in qposlimits]

        # Create perturbed spin pair data
        perturbed_rr = [(jet[11], jet[7]) for jet in perturbed_reflections]
        perturbed_rrl = [(jet[11], jet[7]) for jet in perturbed_reflectionslimits]

        perturbed_cc = [(jet[12], jet[7]) for jet in perturbed_continuums]
        perturbed_ccl = [(jet[12], jet[7]) for jet in perturbed_continuumslimits]

        perturbed_qq = [(jet[13], jet[7]) for jet in perturbed_qpos]
        perturbed_qql = [(jet[13], jet[7]) for jet in perturbed_qposlimits]

        # Run correlation tests for spin vs βγ
        # Only run if we have at least 2 data points
        if len(perturbed_rr) > 1:
            # Extract x and y values for correlation tests
            x_vals = [x for x, y in perturbed_rr]
            y_vals = [y for x, y in perturbed_rr]

            # Pearson correlation
            r_val, p_val = stats.pearsonr(x_vals, y_vals)

            # Linear regression to get slope if we have enough points
            limits_consistent = True
            iteration_consistency = []

            if len(perturbed_rr) > 2:  # Need at least 3 points for regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

                # Check consistency of limits with regression
                if perturbed_rrl:
                    for spin, betagamma in perturbed_rrl:
                        expected_y = slope * spin + intercept
                        consistent = betagamma >= expected_y
                        iteration_consistency.append(consistent)

                    # If any limits are inconsistent, mark as inconsistent
                    limits_consistent = all(iteration_consistency)

            # Check if correlation is significant AND consistent with limits
            correlation_significant = p_val < 0.05
            result = "significant" if correlation_significant and limits_consistent else "not significant"
            reflection_corr_results[result] += 1

            # Store correlation values
            reflection_pearson_r.append(r_val)
            reflection_pearson_p.append(p_val)

            # Store slope information
            if len(perturbed_rr) > 2:
                reflection_slopes.append(slope)

                # Store the percentage of consistent limits
                if perturbed_rrl:
                    reflection_limit_consistency.append(
                        sum(iteration_consistency) / len(iteration_consistency) * 100
                    )

        if len(perturbed_cc) > 1:
            # Extract x and y values for correlation tests
            x_vals = [x for x, y in perturbed_cc]
            y_vals = [y for x, y in perturbed_cc]

            # Pearson correlation
            r_val, p_val = stats.pearsonr(x_vals, y_vals)

            # Linear regression to get slope if we have enough points
            limits_consistent = True
            iteration_consistency = []

            if len(perturbed_cc) > 2:  # Need at least 3 points for regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

                # Check consistency of limits with regression
                if perturbed_ccl:
                    for spin, betagamma in perturbed_ccl:
                        expected_y = slope * spin + intercept
                        consistent = betagamma >= expected_y
                        iteration_consistency.append(consistent)

                    # If any limits are inconsistent, mark as inconsistent
                    limits_consistent = all(iteration_consistency)

            # Check if correlation is significant AND consistent with limits
            correlation_significant = p_val < 0.05
            result = "significant" if correlation_significant and limits_consistent else "not significant"
            continuum_corr_results[result] += 1

            # Store correlation values
            continuum_pearson_r.append(r_val)
            continuum_pearson_p.append(p_val)

            # Store slope information
            if len(perturbed_cc) > 2:
                continuum_slopes.append(slope)

                # Store the percentage of consistent limits
                if perturbed_ccl:
                    continuum_limit_consistency.append(
                        sum(iteration_consistency) / len(iteration_consistency) * 100
                    )

        if len(perturbed_qq) > 1:
            # Extract x and y values for correlation tests
            x_vals = [x for x, y in perturbed_qq]
            y_vals = [y for x, y in perturbed_qq]

            # Pearson correlation
            r_val, p_val = stats.pearsonr(x_vals, y_vals)

            # Linear regression to get slope if we have enough points
            limits_consistent = True
            iteration_consistency = []

            if len(perturbed_qq) > 2:  # Need at least 3 points for regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

                # Check consistency of limits with regression
                if perturbed_qql:
                    for spin, betagamma in perturbed_qql:
                        expected_y = slope * spin + intercept
                        consistent = betagamma >= expected_y
                        iteration_consistency.append(consistent)

                    # If any limits are inconsistent, mark as inconsistent
                    limits_consistent = all(iteration_consistency)

            # Check if correlation is significant AND consistent with limits
            correlation_significant = p_val < 0.05
            result = "significant" if correlation_significant and limits_consistent else "not significant"
            qpo_corr_results[result] += 1

            # Store correlation values
            qpo_pearson_r.append(r_val)
            qpo_pearson_p.append(p_val)

            # Store slope information
            if len(perturbed_qq) > 2:
                qpo_slopes.append(slope)

                # Store the percentage of consistent limits
                if perturbed_qql:
                    qpo_limit_consistency.append(
                        sum(iteration_consistency) / len(iteration_consistency) * 100
                    )

    # Calculate percentages for correlation tests
    reflection_sig_pct = (reflection_corr_results.get('significant',
                                                      0) / num_iterations) * 100 if reflection_pearson_r else 0
    continuum_sig_pct = (continuum_corr_results.get('significant',
                                                    0) / num_iterations) * 100 if continuum_pearson_r else 0
    qpo_sig_pct = (qpo_corr_results.get('significant', 0) / num_iterations) * 100 if qpo_pearson_r else 0

    # Print summary results
    print(f"\n{'=' * 80}")
    print("SPIN CORRELATION UNCERTAINTY ANALYSIS RESULTS")
    print(f"{'=' * 80}")

    # Check if we have any correlation data
    if reflection_pearson_r or continuum_pearson_r or qpo_pearson_r:
        corr_results_df = pd.DataFrame({
            'Spin Measurement': [
                'Reflection',
                'Continuum',
                'QPO'
            ],
            'Direct Measurements': [
                len(reflections),
                len(continuums),
                len(qpos)
            ],
            'Lower Limits (βγ)': [
                len(reflectionslimits),
                len(continuumslimits),
                len(qposlimits)
            ],
            'Significant Correlation (%)': [
                f"{reflection_sig_pct:.1f}%" if reflection_pearson_r else "N/A",
                f"{continuum_sig_pct:.1f}%" if continuum_pearson_r else "N/A",
                f"{qpo_sig_pct:.1f}%" if qpo_pearson_r else "N/A"
            ],
            'Mean Correlation (r)': [
                f"{np.mean(reflection_pearson_r):.4f}" if reflection_pearson_r else "N/A",
                f"{np.mean(continuum_pearson_r):.4f}" if continuum_pearson_r else "N/A",
                f"{np.mean(qpo_pearson_r):.4f}" if qpo_pearson_r else "N/A"
            ],
            'Mean p-value': [
                f"{np.mean(reflection_pearson_p):.4f}" if reflection_pearson_p else "N/A",
                f"{np.mean(continuum_pearson_p):.4f}" if continuum_pearson_p else "N/A",
                f"{np.mean(qpo_pearson_p):.4f}" if qpo_pearson_p else "N/A"
            ],
            'Mean Slope': [
                f"{np.mean(reflection_slopes):.4f}" if reflection_slopes else "N/A",
                f"{np.mean(continuum_slopes):.4f}" if continuum_slopes else "N/A",
                f"{np.mean(qpo_slopes):.4f}" if qpo_slopes else "N/A"
            ]
        })

        print(tabulate(corr_results_df, headers='keys', tablefmt='grid', showindex=False))

        # Display limit consistency if available
        if reflection_limit_consistency or continuum_limit_consistency or qpo_limit_consistency:
            print("\nConsistency of lower limits with regression line:")
            limits_df = pd.DataFrame({
                'Spin Measurement': [
                    'Reflection',
                    'Continuum',
                    'QPO'
                ],
                'Lower Limits (βγ)': [
                    len(reflectionslimits),
                    len(continuumslimits),
                    len(qposlimits)
                ],
                'Mean Consistency (%)': [
                    f"{np.mean(reflection_limit_consistency):.1f}%" if reflection_limit_consistency else "N/A",
                    f"{np.mean(continuum_limit_consistency):.1f}%" if continuum_limit_consistency else "N/A",
                    f"{np.mean(qpo_limit_consistency):.1f}%" if qpo_limit_consistency else "N/A"
                ]
            })
            print(tabulate(limits_df, headers='keys', tablefmt='grid', showindex=False))
    else:
        print("No spin correlation data available for analysis.")

    # Save results to CSV
    if reflection_pearson_r or continuum_pearson_r or qpo_pearson_r:
        corr_results_df.to_csv('spin_uncertainty_results.csv', index=False)
        if reflection_limit_consistency or continuum_limit_consistency or qpo_limit_consistency:
            limits_df.to_csv('limit_consistency_results.csv', index=False)
        print("\nResults saved to CSV files")

    # Return combined results
    return {
        'reflection_corr_results': reflection_corr_results,
        'continuum_corr_results': continuum_corr_results,
        'qpo_corr_results': qpo_corr_results,
        'reflection_pearson_r': reflection_pearson_r,
        'reflection_pearson_p': reflection_pearson_p,
        'reflection_slopes': reflection_slopes,
        'continuum_pearson_r': continuum_pearson_r,
        'continuum_pearson_p': continuum_pearson_p,
        'continuum_slopes': continuum_slopes,
        'qpo_pearson_r': qpo_pearson_r,
        'qpo_pearson_p': qpo_pearson_p,
        'qpo_slopes': qpo_slopes,
        'reflection_limit_consistency': reflection_limit_consistency,
        'continuum_limit_consistency': continuum_limit_consistency,
        'qpo_limit_consistency': qpo_limit_consistency,
        'original_data': {
            'rr': rr_original,
            'rrl': rrl_original,
            'cc': cc_original,
            'ccl': ccl_original,
            'qq': qq_original,
            'qql': qql_original
        }
    }


# ------------------------------------------------------------------------------
# Threshold Analysis Functions
# ------------------------------------------------------------------------------

def explore_uncertainty_threshold(all_jets_list, categories,
                                  angle_uncertainty_range=(0, 40), angle_step=5,
                                  distance_uncertainty_range=(0, 0.5), distance_step=0.05,
                                  num_iterations=500,
                                  test_method="ks",
                                  suppress_interim_plots=True):
    """
    Explore how different levels of uncertainty affect statistical test results

    Parameters:
    all_jets_list: Original jet data
    categories: Dictionary of categorized data
    angle_uncertainty_range: Tuple of (min, max) angle uncertainty in degrees
    angle_step: Step size for angle uncertainty in degrees
    distance_uncertainty_range: Tuple of (min, max) fractional distance uncertainty
                               (e.g., 0.1 = 10%)
    distance_step: Step size for fractional distance uncertainty
    num_iterations: Number of Monte Carlo iterations per parameter set
    test_method: Statistical test to use - "ks", "ad", or "average"
    suppress_interim_plots: Whether to suppress intermediate plots

    Returns:
    DataFrame with results for each parameter combination
    """
    results = []

    # Generate parameter grid
    angle_uncertainties = np.arange(
        angle_uncertainty_range[0],
        angle_uncertainty_range[1] + angle_step,
        angle_step
    )

    distance_uncertainties = np.arange(
        distance_uncertainty_range[0],
        distance_uncertainty_range[1] + distance_step,
        distance_step
    )

    total_combinations = len(angle_uncertainties) * len(distance_uncertainties)
    print(f"Exploring {total_combinations} parameter combinations...")
    print(f"Total Monte Carlo iterations will be approximately: {total_combinations * num_iterations}")

    # Store the original matplotlib show function
    original_plt_show = plt.show
    # Replace it with a no-op function if suppressing plots
    if suppress_interim_plots:
        plt.show = lambda: None

    try:
        # Setup progress tracking
        progress_counter = 0

        # Test each combination
        for angle_uncertainty in angle_uncertainties:
            for distance_uncertainty in distance_uncertainties:
                progress_counter += 1
                print(
                    f"\n[{progress_counter}/{total_combinations}] Testing angle±{angle_uncertainty}°, distance±{distance_uncertainty * 100:.0f}%")

                # Run uncertainty analysis for this parameter set
                uncertainty_results = run_population_uncertainty(
                    all_jets_list,
                    categories,
                    num_iterations=num_iterations,
                    angle_uncertainty=angle_uncertainty,
                    distance_uncertainty=distance_uncertainty,
                    test_method=test_method
                )

                # Extract the percentage of non-rejected null hypotheses for prec vs fixed
                non_rejected_percentage = (
                        uncertainty_results['prec_vs_fixed_results']['not rejected'] / num_iterations * 100
                )

                # Record results
                results.append({
                    'angle_uncertainty': angle_uncertainty,
                    'distance_uncertainty': distance_uncertainty,
                    'non_rejected_percentage': non_rejected_percentage
                })

                print(f"Non-rejected percentage: {non_rejected_percentage:.1f}%")

    finally:
        # Restore matplotlib show function
        if suppress_interim_plots:
            plt.show = original_plt_show

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Save results to CSV
    filename_base = f'uncertainty_{test_method}_exploration'
    results_df.to_csv(f'{filename_base}.csv', index=False)
    print(f"\nResults saved to '{filename_base}.csv'")

    # Create visualization of the results
    if not suppress_interim_plots:
        create_threshold_heatmap(results_df, test_method=test_method)

    return results_df


def explore_spin_uncertainty_threshold(all_jets_list, categories,
                                       spin_method="reflection",
                                       angle_uncertainty_range=(0, 40), angle_step=5,
                                       distance_uncertainty_range=(0, 0.5), distance_step=0.05,
                                       num_iterations=500,
                                       suppress_interim_plots=True):
    """
    Explore how different levels of uncertainty affect spin-speed correlation significance

    Parameters:
    all_jets_list: Original jet data
    categories: Dictionary of categorized data
    spin_method: Which spin measurement to analyze - "reflection", "continuum", or "qpo"
    angle_uncertainty_range: Tuple of (min, max) angle uncertainty in degrees
    angle_step: Step size for angle uncertainty in degrees
    distance_uncertainty_range: Tuple of (min, max) fractional distance uncertainty
    distance_step: Step size for fractional distance uncertainty
    num_iterations: Number of Monte Carlo iterations per parameter set
    suppress_interim_plots: Whether to suppress intermediate plots

    Returns:
    DataFrame with results for each parameter combination
    """
    # Validate spin method
    valid_methods = ["reflection", "continuum", "qpo"]
    if spin_method not in valid_methods:
        raise ValueError(f"Invalid spin_method '{spin_method}'. Must be one of {valid_methods}")

    results = []

    # Generate parameter grid
    angle_uncertainties = np.arange(
        angle_uncertainty_range[0],
        angle_uncertainty_range[1] + angle_step,
        angle_step
    )

    distance_uncertainties = np.arange(
        distance_uncertainty_range[0],
        distance_uncertainty_range[1] + distance_step,
        distance_step
    )

    total_combinations = len(angle_uncertainties) * len(distance_uncertainties)
    print(f"Exploring {total_combinations} parameter combinations for {spin_method} spin correlation...")
    print(f"Total Monte Carlo iterations will be approximately: {total_combinations * num_iterations}")

    # Store the original matplotlib show function
    original_plt_show = plt.show
    # Replace it with a no-op function if suppressing plots
    if suppress_interim_plots:
        plt.show = lambda: None

    try:
        # Setup progress tracking
        progress_counter = 0

        # Test each combination
        for angle_uncertainty in angle_uncertainties:
            for distance_uncertainty in distance_uncertainties:
                progress_counter += 1
                print(
                    f"\n[{progress_counter}/{total_combinations}] Testing angle±{angle_uncertainty}°, distance±{distance_uncertainty * 100:.0f}%")

                # Run spin uncertainty analysis for this parameter set
                uncertainty_results = run_spin_uncertainty(
                    all_jets_list,
                    categories,
                    num_iterations=num_iterations,
                    angle_uncertainty=angle_uncertainty,
                    distance_uncertainty=distance_uncertainty
                )

                # Extract the percentage of significant correlations for the chosen spin method
                significance_percentage = 0
                if spin_method == "reflection" and 'reflection_corr_results' in uncertainty_results:
                    total_iterations = sum(uncertainty_results['reflection_corr_results'].values())
                    if total_iterations > 0:
                        significance_percentage = (
                                uncertainty_results['reflection_corr_results'].get('significant', 0) /
                                total_iterations * 100
                        )
                elif spin_method == "continuum" and 'continuum_corr_results' in uncertainty_results:
                    total_iterations = sum(uncertainty_results['continuum_corr_results'].values())
                    if total_iterations > 0:
                        significance_percentage = (
                                uncertainty_results['continuum_corr_results'].get('significant', 0) /
                                total_iterations * 100
                        )
                elif spin_method == "qpo" and 'qpo_corr_results' in uncertainty_results:
                    total_iterations = sum(uncertainty_results['qpo_corr_results'].values())
                    if total_iterations > 0:
                        significance_percentage = (
                                uncertainty_results['qpo_corr_results'].get('significant', 0) /
                                total_iterations * 100
                        )

                # Record results
                results.append({
                    'angle_uncertainty': angle_uncertainty,
                    'distance_uncertainty': distance_uncertainty,
                    'significance_percentage': significance_percentage
                })

                print(f"Significance percentage: {significance_percentage:.1f}%")

    finally:
        # Restore matplotlib show function
        if suppress_interim_plots:
            plt.show = original_plt_show

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Save results to CSV
    filename_base = f'spin_{spin_method}_uncertainty'
    results_df.to_csv(f'{filename_base}.csv', index=False)
    print(f"\nResults saved to '{filename_base}.csv'")

    # Create visualization of the results
    if not suppress_interim_plots:
        create_spin_threshold_heatmap(results_df, spin_method)

    return results_df


# ------------------------------------------------------------------------------
# Unified Analysis Function
# ------------------------------------------------------------------------------

def unified_analysis(all_jets_list, categories,
                     analysis_type="population",  # "population", "spin", or "both"
                     uncertainty_mode="none",  # "none", "single", or "threshold"
                     angle_uncertainty=0.0,
                     distance_uncertainty=0.0,
                     angle_uncertainty_range=None,
                     angle_step=5.0,
                     distance_uncertainty_range=None,
                     distance_step=0.05,
                     num_iterations=1,
                     test_method="ad",
                     spin_method="reflection",  # For spin analysis: "reflection", "continuum", or "qpo"
                     suppress_interim_plots=True):
    """
    Unified analysis function that handles all analysis types:
    1. Basic analysis (when uncertainty_mode="none")
    2. Uncertainty analysis (when uncertainty_mode="single")
    3. Threshold/heatmap analysis (when uncertainty_mode="threshold")

    Parameters:
    all_jets_list: List of jet data
    categories: Dictionary of categorized data
    analysis_type: Type of analysis to perform ("population", "spin", or "both")
    uncertainty_mode: Type of uncertainty analysis to perform
                    - "none": Basic analysis with no perturbations
                    - "single": Single uncertainty analysis with fixed angle/distance uncertainty
                    - "threshold": Threshold analysis across a range of uncertainties
    angle_uncertainty: Single value for angle uncertainty in degrees
    distance_uncertainty: Single value for fractional distance uncertainty
    angle_uncertainty_range: Tuple of (min, max) angle uncertainty for threshold analysis
    angle_step: Step size for angle uncertainty in threshold analysis
    distance_uncertainty_range: Tuple of (min, max) distance uncertainty for threshold analysis
    distance_step: Step size for distance uncertainty in threshold analysis
    num_iterations: Number of Monte Carlo iterations
    test_method: Statistical test to use ("ks", "ad", or "average")
    spin_method: Spin measurement method to analyze ("reflection", "continuum", or "qpo")
    suppress_interim_plots: Whether to suppress intermediate plots in threshold analysis

    Returns:
    Dictionary with analysis results
    """
    results = {}

    print(f"\n{'=' * 80}")
    print(f"UNIFIED ANALYSIS: Type={analysis_type}, Mode={uncertainty_mode}")
    print(f"{'=' * 80}")

    # Validate analysis type
    valid_types = ["population", "spin", "both"]
    if analysis_type not in valid_types:
        raise ValueError(f"Invalid analysis_type '{analysis_type}'. Must be one of {valid_types}")

    # Validate uncertainty mode
    valid_modes = ["none", "single", "threshold"]
    if uncertainty_mode not in valid_modes:
        raise ValueError(f"Invalid uncertainty_mode '{uncertainty_mode}'. Must be one of {valid_modes}")

    # Handle population analysis
    if analysis_type in ["population", "both"]:
        if uncertainty_mode == "none":
            # Basic population analysis
            pop_results = perform_population_analysis(all_jets_list, categories)
            results["population"] = pop_results

        elif uncertainty_mode == "single":
            # Single uncertainty analysis
            pop_uncertainty = run_population_uncertainty(
                all_jets_list,
                categories,
                num_iterations=num_iterations,
                angle_uncertainty=angle_uncertainty,
                distance_uncertainty=distance_uncertainty,
                test_method=test_method
            )
            results["population_uncertainty"] = pop_uncertainty

        elif uncertainty_mode == "threshold":
            # Threshold exploration
            # Set default ranges if not provided
            if angle_uncertainty_range is None:
                angle_uncertainty_range = (0, 40.0)
            if distance_uncertainty_range is None:
                distance_uncertainty_range = (0, 0.4)

            pop_threshold = explore_uncertainty_threshold(
                all_jets_list,
                categories,
                angle_uncertainty_range=angle_uncertainty_range,
                angle_step=angle_step,
                distance_uncertainty_range=distance_uncertainty_range,
                distance_step=distance_step,
                num_iterations=num_iterations,
                test_method=test_method,
                suppress_interim_plots=suppress_interim_plots
            )
            results["population_threshold"] = pop_threshold

    # Handle spin analysis
    if analysis_type in ["spin", "both"]:
        if uncertainty_mode == "none":
            # Basic spin analysis
            spin_results = perform_spin_analysis(all_jets_list, categories)
            results["spin"] = spin_results

        elif uncertainty_mode == "single":
            # Single uncertainty analysis
            spin_uncertainty = run_spin_uncertainty(
                all_jets_list,
                categories,
                num_iterations=num_iterations,
                angle_uncertainty=angle_uncertainty,
                distance_uncertainty=distance_uncertainty
            )
            results["spin_uncertainty"] = spin_uncertainty

        elif uncertainty_mode == "threshold":
            # Threshold exploration for spin
            # Set default ranges if not provided
            if angle_uncertainty_range is None:
                angle_uncertainty_range = (0, 40.0)
            if distance_uncertainty_range is None:
                distance_uncertainty_range = (0, 0.4)

            spin_threshold = explore_spin_uncertainty_threshold(
                all_jets_list,
                categories,
                spin_method=spin_method,
                angle_uncertainty_range=angle_uncertainty_range,
                angle_step=angle_step,
                distance_uncertainty_range=distance_uncertainty_range,
                distance_step=distance_step,
                num_iterations=num_iterations,
                suppress_interim_plots=suppress_interim_plots
            )
            results["spin_threshold"] = spin_threshold

    return results


# ------------------------------------------------------------------------------
# Remaining Essential Visualization Functions - Only Heatmaps
# ------------------------------------------------------------------------------

def create_threshold_heatmap(results_df, test_method='ks'):
    """Create a heatmap showing how uncertainty affects statistical test results"""
    print(f"Creating heatmap with {len(results_df)} data points...")

    if results_df.empty:
        print("ERROR: No data available to create heatmap.")
        return

    # Extract unique values for each axis
    angle_values = sorted(results_df['angle_uncertainty'].unique())
    distance_values = sorted(results_df['distance_uncertainty'].unique())

    print(f"Heatmap will have dimensions: {len(angle_values)} × {len(distance_values)}")

    # Create a 2D array for the heatmap
    heatmap_data = np.full((len(angle_values), len(distance_values)), np.nan)

    # Fill in the heatmap data
    for i, angle in enumerate(angle_values):
        for j, distance in enumerate(distance_values):
            row = results_df[(results_df['angle_uncertainty'] == angle) &
                             (results_df['distance_uncertainty'] == distance)]
            if not row.empty:
                heatmap_data[i, j] = row['non_rejected_percentage'].values[0]

    try:
        # Create the heatmap
        print("Generating heatmap figure...")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a masked array to handle missing values
        masked_data = np.ma.masked_invalid(heatmap_data)
        im = ax.imshow(masked_data, cmap='YlOrRd', origin='lower')

        # Set axis labels
        ax.set_xticks(np.arange(len(distance_values)))
        ax.set_yticks(np.arange(len(angle_values)))
        ax.set_xticklabels([f"{d * 100:.0f}%" for d in distance_values])
        ax.set_yticklabels([f"{a}°" for a in angle_values])

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Non-rejected percentage (%)", rotation=-90, va="bottom")

        # Try to add reference contour at 50%
        try:
            if np.any(masked_data >= 50) and np.any(masked_data < 50):
                CS = ax.contour(np.arange(len(distance_values)), np.arange(len(angle_values)),
                                masked_data, levels=[50], colors='blue', linewidths=2)
                ax.clabel(CS, inline=True, fontsize=10)
        except Exception as e:
            print(f"Could not draw contour: {e}")

        # Get test method name for the title
        test_name = {
            "ks": "Kolmogorov-Smirnov",
            "ad": "Anderson-Darling",
            "average": "Combined K-S & A-D"
        }.get(test_method, test_method.upper())

        # Labels and title
        ax.set_xlabel("Distance Uncertainty")
        ax.set_ylabel("Angle Uncertainty (degrees)")
        ax.set_title(f"Effect of Measurement Uncertainties on Statistical Tests\n" +
                     f"Precessing vs. Fixed Jets ({test_name} Test)")

        # Add text annotations with values where we have data
        for i in range(len(angle_values)):
            for j in range(len(distance_values)):
                if not np.isnan(heatmap_data[i, j]):
                    val = heatmap_data[i, j]
                    text_color = "white" if val > 60 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color)

        fig.tight_layout()

        # Save figures
        pdf_filename = f'figures/uncertainty_{test_method}_heatmap.pdf'
        png_filename = f'figures/uncertainty_{test_method}_heatmap.png'

        print(f"Saving heatmap to: {pdf_filename} and {png_filename}")
        plt.savefig(pdf_filename)
        plt.savefig(png_filename, dpi=300)
        print(f"Heatmap saved successfully.")

        # Display the plot
        plt.show()

    except Exception as e:
        import traceback
        print(f"ERROR creating heatmap: {e}")
        traceback.print_exc()


def create_spin_threshold_heatmap(results_df, spin_method):
    """Create a heatmap showing how uncertainty affects spin-speed correlations"""
    print(f"Creating spin correlation heatmap with {len(results_df)} data points...")

    if results_df.empty:
        print("ERROR: No data available to create heatmap.")
        return

    # Extract unique values for each axis
    angle_values = sorted(results_df['angle_uncertainty'].unique())
    distance_values = sorted(results_df['distance_uncertainty'].unique())

    print(f"Heatmap will have dimensions: {len(angle_values)} × {len(distance_values)}")

    # Create a 2D array for the heatmap
    heatmap_data = np.full((len(angle_values), len(distance_values)), np.nan)

    # Fill in the heatmap data
    for i, angle in enumerate(angle_values):
        for j, distance in enumerate(distance_values):
            row = results_df[(results_df['angle_uncertainty'] == angle) &
                             (results_df['distance_uncertainty'] == distance)]
            if not row.empty:
                heatmap_data[i, j] = row['significance_percentage'].values[0]

    try:
        # Create the heatmap
        print("Generating heatmap figure...")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a masked array to handle missing values
        masked_data = np.ma.masked_invalid(heatmap_data)
        im = ax.imshow(masked_data, cmap='YlOrRd', origin='lower')

        # Set axis labels
        ax.set_xticks(np.arange(len(distance_values)))
        ax.set_yticks(np.arange(len(angle_values)))
        ax.set_xticklabels([f"{d * 100:.0f}%" for d in distance_values])
        ax.set_yticklabels([f"{a}°" for a in angle_values])

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Significance percentage (%)", rotation=-90, va="bottom")

        # Try to add reference contour at 50%
        try:
            if np.any(masked_data >= 50) and np.any(masked_data < 50):
                CS = ax.contour(np.arange(len(distance_values)), np.arange(len(angle_values)),
                                masked_data, levels=[50], colors='blue', linewidths=2)
                ax.clabel(CS, inline=True, fontsize=10)
        except Exception as e:
            print(f"Could not draw contour: {e}")

        # Labels and title
        spin_method_title = {
            'reflection': 'Reflection',
            'continuum': 'Continuum',
            'qpo': 'QPO'
        }.get(spin_method, spin_method.capitalize())

        ax.set_xlabel("Distance Uncertainty")
        ax.set_ylabel("Angle Uncertainty (degrees)")
        ax.set_title(f"Effect of Measurement Uncertainties on Spin-Speed Correlation\n" +
                     f"{spin_method_title} Spin Measurement Method")

        # Add text annotations with values where we have data
        for i in range(len(angle_values)):
            for j in range(len(distance_values)):
                if not np.isnan(heatmap_data[i, j]):
                    val = heatmap_data[i, j]
                    text_color = "white" if val < 60 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color)

        fig.tight_layout()

        # Save figures
        pdf_filename = f'figures/spin_{spin_method}_uncertainty_heatmap.pdf'
        png_filename = f'figures/spin_{spin_method}_uncertainty_heatmap.png'

        print(f"Saving heatmap to: {pdf_filename} and {png_filename}")
        plt.savefig(pdf_filename)
        plt.savefig(png_filename, dpi=300)
        print(f"Heatmap saved successfully.")

        # Display the plot
        plt.show()

    except Exception as e:
        import traceback
        print(f"ERROR creating heatmap: {e}")
        traceback.print_exc()
