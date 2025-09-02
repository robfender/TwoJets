#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spin Analysis Module

This module performs correlation analyses between black hole spin and jet speed.
"""

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

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
    print(f"\n{'='*80}")
    print("SPIN CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Extract paired data
    rr = categories['rr']  # Regular measurements (Reflection)
    rrl = categories['rrl']  # Lower limits (Reflection)
    cc = categories['cc']  # Regular measurements (Continuum)
    ccl = categories['ccl']  # Lower limits (Continuum) 
    qq = categories['qq']  # Regular measurements (QPO)
    qql = categories['qql']  # Lower limits (QPO)
    
    # Analyze each spin measurement method
    reflection_stats = analyze_spin_correlation(rr, rrl, "Reflection Spin vs. βγ", alpha)
    continuum_stats = analyze_spin_correlation(cc, ccl, "Continuum Spin vs. βγ", alpha)
    qpo_stats = analyze_spin_correlation(qq, qql, "QPO Spin vs. βγ", alpha)
    
    # Create a summary of all correlation results
    print(f"\n{'='*80}")
    print("SUMMARY OF SPIN CORRELATION ANALYSES")
    print(f"{'='*80}")
    
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
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS: {label}")
    print(f"{'='*80}")
    
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
                f"{'Significant' if pearson_p < alpha else 'Not significant'} at {alpha*100:.0f}% level",
                f"{'Significant' if spearman_p < alpha else 'Not significant'} at {alpha*100:.0f}% level",
                f"{'Significant' if kendall_p < alpha else 'Not significant'} at {alpha*100:.0f}% level"
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
            print(f"R-squared: {r_value**2:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Regression is {'statistically significant' if p_value < alpha else 'not statistically significant'} at {alpha*100:.0f}% level")
            
            # Check if limits are consistent with regression
            if data_limits:
                print("\nConsistency check for lower limits:")
                for i, (x_limit, y_limit) in enumerate(data_limits):
                    expected_y = slope * x_limit + intercept
                    consistent = y_limit >= expected_y
                    if not consistent:
                        inconsistent_limits += 1
                    print(f"  Limit {i+1}: Spin ≥ {x_limit:.2f}, βγ = {y_limit:.2f}, Predicted βγ = {expected_y:.2f}, {'Consistent' if consistent else 'Inconsistent'}")
                
                # If any limits are inconsistent, mark the regression as inconsistent
                if inconsistent_limits > 0:
                    limits_consistent = False
                    print(f"\nWARNING: {inconsistent_limits} of {len(data_limits)} lower limits are inconsistent with the regression.")
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
                    print("\nFinal determination: Correlation is statistically significant and consistent with all lower limits.")
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
                'r_squared': r_value**2,
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