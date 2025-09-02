#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:48:37 2025

@author: rob
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population Analysis Module

This module performs statistical tests comparing different populations of relativistic jets.
"""

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

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
    slowbh = categories['slowbh']
    nsnourf = categories['nsnourf']
    prec = categories['prec']
    fixednourf = categories['fixednourf']
    nonprecnourf = categories['nonprecnourf']
    
    # Extract beta*gamma values for statistical tests
    bh_betagammas = [jet[7] for jet in bh]
    ns_betagammas = [jet[7] for jet in nsnourf]
    slowbh_betagammas = [jet[7] for jet in slowbh]
    prec_betagammas = [jet[7] for jet in prec]
    fixed_betagammas = [jet[7] for jet in fixednourf]
    nonprec_betagammas = [jet[7] for jet in nonprecnourf]
    
    # Run statistical tests
    print("\nRunning statistical tests...")
    
    #1. All BH vs NS
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
    print(f"\n{'='*80}")
    print("SUMMARY OF STATISTICAL TESTS")
    print(f"{'='*80}")

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
    print(f"\n{'='*80}")
    print(f"STATISTICAL COMPARISON: {label}")
    print(f"{'='*80}")
    
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
            f"The null hypothesis is {ks_result} at the {alpha*100:.0f}% significance level",
            f"The null hypothesis is {ad_result} at the {alpha*100:.0f}% significance level"
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