#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 13:08:00 2025

@author: rob
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module

This module creates all plots for the jet analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon
from scipy import stats

def create_all_plots(all_jets_list, categories, population_results, spin_results=None):
    """Create all plots for jet analysis"""
    
    # Set up custom styling to match Mathematica
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'figure.figsize': (10, 7),
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
    })

    # Define Mathematica-like colors
    light_gray = '#CCCCCC'
    gray = '#999999'
    blue = '#6495ED'      # Brighter blue to match Mathematica
    green = '#90EE90'     # Light green to match Mathematica
    red = '#FF6347'       # Tomato red for the third category in 3-way plot
    
    # Extract data for plotting
    # We need to recreate the original data structure that visualization functions expect
    extracted_data = {}
    
    # Extract all necessary data from categories directly
    alljets = categories['alljets']
    bh = categories['bh']
    slowbh = categories['slowbh']
    nsnourf = categories['nsnourf']
    prec = categories['prec']
    fixednourf = categories['fixednourf']
    nonprecnourf = categories['nonprecnourf']
    alljetsmeerkat = categories['alljetsmeerkat']
    alljetsnomeerkat = categories['alljetsnomeerkat']
    
    # Proper motions
    extracted_data['all_proper_motions'] = [jet[1] for jet in alljets]
    extracted_data['meerkat_proper_motions'] = [jet[1] for jet in alljetsmeerkat]
    extracted_data['no_meerkat_proper_motions'] = [jet[1] for jet in alljetsnomeerkat]
    extracted_data['bh_proper_motions'] = [jet[1] for jet in bh]
    extracted_data['ns_proper_motions'] = [jet[1] for jet in nsnourf]
    extracted_data['unknown_proper_motions'] = [jet[1] for jet in categories['unknourf']]

    # Apparent speeds
    extracted_data['all_speeds'] = [jet[3] for jet in alljets]
    extracted_data['bh_speeds'] = [jet[3] for jet in bh]
    extracted_data['ns_speeds'] = [jet[3] for jet in nsnourf]
    extracted_data['unknown_speeds'] = [jet[3] for jet in categories['unknourf']]
    extracted_data['prec_speeds'] = [jet[3] for jet in prec]
    extracted_data['nonprec_speeds'] = [jet[3] for jet in nonprecnourf]

    # Physics data
    extracted_data['all_betas'] = [jet[5] for jet in alljets]
    extracted_data['all_gammas'] = [jet[6] for jet in alljets]
    extracted_data['all_betagammas'] = [jet[7] for jet in alljets]

    extracted_data['bh_betas'] = [jet[5] for jet in bh]
    extracted_data['bh_gammas'] = [jet[6] for jet in bh]
    extracted_data['bh_betagammas'] = [jet[7] for jet in bh]

    extracted_data['ns_betas'] = [jet[5] for jet in nsnourf]
    extracted_data['ns_gammas'] = [jet[6] for jet in nsnourf]
    extracted_data['ns_betagammas'] = [jet[7] for jet in nsnourf]

    extracted_data['unknown_betas'] = [jet[5] for jet in categories['unknourf']]
    extracted_data['unknown_gammas'] = [jet[6] for jet in categories['unknourf']]
    extracted_data['unknown_betagammas'] = [jet[7] for jet in categories['unknourf']]

    extracted_data['slowbh_betagammas'] = [jet[7] for jet in slowbh]
    extracted_data['prec_betagammas'] = [jet[7] for jet in prec]
    extracted_data['nonprec_betagammas'] = [jet[7] for jet in nonprecnourf]
    extracted_data['fixed_betagammas'] = [jet[7] for jet in fixednourf]
    extracted_data['single_betagammas'] = [jet[7] for jet in categories['singlenourf']]
    
    # Generate proper motion histograms
    print("\nGenerating proper motion histograms...")
    create_proper_motion_plots(extracted_data, light_gray, gray, blue, green)
    
    # Generate apparent speed histograms
    print("\nGenerating apparent speed histograms...")
    create_apparent_speed_plots(extracted_data, light_gray, gray, blue, green)
    
    # Generate physics plots with triangle annotations
    print("\nGenerating physics plots with triangle annotations...")
    create_physics_plots(
        extracted_data, 
        categories['triangle_points'],
        light_gray, gray, blue, green, red
    )
    
    # Generate spin-speed correlation plots if spin results are provided
    if spin_results:
        print("\nGenerating spin vs. speed correlation plots...")
        # Remove the spin_results parameter from the call
        create_spin_speed_plots(
        categories['rr'], categories['rrl'],
        categories['cc'], categories['ccl'],
        categories['qq'], categories['qql']
    )
    
    print("All plots generated successfully.")

def add_triangles(ax, triangles):
    """Add multiple triangles to the plot"""
    for points in triangles:
        # Create a polygon patch with the triangle vertices
        triangle = Polygon(
            points,  # Array of (x,y) vertices
            closed=True,  
            facecolor='gray',
            alpha=0.2,
            edgecolor='none'  # No edge line
        )
        ax.add_patch(triangle)

def create_histogram(data, bins, color, title, xlabel, range_x, range_y, legend=None, edgecolor='black'):
    """Create a histogram with consistent styling"""
    fig, ax = plt.subplots()
    
    if isinstance(data[0], list):  # Stacked histogram
        n, bins, patches = ax.hist(
            data, 
            bins=bins, 
            stacked=True, 
            color=color, 
            edgecolor=edgecolor,
            linewidth=1.0
        )
    else:  # Single histogram
        n, bins, patches = ax.hist(
            data, 
            bins=bins, 
            color=color, 
            edgecolor=edgecolor,
            linewidth=1.0
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title, pad=10)
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    
    if legend:
        ax.legend(legend, loc='upper right', frameon=False)
    
    ax.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    return fig, ax

def create_spin_plot(data_points, data_limits, x_label, y_label="βγ", y_max=None):
    """Create a scatter plot with different markers for points and lower limits"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data 
    x_regular = [x for x, y in data_points]
    y_regular = [y for x, y in data_points]
    
    x_limits = [x for x, y in data_limits]
    y_limits = [y for x, y in data_limits]
    
    # Plot regular points with filled circles
    ax.scatter(x_regular, y_regular, color='blue', s=100, marker='o', label='Measurements')
    
    # Plot lower limits with filled up-triangles
    ax.scatter(x_limits, y_limits, color='blue', s=150, marker='^', label='βγ Lower Limits')
    
    # Add arrows to indicate these are limits
    for x, y in data_limits:
        ax.annotate('', xy=(x, y+0.2), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color='blue', alpha=0.5))
    
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlim(0, 1)
    if y_max:
        ax.set_ylim(0, y_max)
    
    # Add grid
    ax.grid(True, linestyle='dotted', alpha=0.6)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Add a correlation line if enough data points AND correlation is significant
    if len(data_points) > 2:
        x_vals = [x for x, y in data_points]
        y_vals = [y for x, y in data_points]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        
        # Check if correlation is significant
        pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)
        spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)
        kendall_tau, kendall_p = stats.kendalltau(x_vals, y_vals)
        
        is_significant = (pearson_p < 0.05) or (spearman_p < 0.05) or (kendall_p < 0.05) or (p_value < 0.05)
        
        # Check if limits are consistent with regression
        limits_consistent = True
        if data_limits and is_significant:
            for x_limit, y_limit in data_limits:
                expected_y = slope * x_limit + intercept
                if y_limit < expected_y:
                    limits_consistent = False
                    break
        
        # Only plot the regression line if significant AND consistent with limits
        if is_significant and limits_consistent:
            # Plot the regression line
            x_line = np.array([0, 1])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', alpha=0.7)
            
            # Add R-squared annotation
            ax.annotate(f'R² = {r_value**2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    return fig, ax


def create_proper_motion_plots(data, light_gray, gray, blue, green):
    """Create all proper motion histograms"""
    
    # Create bin edges similar to the Mathematica histograms
    # Using 20 bins from 0 to 200 to match the original spacing
    bins = np.linspace(0, 200, 21)

    # Plot 1: All proper motions
    fig1, ax1 = plt.subplots()
    n, bins, patches = ax1.hist(
        data['all_proper_motions'], 
        bins=bins, 
        color=light_gray, 
        edgecolor='black',
        linewidth=1.0
    )
    ax1.set_xlabel('Proper motions (mas $d^{-1}$)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Proper motions (mas $d^{-1}$)', pad=10)
    ax1.legend(['All measured proper motions'], loc='upper right', frameon=False)
    ax1.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, 5.5)
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig('figures/pm_updated.pdf')
    plt.savefig('figures/pm_updated.png', dpi=300)

    # Plot 2: MeerKAT vs other proper motions
    fig2, ax2 = plt.subplots()
    n, bins, patches = ax2.hist(
        [data['meerkat_proper_motions'], data['no_meerkat_proper_motions']], 
        bins=bins, 
        stacked=True, 
        color=['#4682B4', light_gray],  # Slightly adjusted blue 
        edgecolor='black',
        linewidth=1.0
    )
    ax2.set_xlabel('Proper motions (mas $d^{-1}$)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Proper motions (mas $d^{-1}$)', pad=10)
    ax2.legend(['Measured proper motions (MeerKAT)', 'Measured proper motions (other)'], 
               loc='upper right', frameon=False)
    ax2.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 5.5)
    ax2.yaxis.set_major_locator(MultipleLocator(1))
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig('figures/pmkat_updated.pdf')
    plt.savefig('figures/pmkat_updated.png', dpi=300)

    # Plot 3: Black Holes vs Neutron Stars vs Unknown
    fig3, ax3 = plt.subplots()
    n, bins, patches = ax3.hist(
        [data['bh_proper_motions'], data['ns_proper_motions'], data['unknown_proper_motions']], 
        bins=bins, 
        stacked=True, 
        color=[gray, blue, green], 
        edgecolor='black',
        linewidth=1.0
    )
    ax3.set_xlabel('Proper motions (mas $d^{-1}$)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Proper motions (mas $d^{-1}$)', pad=10)
    ax3.legend(['Black Holes', 'Neutron Stars', 'Unknown'], 
               loc='upper right', frameon=False)
    ax3.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    ax3.set_axisbelow(True)
    ax3.set_ylim(0, 5.5)
    ax3.yaxis.set_major_locator(MultipleLocator(1))
    for spine in ax3.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig('figures/pmsep_updated.pdf')
    plt.savefig('figures/pmsep_updated.png', dpi=300)

    # # Plot 4: No URF version with same styling as Plot 3
    # fig4, ax4 = plt.subplots()
    # n, bins, patches = ax4.hist(
    #     [data['bh_proper_motions'], data['ns_proper_motions'], data['unknown_proper_motions']], 
    #     bins=bins, 
    #     stacked=True, 
    #     color=[gray, blue, green], 
    #     edgecolor='black',
    #     linewidth=1.0
    # )
    # ax4.set_xlabel('Proper motions (mas $d^{-1}$)')
    # ax4.set_ylabel('Frequency')
    # ax4.set_title('Proper motions (mas $d^{-1}$), no URF', pad=10)
    # ax4.legend(['Black Holes', 'Neutron Stars', 'Unknown'], 
    #            loc='upper right', frameon=False)
    # ax4.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    # ax4.set_axisbelow(True)
    # ax4.set_ylim(0, 5.5)
    # ax4.yaxis.set_major_locator(MultipleLocator(1))
    # for spine in ax4.spines.values():
    #     spine.set_linewidth(1.0)
    #     spine.set_color('black')
    # plt.tight_layout()
    # plt.savefig('pmsepnourf_updated.pdf')
    # plt.savefig('pmsepnourf_updated.png', dpi=300)

def create_apparent_speed_plots(data, light_gray, gray, blue, green):
    """Create all apparent speed histograms"""
    
    # Calculate max speed for bin range with a slight buffer
    max_speed = max(data['all_speeds']) * 1.1
    
    # Plot 1: All apparent speeds
    fig1, ax1 = plt.subplots()
    # Create bins with width 0.5 to match Mathematica
    bins = np.arange(0, max_speed + 0.25, 0.25)

    n, bins, patches = ax1.hist(
        data['all_speeds'], 
        bins=bins, 
        color=light_gray, 
        edgecolor='black',
        linewidth=1.0
    )
    ax1.set_xlabel('Apparent speed (c)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Apparent speed (c)', pad=10)
    ax1.legend(['All apparent speeds'], loc='upper right', frameon=False)
    ax1.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    ax1.set_axisbelow(True)
    # Set y-axis to show integer values
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig('figures/speed.pdf')
    plt.savefig('figures/speed.png', dpi=300)

    # Plot 2: Black Holes vs Neutron Stars vs Unknown 
    fig2, ax2 = plt.subplots()
    # Use smaller bins (0.25) as in Mathematica
    bins = np.arange(0, max_speed + 0.25, 0.25)

    n, bins, patches = ax2.hist(
        [data['bh_speeds'], data['ns_speeds'], data['unknown_speeds']], 
        bins=bins, 
        stacked=True, 
        color=[gray, blue, green], 
        edgecolor='black',
        linewidth=1.0
    )
    ax2.set_xlabel('Apparent speed (c)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Apparent speed (c)', pad=10)
    ax2.legend(['Black Holes', 'Neutron Stars', 'Unknown'], 
               loc='upper right', frameon=False)
    ax2.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.yaxis.set_major_locator(MultipleLocator(1))
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig('figures/speedsepnourf.pdf')
    plt.savefig('figures/speedsepnourf.png', dpi=300)

    # # Plot 3: Precession vs No evidence for precession
    # fig3, ax3 = plt.subplots()
    # # Use bins with width 0.5 as in Mathematica
    # bins = np.arange(0, max_speed + 0.5, 0.5)

    # n, bins, patches = ax3.hist(
    #     [data['prec_speeds'], data['nonprec_speeds']], 
    #     bins=bins, 
    #     stacked=True, 
    #     color=['#FFA07A', '#20B2AA'],  # Light salmon and light sea green
    #     edgecolor='black',
    #     linewidth=1.0
    # )
    # ax3.set_xlabel('Apparent speed (c)')
    # ax3.set_ylabel('Frequency')
    # ax3.set_title('Apparent speed (c)', pad=10)
    # ax3.legend(['Precession', 'No evidence for precession'], 
    #            loc='upper right', frameon=False)
    # ax3.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    # ax3.set_axisbelow(True)
    # ax3.yaxis.set_major_locator(MultipleLocator(1))
    # for spine in ax3.spines.values():
    #     spine.set_linewidth(1.0)
    #     spine.set_color('black')
    # plt.tight_layout()
    # plt.savefig('prec.pdf')
    # plt.savefig('prec.png', dpi=300)

    # # Plot 4: Precession vs No evidence for precession (no URF)
    # fig4, ax4 = plt.subplots()
    # # Use smaller bins (0.1) as in Mathematica
    # bins = np.arange(0, max_speed + 0.1, 0.1)

    # n, bins, patches = ax4.hist(
    #     [data['prec_speeds'], data['nonprec_speeds']], 
    #     bins=bins, 
    #     stacked=True, 
    #     color=['#FFA07A', '#20B2AA'],  # Light salmon and light sea green
    #     edgecolor='black',
    #     linewidth=1.0
    # )
    # ax4.set_xlabel('Apparent speed (c)')
    # ax4.set_ylabel('Frequency')
    # ax4.set_title('Apparent speed (c), no URF', pad=10)
    # ax4.legend(['Precession', 'No evidence for precession'], 
    #            loc='upper right', frameon=False)
    # ax4.grid(True, linestyle='dotted', color='#CCCCCC', alpha=0.6)
    # ax4.set_axisbelow(True)
    # ax4.yaxis.set_major_locator(MultipleLocator(1))
    # for spine in ax4.spines.values():
    #     spine.set_linewidth(1.0)
    #     spine.set_color('black')
    # plt.tight_layout()
    # plt.savefig('precnourf.pdf')
    # plt.savefig('precnourf.png', dpi=300)

def create_physics_plots(data, triangle_points, light_gray, gray, blue, green, red):
    """Create physics plots with triangle annotations"""
    
    # # Plot 1: Beta histogram
    # beta_bins = np.arange(0, 1.1 + 0.05, 0.05)
    # fig1, ax1 = create_histogram(
    #     data['all_betas'], 
    #     beta_bins, 
    #     light_gray, 
    #     "Fractional speed of light β", 
    #     "β", 
    #     [0.0, 1.1], 
    #     [0, 8]
    # )
    # plt.tight_layout()
    # plt.savefig('physics1a.pdf')
    # plt.savefig('physics1a.png', dpi=300)

    # # Plot 2: Gamma histogram with triangles
    # gamma_bins = np.arange(1.0, 6.0 + 0.25, 0.25)
    # fig2, ax2 = create_histogram(
    #     data['all_gammas'], 
    #     gamma_bins, 
    #     light_gray, 
    #     "Lorentz factor γ", 
    #     "γ", 
    #     [1.0, 6.0], 
    #     [0, 10]
    # )
    # add_triangles(ax2, [triangle_points['t1_points'], triangle_points['t1a_points']])
    # plt.tight_layout()
    # plt.savefig('physics1b.pdf')
    # plt.savefig('physics1b.png', dpi=300)

    # # Plot 3: Beta*Gamma histogram with triangles
    betagamma_bins = np.arange(0.0, 6.0 + 0.25, 0.25)
    # fig3, ax3 = create_histogram(
    #     data['all_betagammas'], 
    #     betagamma_bins, 
    #     light_gray, 
    #     "Measured βγ all sources", 
    #     "βγ", 
    #     [0.0, 6.0], 
    #     [0, 8]
    # )
    # add_triangles(ax3, [triangle_points['t2_points'], triangle_points['t2a_points']])
    # plt.tight_layout()
    # plt.savefig('physics1c.pdf')
    # plt.savefig('physics1c.png', dpi=300)

    # # Plot 4: Beta by source type (no URF)
    # fig4, ax4 = create_histogram(
    #     [data['bh_betas'], data['ns_betas'], data['unknown_betas']], 
    #     beta_bins, 
    #     [gray, blue, green], 
    #     "Fractional speed of light β (no URF)", 
    #     "β", 
    #     [0.0, 1.1], 
    #     [0, 6],
    #     ["BH", "NS", "BH/NS?"]
    # )
    # plt.tight_layout()
    # plt.savefig('physics3a.pdf')
    # plt.savefig('physics3a.png', dpi=300)

    # # Plot 5: Gamma by source type with triangles (no URF)
    # fig5, ax5 = create_histogram(
    #     [data['bh_gammas'], data['ns_gammas'], data['unknown_gammas']], 
    #     gamma_bins, 
    #     [gray, blue, green], 
    #     "Lorentz factor γ (no URF)", 
    #     "γ", 
    #     [1.0, 6.0], 
    #     [0, 7],
    #     ["BH", "NS", "BH/NS?"]
    # )
    # add_triangles(ax5, [triangle_points['t3_points'], triangle_points['t3a_points']])
    # plt.tight_layout()
    # plt.savefig('physics3b.pdf')
    # plt.savefig('physics3b.png', dpi=300)

    # Plot 6: Beta*Gamma by source type with triangles (no URF) - Science Figure 1
    fig6, ax6 = create_histogram(
        [data['bh_betagammas'], data['ns_betagammas'], data['unknown_betagammas']], 
        betagamma_bins, 
        [gray, blue, green], 
        "Measured βγ", 
        "βγ", 
        [0.0, 6.0], 
        [0, 6],
        ["BH", "NS", "unknown CO"]
    )
    add_triangles(ax6, [triangle_points['t4_points'], triangle_points['t4a_points']])
    plt.tight_layout()
    plt.savefig('figures/physics3c.pdf')
    plt.savefig('figures/physics3c.png', dpi=300)
    plt.savefig('figures/SCIENCEFIG1.pdf')
    plt.savefig('figures/SCIENCEFIG1.png', dpi=300)

    # # Plot 7: Precession vs No Precession Beta*Gamma with triangles
    # fig7, ax7 = create_histogram(
    #     [data['prec_betagammas'], data['nonprec_betagammas']], 
    #     betagamma_bins, 
    #     ['#FFA07A', '#20B2AA'],  # Light salmon and light sea green
    #     "βγ", 
    #     "βγ", 
    #     [0.0, 16.0], 
    #     [0, 6],
    #     ["precessing", "no precession"]
    # )
    # add_triangles(ax7, [triangle_points['t2_points']])
    # plt.tight_layout()
    # plt.savefig('physics4a.pdf')
    # plt.savefig('physics4a.png', dpi=300)

    # Plot 8: Precession vs No Precession Beta*Gamma with triangles (no URF)
    fig8, ax8 = create_histogram(
        [data['prec_betagammas'], data['nonprec_betagammas']], 
        betagamma_bins, 
        ['#FFA07A', '#20B2AA'],  # Light salmon and light sea green
        "Measured βγ", 
        "βγ", 
        [0.0, 6.0], 
        [0, 6],
        ["precessing", "fixed axis"]
    )
    add_triangles(ax8, [triangle_points['t4_points'], triangle_points['t4a_points']])
    plt.tight_layout()
    plt.savefig('physics4b.pdf')
    plt.savefig('physics4b.png', dpi=300)

    # Plot 9: Three-way classification with triangles - Science Figure 2
    fig9, ax9 = create_histogram(
        [data['prec_betagammas'], data['fixed_betagammas'], data['single_betagammas']], 
        betagamma_bins, 
        [red, blue, 'white'], 
        "Measured βγ", 
        "βγ", 
        [0.0, 6.0], 
        [0, 6],
        ["precessing", "fixed axis", "single"]
    )
    add_triangles(ax9, [triangle_points['t4_points'], triangle_points['t4a_points']])
    plt.tight_layout()
    plt.savefig('figures/prec3way.pdf')
    plt.savefig('figures/prec3way.png', dpi=300)
    plt.savefig('figures/SCIENCEFIG2.pdf')
    plt.savefig('figures/SCIENCEFIG2.png', dpi=300)


def create_spin_speed_plots(rr, rrl, cc, ccl, qq, qql):
    """Create spin vs. speed correlation plots"""
    
    # Create the plots for spin vs. betagamma
    fig_reflection, ax_reflection = create_spin_plot(rr, rrl, "BH spin (reflection)")
    plt.tight_layout()
    plt.savefig('figures/spin_reflection.pdf')
    plt.savefig('figures/spin_reflection.png', dpi=300)

    fig_continuum, ax_continuum = create_spin_plot(cc, ccl, "BH spin (continuum)")
    plt.tight_layout()
    plt.savefig('figures/spin_continuum.pdf')
    plt.savefig('figures/spin_continuum.png', dpi=300)

    fig_qpo, ax_qpo = create_spin_plot(qq, qql, "BH spin (QPO)")
    plt.tight_layout()
    plt.savefig('figures/spin_qpo.pdf')
    plt.savefig('figures/spin_qpo.png', dpi=300)

    # Create a combined figure with all three plots (similar to GraphicsColumn in Mathematica)
    fig_combined = plt.figure(figsize=(8, 18))
    gs = gridspec.GridSpec(3, 1, figure=fig_combined)

    # Reflection plot - NO REGRESSION LINES
    ax1 = fig_combined.add_subplot(gs[0, 0])
    ax1.scatter([x for x, y in rr], [y for x, y in rr], color='blue', s=100, marker='o', label='Measurements')
    ax1.scatter([x for x, y in rrl], [y for x, y in rrl], color='blue', s=150, marker='^', label='βγ Lower Limits')
    ax1.set_xlabel("BH spin (reflection)", fontsize=14)
    ax1.set_ylabel("βγ", fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.grid(True, linestyle='dotted', alpha=0.6)
    ax1.legend()

    # Continuum plot - NO REGRESSION LINES
    ax2 = fig_combined.add_subplot(gs[1, 0])
    ax2.scatter([x for x, y in cc], [y for x, y in cc], color='blue', s=100, marker='o', label='Measurements')
    ax2.scatter([x for x, y in ccl], [y for x, y in ccl], color='blue', s=150, marker='^', label='βγ Lower Limits')
    ax2.set_xlabel("BH spin (continuum)", fontsize=14)
    ax2.set_ylabel("βγ", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.grid(True, linestyle='dotted', alpha=0.6)
    ax2.legend()

    # QPO plot - NO REGRESSION LINES
    ax3 = fig_combined.add_subplot(gs[2, 0])
    ax3.scatter([x for x, y in qq], [y for x, y in qq], color='blue', s=100, marker='o', label='Measurements')
    ax3.scatter([x for x, y in qql], [y for x, y in qql], color='blue', s=150, marker='^', label='βγ Lower Limits')
    ax3.set_xlabel("BH spin (QPO)", fontsize=14)
    ax3.set_ylabel("βγ", fontsize=14)
    ax3.set_xlim(0, 1)
    ax3.grid(True, linestyle='dotted', alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('figures/spintests.pdf')
    plt.savefig('figures/spintests.png', dpi=300)
    plt.savefig('figures/SCIENCEFIG3.pdf')
    plt.savefig('figures/SCIENCEFIG3.png', dpi=300)
    