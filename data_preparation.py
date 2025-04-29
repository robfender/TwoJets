#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preparation Module

This module loads and prepares data for the jet analysis from CSV files.
"""

import os
import numpy as np
import pandas as pd
from tabulate import tabulate
from utils import clip, breal, lorentz

def prepare_data(data_file='jet_data_basic.csv'):
#def prepare_data(data_file='jet_data_basic_no1543.csv'):
#def prepare_data(data_file='jet_data_fakespincorr.csv'):
    """
    Load and prepare data for jet analysis from a CSV file
    
    Parameters:
    data_file: Path to CSV file with jet data (default: 'jet_data_basic.csv')
    
    Returns:
    all_jets_list, categories
    """
    print(f"Loading data from {data_file}...")
    
    # Check if file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Read data from CSV
    df = pd.read_csv(data_file)
    
    # Convert DataFrame back to list format expected by other modules
    all_jets_list = []
    
    # Create the list of lists structure with initial parameters
    for _, row in df.iterrows():
        # Extract the fundamental parameters
        name = row['name']
        proper_motion = row['proper_motion']
        distance = row['distance']
        inclination = row['inclination']
        accretor_type = row['accretor_type']
        precession_class = row['precession_class']
        spin_reflection = row['spin_reflection']
        spin_continuum = row['spin_continuum']
        spin_qpo = row['spin_qpo']
        
        # Initial values for derived quantities 
        apparent_speed = 0
        beta = 0
        gamma = 0
        beta_gamma = 0
        
        # Fast/slow will be determined after calculating beta*gamma
        fast_slow = ""
        
        # Create jet entry in original format
        jet = [
            name, proper_motion, distance, apparent_speed,
            inclination, beta, gamma, beta_gamma,
            accretor_type, fast_slow, precession_class,
            spin_reflection, spin_continuum, spin_qpo
        ]
        
        all_jets_list.append(jet)
    
    # Calculate derived quantities
    
    # Convert proper motion and distance to apparent speed in units of c
    for jet in all_jets_list:
        jet[3] = (jet[1]/173.0) * jet[2]

    # Calculate intrinsic beta
    for jet in all_jets_list:
        theta_rad = np.radians(jet[4])
        if jet[3] > 0:
            beta = jet[3] / (np.sin(theta_rad) + jet[3] * np.cos(theta_rad))
            jet[5] = clip(beta, 0, 0.90)
        else:
            jet[5] = 0

    # Calculate intrinsic gamma
    for jet in all_jets_list:
        if jet[5] > 0:
            jet[6] = 1 / np.sqrt(1 - jet[5]**2)
        else:
            jet[6] = 1

    # Calculate beta*gamma
    for jet in all_jets_list:
        jet[7] = jet[5] * jet[6]
    
    # Determine fast/slow classification
    for jet in all_jets_list:
        if jet[7] > 2.0:
            jet[9] = ">F"
        elif jet[7] > 1.0:
            jet[9] = "F"
        else:
            jet[9] = "S"

    # Manually set the lower limits for 4U1543 as it uniquely has a higher lower limit
    for jet in all_jets_list:
        if jet[0] == "4U1543":
            jet[5] = 0.982
            jet[6] = 5.3
            jet[7] = 5.2
            jet[9] = ">F"
            
    # Ideally would have a better way of doing this ^ (for future)
    
    # Output a file including the derived products (apparent speed, true beta, true gamma)
    output_data_with_derived_products(all_jets_list)
    
    # Identify specific jets by name for categorization
    jet_dict = {jet[0]: jet for jet in all_jets_list}
    
    # Create categories exactly as in the original code
    categories = {}
    
    # note several of these categories are redundant in the context
    # of the specific analysis in Fender & Motta (2025)
    
    # Compilation of all the jets
    alljets = all_jets_list
    categories['alljets'] = alljets
    
    # All jets with URFs removed (THIS IS REDUNDANT SHOULD BE REMOVED)
    alljetsnourf = all_jets_list.copy()
    categories['alljetsnourf'] = alljetsnourf
    
    # All black holes - find jets where accretor_type (index 8) is "BH"
    bh = [jet for jet in all_jets_list if jet[8] == "BH" or jet[8] == "BH?"]
    categories['bh'] = bh
    
    # Black holes with beta * gamma < 1.0
    slowbh = [jet for jet in bh if jet[7] < 1.0]
    categories['slowbh'] = slowbh
    
    # All the superluminal black holes
    superluminalbh = [jet for jet in bh if jet[7] > 1.0]
    categories['superluminalbh'] = superluminalbh
    
    # Unknown types
    unknourf = [jet for jet in all_jets_list if jet[8] == "?"]
    categories['unknourf'] = unknourf
    
    # All neutron stars
    nsnourf = [jet for jet in all_jets_list if jet[8] == "NS"]
    categories['nsnourf'] = nsnourf
    
    # Precessing or not
    prec = [jet for jet in all_jets_list if jet[10] == "P"]
    categories['prec'] = prec
    
    # Single ejection only
    singlenourf = [jet for jet in all_jets_list if jet[10] == "?"]
    categories['singlenourf'] = singlenourf
    
    # Multiple ejections in same direction
    fixednourf = [jet for jet in all_jets_list if jet[10] == "L" or jet[10] == "L?"]
    categories['fixednourf'] = fixednourf
    
    # No evidence for precession
    nonprecnourf = [jet for jet in all_jets_list if jet[10] != "P"]
    categories['nonprecnourf'] = nonprecnourf
    
    # MEERKAT classifications - by name
    # these are the sources whose jet speeds are from MeerKAT
    # for the 'MeerKAT promo' graphic
    meerkat_names = ["4U1543", "GX339-4", "MAXIJJ1820(fast)", "MAXIJ1348", 
                    "GRS1915", "MAXIJ1535", "J1848"]
    alljetsmeerkat = [jet for jet in all_jets_list if jet[0] in meerkat_names]
    alljetsnomeerkat = [jet for jet in all_jets_list if jet[0] not in meerkat_names]
    categories['alljetsmeerkat'] = alljetsmeerkat
    categories['alljetsnomeerkat'] = alljetsnomeerkat
    
    # Reflection spins
    reflection_jets = [jet for jet in bh if jet[11] > 0]  # All jets with reflection spin measurements
    reflections = [jet for jet in reflection_jets if jet[9] != ">F"]  # Direct measurements (not βγ lower limits)
    reflectionslimits = [jet for jet in reflection_jets if jet[9] == ">F"]  # Jets with βγ lower limits
    categories['reflections'] = reflections
    categories['reflectionslimits'] = reflectionslimits

    # Continuum spins
    continuum_jets = [jet for jet in bh if jet[12] > 0]  # All jets with continuum spin measurements
    continuums = [jet for jet in continuum_jets if jet[9] != ">F"]  # Direct measurements (not βγ lower limits)
    continuumslimits = [jet for jet in continuum_jets if jet[9] == ">F"]  # Jets with βγ lower limits
    categories['continuums'] = continuums
    categories['continuumslimits'] = continuumslimits

    # QPO spins
    qpo_jets = [jet for jet in bh if jet[13] > 0]  # All jets with QPO spin measurements
    qpos = [jet for jet in qpo_jets if jet[9] != ">F"]  # Direct measurements (not βγ lower limits)
    qposlimits = [jet for jet in qpo_jets if jet[9] == ">F"]  # Jets with βγ lower limits
    categories['qpos'] = qpos
    categories['qposlimits'] = qposlimits
    
   # Create paired data (spin value, βγ value)
    rr = [(jet[11], jet[7]) for jet in reflections]
    rrl = [(jet[11], jet[7]) for jet in reflectionslimits]
    categories['rr'] = rr
    categories['rrl'] = rrl
    
    cc = [(jet[12], jet[7]) for jet in continuums]
    ccl = [(jet[12], jet[7]) for jet in continuumslimits]
    categories['cc'] = cc
    categories['ccl'] = ccl
    
    qq = [(jet[13], jet[7]) for jet in qpos]
    qql = [(jet[13], jet[7]) for jet in qposlimits]
    categories['qq'] = qq
    categories['qql'] = qql

    
    # Define triangle coordinates for plots (used in visualization)
    # theses are essentially manually set for the columns I know
    # for _this_ data set are lower limits
    # would have to be tailored to new data sets
    triangle_points = {
        't1_points': np.array([[3.25, 5], [3.25, 0], [4, 2.5]]),
        't1a_points': np.array([[5.775, 1], [5.75, 0], [6.25, 0.5]]),
        't2_points': np.array([[3.25, 5], [3.25, 0], [4.05, 2.5]]),
        't2a_points': np.array([[5.5, 1], [5.5, 0], [6.3, 0.5]]),
        't3_points': np.array([[3.25, 4], [3.25, 0], [3.5, 2]]),
        't3a_points': np.array([[5.75, 1], [5.75, 0], [6, 0.5]]),
        't4_points': np.array([[2.3, 6], [2.3, 0], [2.59, 3]]),
        't4a_points': np.array([[5.3, 1], [5.3, 0], [5.5, 0.5]])
    }
    categories['triangle_points'] = triangle_points
    
    
    # Print the data table
    print_data_table(all_jets_list)
    
    # Print category assignments
    print_category_assignments(categories)
    
    return all_jets_list, categories

def print_data_table(all_jets_list):
    """Create a formatted table from the jet data"""
    # Create a DataFrame for display
    df = pd.DataFrame(all_jets_list, columns=[
        "Name", "Proper_motion", "Dist", "Apparent_speed", 
        "Inclination", "Beta", "Gamma", "Beta_Gamma", 
        "Accretor", "Fast_Slow", "Direction", 
        "a_refl", "a_cont", "a_QPO"
    ])

    # Print with nice formatting
    print("\nJet Data Table:")
    pd.set_option('display.precision', 3)
    df_formatted = df.round(3)
    print(df_formatted.to_string())
    print(" |BH = Black Hole, NS = Neutron Star, S = Slow, F = Fast, >F = Fast and unconstrained, L = Locked, P = Precessing, ? = Unknown, a* reported BH spins |")

def print_category_assignments(categories):
    """Print which sources fall into which categories"""
    print("\n===== JET CATEGORY ASSIGNMENTS =====")
    
    # Skip paired data and triangle points
    skip_categories = ['rr', 'rrl', 'cc', 'ccl', 'qq', 'qql', 'triangle_points']
    
    # Process each category
    for category_name, category_jets in sorted(categories.items()):
        if category_name in skip_categories:
            continue
            
        # Skip the full jet list to avoid overwhelming output
        if category_name == 'alljets' or category_name == 'alljetsnourf':
            print(f"\n{category_name}: [All {len(category_jets)} jets]")
            continue
        
        # Get names of jets in this category
        jet_names = [jet[0] for jet in category_jets]
        
        # Print category and its members
        # Comment everything below if you don't want to output all these categories
        print(f"\n{category_name} ({len(jet_names)} jets):")
        if jet_names:
            # Format nicely with wrapping for long lists
            max_width = 80
            current_line = "  "
            
            for name in sorted(jet_names):
                if len(current_line + name) > max_width:
                    print(current_line)
                    current_line = "  " + name + ", "
                else:
                    current_line += name + ", "
            
            # Print the last line without trailing comma
            if current_line != "  ":
                print(current_line.rstrip(", "))
        else:
            print("  [Empty category]")
    
    # Print paired data categories sizes
    print("\n===== PAIRED DATA CATEGORIES =====")
    for category_name in ['rr', 'rrl', 'cc', 'ccl', 'qq', 'qql']:
        if category_name in categories:
            print(f"{category_name}: {len(categories[category_name])} pairs")
            
            
def output_data_with_derived_products(all_jets_list):
    """
    Write a CSV file with input data and all derived products
    
    Parameters:
    all_jets_list: List of jet data including derived products
    """
    # Create a DataFrame from the jet data
    df = pd.DataFrame(all_jets_list, columns=[
        "Name", "Proper_motion", "Distance", "Apparent_speed", 
        "Inclination", "Beta", "Gamma", "Beta_Gamma", 
        "Accretor", "Fast_Slow", "Direction", 
        "a_refl", "a_cont", "a_QPO"
    ])
    
    # Format with reasonable precision
    df_output = df.round({
        "Proper_motion": 3,
        "Distance": 2,
        "Apparent_speed": 3,
        "Inclination": 1,
        "Beta": 4,
        "Gamma": 3,
        "Beta_Gamma": 3,
        "a_refl": 3,
        "a_cont": 3,
        "a_QPO": 3
    })
    
    # Write to CSV
    output_filename = 'jet_data_with_derived_products.csv'
    df_output.to_csv(output_filename, index=False)
    print(f"\nData with derived products saved to '{output_filename}'")
            
            
            
            
            
            
            