#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jet Analysis Code - Main Script

This code analyzes relativistic jets from black holes and neutron stars.
"""

import os
import data_preparation as dp
import visualisation as vis
import analysis

# ====================================================
# ANALYSIS CONFIGURATION - ADJUST PARAMETERS AS NEEDED
# ====================================================

# What to analyze
RUN_POPULATION_ANALYSIS = False   # Run population distribution analysis
RUN_SPIN_ANALYSIS = False          # Run black hole spin correlation analysis

# Statistical test method for population analysis
TEST_METHOD = "ad"                 # "ks" (Kolmogorov-Smirnov), "ad" (Anderson-Darling)

# Uncertainty parameters
# For a single analysis point:
#   - Set MIN/MAX to the same value
#   - Set STEP to 0
# For a threshold exploration (heatmap):
#   - Set MIN/MAX to different values
#   - Set STEP to a non-zero value

# Angle uncertainty (degrees)
ANGLE_UNCERTAINTY_MIN = 0.0        # Minimum angle uncertainty to test
ANGLE_UNCERTAINTY_MAX = 30.0        # Maximum angle uncertainty to test
ANGLE_UNCERTAINTY_STEP = 5.0       # Step size for angle uncertainty

# Distance uncertainty (as a fraction)
DISTANCE_UNCERTAINTY_MIN = 0.0     # Minimum distance uncertainty to test
DISTANCE_UNCERTAINTY_MAX = 0.5   # Maximum distance uncertainty to test
DISTANCE_UNCERTAINTY_STEP = 0.05    # Step size for distance uncertainty

# Simulation parameters
ITERATIONS = 1000                    # Monte Carlo iterations per parameter set
SPIN_METHOD = "qpo"         # Spin measurement method for threshold ("reflection", "continuum", "qpo")

def main():
   # Create figures directory if it doesn't exist
   if not os.path.exists('figures'):
       os.makedirs('figures')
       print("Created 'figures' directory for plot output")
   
   print(" ********************** ANALYSIS BEGINNING ********************** ")
   
   # Load and prepare data
   print("Starting data preparation...")
   all_jets_list, categories = dp.prepare_data()
   
   # ALWAYS GENERATE KEY PLOTS
   print("\n ********************** GENERATING KEY PLOTS ********************** ")
   
   # Always generate spin-speed plots
   vis.create_spin_speed_plots(
       categories['rr'], categories['rrl'],
       categories['cc'], categories['ccl'],
       categories['qq'], categories['qql']
   )
   
   # Always generate the physics plots using existing function
   print("Generating physics plots...")
   
   # Extract data using existing structure from create_all_plots
   alljets = categories['alljets']
   bh = categories['bh']
   nsnourf = categories['nsnourf']
   prec = categories['prec']
   fixednourf = categories['fixednourf']
   
   extracted_data = {
       'bh_betagammas': [jet[7] for jet in bh],
       'ns_betagammas': [jet[7] for jet in nsnourf],
       'unknown_betagammas': [jet[7] for jet in categories['unknourf']],
       'prec_betagammas': [jet[7] for jet in prec],
       'nonprec_betagammas': [jet[7] for jet in categories['nonprecnourf']],
       'fixed_betagammas': [jet[7] for jet in fixednourf],
       'single_betagammas': [jet[7] for jet in categories['singlenourf']]
   }
   
   # Define colors (from existing code)
   light_gray = '#CCCCCC'
   gray = '#999999'
   blue = '#6495ED'
   green = '#90EE90'
   red = '#FF6347'
   
   # Call existing physics plots function
   vis.create_physics_plots(
       extracted_data, 
       categories['triangle_points'],
       light_gray, gray, blue, green, red
   )
   
   print(" ********************** KEY PLOTS GENERATED ********************** ")
   
   # Determine analysis mode based on parameters
   is_threshold_analysis = (
       (ANGLE_UNCERTAINTY_MIN != ANGLE_UNCERTAINTY_MAX and ANGLE_UNCERTAINTY_STEP > 0) or
       (DISTANCE_UNCERTAINTY_MIN != DISTANCE_UNCERTAINTY_MAX and DISTANCE_UNCERTAINTY_STEP > 0)
   )
   
   # Create uncertainty ranges whether we're doing threshold analysis or not
   angle_range = (
       ANGLE_UNCERTAINTY_MIN, 
       ANGLE_UNCERTAINTY_MAX if is_threshold_analysis else ANGLE_UNCERTAINTY_MIN
   )
   
   distance_range = (
       DISTANCE_UNCERTAINTY_MIN, 
       DISTANCE_UNCERTAINTY_MAX if is_threshold_analysis else DISTANCE_UNCERTAINTY_MIN
   )
   
   # Determine mode description for output
   if is_threshold_analysis:
       mode_description = (
           f"exploring thresholds (angle: {ANGLE_UNCERTAINTY_MIN}° to {ANGLE_UNCERTAINTY_MAX}°, "
           f"distance: {DISTANCE_UNCERTAINTY_MIN*100:.0f}% to {DISTANCE_UNCERTAINTY_MAX*100:.0f}%)"
       )
       uncertainty_mode = "threshold"
   elif ANGLE_UNCERTAINTY_MIN > 0 or DISTANCE_UNCERTAINTY_MIN > 0 or ITERATIONS > 1:
       mode_description = (
           f"with uncertainties (angle±{ANGLE_UNCERTAINTY_MIN}°, "
           f"distance±{DISTANCE_UNCERTAINTY_MIN*100:.0f}%, {ITERATIONS} iterations)"
       )
       uncertainty_mode = "single"
   else:
       mode_description = "without uncertainties"
       uncertainty_mode = "none"
   
   # Run population analysis if enabled
   if RUN_POPULATION_ANALYSIS:
       print(f"\n ********************** RUNNING POPULATION ANALYSIS {mode_description} ********************** ")
       
       population_results = analysis.unified_analysis(
           all_jets_list,
           categories,
           analysis_type="population",
           uncertainty_mode=uncertainty_mode,
           angle_uncertainty=ANGLE_UNCERTAINTY_MIN,
           distance_uncertainty=DISTANCE_UNCERTAINTY_MIN,
           angle_uncertainty_range=angle_range,
           angle_step=ANGLE_UNCERTAINTY_STEP,
           distance_uncertainty_range=distance_range,
           distance_step=DISTANCE_UNCERTAINTY_STEP,
           num_iterations=ITERATIONS,
           test_method=TEST_METHOD,
           suppress_interim_plots=False
       )
   else:
       population_results = None
   
   # Run spin analysis if enabled
   if RUN_SPIN_ANALYSIS:
       print(f"\n ********************** RUNNING SPIN ANALYSIS {mode_description} ********************** ")
       
       spin_results = analysis.unified_analysis(
           all_jets_list,
           categories,
           analysis_type="spin",
           uncertainty_mode=uncertainty_mode,
           angle_uncertainty=ANGLE_UNCERTAINTY_MIN,
           distance_uncertainty=DISTANCE_UNCERTAINTY_MIN,
           angle_uncertainty_range=angle_range,
           angle_step=ANGLE_UNCERTAINTY_STEP,
           distance_uncertainty_range=distance_range,
           distance_step=DISTANCE_UNCERTAINTY_STEP,
           num_iterations=ITERATIONS,
           spin_method=SPIN_METHOD,
           suppress_interim_plots=False
       )
   else:
       spin_results = None
   
   # Generate additional visualizations for population and spin analysis if needed
   # Only do this for non-threshold analysis as threshold creates its own plots
   if (RUN_POPULATION_ANALYSIS or RUN_SPIN_ANALYSIS) and not is_threshold_analysis:
       print("\n ********************** ADDITIONAL PLOTTING BEGINNING ********************** ")
       
       # Extract the relevant results depending on the mode
       if uncertainty_mode == "none":
           pop_data = population_results.get("population") if population_results else None
           spin_data = spin_results.get("spin") if spin_results else None
       else:
           pop_data = population_results.get("population_uncertainty") if population_results else None
           spin_data = spin_results.get("spin_uncertainty") if spin_results else None
       
       vis.create_all_plots(all_jets_list, categories, pop_data, spin_data)
       print(" ********************** ADDITIONAL PLOTTING COMPLETE ********************** ")
       

if __name__ == "__main__":
   main()