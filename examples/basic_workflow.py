"""
Basic History Matching Workflow Example

This example demonstrates the simplest way to set up and run
a history matching workflow using the new object-oriented API.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the history matching library
import history_matching as hm


def simple_epidemic_model(samples: pd.DataFrame) -> pd.DataFrame:
    """
    Simple epidemic model for demonstration.

    Args:
        samples: DataFrame with parameters ['transmission_rate', 'recovery_rate', 'initial_infected']

    Returns:
        DataFrame with simulation outputs
    """
    results = []

    for _, row in samples.iterrows():
        # Extract parameters
        beta = row['transmission_rate']
        gamma = row['recovery_rate']
        I0 = row['initial_infected']

        # Simple SIR model simulation (deterministic)
        N = 10000 # Total population
        S0 = N - I0
        R0_basic = beta / gamma

        # Calculate peak and final values analytically for speed
        if R0_basic > 1:
            peak_infected = N * (1 - 1/R0_basic - (1/R0_basic) * np.log(R0_basic))
            final_recovered = N * (1 - np.exp(-R0_basic))
        else:
            peak_infected = I0
            final_recovered = I0

        # Add some realistic noise
        peak_infected += np.random.normal(0, peak_infected * 0.1)
        final_recovered += np.random.normal(0, final_recovered * 0.05)
        total_deaths = final_recovered * 0.02 + np.random.normal(0, 10)

        # Calculate additional model outputs
        attack_rate = final_recovered / N
        epidemic_duration = 200 if R0_basic > 1 else 50 # days
        epidemic_duration += np.random.normal(0, 20)

        results.append({
            'peak_infections': max(0, peak_infected),
            'total_deaths': max(0, total_deaths),
            'final_recovered': max(0, final_recovered),
            'attack_rate': max(0, min(1, attack_rate)),
            'epidemic_duration': max(0, epidemic_duration),
            'reproduction_number': R0_basic # This is a calculated output, not observed
        })

    return pd.DataFrame(results)


def main():
    """Run basic history matching workflow."""
    print(" Basic History Matching Workflow Example")
    print("=" * 50)

    # Step 1: Define parameter space and observations
    print(" Setting up parameter space and observations...")

    parameter_bounds = {
        'transmission_rate': (0.1, 1.0), # per day
        'recovery_rate': (0.05, 0.3), # per day
        'initial_infected': (1, 100) # number of people
    }

    # Target observations from "data" - these are directly observable quantities
    observations = {
        'peak_infections': (2500, 300), # (target, std) - from hospital data
        'total_deaths': (150, 25), # from death certificates
        'final_recovered': (8000, 500) # from serosurvey data
    }

    print(f" Parameters: {list(parameter_bounds.keys())}")
    print(f" Observations: {list(observations.keys())}")

    # Step 2: Build the engine using the builder pattern
    print("\nBuilding history matching engine...")

    builder = hm.HistoryMatchingBuilder.from_data(parameter_bounds, observations)
    engine = builder \
        .with_sampling_strategy('lhs') \
        .with_emulator_type('gpr') \
        .with_samples_per_iteration(1000) \
        .with_max_iterations(5) \
        .with_random_seed(42) \
        .build()

    print(f" Engine created: {engine}")

    # Step 3: Set simulation function
    print("\n Configuring simulation model...")
    engine.set_simulation_function(simple_epidemic_model)

    # Step 4: Run automated workflow
    print("\n Running automated history matching...")
    print(" This will run multiple iterations automatically...")

    results = engine.run()

    # Step 5: Analyze results
    print(f"\n Analysis complete! Ran {len(results)} iterations")
    print(f" Final acceptance rate: {engine.acceptance_rate:.3f}")
    print(f" Total samples generated: {engine.progress.total_samples_generated}")
    print(f" Total samples accepted: {engine.progress.total_samples_accepted}")

    # Print iteration summary
    print("\n Iteration Summary:")
    for i, result in enumerate(results, 1):
        print(f" Iteration {i}: {len(result.samples)} samples, "
              f"features {result.selected_features}")

    # Step 6: Extract final plausible parameter ranges
    if results:
        final_result = results[-1]
        print(f"\n Final plausible parameter ranges (from {len(final_result.samples)} samples):")

        for param in parameter_bounds.keys():
            param_values = final_result.samples[param]
            print(f" {param}: [{param_values.min():.3f}, {param_values.max():.3f}] "
                  f"(original: [{parameter_bounds[param][0]}, {parameter_bounds[param][1]}])")

    print("\n Basic workflow completed successfully!")
    return engine, results


def interactive_example():
    """Demonstrate interactive step-by-step workflow."""
    print("\n" + "=" * 50)
    print(" Interactive Workflow Example")
    print("=" * 50)

    # Setup
    parameter_bounds = {
        'transmission_rate': (0.2, 0.8),
        'recovery_rate': (0.1, 0.2),
        'initial_infected': (10, 50)
    }

    # Only directly observable quantities
    observations = {
        'peak_infections': (3000, 400),
        'total_deaths': (120, 20) # From death certificates - directly observable
    }

    # Build engine with more explicit configuration
    builder = hm.HistoryMatchingBuilder.from_data(parameter_bounds, observations)
    engine = builder \
        .with_sampling_strategy('lhs') \
        .with_feature_selection(['peak_infections']) \
        .with_emulator_type('linear') \
        .with_samples_per_iteration(500) \
        .build()

    engine.set_simulation_function(simple_epidemic_model)

    print(" Interactive engine built")
    print(f" Initial parameter space: {len(engine.parameter_space.get_parameter_names())} parameters")

    # Step 1: Run first iteration
    print("\n Step 1: Running first iteration...")
    result1 = engine.step()

    print(f" Iteration 1 completed:")
    print(f" - Generated {len(result1.samples)} samples")
    print(f" - Selected features: {result1.selected_features}")
    print(f" - Emulators trained: {len(result1.emulators)}")

    # Decision point: accept or modify?
    print(f"\n Should we accept this iteration? Let's check the emulator quality...")
    # In a real workflow, you might inspect result1.plot_emulator_diagnostics()

    print(" Looks good! Committing iteration 1...")
    engine.commit_step()

    # Step 2: Modify strategy and run again
    print(f"\n Step 2: Let's try both features this time...")
    engine.update_feature_selection(['peak_infections', 'total_deaths'])
    engine.update_emulator_type('gpr') # Switch to GPR for better flexibility

    result2 = engine.step()

    print(f" Iteration 2 completed:")
    print(f" - Generated {len(result2.samples)} samples")
    print(f" - Acceptance rate: {engine.acceptance_rate:.3f}")
    print(f" - Selected features: {result2.selected_features}")

    print(" Committing iteration 2...")
    engine.commit_step()

    print(f"\n Interactive workflow completed!")
    print(f" Final state: {engine.current_iteration} iterations, "
          f"{engine.progress.total_samples_accepted} total accepted samples")

    return engine


def advanced_configuration_example():
    """Demonstrate advanced configuration options."""
    print("\n" + "=" * 50)
    print(" Advanced Configuration Example")
    print("=" * 50)

    # Create parameter space DataFrame for more control
    parameter_df = pd.DataFrame({
        'parameter': ['beta', 'gamma', 'I0', 'contact_rate'],
        'minimum': [0.1, 0.05, 1, 5],
        'maximum': [1.0, 0.3, 100, 20]
    })

    # Create observations DataFrame - only directly observable quantities
    observations_df = pd.DataFrame({
        'feature': ['peak_infections', 'total_deaths', 'attack_rate'],
        'mean': [2500, 150, 0.75],
        'std': [300, 25, 0.1] # Note: std not variance
    })

    print(" Using DataFrame inputs for more control...")
    print(f" Parameters: {parameter_df['parameter'].tolist()}")
    print(f" Observations: {observations_df['feature'].tolist()}")

    # Advanced builder configuration with preview
    builder = hm.HistoryMatchingBuilder.from_dataframes(
        parameter_df,
        observations_df,
    )
    builder \
        .with_sampling_strategy({'type': 'lhs', 'criterion': 'maximin', 'iterations': 10}) \
        .with_feature_selection({'method': 'fano', 'max_features': 2, 'correlation_threshold': 0.7}) \
        .with_emulator_type('gpr') \
        .with_samples_per_iteration(800) \
        .with_max_iterations(8) \
        .with_implausibility_threshold(2.8) \
        .with_space_reduction(True) \
        .with_oversample_factor(4.0) \
        .with_random_seed(123)

    # Preview configuration before building
    print(f"\n Configuration preview:")
    config = builder.preview_configuration()
    for key, value in config.items():
        if isinstance(value, dict):
            print(f" {key}:")
            for sub_key, sub_value in value.items():
                print(f" {sub_key}: {sub_value}")
        else:
            print(f" {key}: {value}")

    # Now build the engine
    engine = builder.build()

    print(f"\n Advanced engine built:")
    print(f" Space reduction: {'enabled' if engine._auto_reduce_space else 'disabled'}")
    print(f" Oversample factor: {engine._oversample_factor}")
    print(f" Implausibility threshold: {engine._implausibility_threshold}")

    print(f"\n Advanced configuration example completed!")
    return engine


if __name__ == "__main__":
    # Run the examples
    np.random.seed(42) # For reproducible results

    # Basic workflow
    main_engine, main_results = main()

    # Interactive workflow
    interactive_engine = interactive_example()

    # Advanced configuration
    advanced_engine = advanced_configuration_example()

    print(f"\n All examples completed successfully!")
    print(f" Check the engine objects and results for detailed analysis.")