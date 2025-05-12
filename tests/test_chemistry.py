"""Tests for chemical mechanisms."""

import numpy as np
import pytest
from phasta.solver.chemistry import (
    ChemicalMechanism,
    DetailedChemistry,
    SootFormation,
    PollutantFormation
)


def test_detailed_chemistry():
    """Test detailed chemistry model."""
    # Create model
    model = DetailedChemistry("mechanism.cti")
    
    # Test data
    temperature = np.array([1000.0, 800.0, 600.0])
    pressure = np.array([101325.0, 101325.0, 101325.0])
    species_mass_fractions = np.array([
        [0.1, 0.2, 0.3],  # CH4
        [0.2, 0.3, 0.4],  # O2
        [0.3, 0.2, 0.1],  # CO2
        [0.2, 0.1, 0.1],  # H2O
        [0.2, 0.2, 0.1]   # N2
    ])
    
    # Test reaction rates
    reaction_rates = model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    
    # Check results
    assert reaction_rates.shape == (1, 3)  # 1 reaction, 3 points
    assert np.all(reaction_rates >= 0.0)
    
    # Test species sources
    source_terms = model.compute_species_sources(reaction_rates)
    
    # Check results
    assert source_terms.shape == (5, 3)  # 5 species, 3 points
    
    # Test heat release
    species_enthalpies = {
        "CH4": -74873.0,
        "O2": 0.0,
        "CO2": -393522.0,
        "H2O": -241826.0,
        "N2": 0.0
    }
    heat_release = model.compute_heat_release(reaction_rates, species_enthalpies)
    
    # Check results
    assert heat_release.shape == (3,)
    assert np.all(heat_release >= 0.0)


def test_soot_formation():
    """Test soot formation model."""
    # Create model
    model = SootFormation()
    
    # Test data
    temperature = np.array([1000.0, 800.0, 600.0])
    pressure = np.array([101325.0, 101325.0, 101325.0])
    species_mass_fractions = np.array([
        [0.1, 0.2, 0.3],  # C2H2
        [0.2, 0.3, 0.4],  # C6H6
        [0.3, 0.2, 0.1],  # C10H8
        [0.2, 0.1, 0.1],  # C16H10
        [0.2, 0.2, 0.1]   # Soot
    ])
    
    # Test reaction rates
    reaction_rates = model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    
    # Check results
    assert reaction_rates.shape == (4, 3)  # 4 reactions, 3 points
    assert np.all(reaction_rates >= 0.0)
    
    # Test species sources
    source_terms = model.compute_species_sources(reaction_rates)
    
    # Check results
    assert source_terms.shape == (5, 3)  # 5 species, 3 points
    
    # Test heat release
    species_enthalpies = {
        "C2H2": 226730.0,
        "C6H6": 82930.0,
        "C10H8": 150960.0,
        "C16H10": 110160.0,
        "Soot": 0.0
    }
    heat_release = model.compute_heat_release(reaction_rates, species_enthalpies)
    
    # Check results
    assert heat_release.shape == (3,)
    assert np.all(heat_release >= 0.0)


def test_pollutant_formation():
    """Test pollutant formation model."""
    # Create model
    model = PollutantFormation()
    
    # Test data
    temperature = np.array([1000.0, 800.0, 600.0])
    pressure = np.array([101325.0, 101325.0, 101325.0])
    species_mass_fractions = np.array([
        [0.1, 0.2, 0.3],  # NO
        [0.2, 0.3, 0.4],  # NO2
        [0.3, 0.2, 0.1],  # N2O
        [0.2, 0.1, 0.1],  # CO
        [0.3, 0.2, 0.1],  # CO2
        [0.2, 0.1, 0.1]   # H2O
    ])
    
    # Test reaction rates
    reaction_rates = model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    
    # Check results
    assert reaction_rates.shape == (3, 3)  # 3 reactions, 3 points
    assert np.all(reaction_rates >= 0.0)
    
    # Test species sources
    source_terms = model.compute_species_sources(reaction_rates)
    
    # Check results
    assert source_terms.shape == (6, 3)  # 6 species, 3 points
    
    # Test heat release
    species_enthalpies = {
        "NO": 90250.0,
        "NO2": 33180.0,
        "N2O": 82050.0,
        "CO": -110530.0,
        "CO2": -393522.0,
        "H2O": -241826.0
    }
    heat_release = model.compute_heat_release(reaction_rates, species_enthalpies)
    
    # Check results
    assert heat_release.shape == (3,)
    assert np.all(heat_release >= 0.0)


def test_edge_cases():
    """Test edge cases."""
    # Create models
    detailed_model = DetailedChemistry("mechanism.cti")
    soot_model = SootFormation()
    pollutant_model = PollutantFormation()
    
    # Test data
    temperature = np.array([0.0, 0.0, 0.0])
    pressure = np.array([0.0, 0.0, 0.0])
    species_mass_fractions = np.zeros((5, 3))
    
    # Test zero temperature
    reaction_rates = detailed_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert np.all(reaction_rates == 0.0)
    
    reaction_rates = soot_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert np.all(reaction_rates == 0.0)
    
    reaction_rates = pollutant_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert np.all(reaction_rates == 0.0)
    
    # Test zero pressure
    temperature = np.array([1000.0, 800.0, 600.0])
    
    reaction_rates = detailed_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert np.all(reaction_rates == 0.0)
    
    reaction_rates = soot_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert np.all(reaction_rates == 0.0)
    
    reaction_rates = pollutant_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert np.all(reaction_rates == 0.0)


def test_memory_management():
    """Test memory management."""
    # Create models
    detailed_model = DetailedChemistry("mechanism.cti")
    soot_model = SootFormation()
    pollutant_model = PollutantFormation()
    
    # Test data
    n_points = 1000
    temperature = np.ones(n_points) * 1000.0
    pressure = np.ones(n_points) * 101325.0
    species_mass_fractions = np.ones((5, n_points)) * 0.2
    
    # Test large arrays
    reaction_rates = detailed_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert reaction_rates.shape == (1, n_points)
    
    reaction_rates = soot_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert reaction_rates.shape == (4, n_points)
    
    reaction_rates = pollutant_model.compute_reaction_rates(
        temperature, pressure, species_mass_fractions
    )
    assert reaction_rates.shape == (3, n_points)


def test_convergence():
    """Test convergence with grid refinement."""
    # Create model
    model = DetailedChemistry("mechanism.cti")
    
    # Test data
    n_points = [10, 20, 40, 80]
    temperature = np.array([1000.0, 800.0, 600.0])
    pressure = np.array([101325.0, 101325.0, 101325.0])
    species_mass_fractions = np.array([
        [0.1, 0.2, 0.3],  # CH4
        [0.2, 0.3, 0.4],  # O2
        [0.3, 0.2, 0.1],  # CO2
        [0.2, 0.1, 0.1],  # H2O
        [0.2, 0.2, 0.1]   # N2
    ])
    
    # Test different grid sizes
    reaction_rates = []
    
    for n in n_points:
        # Interpolate data to finer grid
        T = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 3), temperature)
        P = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 3), pressure)
        Y = np.array([
            np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 3), y)
            for y in species_mass_fractions
        ])
        
        # Compute reaction rates
        rates = model.compute_reaction_rates(T, P, Y)
        reaction_rates.append(np.mean(rates))
    
    # Check convergence
    for i in range(len(reaction_rates) - 1):
        ratio = abs(reaction_rates[i] - reaction_rates[i+1]) / reaction_rates[i+1]
        assert ratio < 0.1  # Should converge with grid refinement 