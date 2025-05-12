"""Tests for chemical reactions module."""

import numpy as np
import pytest
from phasta.solver.chemical import (
    ChemicalSpecies, Reaction, ChemicalMechanism, SpeciesTransport
)


def test_chemical_species():
    """Test chemical species class."""
    # Create test species
    h2 = ChemicalSpecies(
        name="H2",
        molecular_weight=2.016e-3,  # kg/mol
        formation_enthalpy=0.0,  # J/mol
        specific_heat=14300.0  # J/(kg·K)
    )
    
    assert h2.name == "H2"
    assert h2.molecular_weight == 2.016e-3
    assert h2.formation_enthalpy == 0.0
    assert h2.specific_heat == 14300.0


def test_reaction():
    """Test reaction class."""
    # Create test species
    h2 = ChemicalSpecies("H2", 2.016e-3, 0.0, 14300.0)
    o2 = ChemicalSpecies("O2", 32.0e-3, 0.0, 920.0)
    h2o = ChemicalSpecies("H2O", 18.016e-3, -241.8e3, 1880.0)
    
    # Create test reaction: H2 + 0.5 O2 -> H2O
    reaction = Reaction(
        reactants={"H2": 1.0, "O2": 0.5},
        products={"H2O": 1.0},
        forward_rate=1.0e6,
        backward_rate=1.0e5,
        activation_energy=50.0e3  # J/mol
    )
    
    # Test reaction rate computation
    concentrations = {
        "H2": 1.0,
        "O2": 1.0,
        "H2O": 0.1
    }
    temperature = 1000.0  # K
    
    rate = reaction.compute_rate(concentrations, temperature)
    assert rate > 0.0  # Forward rate should dominate at high temperature
    
    # Test at lower temperature
    rate_low_temp = reaction.compute_rate(concentrations, 300.0)
    assert rate_low_temp < rate  # Rate should be lower at lower temperature


def test_chemical_mechanism():
    """Test chemical mechanism class."""
    # Create test species
    h2 = ChemicalSpecies("H2", 2.016e-3, 0.0, 14300.0)
    o2 = ChemicalSpecies("O2", 32.0e-3, 0.0, 920.0)
    h2o = ChemicalSpecies("H2O", 18.016e-3, -241.8e3, 1880.0)
    
    # Create test reaction
    reaction = Reaction(
        reactants={"H2": 1.0, "O2": 0.5},
        products={"H2O": 1.0},
        forward_rate=1.0e6,
        backward_rate=1.0e5,
        activation_energy=50.0e3
    )
    
    # Create mechanism
    mechanism = ChemicalMechanism()
    mechanism.add_species(h2)
    mechanism.add_species(o2)
    mechanism.add_species(h2o)
    mechanism.add_reaction(reaction)
    
    # Test source terms
    concentrations = {
        "H2": 1.0,
        "O2": 1.0,
        "H2O": 0.1
    }
    temperature = 1000.0
    
    source_terms = mechanism.compute_source_terms(concentrations, temperature)
    assert source_terms["H2"] < 0.0  # H2 should be consumed
    assert source_terms["O2"] < 0.0  # O2 should be consumed
    assert source_terms["H2O"] > 0.0  # H2O should be produced
    
    # Test heat release
    heat_release = mechanism.compute_heat_release(concentrations, temperature)
    assert heat_release > 0.0  # Exothermic reaction


def test_species_transport():
    """Test species transport class."""
    # Create test mechanism
    mechanism = ChemicalMechanism()
    h2 = ChemicalSpecies("H2", 2.016e-3, 0.0, 14300.0)
    o2 = ChemicalSpecies("O2", 32.0e-3, 0.0, 920.0)
    mechanism.add_species(h2)
    mechanism.add_species(o2)
    
    # Create transport model
    transport = SpeciesTransport(mechanism)
    
    # Test diffusion
    concentrations = {"H2": 1.0, "O2": 1.0}
    temperature = 1000.0
    pressure = 101325.0  # Pa
    
    diffusion = transport.compute_diffusion(concentrations, temperature, pressure)
    assert diffusion["H2"] > diffusion["O2"]  # H2 should diffuse faster
    
    # Test convection
    velocity = np.array([1.0, 0.0, 0.0])
    convection = transport.compute_convection(concentrations, velocity)
    assert np.allclose(convection["H2"], concentrations["H2"] * velocity)
    assert np.allclose(convection["O2"], concentrations["O2"] * velocity)


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test zero concentration
    mechanism = ChemicalMechanism()
    h2 = ChemicalSpecies("H2", 2.016e-3, 0.0, 14300.0)
    mechanism.add_species(h2)
    
    reaction = Reaction(
        reactants={"H2": 1.0},
        products={},
        forward_rate=1.0e6,
        backward_rate=0.0,
        activation_energy=50.0e3
    )
    mechanism.add_reaction(reaction)
    
    # Test with zero concentration
    concentrations = {"H2": 0.0}
    temperature = 1000.0
    
    source_terms = mechanism.compute_source_terms(concentrations, temperature)
    assert source_terms["H2"] == 0.0  # No reaction with zero concentration
    
    # Test with missing species
    with pytest.raises(KeyError):
        mechanism.compute_source_terms({"O2": 1.0}, temperature)
    
    # Test with invalid temperature
    with pytest.raises(ValueError):
        reaction.compute_rate(concentrations, -1.0)


def test_memory_management():
    """Test memory management during computations."""
    # Create large mechanism
    mechanism = ChemicalMechanism()
    n_species = 100
    
    # Add many species
    for i in range(n_species):
        species = ChemicalSpecies(
            name=f"Species{i}",
            molecular_weight=1.0e-3,
            formation_enthalpy=0.0,
            specific_heat=1000.0
        )
        mechanism.add_species(species)
    
    # Add many reactions
    for i in range(n_species - 1):
        reaction = Reaction(
            reactants={f"Species{i}": 1.0},
            products={f"Species{i+1}": 1.0},
            forward_rate=1.0e6,
            backward_rate=1.0e5,
            activation_energy=50.0e3
        )
        mechanism.add_reaction(reaction)
    
    # Test computation with large mechanism
    concentrations = {f"Species{i}": 1.0 for i in range(n_species)}
    temperature = 1000.0
    
    source_terms = mechanism.compute_source_terms(concentrations, temperature)
    assert len(source_terms) == n_species 