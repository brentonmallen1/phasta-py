"""Tests for advanced combustion models."""

import numpy as np
import pytest
from phasta.solver.combustion import (
    SootModel, MossSootModel, PollutantModel, ZeldovichNOxModel,
    DetailedChemistry
)
from phasta.solver.chemical import ChemicalSpecies, Reaction


def test_moss_soot_model():
    """Test Moss soot model."""
    # Create model
    model = MossSootModel(
        nucleation_rate=1e-3,
        growth_rate=1e-2,
        oxidation_rate=1e-2,
        coagulation_rate=1e-3
    )
    
    # Test source terms
    T = 1500.0  # K
    Y = {
        'C2H2': 0.1,
        'O2': 0.2
    }
    rho = 1.0  # kg/m^3
    
    nucleation, growth = model.compute_source_terms(T, Y, rho)
    
    # Check rates
    assert nucleation > 0  # Nucleation should be positive
    assert growth > 0  # Growth should be positive
    
    # Test temperature dependence
    T_low = 1000.0
    nucleation_low, growth_low = model.compute_source_terms(T_low, Y, rho)
    assert nucleation_low < nucleation
    assert growth_low < growth
    
    # Test species dependence
    Y_no_C2H2 = {'O2': 0.2}
    nucleation_no_C2H2, growth_no_C2H2 = model.compute_source_terms(T, Y_no_C2H2, rho)
    assert nucleation_no_C2H2 < nucleation
    assert growth_no_C2H2 < growth


def test_zeldovich_nox_model():
    """Test Zeldovich NOx model."""
    # Create model
    model = ZeldovichNOxModel(
        nox_rate=1e-2,
        co_rate=1e-2,
        unburned_hc_rate=1e-3
    )
    
    # Test source terms
    T = 1500.0  # K
    Y = {
        'N2': 0.7,
        'O2': 0.2,
        'O': 0.01,
        'CO2': 0.05,
        'CxHy': 0.04
    }
    rho = 1.0  # kg/m^3
    
    source_terms = model.compute_source_terms(T, Y, rho)
    
    # Check rates
    assert 'NO' in source_terms
    assert 'CO' in source_terms
    assert 'UHC' in source_terms
    
    # Test temperature dependence
    T_low = 1000.0
    source_terms_low = model.compute_source_terms(T_low, Y, rho)
    assert source_terms_low['NO'] < source_terms['NO']
    assert source_terms_low['CO'] < source_terms['CO']
    
    # Test species dependence
    Y_no_N2 = {k: v for k, v in Y.items() if k != 'N2'}
    source_terms_no_N2 = model.compute_source_terms(T, Y_no_N2, rho)
    assert source_terms_no_N2['NO'] < source_terms['NO']


def test_detailed_chemistry():
    """Test detailed chemistry mechanism."""
    # Create species
    species = [
        ChemicalSpecies('H2', 2.016, 0.0, 14.3),
        ChemicalSpecies('O2', 32.0, 0.0, 0.92),
        ChemicalSpecies('H2O', 18.016, -241.8, 1.87)
    ]
    
    # Create reaction
    reactions = [
        Reaction(
            reactants={'H2': 2, 'O2': 1},
            products={'H2O': 2},
            A=1e10,
            b=0.0,
            E=10000.0
        )
    ]
    
    # Create soot model
    soot_model = MossSootModel()
    
    # Create pollutant model
    pollutant_model = ZeldovichNOxModel()
    
    # Create detailed chemistry mechanism
    mechanism = DetailedChemistry(
        species=species,
        reactions=reactions,
        soot_model=soot_model,
        pollutant_model=pollutant_model
    )
    
    # Test source terms
    T = 1500.0  # K
    Y = {
        'H2': 0.1,
        'O2': 0.2,
        'H2O': 0.0,
        'C2H2': 0.1,
        'N2': 0.7,
        'O': 0.01,
        'CO2': 0.05,
        'CxHy': 0.04
    }
    rho = 1.0  # kg/m^3
    
    source_terms = mechanism.compute_source_terms(T, Y, rho)
    
    # Check rates
    assert 'H2' in source_terms
    assert 'O2' in source_terms
    assert 'H2O' in source_terms
    assert 'soot' in source_terms
    assert 'NO' in source_terms
    assert 'CO' in source_terms
    assert 'UHC' in source_terms
    
    # Test heat release
    heat_release = mechanism.compute_heat_release(T, Y, rho)
    assert heat_release > 0  # Heat release should be positive
    
    # Test without soot and pollutant models
    mechanism_no_models = DetailedChemistry(species=species, reactions=reactions)
    source_terms_no_models = mechanism_no_models.compute_source_terms(T, Y, rho)
    assert 'soot' not in source_terms_no_models
    assert 'NO' not in source_terms_no_models
    assert 'CO' not in source_terms_no_models
    assert 'UHC' not in source_terms_no_models


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test zero temperature
    model = MossSootModel()
    with pytest.raises(ZeroDivisionError):
        model.compute_source_terms(0.0, {'C2H2': 0.1}, 1.0)
    
    # Test negative temperature
    with pytest.raises(ValueError):
        model.compute_source_terms(-100.0, {'C2H2': 0.1}, 1.0)
    
    # Test zero density
    source_terms = model.compute_source_terms(1500.0, {'C2H2': 0.1}, 0.0)
    assert source_terms[0] == 0.0
    assert source_terms[1] == 0.0
    
    # Test missing species
    source_terms = model.compute_source_terms(1500.0, {}, 1.0)
    assert source_terms[0] == 0.0
    assert source_terms[1] == 0.0


def test_memory_management():
    """Test memory management during computations."""
    # Create large mechanism
    n_species = 100
    species = [
        ChemicalSpecies(f'Species{i}', 1.0, 0.0, 1.0)
        for i in range(n_species)
    ]
    
    n_reactions = 1000
    reactions = [
        Reaction(
            reactants={f'Species{i}': 1},
            products={f'Species{i+1}': 1},
            A=1e10,
            b=0.0,
            E=10000.0
        )
        for i in range(n_reactions)
    ]
    
    # Create mechanism
    mechanism = DetailedChemistry(
        species=species,
        reactions=reactions,
        soot_model=MossSootModel(),
        pollutant_model=ZeldovichNOxModel()
    )
    
    # Test source terms computation
    T = 1500.0
    Y = {f'Species{i}': 1.0/n_species for i in range(n_species)}
    Y.update({
        'C2H2': 0.1,
        'O2': 0.2,
        'N2': 0.7,
        'O': 0.01,
        'CO2': 0.05,
        'CxHy': 0.04
    })
    rho = 1.0
    
    source_terms = mechanism.compute_source_terms(T, Y, rho)
    assert len(source_terms) == n_species + 4  # +4 for soot and pollutants 