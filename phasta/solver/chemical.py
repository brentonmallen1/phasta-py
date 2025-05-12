"""Chemical reactions module.

This module provides functionality for:
- Basic reaction mechanisms
- Species transport
- Heat release
- Chemical kinetics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ChemicalSpecies:
    """Class representing a chemical species."""
    
    def __init__(self, 
                 name: str,
                 molecular_weight: float,
                 formation_enthalpy: float,
                 specific_heat: float):
        """Initialize chemical species.
        
        Args:
            name: Species name
            molecular_weight: Molecular weight in kg/mol
            formation_enthalpy: Formation enthalpy in J/mol
            specific_heat: Specific heat at constant pressure in J/(kg·K)
        """
        self.name = name
        self.molecular_weight = molecular_weight
        self.formation_enthalpy = formation_enthalpy
        self.specific_heat = specific_heat


class Reaction:
    """Class representing a chemical reaction."""
    
    def __init__(self,
                 reactants: Dict[str, float],
                 products: Dict[str, float],
                 forward_rate: float,
                 backward_rate: float,
                 activation_energy: float):
        """Initialize chemical reaction.
        
        Args:
            reactants: Dictionary of reactant species and their stoichiometric coefficients
            products: Dictionary of product species and their stoichiometric coefficients
            forward_rate: Forward reaction rate constant
            backward_rate: Backward reaction rate constant
            activation_energy: Activation energy in J/mol
        """
        self.reactants = reactants
        self.products = products
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate
        self.activation_energy = activation_energy
    
    def compute_rate(self, 
                    concentrations: Dict[str, float],
                    temperature: float) -> float:
        """Compute reaction rate.
        
        Args:
            concentrations: Dictionary of species concentrations
            temperature: Temperature in K
            
        Returns:
            Reaction rate
        """
        # Forward rate
        forward = self.forward_rate * np.exp(-self.activation_energy / (8.314 * temperature))
        for species, coeff in self.reactants.items():
            forward *= concentrations[species]**coeff
        
        # Backward rate
        backward = self.backward_rate
        for species, coeff in self.products.items():
            backward *= concentrations[species]**coeff
        
        return forward - backward


class ChemicalMechanism:
    """Class representing a chemical reaction mechanism."""
    
    def __init__(self):
        """Initialize chemical mechanism."""
        self.species: Dict[str, ChemicalSpecies] = {}
        self.reactions: List[Reaction] = []
    
    def add_species(self, species: ChemicalSpecies) -> None:
        """Add species to mechanism.
        
        Args:
            species: Chemical species to add
        """
        self.species[species.name] = species
    
    def add_reaction(self, reaction: Reaction) -> None:
        """Add reaction to mechanism.
        
        Args:
            reaction: Chemical reaction to add
        """
        self.reactions.append(reaction)
    
    def compute_source_terms(self,
                           concentrations: Dict[str, float],
                           temperature: float) -> Dict[str, float]:
        """Compute source terms for all species.
        
        Args:
            concentrations: Dictionary of species concentrations
            temperature: Temperature in K
            
        Returns:
            Dictionary of source terms for each species
        """
        source_terms = {name: 0.0 for name in self.species}
        
        for reaction in self.reactions:
            rate = reaction.compute_rate(concentrations, temperature)
            
            # Add contribution to reactants
            for species, coeff in reaction.reactants.items():
                source_terms[species] -= coeff * rate
            
            # Add contribution to products
            for species, coeff in reaction.products.items():
                source_terms[species] += coeff * rate
        
        return source_terms
    
    def compute_heat_release(self,
                           concentrations: Dict[str, float],
                           temperature: float) -> float:
        """Compute heat release rate.
        
        Args:
            concentrations: Dictionary of species concentrations
            temperature: Temperature in K
            
        Returns:
            Heat release rate in W/m³
        """
        heat_release = 0.0
        
        for reaction in self.reactions:
            rate = reaction.compute_rate(concentrations, temperature)
            
            # Compute enthalpy change
            enthalpy_change = 0.0
            for species, coeff in reaction.reactants.items():
                enthalpy_change -= coeff * self.species[species].formation_enthalpy
            for species, coeff in reaction.products.items():
                enthalpy_change += coeff * self.species[species].formation_enthalpy
            
            heat_release += rate * enthalpy_change
        
        return heat_release


class SpeciesTransport:
    """Class for handling species transport."""
    
    def __init__(self, mechanism: ChemicalMechanism):
        """Initialize species transport.
        
        Args:
            mechanism: Chemical reaction mechanism
        """
        self.mechanism = mechanism
    
    def compute_diffusion(self,
                         concentrations: Dict[str, float],
                         temperature: float,
                         pressure: float) -> Dict[str, float]:
        """Compute diffusion coefficients.
        
        Args:
            concentrations: Dictionary of species concentrations
            temperature: Temperature in K
            pressure: Pressure in Pa
            
        Returns:
            Dictionary of diffusion coefficients
        """
        # Simplified diffusion model
        # In practice, this would use more sophisticated models
        diffusion = {}
        for name, species in self.mechanism.species.items():
            # Simple temperature-dependent diffusion
            diffusion[name] = 1e-5 * (temperature / 300.0)**1.5 * (101325.0 / pressure)
        
        return diffusion
    
    def compute_convection(self,
                          concentrations: Dict[str, float],
                          velocity: np.ndarray) -> Dict[str, float]:
        """Compute convective fluxes.
        
        Args:
            concentrations: Dictionary of species concentrations
            velocity: Velocity vector
            
        Returns:
            Dictionary of convective fluxes
        """
        convection = {}
        for name, conc in concentrations.items():
            convection[name] = conc * velocity
        
        return convection 