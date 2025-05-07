"""Configuration management for PHASTA-Py.

This module provides functionality for loading and managing simulation
configuration settings.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Config:
    """Configuration manager for PHASTA-Py simulations."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration with optional initial settings.
        
        Args:
            config_dict: Optional dictionary of initial configuration settings
        """
        self._config = config_dict or {}
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Config':
        """Load configuration from a file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Config object with loaded settings
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path) as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
        
        return cls(config_dict)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a file.
        
        Args:
            file_path: Path to save configuration file
        """
        file_path = Path(file_path)
        with open(file_path, 'w') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self._config, f)
            elif file_path.suffix.lower() == '.json':
                json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with a dictionary of settings.
        
        Args:
            config_dict: Dictionary of configuration settings
        """
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary of configuration settings
        """
        return self._config.copy()


def create_default_config() -> Config:
    """Create a default configuration.
    
    Returns:
        Config object with default settings
    """
    return Config({
        'simulation': {
            'type': 'incompressible',
            'time_integration': {
                'method': 'implicit',
                'dt': 0.001,
                'max_steps': 1000
            },
            'output': {
                'frequency': 100,
                'format': 'vtk'
            }
        },
        'mesh': {
            'type': 'unstructured',
            'dimension': 3
        },
        'solver': {
            'linear_solver': {
                'type': 'gmres',
                'max_iterations': 1000,
                'tolerance': 1e-6
            },
            'preconditioner': {
                'type': 'ilu',
                'fill_level': 1
            }
        },
        'acceleration': {
            'backend': 'auto',
            'device': 'auto'
        }
    })
