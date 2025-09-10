"""
Dimensional Analysis System for Financial Operators
==================================================

Fixed dimensional compatibility rules:
1. Constants (DIMENSIONLESS) are compatible with ANY dimension
2. SCALAR values can be used in arithmetic with any dimension  
3. Same dimensions are always compatible
4. Some special compatibility rules for financial data
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Optional, Union


class Dimension(Enum):
    """Dimension types for financial operators"""
    # Basic financial dimensions
    PRICE = auto()          # Price data (open, high, low, close)
    VOLUME = auto()         # Volume data  
    RETURN = auto()         # Returns, changes
    RATIO = auto()          # Ratios, percentages
    
    # Processed dimensions
    RANK = auto()           # Ranks, percentiles
    INDICATOR = auto()      # Binary indicators (0/1)
    TIME_SERIES = auto()    # Time series aggregates
    CROSS_SECTION = auto()  # Cross-sectional aggregates
    
    # Universal dimensions
    SCALAR = auto()         # Constants, scalars (compatible with everything)
    DIMENSIONLESS = auto()  # Normalized, unitless values
    ANY = auto()           # Wildcard - accepts any dimension


@dataclass(frozen=True)
class DimensionSpec:
    """Enhanced dimension specification with flexible compatibility"""
    input_dims: Tuple[Dimension, ...]
    output_dim: Dimension
    
    # Compatibility matrix - which dimensions can work together
    _COMPATIBILITY_RULES = {
        # Scalars/constants work with everything
        (Dimension.SCALAR, Dimension.ANY): True,
        (Dimension.ANY, Dimension.SCALAR): True,
        (Dimension.DIMENSIONLESS, Dimension.ANY): True,
        (Dimension.ANY, Dimension.DIMENSIONLESS): True,
        
        # Same dimensions are always compatible
        'same_dimension': True,
        
        # Arithmetic compatibility
        (Dimension.PRICE, Dimension.PRICE): True,
        (Dimension.PRICE, Dimension.RETURN): True,  # price + return
        (Dimension.VOLUME, Dimension.VOLUME): True,
        (Dimension.RETURN, Dimension.RETURN): True,
        (Dimension.RATIO, Dimension.RATIO): True,
        
        # Cross-type compatibility for processed dimensions
        (Dimension.DIMENSIONLESS, Dimension.RANK): True,
        (Dimension.DIMENSIONLESS, Dimension.INDICATOR): True,
        (Dimension.RANK, Dimension.DIMENSIONLESS): True,
        (Dimension.INDICATOR, Dimension.DIMENSIONLESS): True,
    }
    
    @staticmethod
    def are_compatible(dim1: Dimension, dim2: Dimension) -> bool:
        """Check if two dimensions are compatible"""
        # Same dimensions are always compatible
        if dim1 == dim2:
            return True
            
        # SCALAR and DIMENSIONLESS are universal
        if dim1 in (Dimension.SCALAR, Dimension.DIMENSIONLESS) or \
           dim2 in (Dimension.SCALAR, Dimension.DIMENSIONLESS):
            return True
            
        # ANY dimension wildcard
        if dim1 == Dimension.ANY or dim2 == Dimension.ANY:
            return True
            
        # Check specific compatibility rules
        if (dim1, dim2) in DimensionSpec._COMPATIBILITY_RULES:
            return DimensionSpec._COMPATIBILITY_RULES[(dim1, dim2)]
            
        if (dim2, dim1) in DimensionSpec._COMPATIBILITY_RULES:
            return DimensionSpec._COMPATIBILITY_RULES[(dim2, dim1)]
            
        # Default: incompatible
        return False
    
    def is_compatible_with(self, input_dims: Tuple[Dimension, ...]) -> bool:
        """Check if input dimensions match this specification"""
        if len(self.input_dims) != len(input_dims):
            return False
            
        for expected, actual in zip(self.input_dims, input_dims):
            if not self.are_compatible(expected, actual):
                return False
                
        return True
    
    def infer_output_dimension(self, input_dims: Tuple[Dimension, ...]) -> Dimension:
        """Infer output dimension based on inputs and operation"""
        if not self.is_compatible_with(input_dims):
            raise ValueError(f"Incompatible input dimensions: {input_dims}")
            
        # If output is explicitly specified, use it
        if self.output_dim != Dimension.ANY:
            return self.output_dim
            
        # Smart inference rules
        actual_dims = [d for d in input_dims if d not in (Dimension.SCALAR, Dimension.DIMENSIONLESS)]
        
        if len(actual_dims) == 0:
            return Dimension.DIMENSIONLESS
        elif len(actual_dims) == 1:
            return actual_dims[0]
        else:
            # Multiple non-scalar dimensions - depends on operation
            return Dimension.DIMENSIONLESS  # Conservative default


def infer_variable_dimension(var_name: str) -> Dimension:
    """Infer dimension from variable name"""
    name_lower = var_name.lower()
    
    # Price-related
    if any(keyword in name_lower for keyword in ['price', 'open', 'high', 'low', 'close']):
        return Dimension.PRICE
        
    # Volume-related  
    if any(keyword in name_lower for keyword in ['volume', 'vol', 'amount', 'turnover']):
        return Dimension.VOLUME
        
    # Return-related
    if any(keyword in name_lower for keyword in ['return', 'ret', 'change', 'pct']):
        return Dimension.RETURN
        
    # Ratio-related
    if any(keyword in name_lower for keyword in ['ratio', 'rate']):
        return Dimension.RATIO
        
    # Rank-related
    if any(keyword in name_lower for keyword in ['rank', 'percentile']):
        return Dimension.RANK
        
    # Default to dimensionless for unknown variables
    return Dimension.DIMENSIONLESS


# Convenience functions for creating common dimension specs

def unary_spec(input_dim: Dimension = Dimension.ANY, output_dim: Dimension = Dimension.ANY) -> DimensionSpec:
    """Create specification for unary operator"""
    return DimensionSpec((input_dim,), output_dim)

def binary_spec(input_dims: Tuple[Dimension, Dimension] = (Dimension.ANY, Dimension.ANY), 
                output_dim: Dimension = Dimension.ANY) -> DimensionSpec:
    """Create specification for binary operator"""
    return DimensionSpec(input_dims, output_dim)

def ternary_spec(input_dims: Tuple[Dimension, Dimension, Dimension] = (Dimension.ANY, Dimension.ANY, Dimension.ANY),
                 output_dim: Dimension = Dimension.ANY) -> DimensionSpec:
    """Create specification for ternary operator"""
    return DimensionSpec(input_dims, output_dim)

# Commonly used specs
FLEXIBLE_UNARY = unary_spec()  # Accepts any input, outputs any
FLEXIBLE_BINARY = binary_spec()  # Accepts any inputs, outputs any
ARITHMETIC_BINARY = binary_spec(output_dim=Dimension.DIMENSIONLESS)  # Arithmetic preserves dimensionality intelligently