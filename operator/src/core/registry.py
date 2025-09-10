"""
Enhanced Operator Registry with Improved Metadata System
========================================================
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Union, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod

import polars as pl
from .dimensions import Dimension, DimensionSpec
from .ast_nodes import AST


class OperatorCategory(Enum):
    """Operator categories for better organization"""
    ARITHMETIC = auto()      # Basic arithmetic: +, -, *, /
    COMPARISON = auto()      # Comparisons: >, <, ==, etc.
    LOGICAL = auto()         # Logical ops: and, or, not
    UNARY_MATH = auto()      # Math functions: abs, log, sqrt
    CONDITIONAL = auto()     # Conditional: if-then-else, clip
    TIME_SERIES = auto()     # Time series: rolling, lag
    CROSS_SECTION = auto()   # Cross-sectional: rank, demean
    AGGREGATION = auto()     # Aggregations: sum, mean, std


@dataclass
class OperatorMeta:
    """Enhanced operator metadata"""
    name: str
    category: OperatorCategory
    arity: Tuple[int, int]  # (min_args, max_args), -1 for unlimited
    dimension_spec: DimensionSpec
    description: str
    examples: List[str] = field(default_factory=list)
    
    def validate_arity(self, num_args: int) -> bool:
        """Validate argument count"""
        min_args, max_args = self.arity
        if num_args < min_args:
            return False
        if max_args != -1 and num_args > max_args:
            return False
        return True
    
    def validate_dimensions(self, arg_dims: Tuple[Dimension, ...]) -> bool:
        """Validate dimensional compatibility"""
        return self.dimension_spec.is_compatible_with(arg_dims)
    
    def get_output_dimension(self, input_dims: Tuple[Dimension, ...]) -> Dimension:
        """Get output dimension for given inputs"""
        return self.dimension_spec.infer_output_dimension(input_dims)


# Enhanced evaluation context
@dataclass
class EvalContext:
    """Evaluation context with two-dimensional system"""
    variables: Dict[str, pl.Expr]
    ticker_col: str = "ticker"
    time_col: str = "time"
    sort_by: List[str] = None
    
    def __post_init__(self):
        if self.sort_by is None:
            self.sort_by = [self.time_col]


# Operator builder signature
OpBuilder = Callable[[EvalContext, List[Union[pl.Expr, int, float]]], pl.Expr]


@dataclass
class Operator:
    """Operator with metadata and implementation"""
    meta: OperatorMeta
    builder: OpBuilder
    
    def validate_call(self, args: List[AST]) -> Tuple[bool, str]:
        """Validate a function call comprehensively"""
        # Check arity
        if not self.meta.validate_arity(len(args)):
            min_args, max_args = self.meta.arity
            if max_args == -1:
                return False, f"{self.meta.name} expects at least {min_args} arguments, got {len(args)}"
            else:
                return False, f"{self.meta.name} expects {min_args}-{max_args} arguments, got {len(args)}"
        
        # Check dimensions (only for nodes that have dimensions)
        arg_dims = []
        for arg in args:
            if arg.dimension is not None:
                arg_dims.append(arg.dimension)
            else:
                # For nodes without dimensions, assume ANY for validation
                arg_dims.append(Dimension.ANY)
                
        arg_dims = tuple(arg_dims)
        
        if not self.meta.validate_dimensions(arg_dims):
            return False, f"{self.meta.name} dimensional mismatch: got {[d.name for d in arg_dims]}"
            
        return True, "OK"


class EnhancedOperatorRegistry:
    """Enhanced registry with better categorization and validation"""
    
    def __init__(self):
        self._ops: Dict[str, Operator] = {}
        self._categories: Dict[OperatorCategory, List[str]] = {
            cat: [] for cat in OperatorCategory
        }
    
    def register(self, meta: OperatorMeta):
        """Decorator for registering operators"""
        def decorator(fn: OpBuilder):
            op = Operator(meta=meta, builder=fn)
            self._ops[meta.name] = op
            self._categories[meta.category].append(meta.name)
            return fn
        return decorator
    
    def get(self, name: str) -> Operator:
        """Get operator by name"""
        if name not in self._ops:
            available = list(self._ops.keys())
            raise KeyError(f"Unknown operator: {name}. Available: {available}")
        return self._ops[name]
    
    def exists(self, name: str) -> bool:
        """Check if operator exists"""
        return name in self._ops
    
    def list_by_category(self, category: OperatorCategory) -> List[str]:
        """List operators by category"""
        return self._categories[category].copy()
    
    def get_all_operators(self) -> Dict[str, OperatorMeta]:
        """Get all operator metadata"""
        return {name: op.meta for name, op in self._ops.items()}
    
    def get_random_operators(self, n: int = 5) -> List[str]:
        """Get random operator names for testing"""
        import random
        all_ops = list(self._ops.keys())
        return random.sample(all_ops, min(n, len(all_ops)))
    
    def validate_call(self, op_name: str, args: List[AST]) -> Tuple[bool, str]:
        """Validate operator call"""
        if not self.exists(op_name):
            return False, f"Unknown operator: {op_name}"
            
        op = self.get(op_name)
        return op.validate_call(args)