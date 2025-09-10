"""
Enhanced AST Nodes with Improved Dimension Tracking
==================================================
"""

from dataclasses import dataclass
from typing import List, Optional
from .dimensions import Dimension, infer_variable_dimension


class AST:
    """Base AST node with dimension tracking"""
    
    def __init__(self):
        self._dimension: Optional[Dimension] = None
    
    @property 
    def dimension(self) -> Optional[Dimension]:
        return self._dimension
    
    @dimension.setter
    def dimension(self, dim: Dimension):
        self._dimension = dim


@dataclass
class Var(AST):
    """Variable node with automatic dimension inference"""
    name: str
    
    def __post_init__(self):
        super().__init__()
        self._dimension = infer_variable_dimension(self.name)
    
    def __str__(self):
        return self.name


@dataclass  
class Const(AST):
    """Constant node - always SCALAR dimension"""
    value: float
    
    def __post_init__(self):
        super().__init__()
        # Constants are always SCALAR (universal compatibility)
        self._dimension = Dimension.SCALAR
        
    def __str__(self):
        return str(self.value)


@dataclass
class Call(AST):
    """Function call node - dimension set during evaluation"""
    name: str
    args: List[AST]
    
    def __post_init__(self):
        super().__init__()
        # Dimension will be set by evaluator based on operator output
    
    def __str__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"


def pretty_print_ast(node: AST, indent: int = 0) -> str:
    """Pretty print AST with dimension information"""
    spaces = "  " * indent
    dim_str = f" [{node.dimension.name if node.dimension else 'None'}]"
    
    if isinstance(node, Var):
        return f"{spaces}{node.name}{dim_str}"
    elif isinstance(node, Const):
        return f"{spaces}{node.value}{dim_str}"
    elif isinstance(node, Call):
        result = f"{spaces}{node.name}{dim_str}(\n"
        for arg in node.args:
            result += pretty_print_ast(arg, indent + 1) + "\n"
        result += f"{spaces})"
        return result
    else:
        return f"{spaces}{str(node)}{dim_str}"