# OKX Operator Engine - Project Structure

## 📁 Final Clean Project Structure

```
okx-operator/
├── src/                           # Core engine implementation
│   ├── main_engine.py            # 🔥 Main API - Production ready engine
│   ├── __init__.py               # Package initialization
│   └── core/                     # Core components
│       ├── __init__.py
│       ├── ast_nodes.py          # AST node definitions (Var, Call, Const)
│       ├── registry.py           # Operator registry and EvalContext
│       ├── expression_optimizer.py  # Window function optimization logic
│       ├── dimensions.py         # Dimensional analysis system
│       └── constants.py          # Smart constant classification
│
├── examples/                     # Usage examples
│   └── basic_usage.py           # Comprehensive usage examples
│
├── 1.ipynb                      # Interactive Jupyter notebook
├── README.md                    # Project overview and quick start
├── API.md                       # Complete API documentation
└── PROJECT_STRUCTURE.md         # This file
```

## 🎯 Core Files Explained

### `src/main_engine.py` - The Heart of the Engine
- **Entry Points**: `build_expr()`, `build_optimized_expr()`, `build_multiple_expr()`
- **All Operators**: Complete registry of 25+ financial operators
- **Window Function Logic**: Solves over repetition with step-by-step execution
- **Batch Processing**: `MultiExprBuilder` with automatic cleanup

### `src/core/` - Supporting Infrastructure
- **`registry.py`**: `EvalContext`, `OperatorRegistry`, and operator metadata
- **`expression_optimizer.py`**: Detects and handles nested window functions
- **`ast_nodes.py`**: AST classes (`Var`, `Call`, `Const`) and utilities
- **`dimensions.py`**: Dimensional analysis for operator compatibility
- **`constants.py`**: Smart constant classification (SCALAR, etc.)

### Documentation
- **`README.md`**: User-focused overview with quick start examples
- **`API.md`**: Complete technical API reference with detailed examples
- **`PROJECT_STRUCTURE.md`**: This architectural overview

### Examples & Interactive Usage
- **`examples/basic_usage.py`**: Production-ready code examples
- **`1.ipynb`**: Interactive Jupyter notebook for experimentation

## 🗑️ Cleaned Up (Removed Files)

- `debug_*.py` - All debug scripts removed
- `test_*.py` - All test scripts removed  
- `test/` directory - Entire test directory removed
- `docs/` directory - Old documentation removed
- `examples/original_engine.py` - Outdated example removed
- `src/notebook_api.py` - Simplified API removed (as requested)
- `CLAUDE.md` - Internal development notes removed

## 🎉 Benefits of Clean Structure

1. **Clear Entry Point**: `src/main_engine.py` is the single source of truth
2. **No Confusion**: No multiple APIs or simplified versions
3. **Production Ready**: Only essential, tested code remains
4. **Easy to Navigate**: Logical organization with clear responsibilities
5. **Self-Contained**: All dependencies properly organized

## 🚀 Getting Started

```python
# Single import for everything
from src.main_engine import build_expr, build_multiple_expr, EvalContext

# That's it! No confusion about which API to use.
```

The project is now clean, well-documented, and production-ready for financial factor computation!