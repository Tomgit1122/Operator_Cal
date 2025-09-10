#!/usr/bin/env python3
"""
OKX Operator Engine - Basic Usage Examples
==========================================

This example demonstrates how to use the OKX Operator Engine for financial factor computation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import polars as pl
from main_engine import build_expr, build_optimized_expr, build_multiple_expr, EvalContext

def create_sample_data():
    """Create sample financial data"""
    return pl.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT'] * 100,
        'time': list(range(300)),
        'close': [100 + i*0.1 + (i%3)*5 for i in range(300)],
        'volume': [1000000 + i*1000 + (i%5)*50000 for i in range(300)],
        'returns': [0.001 * (i % 40 - 20) for i in range(300)],
        'high': [105 + i*0.1 + (i%3)*5 for i in range(300)],
        'low': [95 + i*0.1 + (i%3)*5 for i in range(300)]
    }).lazy()

def example_1_simple_expressions():
    """Example 1: Simple expressions using build_expr()"""
    print("=== Example 1: Simple Expressions ===")
    
    data = create_sample_data()
    ctx = EvalContext(
        variables={
            "close": pl.col("close"),
            "volume": pl.col("volume"),
            "returns": pl.col("returns")
        },
        ticker_col="ticker",
        time_col="time"
    )
    
    # Simple time series operations
    ma20 = build_expr("TSMean(close, 20)", ctx)
    vol_std = build_expr("TSStd(returns, 10)", ctx)
    
    # Simple cross-sectional operations  
    price_rank = build_expr("CSRank(close)", ctx)
    
    # Apply all at once
    result = data.with_columns(
        ma20=ma20,
        volatility=vol_std,
        price_rank=price_rank
    ).select(['ticker', 'time', 'close', 'ma20', 'volatility', 'price_rank']).limit(10).collect()
    
    print(f"Result shape: {result.shape}")
    print(result)

def example_2_complex_expressions():
    """Example 2: Complex nested expressions using build_optimized_expr()"""
    print("\n=== Example 2: Complex Nested Expressions ===")
    
    data = create_sample_data()
    ctx = EvalContext(
        variables={
            "close": pl.col("close"),
            "volume": pl.col("volume"),
            "returns": pl.col("returns")
        },
        ticker_col="ticker",
        time_col="time"
    )
    
    # Complex nested expression that would fail with basic build_expr
    complex_expr = "TSMean(TSStd(close, 10), 5)"
    result = build_optimized_expr(complex_expr, ctx, data, "complex_factor")
    
    final = result.select(['ticker', 'time', 'complex_factor']).filter(
        pl.col('time') > 50  # Skip initial null values due to rolling windows
    ).limit(10).collect()
    
    print(f"Complex expression: {complex_expr}")
    print(f"Result shape: {final.shape}")
    print(final)

def example_3_batch_processing():
    """Example 3: Batch processing multiple factors"""
    print("\n=== Example 3: Batch Processing Multiple Factors ===")
    
    data = create_sample_data()
    ctx = EvalContext(
        variables={
            "close": pl.col("close"),
            "volume": pl.col("volume"),
            "returns": pl.col("returns"),
            "high": pl.col("high"),
            "low": pl.col("low")
        },
        ticker_col="ticker",
        time_col="time"
    )
    
    # Define multiple factors including complex ones
    factors = {
        # Simple factors
        'momentum': 'Div(close, TSLag(close, 20))',
        'volatility': 'TSStd(returns, 20)',
        'price_rank': 'CSRank(close)',
        
        # Complex nested factors
        'complex_momentum': 'TSMean(TSStd(close, 10), 5)',
        'mixed_factor': 'CSRank(TSMean(volume, 10))',
        'advanced_factor': 'TSMean(CSScale(returns), 5)',
        
        # Very complex factor
        'super_complex': 'Add(CSRank(TSMean(close, 10)), Mult(TSStd(returns, 5), 0.5))'
    }
    
    # Build all factors at once
    builder = build_multiple_expr(factors, ctx)
    result = builder.apply(data)
    
    final = result.select(['ticker', 'time'] + list(factors.keys())).filter(
        pl.col('time') > 50  # Skip initial rows with nulls
    ).limit(10).collect()
    
    print(f"Built {len(factors)} factors:")
    for name, expr in factors.items():
        print(f"  {name}: {expr}")
    
    print(f"\nResult shape: {final.shape}")
    print(f"Columns: {final.columns}")
    print("\nSample data:")
    print(final)

if __name__ == "__main__":
    print("OKX Operator Engine - Usage Examples")
    print("=" * 50)
    
    try:
        example_1_simple_expressions()
        example_2_complex_expressions()
        example_3_batch_processing()
        
        print("\n" + "=" * 50)
        print("SUCCESS: All examples executed successfully!")
        print("The engine handles simple expressions, complex nested expressions,")
        print("and large-scale batch processing seamlessly.")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()