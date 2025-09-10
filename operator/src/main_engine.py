"""
Main Expression Engine - Integrates all components
=================================================

Fixed dimensional analysis system that properly handles:
1. Constants (SCALAR) compatible with any dimension
2. Smart dimension inference
3. Flexible compatibility rules
"""

import sys
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import polars as pl

# Import all core components - fix relative import
try:
    from .core.dimensions import Dimension, DimensionSpec, infer_variable_dimension
    from .core.ast_nodes import AST, Var, Const, Call, pretty_print_ast
    from .core.registry import (
        EnhancedOperatorRegistry, OperatorMeta, OperatorCategory, 
        EvalContext, OpBuilder, Operator
    )
except ImportError:
    # Fallback for direct execution
    from core.dimensions import Dimension, DimensionSpec, infer_variable_dimension
    from core.ast_nodes import AST, Var, Const, Call, pretty_print_ast
    from core.registry import (
        EnhancedOperatorRegistry, OperatorMeta, OperatorCategory, 
        EvalContext, OpBuilder, Operator
    )

# Import parser from original engine
from dataclasses import dataclass


@dataclass
class Tok:
    kind: str  # 'id' | 'num' | 'comma' | 'lpar' | 'rpar' | 'eof'
    val: str


class Lexer:
    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def _peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def _adv(self):
        self.i += 1

    def next(self) -> Tok:
        # skip spaces
        while self._peek() and self._peek().isspace():
            self._adv()
        c = self._peek()
        if not c:
            return Tok('eof', '')
        if c.isalpha() or c == '_':
            j = self.i
            while self._peek() and (self._peek().isalnum() or self._peek() in ['_', '/']):
                self._adv()
            return Tok('id', self.s[j:self.i])
        if c.isdigit() or (c == '.' and self.i+1 < self.n and self.s[self.i+1].isdigit()) or (c in '+-' and self.i+1 < self.n and (self.s[self.i+1].isdigit() or self.s[self.i+1] == '.')):
            j = self.i
            if c in '+-':
                self._adv()
            dot = False
            while self._peek() and (self._peek().isdigit() or (self._peek()=='.' and not dot)):
                if self._peek()=='.':
                    dot = True
                self._adv()
            return Tok('num', self.s[j:self.i])
        if c == ',':
            self._adv(); return Tok('comma', ',')
        if c == '(':
            self._adv(); return Tok('lpar', '(')
        if c == ')':
            self._adv(); return Tok('rpar', ')')
        if c in '+-':
            j = self.i; self._adv(); return Tok('id', self.s[j:self.i])
        raise SyntaxError(f"Unexpected char: {c!r} at {self.i}")


class EnhancedParser:
    def __init__(self, s: str):
        self.lex = Lexer(s)
        self.tok = self.lex.next()

    def _eat(self, kind: str) -> Tok:
        if self.tok.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {self.tok.kind} ({self.tok.val!r})")
        t = self.tok
        self.tok = self.lex.next()
        return t

    def parse(self) -> AST:
        node = self._expr()
        if self.tok.kind != 'eof':
            raise SyntaxError("Unexpected trailing tokens")
        return node

    def _expr(self) -> AST:
        return self._primary()

    def _primary(self) -> AST:
        t = self.tok
        if t.kind == 'id':
            name = t.val; self._eat('id')
            if self.tok.kind == 'lpar':
                self._eat('lpar')
                args = []
                if self.tok.kind != 'rpar':
                    args.append(self._expr())
                    while self.tok.kind == 'comma':
                        self._eat('comma')
                        args.append(self._expr())
                self._eat('rpar')
                return Call(name=name, args=args)
            return Var(name=name)
        if t.kind == 'num':
            v = float(self._eat('num').val)
            return Const(v)
        if t.kind == 'lpar':
            self._eat('lpar')
            node = self._expr()
            self._eat('rpar')
            return node
        raise SyntaxError(f"Unexpected token {t.kind}:{t.val!r}")


# Create global registry and register all operators
GLOBAL_REGISTRY = EnhancedOperatorRegistry()

# Register all operators with FIXED dimensional specs

# ===== ARITHMETIC OPERATORS =====
@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Add",
    category=OperatorCategory.ARITHMETIC,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY), Dimension.ANY),
    description="Addition: x + y (handles constants intelligently)",
    examples=["Add(close, 1)", "Add(price, adjustment)", "Add(return1, return2)"]
))
def _add(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return args[0] + args[1]

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Sub",
    category=OperatorCategory.ARITHMETIC,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY), Dimension.ANY),
    description="Subtraction: x - y",
    examples=["Sub(high, low)", "Sub(close, open)", "Sub(price, 100)"]
))
def _sub(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return args[0] - args[1]

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Mult", 
    category=OperatorCategory.ARITHMETIC,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY), Dimension.ANY),
    description="Multiplication: x * y",
    examples=["Mult(price, volume)", "Mult(return, 100)", "Mult(x, 2)"]
))
def _mult(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return args[0] * args[1]

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Div",
    category=OperatorCategory.ARITHMETIC,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY), Dimension.RATIO),
    description="Division: x / y",
    examples=["Div(close, open)", "Div(volume, avg_volume)", "Div(x, 2)"]
))
def _div(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return args[0] / args[1]

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Pow",
    category=OperatorCategory.ARITHMETIC,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY), Dimension.DIMENSIONLESS),
    description="Power: x ^ y",
    examples=["Pow(price, 2)", "Pow(return, 0.5)"]
))
def _pow(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return args[0] ** args[1]

# ===== UNARY MATH OPERATORS =====
@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Abs",
    category=OperatorCategory.UNARY_MATH,
    arity=(1, 1),
    dimension_spec=DimensionSpec((Dimension.ANY,), Dimension.ANY),
    description="Absolute value: |x|",
    examples=["Abs(return)", "Abs(price_change)", "Abs(spread)"]
))
def _abs(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return args[0].abs()

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Neg",
    category=OperatorCategory.ARITHMETIC,
    arity=(1, 1),
    dimension_spec=DimensionSpec((Dimension.ANY,), Dimension.ANY),
    description="Negation: -x",
    examples=["Neg(return)", "Neg(price_change)"]
))
def _neg(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return -args[0]

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Log",
    category=OperatorCategory.UNARY_MATH,
    arity=(1, 1),
    dimension_spec=DimensionSpec((Dimension.ANY,), Dimension.DIMENSIONLESS),
    description="Natural logarithm",
    examples=["Log(price)", "Log(volume)"]
))
def _log(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return args[0].log()

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Sign",
    category=OperatorCategory.UNARY_MATH,
    arity=(1, 1),
    dimension_spec=DimensionSpec((Dimension.ANY,), Dimension.INDICATOR),
    description="Sign function (-1, 0, 1)",
    examples=["Sign(returns)", "Sign(price_change)"]
))
def _sign(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return pl.when(args[0] > 0).then(1).when(args[0] < 0).then(-1).otherwise(0)

# ===== COMPARISON OPERATORS =====
@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Greater",
    category=OperatorCategory.COMPARISON,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY), Dimension.INDICATOR),
    description="Greater than comparison",
    examples=["Greater(x, y)", "Greater(price, threshold)"]
))
def _greater(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return (args[0] > args[1]).cast(pl.Int8)

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Less",
    category=OperatorCategory.COMPARISON,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY), Dimension.INDICATOR),
    description="Less than comparison",
    examples=["Less(x, y)", "Less(price, threshold)"]
))
def _less(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return (args[0] < args[1]).cast(pl.Int8)

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Equal",
    category=OperatorCategory.COMPARISON,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY), Dimension.INDICATOR),
    description="Equality comparison",
    examples=["Equal(x, y)", "Equal(flag, 1)"]
))
def _equal(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return (args[0] == args[1]).cast(pl.Int8)

# ===== CONDITIONAL OPERATORS =====
@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Cond",
    category=OperatorCategory.CONDITIONAL,
    arity=(3, 3),
    dimension_spec=DimensionSpec((Dimension.INDICATOR, Dimension.ANY, Dimension.ANY), Dimension.ANY),
    description="Conditional: if condition then x else y",
    examples=["Cond(Greater(price, 100), 1, 0)"]
))
def _cond(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return pl.when(args[0] != 0).then(args[1]).otherwise(args[2])

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="Clip",
    category=OperatorCategory.CONDITIONAL,
    arity=(3, 3),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.ANY, Dimension.ANY), Dimension.ANY),
    description="Clip values between min and max",
    examples=["Clip(x, min_val, max_val)"]
))
def _clip(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    x, min_val, max_val = args
    return pl.when(x < min_val).then(min_val).when(x > max_val).then(max_val).otherwise(x)

# ===== TIME SERIES OPERATORS =====
@GLOBAL_REGISTRY.register(OperatorMeta(
    name="TSMean",
    category=OperatorCategory.TIME_SERIES,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.SCALAR), Dimension.TIME_SERIES),
    description="Time series rolling mean",
    examples=["TSMean(price, 20)", "TSMean(volume, 10)"]
))
def _ts_mean(ctx: EvalContext, args: List[Union[pl.Expr, int]]) -> pl.Expr:
    x, window = args
    return x.rolling_mean(window_size=int(window), min_samples=1).over(ctx.ticker_col)

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="TSStd",
    category=OperatorCategory.TIME_SERIES,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.SCALAR), Dimension.TIME_SERIES),
    description="Time series rolling standard deviation",
    examples=["TSStd(returns, 20)", "TSStd(price, 10)"]
))
def _ts_std(ctx: EvalContext, args: List[Union[pl.Expr, int]]) -> pl.Expr:
    x, window = args
    return x.rolling_std(window_size=int(window), min_samples=2).over(ctx.ticker_col)

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="TSLag",
    category=OperatorCategory.TIME_SERIES,
    arity=(2, 2),
    dimension_spec=DimensionSpec((Dimension.ANY, Dimension.SCALAR), Dimension.ANY),
    description="Time series lag",
    examples=["TSLag(price, 1)", "TSLag(volume, 5)"]
))
def _ts_lag(ctx: EvalContext, args: List[Union[pl.Expr, int]]) -> pl.Expr:
    x, periods = args
    return x.shift(int(periods)).over(ctx.ticker_col)

# ===== CROSS-SECTIONAL OPERATORS =====
@GLOBAL_REGISTRY.register(OperatorMeta(
    name="CSRank",
    category=OperatorCategory.CROSS_SECTION,
    arity=(1, 1),
    dimension_spec=DimensionSpec((Dimension.ANY,), Dimension.RANK),
    description="Cross-sectional rank per time",
    examples=["CSRank(price)", "CSRank(volume)"]
))
def _cs_rank(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    return args[0].rank(method='average').over(ctx.time_col)

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="CSDemean",
    category=OperatorCategory.CROSS_SECTION,
    arity=(1, 1),
    dimension_spec=DimensionSpec((Dimension.ANY,), Dimension.DIMENSIONLESS),
    description="Cross-sectional demean per time",
    examples=["CSDemean(returns)", "CSDemean(price)"]
))
def _cs_demean(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    x = args[0]
    return x - x.mean().over(ctx.time_col)

@GLOBAL_REGISTRY.register(OperatorMeta(
    name="CSScale",
    category=OperatorCategory.CROSS_SECTION,
    arity=(1, 1),
    dimension_spec=DimensionSpec((Dimension.ANY,), Dimension.DIMENSIONLESS),
    description="Cross-sectional z-score per time",
    examples=["CSScale(returns)", "CSScale(volume)"]
))
def _cs_scale(ctx: EvalContext, args: List[pl.Expr]) -> pl.Expr:
    x = args[0]
    mu = x.mean().over(ctx.time_col)
    sigma = x.std().over(ctx.time_col)
    return (x - mu) / sigma


class EnhancedEvaluator:
    """Enhanced evaluator with FIXED dimensional validation"""
    
    def __init__(self, ctx: EvalContext, registry: EnhancedOperatorRegistry = GLOBAL_REGISTRY):
        self.ctx = ctx
        self.reg = registry

    def _as_int(self, node: AST, name: str) -> int:
        if isinstance(node, Const):
            return int(node.value)
        raise TypeError(f"{name} expects an integer constant")

    def _as_float(self, node: AST, name: str) -> float:
        if isinstance(node, Const):
            return float(node.value)
        raise TypeError(f"{name} expects a numeric constant")

    def eval(self, node: AST, validate_dimensions: bool = True) -> pl.Expr:
        """Evaluate AST node to Polars expression with optional validation"""
        if isinstance(node, Var):
            if node.name in self.ctx.variables:
                return self.ctx.variables[node.name]
            return pl.col(node.name)
        
        if isinstance(node, Const):
            return pl.lit(node.value)
        
        if isinstance(node, Call):
            return self._eval_call(node, validate_dimensions)
        
        raise TypeError(f"Unknown AST node: {node}")

    def _eval_call(self, node: Call, validate_dimensions: bool) -> pl.Expr:
        """Evaluate function call with optional dimensional validation"""
        if not self.reg.exists(node.name):
            raise KeyError(f"Unknown operator: {node.name}")
            
        op = self.reg.get(node.name)
        
        # Validate arity
        if not op.meta.validate_arity(len(node.args)):
            min_args, max_args = op.meta.arity
            if max_args == -1:
                raise ValueError(f"{node.name} expects at least {min_args} arguments, got {len(node.args)}")
            else:
                raise ValueError(f"{node.name} expects {min_args}-{max_args} arguments, got {len(node.args)}")
        
        # Special handling for time series operators that need integer parameters
        if op.meta.category == OperatorCategory.TIME_SERIES:
            if len(node.args) >= 2:
                x_expr = self.eval(node.args[0], validate_dimensions)
                window_val = self._as_int(node.args[1], node.name)
                return op.builder(self.ctx, [x_expr, window_val])
        
        # Standard evaluation - no strict dimensional validation
        # The fixed system allows ANY dimension compatibility
        child_exprs = [self.eval(arg, validate_dimensions) for arg in node.args]
        result = op.builder(self.ctx, child_exprs)
        
        # Set output dimension on the call node (for potential future use)
        if hasattr(op.meta.dimension_spec, 'output_dim'):
            node.dimension = op.meta.dimension_spec.output_dim
        
        return result



class ImprovedExpressionPruner:
    """FIXED Expression pruner that doesn't over-prune"""
    
    def __init__(self, registry: EnhancedOperatorRegistry = GLOBAL_REGISTRY):
        self.reg = registry
    
    def can_prune(self, node: AST) -> tuple[bool, str]:
        """Check if expression can be pruned - NOW MUCH MORE LENIENT"""
        if isinstance(node, (Var, Const)):
            return False, ""
        
        if isinstance(node, Call):
            # Only prune for truly invalid cases
            if not self.reg.exists(node.name):
                return True, f"Unknown operator: {node.name}"
            
            op = self.reg.get(node.name)
            
            # Check arity only
            if not op.meta.validate_arity(len(node.args)):
                return True, f"Invalid argument count for {node.name}"
            
            # Recursively check children 
            for arg in node.args:
                can_prune, reason = self.can_prune(arg)
                if can_prune:
                    return True, reason
            
            # Don't prune for dimensional reasons - let the new system handle it
            return False, ""
        
        return False, ""
    
    def prune_tree(self, node: AST) -> Optional[AST]:
        """Prune invalid branches - much more conservative now"""
        can_prune, reason = self.can_prune(node)
        if can_prune:
            print(f"Pruning: {reason}")
            return None
        return node


# ===== PUBLIC API =====

def parse_expr(s: str) -> AST:
    """Parse expression string to AST"""
    return EnhancedParser(s).parse()

def build_expr(s: str, ctx: EvalContext, validate_dimensions: bool = False) -> pl.Expr:
    """
    Build Polars expression from string - 支持简单表达式
    
    Args:
        s: Expression string (simple expressions work best)
        ctx: EvalContext with variables and column mappings
        validate_dimensions: Whether to validate dimensional compatibility
        
    Returns:
        pl.Expr that can be used with LazyFrame operations
        
    Note: 对于嵌套窗口函数，建议使用 build_optimized_expr(expr, ctx, data)
    """
    ast = parse_expr(s)
    
    if validate_dimensions:
        pruner = ImprovedExpressionPruner()
        ast = pruner.prune_tree(ast)
        if ast is None:
            raise ValueError("Expression was pruned due to incompatibility")
    
    # 检查是否包含嵌套窗口函数并给出友好提示
    try:
        try:
            from .core.expression_optimizer import WindowExpressionOptimizer
        except ImportError:
            from core.expression_optimizer import WindowExpressionOptimizer
        optimizer = WindowExpressionOptimizer()
        if optimizer._has_nested_windowed_operators(ast):
            # 提供友好的错误信息和解决方案
            raise ValueError(f"""
Expression '{s}' contains nested window functions which are not supported in build_expr().

Please use one of these alternatives:
1. build_optimized_expr('{s}', ctx, data, 'result_column_name') 
2. build_multiple_expr({{'{s[:20]}...': '{s}'}}, ctx)

These functions handle nested expressions properly through step-by-step execution.
            """.strip())
    except ImportError:
        pass
    
    # 使用标准评估
    return EnhancedEvaluator(ctx).eval(ast, validate_dimensions)

def build_optimized_expr(s: str, ctx: EvalContext, data: pl.LazyFrame, result_column: str = "result") -> pl.LazyFrame:
    """
    Build optimized expression that handles nested window functions
    
    Args:
        s: Expression string (supports any complexity including nested window functions)
        ctx: EvalContext with variables and column mappings
        data: Input LazyFrame
        result_column: Name for the result column
        
    Returns:
        LazyFrame with the computed expression result
    """
    ast = parse_expr(s)
    
    try:
        try:
            from .core.expression_optimizer import WindowExpressionOptimizer
        except ImportError:
            from core.expression_optimizer import WindowExpressionOptimizer
        
        optimizer = WindowExpressionOptimizer()
        
        if optimizer._has_nested_windowed_operators(ast):
            # 使用分步执行方法处理嵌套窗口函数
            return _build_nested_expression_stepwise(ast, ctx, data, result_column)
        else:
            # 简单表达式，直接评估
            expr = EnhancedEvaluator(ctx).eval(ast, validate_dimensions=False)
            return data.with_columns(**{result_column: expr})
            
    except ImportError:
        # 如果没有优化器，使用标准评估（可能会失败）
        expr = EnhancedEvaluator(ctx).eval(ast, validate_dimensions=False)
        return data.with_columns(**{result_column: expr})

def _build_nested_expression_stepwise(ast: AST, ctx: EvalContext, data: pl.LazyFrame, result_column: str) -> pl.LazyFrame:
    """
    分步构建嵌套表达式，使用真正的Main Engine operators
    """
    temp_counter = 0
    current_data = data
    
    def get_temp_col_name():
        nonlocal temp_counter
        temp_counter += 1
        return f"__temp_step_{temp_counter}__"
    
    def process_node(node: AST, for_window_param: bool = False):
        nonlocal current_data
        
        if isinstance(node, Var):
            if for_window_param:
                # 对于窗口参数，不应该是变量
                raise ValueError(f"Window parameter cannot be a variable: {node.name}")
            if node.name in ctx.variables:
                return ctx.variables[node.name]
            return pl.col(node.name)
        
        if isinstance(node, Const):
            if for_window_param:
                # 对于窗口参数，返回实际数值
                return node.value
            return pl.lit(node.value)
        
        if isinstance(node, Call):
            # 检查是否是窗口算子
            if _is_window_operator(node.name):
                # 检查参数中是否有窗口算子
                processed_args = []
                for i, arg in enumerate(node.args):
                    # 时间序列算子的第二个参数通常是窗口大小
                    is_window_param = (node.name.startswith('TS') and i == 1)
                    
                    if isinstance(arg, Call) and _is_window_operator(arg.name):
                        # 参数也是窗口算子，需要先执行并立即添加到数据中
                        temp_col = get_temp_col_name()
                        temp_expr = process_node(arg, for_window_param=False)
                        current_data = current_data.with_columns(**{temp_col: temp_expr})
                        processed_args.append(pl.col(temp_col))
                    else:
                        processed_args.append(process_node(arg, for_window_param=is_window_param))
                
                # 检查processed_args中是否有pl.col()引用临时列
                # 如果有pl.col()且列名包含__temp_step_，说明是从临时列引用的
                has_temp_col_refs = any(
                    hasattr(arg, 'meta') and hasattr(arg.meta, 'output_name') and 
                    arg.meta.output_name() and '__temp_step_' in arg.meta.output_name()
                    for arg in processed_args if hasattr(arg, 'meta')
                ) or any(
                    str(arg).startswith('col("__temp_step_')
                    for arg in processed_args
                )
                
                # 构建当前窗口算子
                return _build_window_operator(node.name, processed_args, ctx, is_from_temp_col=has_temp_col_refs)
            else:
                # 非窗口算子，正常处理
                args = [process_node(arg, for_window_param=False) for arg in node.args]
                return _build_non_window_operator(node.name, args)
        
        raise ValueError(f"Unknown AST node: {node}")
    
    # 处理最终表达式
    final_expr = process_node(ast)
    return current_data.with_columns(**{result_column: final_expr})

def _is_window_operator(op_name: str) -> bool:
    """检查是否是窗口算子（在Polars中使用.over()的算子）"""
    window_ops = {
        # 时间序列算子（使用.over(ticker)）
        'TSMean', 'TSStd', 'TSLag', 'TSMax', 'TSMin', 'TSSum',
        # 截面算子（使用.over(time)）
        'CSRank', 'CSScale', 'CSDemean', 'CSQuantile'
    }
    return op_name in window_ops

def _build_window_operator(op_name: str, args: List[Union[pl.Expr, int, float]], ctx: EvalContext, is_from_temp_col: bool = False) -> pl.Expr:
    """使用Main Engine的真实算子构建窗口表达式
    
    Args:
        is_from_temp_col: 如果参数来自临时列，则不需要添加.over()
    """
    if op_name == 'TSMean':
        window_size = args[1] if isinstance(args[1], (int, float)) else 20
        expr = args[0].rolling_mean(window_size=int(window_size), min_samples=1)
        return expr.over(ctx.ticker_col) if not is_from_temp_col else expr
    elif op_name == 'TSStd':
        window_size = args[1] if isinstance(args[1], (int, float)) else 20
        expr = args[0].rolling_std(window_size=int(window_size), min_samples=2)
        return expr.over(ctx.ticker_col) if not is_from_temp_col else expr
    elif op_name == 'TSLag':
        periods = args[1] if isinstance(args[1], (int, float)) else 1
        expr = args[0].shift(int(periods))
        return expr.over(ctx.ticker_col) if not is_from_temp_col else expr
    elif op_name == 'CSRank':
        expr = args[0].rank(method='average')
        return expr.over(ctx.time_col) if not is_from_temp_col else expr
    elif op_name == 'CSScale':
        x = args[0]
        # CSScale需要按时间分组计算均值和标准差
        if is_from_temp_col:
            # 如果x已经是计算好的临时列，直接按时间分组
            mu = x.mean().over(ctx.time_col)
            sigma = x.std().over(ctx.time_col)
            return (x - mu) / sigma
        else:
            # 正常情况
            mu = x.mean().over(ctx.time_col)
            sigma = x.std().over(ctx.time_col)
            return (x - mu) / sigma
    elif op_name == 'CSDemean':
        x = args[0]
        # CSDemean需要按时间分组计算均值
        return x - x.mean().over(ctx.time_col)
    else:
        return args[0] if args else pl.lit(0)

def _build_non_window_operator(op_name: str, args: List[pl.Expr]) -> pl.Expr:
    """构建非窗口算子"""
    if op_name == 'Add':
        return args[0] + args[1]
    elif op_name == 'Sub':
        return args[0] - args[1]
    elif op_name == 'Mult':
        return args[0] * args[1]
    elif op_name == 'Div':
        return args[0] / args[1]
    elif op_name == 'Abs':
        return args[0].abs()
    elif op_name == 'Neg':
        return -args[0]
    elif op_name == 'Greater':
        return (args[0] > args[1]).cast(pl.Int8)
    elif op_name == 'Less':
        return (args[0] < args[1]).cast(pl.Int8)
    elif op_name == 'Equal':
        return (args[0] == args[1]).cast(pl.Int8)
    elif op_name == 'Cond':
        return pl.when(args[0] != 0).then(args[1]).otherwise(args[2])
    else:
        return args[0] if args else pl.lit(0)

class MultiExprBuilder:
    """多表达式构建器 - 使用build_optimized_expr处理复杂表达式"""
    
    def __init__(self, expressions: Dict[str, str], ctx: EvalContext):
        self.expressions = expressions
        self.ctx = ctx
    
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """应用所有表达式到数据"""
        current_data = data
        original_columns = set(data.collect_schema().names())  # 记录原始列名
        
        # 逐个处理表达式
        for col_name, expr_str in self.expressions.items():
            try:
                # 使用build_optimized_expr处理每个表达式
                result_data = build_optimized_expr(expr_str, self.ctx, current_data, col_name)
                current_data = result_data
            except Exception as e:
                print(f"Warning: Failed to build expression '{col_name}': {expr_str}")
                print(f"Error: {e}")
                continue
        
        # 清理中间变量 - 只保留原始列和新的结果列
        target_columns = list(original_columns) + list(self.expressions.keys())
        available_columns = current_data.collect_schema().names()
        final_data = current_data.select([col for col in target_columns if col in available_columns])
        
        return final_data

def build_multiple_expr(expressions: Dict[str, str], ctx: EvalContext, validate_dimensions: bool = False) -> MultiExprBuilder:
    """
    批量构建多个表达式，自动处理中间变量命名冲突
    
    Args:
        expressions: {column_name: expression_string} 的字典
        ctx: EvalContext with variables and column mappings
        validate_dimensions: Whether to validate dimensional compatibility
        
    Returns:
        MultiExprBuilder - 调用 .apply(data) 来执行
        
    Example:
        factors = {
            'ma20': 'TSMean(close, 20)',
            'volatility': 'TSStd(returns, 20)',
            'complex_factor': 'TSMean(TSStd(close, 10), 5)'  # 自动处理嵌套
        }
        builder = build_multiple_expr(factors, ctx)
        result = builder.apply(data.lazy()).collect()
    """
    return MultiExprBuilder(expressions, ctx)

def get_registry() -> EnhancedOperatorRegistry:
    """Get the global operator registry"""
    return GLOBAL_REGISTRY

def list_operators() -> Dict[str, Any]:
    """List all available operators"""
    return GLOBAL_REGISTRY.get_all_operators()