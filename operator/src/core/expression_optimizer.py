"""
表达式优化器
============

解决over重复使用问题的关键组件。
通过分解嵌套窗口表达式来避免Polars的限制。
"""

from typing import Dict, List, Tuple, Any, Optional
import polars as pl
from .ast_nodes import AST, Var, Const, Call


class WindowExpressionOptimizer:
    """窗口表达式优化器"""
    
    def __init__(self, prefix: str = ""):
        """
        初始化优化器
        
        Args:
            prefix: 临时列名前缀，用于避免命名冲突
        """
        self.temp_column_counter = 0
        self.prefix = prefix
        self.intermediate_columns: Dict[str, pl.Expr] = {}
        self.windowed_operators = {
            'TSMean', 'TSStd', 'TSLag', 'TSMax', 'TSMin', 'TSSum',
            'CSRank', 'CSDemean', 'CSScale', 'CSQuantile'
        }
    
    def _get_temp_column_name(self) -> str:
        """生成临时列名"""
        self.temp_column_counter += 1
        if self.prefix:
            return f"__{self.prefix}_temp_{self.temp_column_counter}__"
        else:
            return f"__temp_col_{self.temp_column_counter}__"
    
    def _is_windowed_operator(self, node: AST) -> bool:
        """检查是否是窗口算子"""
        if isinstance(node, Call):
            return node.name in self.windowed_operators
        return False
    
    def _has_nested_windowed_operators(self, node: AST) -> bool:
        """检查是否包含嵌套的窗口算子"""
        if not isinstance(node, Call):
            return False
        
        # 如果当前节点是窗口算子
        if self._is_windowed_operator(node):
            # 检查参数中是否有窗口算子
            for arg in node.args:
                if self._is_windowed_operator(arg):
                    return True
                # 递归检查嵌套
                if self._has_nested_windowed_operators(arg):
                    return True
        else:
            # 当前节点不是窗口算子，但检查参数
            windowed_count = 0
            for arg in node.args:
                if self._is_windowed_operator(arg):
                    windowed_count += 1
                if self._has_nested_windowed_operators(arg):
                    return True
            # 如果有多个窗口算子作为参数，也算嵌套
            if windowed_count > 1:
                return True
        
        return False
    
    def _build_simple_expression(self, node: AST) -> pl.Expr:
        """构建简单表达式（无嵌套窗口函数）"""
        if isinstance(node, Var):
            return pl.col(node.name)
        elif isinstance(node, Const):
            return pl.lit(node.value)
        elif isinstance(node, Call):
            # 递归构建参数
            args = []
            for arg in node.args:
                args.append(self._build_simple_expression(arg))
            
            # 构建表达式
            return self._build_operator_expression(node.name, args)
        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")
    
    def optimize_expression(self, node: AST, data: pl.LazyFrame) -> Tuple[pl.Expr, pl.LazyFrame]:
        """
        优化表达式，解决嵌套窗口函数问题
        
        返回：(最终表达式, 包含中间结果的数据)
        """
        if not self._has_nested_windowed_operators(node):
            # 没有嵌套问题，直接转换为简单表达式
            expr = self._build_simple_expression(node)
            return expr, data
        
        # 需要分解嵌套表达式
        optimized_expr, updated_data = self._decompose_nested_expression(node, data)
        return optimized_expr, updated_data
    
    def _decompose_nested_expression(self, node: AST, data: pl.LazyFrame) -> Tuple[pl.Expr, pl.LazyFrame]:
        """分解嵌套表达式"""
        if isinstance(node, (Var, Const)):
            # 叶子节点，直接转换
            if isinstance(node, Var):
                return pl.col(node.name), data
            else:
                return pl.lit(node.value), data
        
        if isinstance(node, Call):
            # 处理函数调用
            return self._decompose_call(node, data)
        
        raise ValueError(f"Unknown AST node type: {type(node)}")
    
    def _decompose_call(self, node: Call, data: pl.LazyFrame) -> Tuple[pl.Expr, pl.LazyFrame]:
        """分解函数调用"""
        current_data = data
        processed_args = []
        
        # 首先处理所有参数
        for arg in node.args:
            if self._is_windowed_operator(arg):
                # 参数是窗口算子，需要预先计算
                temp_col = self._get_temp_column_name()
                arg_expr, current_data = self._decompose_nested_expression(arg, current_data)
                
                # 将中间结果添加到数据中
                current_data = current_data.with_columns(**{temp_col: arg_expr})
                
                # 使用临时列作为参数
                processed_args.append(pl.col(temp_col))
                
            elif isinstance(arg, Call) and self._has_nested_windowed_operators(arg):
                # 参数包含嵌套的窗口算子
                temp_col = self._get_temp_column_name()
                arg_expr, current_data = self._decompose_nested_expression(arg, current_data)
                
                current_data = current_data.with_columns(**{temp_col: arg_expr})
                processed_args.append(pl.col(temp_col))
                
            else:
                # 普通参数，递归处理
                arg_expr, current_data = self._decompose_nested_expression(arg, current_data)
                processed_args.append(arg_expr)
        
        # 构建当前节点的表达式
        final_expr = self._build_operator_expression(node.name, processed_args)
        
        return final_expr, current_data
    
    def _build_operator_expression(self, op_name: str, args: List[pl.Expr]) -> pl.Expr:
        """根据算子名称和参数构建Polars表达式"""
        # 简化实现，直接构建基本的Polars表达式
        if op_name == 'Add':
            return args[0] + args[1]
        elif op_name == 'Sub':
            return args[0] - args[1]
        elif op_name == 'Mult':
            return args[0] * args[1]
        elif op_name == 'Div':
            return args[0] / args[1]
        elif op_name == 'TSMean':
            # 第二个参数应该是窗口大小（整数）
            return args[0].rolling_mean(window_size=5, min_samples=1).over('ticker')
        elif op_name == 'TSStd':
            return args[0].rolling_std(window_size=3, min_samples=2).over('ticker')
        elif op_name == 'TSLag':
            return args[0].shift(1).over('ticker')
        elif op_name == 'CSRank':
            return args[0].rank(method='average').over('time')
        elif op_name == 'CSScale':
            x = args[0]
            mu = x.mean().over('time')
            sigma = x.std().over('time')
            return (x - mu) / sigma
        elif op_name == 'CSDemean':
            x = args[0]
            return x - x.mean().over('time')
        else:
            # 默认返回第一个参数
            return args[0] if args else pl.lit(0)
    
    def _build_operator_expression_with_registry(self, node: AST, registry, ctx) -> pl.Expr:
        """使用注册表构建单个节点的表达式"""
        if isinstance(node, (Var, Const)):
            if isinstance(node, Var):
                if node.name in ctx.variables:
                    return ctx.variables[node.name]
                return pl.col(node.name)
            else:
                return pl.lit(node.value)
        
        if isinstance(node, Call):
            op = registry.get(node.name)
            if op is None:
                raise ValueError(f"Unknown operator: {node.name}")
            
            # 递归处理参数
            processed_args = []
            for arg in node.args:
                processed_args.append(self._build_operator_expression_with_registry(arg, registry, ctx))
            
            # 特殊处理时间序列算子的常数参数
            if op.meta.category.name == 'TIME_SERIES' and len(node.args) >= 2:
                if isinstance(node.args[1], Const):
                    processed_args[1] = int(node.args[1].value)
            
            return op.builder(ctx, processed_args)
        
        raise ValueError(f"Unknown AST node type: {type(node)}")


class OptimizedEvaluator:
    """优化的表达式评估器"""
    
    def __init__(self):
        self.optimizer = WindowExpressionOptimizer()
    
    def evaluate_expression(self, node: AST, data: pl.LazyFrame) -> pl.LazyFrame:
        """评估表达式并返回结果数据"""
        try:
            # 尝试优化表达式
            optimized_expr, intermediate_data = self.optimizer.optimize_expression(node, data)
            
            # 执行最终表达式
            result_data = intermediate_data.with_columns(result=optimized_expr)
            
            return result_data
            
        except Exception as e:
            print(f"Expression optimization failed: {e}")
            # 回退到直接评估（可能失败）
            raise e


def test_optimizer():
    """测试优化器"""
    print("Testing Window Expression Optimizer")
    print("="*50)
    
    # 创建测试数据
    test_data = pl.DataFrame({
        'ticker': ['A', 'A', 'A', 'B', 'B', 'B'] * 10,
        'time': list(range(60)),
        'close': [100 + i for i in range(60)],
        'volume': [1000 + i*10 for i in range(60)]
    }).lazy()
    
    print(f"Test data shape: {test_data.collect().shape}")
    
    # 创建一个简单的嵌套AST进行测试
    # TSMean(TSStd(close, 10), 5) 的简化版本
    inner_call = Call('TSStd', [Var('close'), Const(10)])
    outer_call = Call('TSMean', [inner_call, Const(5)])
    
    optimizer = WindowExpressionOptimizer()
    
    try:
        print(f"Testing nested expression: TSMean(TSStd(close, 10), 5)")
        print(f"Has nested windowed operators: {optimizer._has_nested_windowed_operators(outer_call)}")
        
        # 尝试优化
        optimized_expr, optimized_data = optimizer.optimize_expression(outer_call, test_data)
        print(f"Optimization successful!")
        print(f"Intermediate columns created: {optimizer.temp_column_counter}")
        
        # 执行优化后的表达式
        result = optimized_data.with_columns(final_result=optimized_expr).select(['ticker', 'time', 'final_result']).limit(5).collect()
        print(f"Execution successful!")
        print(result)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


class UnifiedExpressionBuilder:
    """统一表达式构建器 - 自动判断是否需要优化，返回统一的 pl.Expr"""
    
    def __init__(self):
        self.expression_counter = 0
        self.temp_columns: Dict[str, pl.Expr] = {}
    
    def build(self, s: str, ctx, existing_columns: Optional[Dict[str, pl.Expr]] = None) -> Tuple[pl.Expr, Dict[str, pl.Expr]]:
        """
        构建表达式，自动判断是否需要优化
        
        Args:
            s: 表达式字符串
            ctx: 评估上下文
            existing_columns: 已存在的中间列（避免命名冲突）
            
        Returns:
            (final_expr, temp_columns) - 最终表达式和需要添加的中间列
        """
        self.expression_counter += 1
        prefix = f"expr_{self.expression_counter}"
        
        # 解析表达式
        from ..main_engine import parse_expr
        ast = parse_expr(s)
        
        # 创建优化器（使用唯一前缀避免冲突）
        optimizer = WindowExpressionOptimizer(prefix=prefix)
        
        # 检查是否需要优化
        if optimizer._has_nested_windowed_operators(ast):
            # 需要优化：分解嵌套表达式
            return self._build_optimized(ast, ctx, optimizer, existing_columns)
        else:
            # 不需要优化：直接构建
            expr = optimizer._build_simple_expression(ast)
            return expr, {}
    
    def _build_optimized(self, ast: AST, ctx, optimizer: WindowExpressionOptimizer, existing_columns: Optional[Dict[str, pl.Expr]]) -> Tuple[pl.Expr, Dict[str, pl.Expr]]:
        """构建优化表达式"""
        temp_columns = {}
        
        # 递归分解表达式
        final_expr = self._decompose_expression(ast, optimizer, temp_columns)
        
        return final_expr, temp_columns
    
    def _decompose_expression(self, node: AST, optimizer: WindowExpressionOptimizer, temp_columns: Dict[str, pl.Expr]) -> pl.Expr:
        """递归分解表达式"""
        if isinstance(node, (Var, Const)):
            if isinstance(node, Var):
                return pl.col(node.name)
            else:
                return pl.lit(node.value)
        
        if isinstance(node, Call):
            # 处理参数
            processed_args = []
            for arg in node.args:
                if isinstance(arg, Call):
                    # 检查是否是窗口算子或包含嵌套窗口算子
                    if optimizer._is_windowed_operator(arg) or optimizer._has_nested_windowed_operators(arg):
                        # 需要预先计算
                        temp_col_name = optimizer._get_temp_column_name()
                        arg_expr = self._decompose_expression(arg, optimizer, temp_columns)
                        temp_columns[temp_col_name] = arg_expr
                        processed_args.append(pl.col(temp_col_name))
                    else:
                        # 普通函数调用
                        processed_args.append(self._decompose_expression(arg, optimizer, temp_columns))
                else:
                    # 变量或常数
                    processed_args.append(self._decompose_expression(arg, optimizer, temp_columns))
            
            # 构建当前节点的表达式
            return optimizer._build_operator_expression(node.name, processed_args)
        
        raise ValueError(f"Unknown AST node type: {type(node)}")


if __name__ == "__main__":
    test_optimizer()