"""
分类常数系统
=============

为不同类型的算子定义合适的常数范围和候选值。
"""

from enum import Enum, auto
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass
import random


class ConstantType(Enum):
    """常数类型分类"""
    WINDOW = auto()         # 时间窗口参数 (TSMean, TSStd等)
    LAG = auto()           # 滞后期数 (TSLag等)  
    QUANTILE = auto()      # 分位数 (0-1之间)
    THRESHOLD = auto()     # 阈值 (比较操作)
    MULTIPLIER = auto()    # 乘数 (放大缩小)
    PERCENTAGE = auto()    # 百分比 (0-100)
    SMALL_INTEGER = auto() # 小整数 (1-10)
    PRICE_LEVEL = auto()   # 价格水平 (股价相关)
    VOLUME_LEVEL = auto()  # 成交量水平


@dataclass
class ConstantSpec:
    """常数规格定义"""
    const_type: ConstantType
    candidates: List[Union[int, float]]  # 候选值列表
    min_val: Union[int, float] = None    # 最小值
    max_val: Union[int, float] = None    # 最大值
    is_integer: bool = True              # 是否必须为整数
    
    def get_random_value(self) -> Union[int, float]:
        """获取随机常数值"""
        if self.candidates:
            value = random.choice(self.candidates)
        else:
            if self.min_val is not None and self.max_val is not None:
                if self.is_integer:
                    value = random.randint(int(self.min_val), int(self.max_val))
                else:
                    value = random.uniform(self.min_val, self.max_val)
            else:
                value = 1  # 默认值
        
        return int(value) if self.is_integer else float(value)


class ConstantRegistry:
    """常数注册表"""
    
    def __init__(self):
        self._specs: Dict[ConstantType, ConstantSpec] = {}
        self._setup_default_specs()
    
    def _setup_default_specs(self):
        """设置默认的常数规格"""
        
        # 时间窗口参数 - 金融中常用的周期
        self._specs[ConstantType.WINDOW] = ConstantSpec(
            const_type=ConstantType.WINDOW,
            candidates=[3, 5, 10, 15, 20, 30, 60, 120, 250],  # 3天到1年
            min_val=1,
            max_val=500,
            is_integer=True
        )
        
        # 滞后期数 - 一般较小
        self._specs[ConstantType.LAG] = ConstantSpec(
            const_type=ConstantType.LAG,
            candidates=[1, 2, 3, 5, 10, 20],
            min_val=1,
            max_val=50,
            is_integer=True
        )
        
        # 分位数 - 0到1之间
        self._specs[ConstantType.QUANTILE] = ConstantSpec(
            const_type=ConstantType.QUANTILE,
            candidates=[0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.75, 0.8, 0.9],
            min_val=0.0,
            max_val=1.0,
            is_integer=False
        )
        
        # 阈值 - 用于比较
        self._specs[ConstantType.THRESHOLD] = ConstantSpec(
            const_type=ConstantType.THRESHOLD,
            candidates=[0, 0.5, 1, 1.5, 2, 5, 10, 50, 100],
            min_val=-100,
            max_val=1000,
            is_integer=False
        )
        
        # 乘数 - 放大缩小因子
        self._specs[ConstantType.MULTIPLIER] = ConstantSpec(
            const_type=ConstantType.MULTIPLIER,
            candidates=[0.1, 0.5, 1, 2, 5, 10, 100, 1000],
            min_val=0.001,
            max_val=10000,
            is_integer=False
        )
        
        # 百分比
        self._specs[ConstantType.PERCENTAGE] = ConstantSpec(
            const_type=ConstantType.PERCENTAGE,
            candidates=[1, 5, 10, 20, 50, 80, 90, 95, 99],
            min_val=0,
            max_val=100,
            is_integer=True
        )
        
        # 小整数
        self._specs[ConstantType.SMALL_INTEGER] = ConstantSpec(
            const_type=ConstantType.SMALL_INTEGER,
            candidates=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            min_val=1,
            max_val=20,
            is_integer=True
        )
        
        # 价格水平 - 典型股价范围
        self._specs[ConstantType.PRICE_LEVEL] = ConstantSpec(
            const_type=ConstantType.PRICE_LEVEL,
            candidates=[1, 5, 10, 20, 50, 100, 200, 500, 1000],
            min_val=0.1,
            max_val=5000,
            is_integer=False
        )
        
        # 成交量水平
        self._specs[ConstantType.VOLUME_LEVEL] = ConstantSpec(
            const_type=ConstantType.VOLUME_LEVEL,
            candidates=[1000, 5000, 10000, 50000, 100000, 500000, 1000000],
            min_val=100,
            max_val=10000000,
            is_integer=True
        )
    
    def get_spec(self, const_type: ConstantType) -> ConstantSpec:
        """获取常数规格"""
        return self._specs.get(const_type, self._specs[ConstantType.SMALL_INTEGER])
    
    def get_random_constant(self, const_type: ConstantType) -> Union[int, float]:
        """获取指定类型的随机常数"""
        spec = self.get_spec(const_type)
        return spec.get_random_value()
    
    def register_spec(self, spec: ConstantSpec):
        """注册新的常数规格"""
        self._specs[spec.const_type] = spec


# 全局常数注册表实例
CONSTANT_REGISTRY = ConstantRegistry()


# 算子与常数类型的映射
OPERATOR_CONSTANT_MAP = {
    # 时间序列算子 - 第二个参数通常是窗口大小
    'TSMean': {1: ConstantType.WINDOW},      # 第1个参数(索引1)是窗口
    'TSStd': {1: ConstantType.WINDOW},       
    'TSMax': {1: ConstantType.WINDOW},
    'TSMin': {1: ConstantType.WINDOW},
    'TSSum': {1: ConstantType.WINDOW},
    'TSRank': {1: ConstantType.WINDOW},
    
    # 滞后算子
    'TSLag': {1: ConstantType.LAG},          # 第1个参数是滞后期
    'TSDelta': {1: ConstantType.LAG},        # 变化期间
    
    # 比较算子 - 阈值
    'Greater': {1: ConstantType.THRESHOLD},   # 第1个参数可能是阈值
    'Less': {1: ConstantType.THRESHOLD},
    'GreaterEq': {1: ConstantType.THRESHOLD},
    'LessEq': {1: ConstantType.THRESHOLD},
    
    # 算术算子 - 乘数
    'Mult': {1: ConstantType.MULTIPLIER},     # 乘法的第1个参数
    'Div': {1: ConstantType.MULTIPLIER},      # 除法的第1个参数
    
    # 分位数相关
    'Quantile': {1: ConstantType.QUANTILE},   # 分位数参数
    'Bound': {1: ConstantType.QUANTILE, 2: ConstantType.QUANTILE},  # 上下界
    
    # 条件算子
    'Clip': {1: ConstantType.THRESHOLD, 2: ConstantType.THRESHOLD},  # 最小值、最大值
}


def get_constant_type_for_operator(op_name: str, param_index: int) -> ConstantType:
    """获取指定算子指定参数位置应该使用的常数类型"""
    if op_name in OPERATOR_CONSTANT_MAP:
        param_map = OPERATOR_CONSTANT_MAP[op_name]
        if param_index in param_map:
            return param_map[param_index]
    
    # 默认返回小整数类型
    return ConstantType.SMALL_INTEGER


def get_smart_constant(op_name: str, param_index: int) -> Union[int, float]:
    """为指定算子的指定参数位置生成智能常数"""
    const_type = get_constant_type_for_operator(op_name, param_index)
    return CONSTANT_REGISTRY.get_random_constant(const_type)


# 便捷函数
def get_window_constant() -> int:
    """获取时间窗口常数"""
    return CONSTANT_REGISTRY.get_random_constant(ConstantType.WINDOW)

def get_lag_constant() -> int:
    """获取滞后期常数"""
    return CONSTANT_REGISTRY.get_random_constant(ConstantType.LAG)

def get_quantile_constant() -> float:
    """获取分位数常数"""
    return CONSTANT_REGISTRY.get_random_constant(ConstantType.QUANTILE)

def get_threshold_constant() -> float:
    """获取阈值常数"""
    return CONSTANT_REGISTRY.get_random_constant(ConstantType.THRESHOLD)

def get_multiplier_constant() -> float:
    """获取乘数常数"""
    return CONSTANT_REGISTRY.get_random_constant(ConstantType.MULTIPLIER)