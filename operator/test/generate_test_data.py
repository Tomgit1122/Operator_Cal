#!/usr/bin/env python3
"""
Financial Test Data Generator - 生成真实的金融测试数据
=====================================================

生成具有真实金融特征的测试数据，包括：
1. 多只股票的价格数据
2. 成交量数据  
3. 衍生指标（收益率、价差等）
4. 符合金融市场规律的数据分布
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import polars as pl
import numpy as np
import random
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

class FinancialDataGenerator:
    """金融数据生成器"""
    
    def __init__(self, seed: int = 42):
        """初始化生成器"""
        np.random.seed(seed)
        random.seed(seed)
        
        # 股票池
        self.tickers = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
            'NVDA', 'META', 'NFLX', 'ORCL', 'CRM',
            'BABA', 'JD', 'BIDU', 'NIO', 'LI',
            'JPM', 'BAC', 'GS', 'MS', 'C'
        ]
        
        # 基础参数
        self.base_prices = {ticker: np.random.uniform(50, 500) for ticker in self.tickers}
        self.volatilities = {ticker: np.random.uniform(0.15, 0.45) for ticker in self.tickers}
        self.trends = {ticker: np.random.uniform(-0.0002, 0.0002) for ticker in self.tickers}
        
    def generate_price_series(self, ticker: str, days: int) -> Dict[str, np.ndarray]:
        """生成单只股票的价格序列"""
        base_price = self.base_prices[ticker]
        volatility = self.volatilities[ticker] 
        trend = self.trends[ticker]
        
        # 生成对数收益率序列
        returns = np.random.normal(trend, volatility / np.sqrt(252), days)
        
        # 添加一些异常值（模拟市场冲击）
        shock_days = np.random.choice(days, size=max(1, days // 50), replace=False)
        returns[shock_days] += np.random.normal(0, volatility * 2, len(shock_days))
        
        # 转换为价格序列
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # 生成 OHLC 数据
        daily_ranges = np.random.exponential(volatility * close_prices / 20)
        
        # Open: 前一日收盘价加上隔夜跳空
        opens = np.zeros_like(close_prices)
        opens[0] = close_prices[0] * (1 + np.random.normal(0, volatility/4))
        opens[1:] = close_prices[:-1] * (1 + np.random.normal(0, volatility/4, days-1))
        
        # High/Low: 基于当日真实波动范围
        highs = np.maximum(opens, close_prices) + daily_ranges * np.random.uniform(0, 1, days)
        lows = np.minimum(opens, close_prices) - daily_ranges * np.random.uniform(0, 1, days)
        
        # VWAP: 加权平均价
        vwap = (opens + highs + lows + close_prices) / 4 + np.random.normal(0, close_prices * 0.001)
        
        return {
            'open': opens,
            'high': highs,
            'low': lows, 
            'close': close_prices,
            'vwap': vwap
        }
    
    def generate_volume_series(self, ticker: str, days: int, close_prices: np.ndarray) -> Dict[str, np.ndarray]:
        """生成成交量相关数据"""
        # 基础成交量（与价格负相关，价格高时成交量相对较低）
        base_volume = np.random.uniform(1e6, 5e7)
        price_factor = (close_prices / close_prices[0]) ** (-0.3)  # 价格越高，成交量相对越低
        
        # 波动率因子（波动率高时成交量大）
        volatility_factor = np.abs(np.diff(np.log(close_prices), prepend=np.log(close_prices[0]))) * 20
        volatility_factor = np.maximum(volatility_factor, 0.5)  # 最小倍数
        
        # 趋势因子（大涨大跌时成交量放大）
        trend_factor = np.abs(np.diff(np.log(close_prices), prepend=np.log(close_prices[0]))) * 10 + 1
        
        # 随机因子
        random_factor = np.random.lognormal(0, 0.3, days)
        
        # 组合成最终成交量
        volume = base_volume * price_factor * volatility_factor * trend_factor * random_factor
        volume = volume.astype(int)
        
        # 成交额 = 成交量 × 平均价格
        turnover = volume * close_prices * np.random.uniform(0.98, 1.02, days)
        amount = turnover * 1000  # 以千元为单位
        
        return {
            'volume': volume,
            'turnover': turnover, 
            'amount': amount
        }
    
    def generate_derived_indicators(self, prices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """生成衍生指标"""
        close_prices = prices['close']
        high_prices = prices['high']
        low_prices = prices['low']
        
        # 收益率
        returns = np.diff(np.log(close_prices), prepend=np.log(close_prices[0]))
        pct_change = np.diff(close_prices, prepend=close_prices[0]) / close_prices[0]
        
        # 价差
        spread = high_prices - low_prices
        
        # 相对价位
        relative_position = (close_prices - low_prices) / (high_prices - low_prices)
        relative_position = np.nan_to_num(relative_position, 0.5)
        
        return {
            'returns': returns,
            'pct_change': pct_change,
            'spread': spread,
            'relative_position': relative_position
        }
    
    def generate_single_stock_data(self, ticker: str, days: int) -> pl.DataFrame:
        """生成单只股票的完整数据"""
        # 价格数据
        prices = self.generate_price_series(ticker, days)
        
        # 成交量数据
        volumes = self.generate_volume_series(ticker, days, prices['close'])
        
        # 衍生指标
        derived = self.generate_derived_indicators(prices)
        
        # 组合数据
        data = {
            'ticker': [ticker] * days,
            'time': list(range(days)),
            **prices,
            **volumes,
            **derived
        }
        
        return pl.DataFrame(data)
    
    def generate_market_data(self, 
                           tickers: List[str] = None, 
                           days: int = 252,
                           add_noise: bool = True) -> pl.DataFrame:
        """生成多只股票的市场数据"""
        if tickers is None:
            tickers = self.tickers[:10]  # 默认使用前10只股票
            
        print(f"Generating market data for {len(tickers)} tickers, {days} days...")
        
        # 生成各股票数据
        dfs = []
        for i, ticker in enumerate(tickers):
            print(f"  Generating {ticker} ({i+1}/{len(tickers)})")
            df = self.generate_single_stock_data(ticker, days)
            dfs.append(df)
        
        # 合并数据
        market_data = pl.concat(dfs)
        
        # 添加市场噪音（相关性、行业效应等）
        if add_noise:
            market_data = self._add_market_effects(market_data)
        
        # 按时间和股票排序
        market_data = market_data.sort(['time', 'ticker'])
        
        print(f"Generated market data: {market_data.shape}")
        return market_data
    
    def _add_market_effects(self, data: pl.DataFrame) -> pl.DataFrame:
        """添加市场效应（相关性、行业因子等）"""
        # 这里可以添加更复杂的市场效应
        # 目前保持简单，仅添加小量噪音
        
        # 添加一些随机扰动
        data = data.with_columns([
            (pl.col('returns') + pl.lit(np.random.normal(0, 0.001, len(data)))).alias('returns'),
            (pl.col('volume') * pl.lit(np.random.uniform(0.98, 1.02, len(data)))).alias('volume')
        ])
        
        return data
    
    def save_data(self, data: pl.DataFrame, filepath: str = None):
        """保存数据到文件"""
        if filepath is None:
            filepath = str(Path(__file__).parent / "data" / "financial_test_data.parquet")
        
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为parquet格式（高效压缩）
        data.write_parquet(filepath)
        
        # 保存数据摘要
        summary_path = filepath.replace('.parquet', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Financial Test Data Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Data shape: {data.shape}\n")
            f.write(f"Tickers: {sorted(data['ticker'].unique().to_list())}\n")
            f.write(f"Time range: {data['time'].min()} - {data['time'].max()}\n")
            f.write(f"Columns: {data.columns}\n\n")
            
            # 基础统计信息
            f.write("Basic Statistics:\n")
            f.write("-" * 20 + "\n")
            numeric_cols = ['close', 'volume', 'returns', 'spread']
            for col in numeric_cols:
                if col in data.columns:
                    col_data = data.select(pl.col(col))
                    mean_val = col_data.mean().item()
                    std_val = col_data.std().item()
                    min_val = col_data.min().item()
                    max_val = col_data.max().item()
                    
                    f.write(f"{col}:\n")
                    f.write(f"  mean: {mean_val:.6f}\n")
                    f.write(f"  std: {std_val:.6f}\n")
                    f.write(f"  min: {min_val:.6f}\n")
                    f.write(f"  max: {max_val:.6f}\n")
                    f.write("\n")
        
        print(f"Data saved to: {filepath}")
        print(f"Summary saved to: {summary_path}")
        return filepath

def create_standard_test_dataset():
    """创建标准测试数据集"""
    generator = FinancialDataGenerator(seed=42)
    
    # 使用前15只股票，生成1年的数据
    data = generator.generate_market_data(
        tickers=generator.tickers[:15],
        days=252*24,
        add_noise=True
    )
    
    # 保存数据
    filepath = generator.save_data(data)
    
    return data, filepath

def main():
    """主函数"""
    print("Financial Test Data Generator")
    print("=" * 40)
    
    # 创建标准测试数据集
    data, filepath = create_standard_test_dataset()
    
    # 显示数据预览
    print(f"\nData preview:")
    print(data.head(10))
    
    print(f"\nData info:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns}")
    print(f"Tickers: {sorted(data['ticker'].unique().to_list())}")
    
    print(f"\nTest data generation complete!")
    return data, filepath

if __name__ == "__main__":
    main()