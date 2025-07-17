"""
Data Manager
Handles data storage and retrieval using DuckDB and Parquet.
"""
import os
import json
import pandas as pd
import duckdb
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pathlib import Path

from src.infra.nt_bridge import MarketData, TradeCompletion


class DataManager:
    """Manages data storage and retrieval for the trading system."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Database connection
        self.db_path = self.data_dir / "trading_data.db"
        self.conn = duckdb.connect(str(self.db_path))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize database tables
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Initialize database tables."""
        # Market data table with 30m fields (volatility calculated on-demand)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP,
                price_1m DOUBLE[],
                price_5m DOUBLE[],
                price_15m DOUBLE[],
                price_30m DOUBLE[],
                price_1h DOUBLE[],
                volume_1m DOUBLE[],
                volume_5m DOUBLE[],
                volume_15m DOUBLE[],
                volume_30m DOUBLE[],
                volume_1h DOUBLE[],
                account_balance DOUBLE,
                buying_power DOUBLE,
                daily_pnl DOUBLE,
                unrealized_pnl DOUBLE,
                open_positions INTEGER,
                current_price DOUBLE
            )
        """)
        
        # Trade completions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_completions (
                timestamp TIMESTAMP,
                pnl DOUBLE,
                entry_price DOUBLE,
                exit_price DOUBLE,
                size INTEGER,
                exit_reason VARCHAR,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                trade_duration_minutes DOUBLE
            )
        """)
        
        # Historical bars table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_bars (
                timestamp TIMESTAMP,
                timeframe VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System state table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                timestamp TIMESTAMP,
                total_trades INTEGER,
                total_pnl DOUBLE,
                strategy_performance_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.logger.info("Database tables initialized")
    
    def store_market_data(self, market_data: MarketData):
        """Store live market data."""
        try:
            timestamp = datetime.fromtimestamp(market_data.timestamp)
            
            self.conn.execute("""
                INSERT INTO market_data VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                timestamp,
                market_data.price_1m,
                market_data.price_5m,
                market_data.price_15m,
                market_data.price_30m,
                market_data.price_1h,
                market_data.volume_1m,
                market_data.volume_5m,
                market_data.volume_15m,
                market_data.volume_30m,
                market_data.volume_1h,
                market_data.account_balance,
                market_data.buying_power,
                market_data.daily_pnl,
                market_data.unrealized_pnl,
                market_data.open_positions,
                market_data.current_price
            ))
            
            # Note: Using DuckDB as primary storage, parquet only for archival
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
    
    def store_trade_completion(self, trade_completion: TradeCompletion):
        """Store trade completion data."""
        try:
            timestamp = datetime.now()
            entry_time = datetime.fromtimestamp(trade_completion.entry_time / 10000000)  # Convert from ticks
            exit_time = datetime.fromtimestamp(trade_completion.exit_time / 10000000)
            
            self.conn.execute("""
                INSERT INTO trade_completions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                timestamp,
                trade_completion.pnl,
                trade_completion.entry_price,
                trade_completion.exit_price,
                trade_completion.size,
                trade_completion.exit_reason,
                entry_time,
                exit_time,
                trade_completion.trade_duration_minutes
            ))
            
            # Note: Using DuckDB as primary storage
            
        except Exception as e:
            self.logger.error(f"Error storing trade completion: {e}")
    
    def store_historical_data(self, historical_data: Dict[str, Any]):
        """Store historical data from NinjaTrader."""
        try:
            timeframes = ['1m', '5m', '15m', '30m', '1h']
            
            for timeframe in timeframes:
                bars_key = f'bars_{timeframe}'
                if bars_key in historical_data:
                    bars = historical_data[bars_key]
                    
                    for bar in bars:
                        # Handle different timestamp formats
                        try:
                            # Convert .NET ticks to Unix timestamp
                            # .NET ticks: 100ns intervals since Jan 1, 0001 AD
                            # Unix timestamp: seconds since Jan 1, 1970 AD
                            if bar['timestamp'] > 621355968000000000:  # .NET ticks for 1970
                                # Convert .NET ticks to Unix timestamp
                                unix_timestamp = (bar['timestamp'] - 621355968000000000) / 10000000
                                timestamp = datetime.fromtimestamp(unix_timestamp)
                            else:
                                # Assume Unix timestamp
                                timestamp = datetime.fromtimestamp(bar['timestamp'])
                        except (ValueError, OSError):
                            # Fallback to current time if timestamp conversion fails
                            timestamp = datetime.now()
                            self.logger.warning(f"Invalid timestamp {bar['timestamp']}, using current time")
                        
                        self.conn.execute("""
                            INSERT INTO historical_bars VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?
                            )
                        """, (
                            timestamp,
                            timeframe,
                            bar['open'],
                            bar['high'],
                            bar['low'],
                            bar['close'],
                            bar['volume'],
                            datetime.now()
                        ))
            
            # Only save to parquet for long-term archival if needed
            # self._save_historical_to_parquet(historical_data)
            
            self.logger.info(f"Historical data stored for {len(timeframes)} timeframes")
            
        except Exception as e:
            self.logger.error(f"Error storing historical data: {e}")
    
    def save_system_state(self, state: Dict[str, Any]):
        """Save system state."""
        try:
            timestamp = datetime.now()
            
            self.conn.execute("""
                INSERT INTO system_state VALUES (
                    ?, ?, ?, ?, ?
                )
            """, (
                timestamp,
                state['total_trades'],
                state['total_pnl'],
                json.dumps(state['strategy_performance']),
                datetime.now()
            ))
            
            # DuckDB is primary storage, no need for redundant JSON files
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")
    
    def get_recent_market_data(self, hours: int = 24) -> pd.DataFrame:
        """Get recent market data."""
        try:
            result = self.conn.execute("""
                SELECT * FROM market_data 
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL ? HOUR
                ORDER BY timestamp DESC
            """, (hours,)).fetchdf()
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting recent market data: {e}")
            return pd.DataFrame()
    
    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        """Get trade history."""
        try:
            result = self.conn.execute("""
                SELECT * FROM trade_completions 
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL ? DAY
                ORDER BY timestamp DESC
            """, (days,)).fetchdf()
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return pd.DataFrame()
    
    def get_historical_bars(self, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Get historical bars for a specific timeframe."""
        try:
            result = self.conn.execute("""
                SELECT * FROM historical_bars 
                WHERE timeframe = ? AND timestamp >= CURRENT_TIMESTAMP - INTERVAL ? DAY
                ORDER BY timestamp ASC
            """, (timeframe, days)).fetchdf()
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting historical bars: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            # Get total trades and PnL
            total_result = self.conn.execute("""
                SELECT COUNT(*) as total_trades, SUM(pnl) as total_pnl
                FROM trade_completions
            """).fetchone()
            
            # Get win rate
            win_rate_result = self.conn.execute("""
                SELECT 
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate
                FROM trade_completions
            """).fetchone()
            
            # Get average trade metrics
            avg_result = self.conn.execute("""
                SELECT 
                    AVG(pnl) as avg_pnl,
                    AVG(trade_duration_minutes) as avg_duration
                FROM trade_completions
            """).fetchone()
            
            # Get daily PnL
            daily_pnl = self.conn.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    SUM(pnl) as daily_pnl
                FROM trade_completions
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 30 DAY
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """).fetchdf()
            
            return {
                'total_trades': total_result[0] if total_result[0] else 0,
                'total_pnl': total_result[1] if total_result[1] else 0.0,
                'win_rate': win_rate_result[0] if win_rate_result[0] else 0.0,
                'avg_pnl': avg_result[0] if avg_result[0] else 0.0,
                'avg_duration': avg_result[1] if avg_result[1] else 0.0,
                'daily_pnl': daily_pnl.to_dict('records') if not daily_pnl.empty else []
            }
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _save_to_parquet(self, data: Any, table_name: str):
        """Save data to parquet file."""
        try:
            parquet_dir = self.data_dir / "parquet" / table_name
            parquet_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H')
            filename = parquet_dir / f"{table_name}_{timestamp}.parquet"
            
            # Convert data to DataFrame (with 30m fields)
            if isinstance(data, MarketData):
                df = pd.DataFrame([{
                    'timestamp': datetime.fromtimestamp(data.timestamp),
                    'price_1m': str(data.price_1m),
                    'price_5m': str(data.price_5m),
                    'price_15m': str(data.price_15m),
                    'price_30m': str(data.price_30m),
                    'price_1h': str(data.price_1h),
                    'volume_1m': str(data.volume_1m),
                    'volume_5m': str(data.volume_5m),
                    'volume_15m': str(data.volume_15m),
                    'volume_30m': str(data.volume_30m),
                    'volume_1h': str(data.volume_1h),
                    'account_balance': data.account_balance,
                    'buying_power': data.buying_power,
                    'daily_pnl': data.daily_pnl,
                    'unrealized_pnl': data.unrealized_pnl,
                    'open_positions': data.open_positions,
                    'current_price': data.current_price
                }])
            elif isinstance(data, TradeCompletion):
                df = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'pnl': data.pnl,
                    'entry_price': data.entry_price,
                    'exit_price': data.exit_price,
                    'size': data.size,
                    'exit_reason': data.exit_reason,
                    'entry_time': datetime.fromtimestamp(data.entry_time / 10000000),
                    'exit_time': datetime.fromtimestamp(data.exit_time / 10000000),
                    'trade_duration_minutes': data.trade_duration_minutes
                }])
            else:
                return
            
            # Append to existing file or create new one
            if filename.exists():
                existing_df = pd.read_parquet(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_parquet(filename, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving to parquet: {e}")
    
    def _save_historical_to_parquet(self, historical_data: Dict[str, Any]):
        """Save historical data to parquet files."""
        try:
            timeframes = ['1m', '5m', '15m', '30m', '1h']
            
            for timeframe in timeframes:
                bars_key = f'bars_{timeframe}'
                if bars_key in historical_data:
                    bars = historical_data[bars_key]
                    
                    if not bars:
                        continue
                    
                    # Convert to DataFrame
                    converted_bars = []
                    for bar in bars:
                        try:
                            # Convert .NET ticks to Unix timestamp
                            # .NET ticks: 100ns intervals since Jan 1, 0001 AD
                            # Unix timestamp: seconds since Jan 1, 1970 AD
                            if bar['timestamp'] > 621355968000000000:  # .NET ticks for 1970
                                # Convert .NET ticks to Unix timestamp
                                unix_timestamp = (bar['timestamp'] - 621355968000000000) / 10000000
                                timestamp = datetime.fromtimestamp(unix_timestamp)
                            else:
                                # Assume Unix timestamp
                                timestamp = datetime.fromtimestamp(bar['timestamp'])
                        except (ValueError, OSError):
                            # Fallback to current time if timestamp conversion fails
                            timestamp = datetime.now()
                            self.logger.warning(f"Invalid timestamp {bar['timestamp']}, using current time")
                        
                        converted_bars.append({
                            'timestamp': timestamp,
                            'open': bar['open'],
                            'high': bar['high'],
                            'low': bar['low'],
                            'close': bar['close'],
                            'volume': bar['volume']
                        })
                    
                    df = pd.DataFrame(converted_bars)
                    
                    # Save to parquet
                    parquet_dir = self.data_dir / "parquet" / "historical"
                    parquet_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = parquet_dir / f"historical_{timeframe}.parquet"
                    df.to_parquet(filename, index=False)
                    
                    self.logger.info(f"Saved {len(bars)} {timeframe} bars to parquet")
            
        except Exception as e:
            self.logger.error(f"Error saving historical data to parquet: {e}")
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data to save space."""
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            
            # Clean up market data
            result = self.conn.execute("""
                DELETE FROM market_data 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            # Clean up old parquet files
            parquet_dir = self.data_dir / "parquet"
            if parquet_dir.exists():
                for file in parquet_dir.rglob("*.parquet"):
                    if file.stat().st_mtime < cutoff_date.timestamp():
                        file.unlink()
            
            self.logger.info(f"Cleaned up data older than {days} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")