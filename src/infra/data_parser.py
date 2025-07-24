"""
Data Parser for NinjaTrader Messages
Parses incoming JSON messages from NinjaTrader strategy.
"""
from typing import Dict, Any, List
from datetime import datetime
from logging_config import get_logger
from src.infra.message_types import MarketData, TradeCompletion


class DataParser:
    """Parses messages from NinjaTrader strategy."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.set_context(component='data_parser')
    
    def parse_market_data(self, message: Dict[str, Any]) -> MarketData:
        """Parse live market data message from NinjaTrader."""
        try:
            return MarketData(
                # Price data
                price_1m=message.get('price_1m', []),
                price_5m=message.get('price_5m', []),
                price_15m=message.get('price_15m', []),
                price_30m=message.get('price_30m', []),
                price_1h=message.get('price_1h', []),
                
                # Volume data
                volume_1m=message.get('volume_1m', []),
                volume_5m=message.get('volume_5m', []),
                volume_15m=message.get('volume_15m', []),
                volume_30m=message.get('volume_30m', []),
                volume_1h=message.get('volume_1h', []),
                
                # Account info
                account_balance=float(message.get('account_balance', 0)),
                buying_power=float(message.get('buying_power', 0)),
                daily_pnl=float(message.get('daily_pnl', 0)),
                unrealized_pnl=float(message.get('unrealized_pnl', 0)),
                open_positions=int(message.get('open_positions', 0)),
                
                # Current data
                current_price=float(message.get('current_price', 0)),
                timestamp=int(message.get('timestamp', 0)),
                
                # Tick data
                current_tick_price=float(message.get('current_tick_price', 0)),
                tick_timestamp=int(message.get('tick_timestamp', 0)),
                tick_volume=float(message.get('tick_volume', 0)),
                
                # Volatility defaults
                volatility_1m=0.0,
                volatility_5m=0.0,
                volatility_15m=0.0,
                volatility_30m=0.0,
                volatility_1h=0.0,
                volatility=0.0,
                volatility_regime='medium',
                volatility_percentile=0.5,
                volatility_breakout=None
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing market data: {e}")
            raise
    
    def parse_trade_completion(self, message: Dict[str, Any]) -> TradeCompletion:
        """Parse trade completion message from NinjaTrader."""
        try:
            return TradeCompletion(
                order_id=message.get('order_id', ''),
                symbol=message.get('symbol', 'MNQ'),
                quantity=int(message.get('size', 0)),
                price=float(message.get('exit_price', 0)),
                pnl=float(message.get('pnl', 0)),
                timestamp=int(message.get('exit_time', 0))
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing trade completion: {e}")
            raise
    
    def parse_realtime_tick(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Parse real-time tick data."""
        try:
            return {
                'current_tick_price': float(message.get('current_tick_price', 0)),
                'tick_timestamp': int(message.get('tick_timestamp', 0)),
                'tick_volume': float(message.get('tick_volume', 0)),
                'open_positions': int(message.get('open_positions', 0)),
                'account_balance': float(message.get('account_balance', 0)),
                'daily_pnl': float(message.get('daily_pnl', 0)),
                'unrealized_pnl': float(message.get('unrealized_pnl', 0)),
                'tick_age_seconds': self._calculate_tick_age(message.get('tick_timestamp', 0))
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing realtime tick: {e}")
            return {}
    
    def parse_historical_data(self, message: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Parse historical data message from NinjaTrader."""
        try:
            historical_data = {}
            
            # Parse each timeframe
            timeframes = ['1m', '5m', '15m', '30m', '1h']
            for tf in timeframes:
                bars_key = f'bars_{tf}'
                if bars_key in message:
                    historical_data[bars_key] = self._parse_bars(message[bars_key])
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error parsing historical data: {e}")
            return {}
    
    def _parse_bars(self, bars_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse bar data from NinjaTrader format."""
        parsed_bars = []
        
        for bar in bars_data:
            try:
                parsed_bar = {
                    'timestamp': int(bar.get('timestamp', 0)),
                    'open': float(bar.get('open', 0)),
                    'high': float(bar.get('high', 0)),
                    'low': float(bar.get('low', 0)),
                    'close': float(bar.get('close', 0)),
                    'volume': float(bar.get('volume', 0))
                }
                parsed_bars.append(parsed_bar)
                
            except Exception as e:
                self.logger.warning(f"Error parsing bar: {e}")
                continue
        
        return parsed_bars
    
    def _calculate_tick_age(self, tick_timestamp: int) -> float:
        """Calculate age of tick data in seconds."""
        if tick_timestamp == 0:
            return 0.0
        
        try:
            # Convert .NET ticks to Python timestamp
            # .NET ticks are 100-nanosecond intervals since 1/1/0001
            # Unix timestamp is seconds since 1/1/1970
            dotnet_epoch = 621355968000000000  # .NET ticks for Unix epoch
            unix_timestamp = (tick_timestamp - dotnet_epoch) / 10000000.0
            
            current_time = datetime.now().timestamp()
            age = current_time - unix_timestamp
            
            return max(0.0, age)
            
        except Exception as e:
            self.logger.warning(f"Error calculating tick age: {e}")
            return 0.0