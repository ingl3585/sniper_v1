// NinjaTraderStrategy.cs
// Rewritten NinjaTrader strategy fully aligned with Python AI backend

using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Web.Script.Serialization;
using System.Threading;
using System.Linq;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using System.Globalization;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class NinjaTraderStrategy : Strategy
    {
        private TcpClient dataClient;
        private TcpClient signalClient;
        private Thread signalListener;
        private bool isConnected;
        private bool isRunning;
        private bool historicalSent;

        private List<double> price1m = new(), price5m = new(), price15m = new(), price30m = new(), price1h = new();
        private List<double> vol1m = new(), vol5m = new(), vol15m = new(), vol30m = new(), vol1h = new();
		private JavaScriptSerializer serializer = new JavaScriptSerializer();

        private int entryId = 0;
        private double sessionStartPnL = 0;
        private bool sessionInitialized = false;
        private int lastTradeCount = 0;

        // Hybrid Execution: Pending signal for tick execution
        private SignalMessage pendingSignal = null;
        private bool hasPendingSignal = false;
        private double currentTickPrice = 0.0;
        private DateTime lastSignalTime = DateTime.MinValue;

        // Real-time Tick Processing: Rate limiting and filtering
        private DateTime lastTickSent = DateTime.MinValue;
        private double lastSentPrice = 0.0;
        private double minPriceMove = 0.25; // Minimum ticks to trigger update
        private TimeSpan minTickInterval = TimeSpan.FromMilliseconds(100); // Max 10 updates/second
        private DateTime lastBarUpdate = DateTime.MinValue;

        private class SignalMessage
        {
            public int action { get; set; }
            public int position_size { get; set; }
            public double confidence { get; set; }
            public bool use_stop { get; set; }
            public double stop_price { get; set; }
            public bool use_target { get; set; }
            public double target_price { get; set; }
        }

        protected override void OnStateChange()
        {
            switch (State)
            {
                case State.SetDefaults:
                    Name = "ResearchStrategy";
                    Description = "Real-time tick-based AI strategy";
                    Calculate = Calculate.OnEachTick; // REAL-TIME: Process every tick
                    EntriesPerDirection = 10;
                    EntryHandling = EntryHandling.AllEntries;
                    break;

                case State.Configure:
                    AddDataSeries(BarsPeriodType.Minute, 15);
                    AddDataSeries(BarsPeriodType.Minute, 5);
                    AddDataSeries(BarsPeriodType.Minute, 1);
                    AddDataSeries(BarsPeriodType.Minute, 30);
                    AddDataSeries(BarsPeriodType.Minute, 60);
                    break;

                case State.Realtime:
                    ConnectSockets();
                    StartSignalThread();
                    InitSession();
                    break;

                case State.Terminated:
                    Disconnect();
                    break;
            }
        }

        private void InitSession()
        {
            if (!sessionInitialized)
            {
                sessionStartPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                sessionInitialized = true;
                Print($"Session start PnL: {sessionStartPnL:C}");
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || State != State.Realtime)
                return;

            if (!isConnected) return;

            // CRITICAL: Preserve ALL existing functionality on bar close
            if (IsFirstTickOfBar)
            {
                // === EXISTING BAR PROCESSING - COMPLETELY PRESERVED ===
                UpdateSeries();  // PRESERVED: Update all price/volume arrays
                
                if (!historicalSent)
                {
                    SendHistorical();  // PRESERVED: Send 10 days historical data
                    return;
                }
                
                SendLiveData();  // PRESERVED: Send complete market data
                // === END EXISTING BAR PROCESSING ===
                
                lastBarUpdate = DateTime.Now;
                // Bar close processing completed (log reduced)
            }
            else
            {
                // === NEW REAL-TIME TICK PROCESSING - ADDED ON TOP ===
                ProcessRealtimeTick();
                // === END NEW TICK PROCESSING ===
            }
        }

        // NEW METHOD: Handle real-time ticks (does NOT replace existing functionality)
        private void ProcessRealtimeTick()
        {
            // Update current tick price for immediate execution
            currentTickPrice = Close[0];
            
            // Execute pending signal if we have one (immediate execution)
            if (hasPendingSignal && pendingSignal != null)
            {
                ExecutePendingSignal();
                return;
            }
            
            // Send filtered tick data to Python for real-time decisions
            if (ShouldSendTickUpdate())
            {
                SendRealtimeTickData();
            }
        }

        // NEW METHOD: Intelligent filtering to avoid overwhelming Python
        private bool ShouldSendTickUpdate()
        {
            double currentPrice = Close[0];
            DateTime now = DateTime.Now;
            
            // Basic sanity check only - trust NinjaTrader data
            if (currentPrice <= 0 || double.IsNaN(currentPrice) || double.IsInfinity(currentPrice))
            {
                Print($"WARNING: Invalid price detected: {currentPrice}");
                return false;
            }
            
            // Rate limiting: Max 10 updates per second
            if (now - lastTickSent < minTickInterval)
                return false;
            
            // Price movement filtering: Only significant moves
            if (lastSentPrice > 0)
            {
                double priceMove = Math.Abs(currentPrice - lastSentPrice);
                if (priceMove < minPriceMove)
                    return false;
            }
            
            lastTickSent = now;
            lastSentPrice = currentPrice;
            return true;
        }

        private void UpdateSeries()
        {
            UpdateList(price1m, Close[0], 1000);
            UpdateList(vol1m, Volume[0], 1000);

            if (BarsArray.Length > 1 && BarsArray[1].Count > 0)
            {
                UpdateList(price15m, Closes[1][0], 300);
                UpdateList(vol15m, Volumes[1][0], 300);
            }

            if (BarsArray.Length > 2 && BarsArray[2].Count > 0)
            {
                UpdateList(price5m, Closes[2][0], 500);
                UpdateList(vol5m, Volumes[2][0], 500);
            }

            if (BarsArray.Length > 3 && BarsArray[3].Count > 0)
            {
                UpdateList(price1m, Closes[3][0], 1000);
                UpdateList(vol1m, Volumes[3][0], 1000);
            }

            if (BarsArray.Length > 4 && BarsArray[4].Count > 0)
            {
                UpdateList(price30m, Closes[4][0], 150);
                UpdateList(vol30m, Volumes[4][0], 150);
            }

            if (BarsArray.Length > 5 && BarsArray[5].Count > 0)
            {
                UpdateList(price1h, Closes[5][0], 100);
                UpdateList(vol1h, Volumes[5][0], 100);
            }
        }

        private void UpdateList(List<double> list, double val, int max)
        {
            list.Add(val);
            if (list.Count > max) list.RemoveAt(0);
        }

        private void ConnectSockets()
        {
            try
            {
                dataClient = new TcpClient("localhost", 5556);
                signalClient = new TcpClient("localhost", 5557);
                isConnected = true;
                Print("Connected to Python AI server");
            }
            catch (Exception ex)
            {
                Print($"Socket connection error: {ex.Message}");
            }
        }

        private void Disconnect()
        {
            isRunning = false;
            try
            {
                dataClient?.Close();
                signalClient?.Close();
                signalListener?.Join();
            }
            catch (Exception ex)
            {
                Print($"Disconnect error: {ex.Message}");
            }
        }

        private void SendHistorical()
        {
            if (historicalSent || BarsArray.Length < 6) return;

            int days = 10;
            var payload = new
            {
                type = "historical_data",
                bars_1m = ExtractBars(BarsArray[3], days * 1440),
                bars_5m = ExtractBars(BarsArray[2], days * 288),
                bars_15m = ExtractBars(BarsArray[1], days * 96),
                bars_30m = ExtractBars(BarsArray[4], days * 48),
                bars_1h = ExtractBars(BarsArray[5], days * 24),
                timestamp = DateTime.UtcNow.Ticks
            };

            SendJson(payload);
            historicalSent = true;
        }

        private List<object> ExtractBars(Bars bars, int count)
        {
            var list = new List<object>();
            int start = Math.Max(0, bars.Count - count);
            for (int i = start; i < bars.Count; i++)
            {
                list.Add(new
                {
                    timestamp = bars.GetTime(i).Ticks,
                    open = bars.GetOpen(i),
                    high = bars.GetHigh(i),
                    low = bars.GetLow(i),
                    close = bars.GetClose(i),
                    volume = bars.GetVolume(i)
                });
            }
            return list;
        }

        private void SendLiveData()
        {
            var md = new
            {
                type = "live_data",
                price_1m = price1m,
                price_5m = price5m,
                price_15m = price15m,
                price_30m = price30m,
                price_1h = price1h,
                volume_1m = vol1m,
                volume_5m = vol5m,
                volume_15m = vol15m,
                volume_30m = vol30m,
                volume_1h = vol1h,
                current_price = Close[0],
                current_tick_price = currentTickPrice > 0 ? currentTickPrice : Close[0],
                tick_timestamp = DateTime.UtcNow.Ticks,
                tick_volume = Volume[0],
                open_positions = Position.Quantity,
                account_balance = Account.Get(AccountItem.CashValue, Currency.UsDollar),
                buying_power = Account.Get(AccountItem.ExcessIntradayMargin, Currency.UsDollar),
                daily_pnl = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar) - sessionStartPnL,
                unrealized_pnl = Account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar),
                timestamp = new DateTimeOffset(Time[0]).ToUnixTimeSeconds()
            };
            SendJson(md);
        }

        // NEW METHOD: Send lightweight real-time tick data (does NOT replace SendLiveData)
        private void SendRealtimeTickData()
        {
            var tickData = new
            {
                type = "realtime_tick",
                current_price = Close[0],
                current_tick_price = currentTickPrice > 0 ? currentTickPrice : Close[0],
                tick_timestamp = DateTime.UtcNow.Ticks,
                tick_volume = Volume[0],
                open_positions = Position.Quantity,
                // Include minimal account info for real-time decisions
                account_balance = Account.Get(AccountItem.CashValue, Currency.UsDollar),
                daily_pnl = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar) - sessionStartPnL,
                unrealized_pnl = Account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar),
                timestamp = DateTime.UtcNow.Ticks
            };
            SendJson(tickData);
            
            // Tick sent (verbose logging disabled for performance)
        }

        private void SendJson(object obj)
        {
            if (!isConnected || dataClient?.Connected != true) return;
            string json = serializer.Serialize(obj);
            byte[] body = Encoding.UTF8.GetBytes(json);
            byte[] header = BitConverter.GetBytes(body.Length);

            var stream = dataClient.GetStream();
            stream.Write(header, 0, header.Length);
            stream.Write(body, 0, body.Length);
        }


        private void StartSignalThread()
        {
            if (isRunning) return;
            isRunning = true;
            signalListener = new Thread(ListenSignals) { IsBackground = true };
            signalListener.Start();
        }

        private void ListenSignals()
        {
            while (isRunning)
            {
                try
                {
                    if (signalClient?.Connected != true)
                    {
                        Thread.Sleep(1000);
                        continue;
                    }

                    byte[] header = new byte[4];
                    int read = signalClient.GetStream().Read(header, 0, 4);
                    if (read != 4) continue;
                    int length = BitConverter.ToInt32(header, 0);
                    byte[] msg = new byte[length];
                    signalClient.GetStream().Read(msg, 0, length);
                    string json = Encoding.UTF8.GetString(msg);
                    ProcessSignal(json);
                }
                catch (Exception ex)
                {
                    Print($"Signal error: {ex.Message}");
                    Thread.Sleep(1000);
                }
            }
        }

        private void ProcessSignal(string json)
        {
            SignalMessage signal = serializer.Deserialize<SignalMessage>(json);
            
            // HYBRID EXECUTION: Set pending signal for tick execution instead of immediate execution
            pendingSignal = signal;
            hasPendingSignal = true;
            lastSignalTime = DateTime.Now;
            
            Print($"SIGNAL: {(signal.action == 1 ? "BUY" : signal.action == 2 ? "SELL" : "CLOSE")} {signal.position_size} @ {signal.confidence:F2}");
        }

        // NEW: OnEachTick method for immediate signal execution
        protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
        {
            if (State != State.Realtime) return;
            
            // Update current tick price
            if (marketDataUpdate.MarketDataType == MarketDataType.Last)
            {
                currentTickPrice = marketDataUpdate.Price;
                
                // Execute pending signal if we have one
                if (hasPendingSignal && pendingSignal != null)
                {
                    ExecutePendingSignal();
                }
            }
        }

        private void ExecutePendingSignal()
        {
            var signal = pendingSignal;
            int action = signal.action;
            int size = signal.position_size;
            double confidence = signal.confidence;
            bool useStop = signal.use_stop;
            double stop = signal.stop_price;
            bool useTarget = signal.use_target;
            double target = signal.target_price;

            string tag = $"AI_{DateTime.Now:HHmmss}_{entryId++}";

            // Validate current price is reasonable (within 1% of signal price)
            double priceDiff = Math.Abs(currentTickPrice - signal.target_price) / currentTickPrice;
            if (priceDiff > 0.01 && signal.target_price > 0)
            {
                Print($"WARNING: Tick price {currentTickPrice} differs significantly from signal price {signal.target_price}");
            }

            if (action == 0)
            {
                ExitLong();
                ExitShort();
                Print("EMERGENCY CLOSE ALL");
            }
            else if (action == 1)
            {
                EnterLong(size, tag);
                if (useStop) SetStopLoss(tag, CalculationMode.Price, stop, false);
                if (useTarget) SetProfitTarget(tag, CalculationMode.Price, target);
                
                string stopStr = useStop ? $" | Stop: ${stop:F2}" : "";
                string targetStr = useTarget ? $" | Target: ${target:F2}" : "";
                Print($"BUY: {size}x @ ${currentTickPrice:F2}{stopStr}{targetStr}");
            }
            else if (action == 2)
            {
                EnterShort(size, tag);
                if (useStop) SetStopLoss(tag, CalculationMode.Price, stop, false);
                if (useTarget) SetProfitTarget(tag, CalculationMode.Price, target);
                
                string stopStr = useStop ? $" | Stop: ${stop:F2}" : "";
                string targetStr = useTarget ? $" | Target: ${target:F2}" : "";
                Print($"SELL: {size}x @ ${currentTickPrice:F2}{stopStr}{targetStr}");
            }

            // Clear pending signal
            hasPendingSignal = false;
            pendingSignal = null;
            
            var executionTime = DateTime.Now - lastSignalTime;
            Print($"Executed in {executionTime.TotalMilliseconds:F0}ms");
        }

        protected override void OnExecutionUpdate(Execution exec, string execId, double price, int qty,
            MarketPosition pos, string orderId, DateTime time)
        {
            if (exec.Order.Name.Contains("AI_"))
                CheckCompletedTrades();
        }

        private void CheckCompletedTrades()
        {
            if (SystemPerformance.AllTrades.Count <= lastTradeCount) return;
            for (int i = lastTradeCount; i < SystemPerformance.AllTrades.Count; i++)
            {
                Trade t = SystemPerformance.AllTrades[i];
                var payload = new
                {
                    type = "trade_completion",
                    pnl = t.ProfitCurrency,
                    entry_price = t.Entry.Price,
                    exit_price = t.Exit.Price,
                    size = t.Quantity,
                    exit_reason = t.Exit.Name.Contains("Stop") ? "stop_hit" : t.Exit.Name.Contains("Target") ? "target_hit" : "manual",
                    entry_time = t.Entry.Time.Ticks,
                    exit_time = t.Exit.Time.Ticks,
                    trade_duration_minutes = (t.Exit.Time - t.Entry.Time).TotalMinutes
                };
                SendJson(payload);
            }
            lastTradeCount = SystemPerformance.AllTrades.Count;
        }
    }
}
