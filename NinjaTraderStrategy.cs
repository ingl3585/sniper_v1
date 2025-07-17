// NinjaTraderStrategy.cs
// Rewritten NinjaTrader strategy fully aligned with Python AI backend

using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Web.Script.Serialization;
using System.Threading;
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

        private List<double> price1m = new(), price5m = new(), price15m = new(), price1h = new();
        private List<double> vol1m = new(), vol5m = new(), vol15m = new(), vol1h = new();
		private JavaScriptSerializer serializer = new JavaScriptSerializer();

        private int entryId = 0;
        private double sessionStartPnL = 0;
        private bool sessionInitialized = false;
        private int lastTradeCount = 0;

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
                    Description = "Rewritten AI strategy";
                    Calculate = Calculate.OnBarClose;
                    EntriesPerDirection = 10;
                    EntryHandling = EntryHandling.AllEntries;
                    break;

                case State.Configure:
                    AddDataSeries(BarsPeriodType.Minute, 15);
                    AddDataSeries(BarsPeriodType.Minute, 5);
                    AddDataSeries(BarsPeriodType.Minute, 1);
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

            UpdateSeries();
            if (!isConnected) return;

            if (!historicalSent)
            {
                SendHistorical();
                return;
            }

            SendLiveData();
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

            if (BarsArray.Length > 4 && BarsArray[4].Count > 0)
            {
                UpdateList(price1h, Closes[4][0], 100);
                UpdateList(vol1h, Volumes[4][0], 100);
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
            if (historicalSent || BarsArray.Length < 4) return;

            int days = 10;
            var payload = new
            {
                type = "historical_data",
                bars_1m = ExtractBars(BarsArray[3], days * 1440),
                bars_5m = ExtractBars(BarsArray[2], days * 288),
                bars_15m = ExtractBars(BarsArray[1], days * 96),
                bars_1h = ExtractBars(BarsArray[4], days * 24),
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
                price_1h = price1h,
                volume_1m = vol1m,
                volume_5m = vol5m,
                volume_15m = vol15m,
                volume_1h = vol1h,
                current_price = Close[0],
                open_positions = Position.Quantity,
                account_balance = Account.Get(AccountItem.CashValue, Currency.UsDollar),
                buying_power = Account.Get(AccountItem.ExcessIntradayMargin, Currency.UsDollar),
                daily_pnl = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar) - sessionStartPnL,
                unrealized_pnl = Account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar),
                volatility = CalculateVolatility(),
                timestamp = new DateTimeOffset(Time[0]).ToUnixTimeSeconds()
            };
            SendJson(md);
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

        private double CalculateVolatility()
        {
            if (price1m.Count < 20) return 0.02;
            double sum = 0;
            for (int i = 1; i < price1m.Count; i++)
                sum += Math.Abs(price1m[i] - price1m[i - 1]) / price1m[i - 1];
            return sum / (price1m.Count - 1);
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
            int action = signal.action;
            int size = signal.position_size;
            double confidence = signal.confidence;
            bool useStop = signal.use_stop;
            double stop = signal.stop_price;
            bool useTarget = signal.use_target;
            double target = signal.target_price;

            string tag = $"AI_{DateTime.Now:HHmmss}_{entryId++}";

            if (action == 0)
            {
                ExitLong();
                ExitShort();
                Print("EMERGENCY CLOSE_ALL received");
            }
            else if (action == 1)
            {
                EnterLong(size, tag);
                if (useStop) SetStopLoss(tag, CalculationMode.Price, stop, false);
                if (useTarget) SetProfitTarget(tag, CalculationMode.Price, target);
            }
            else if (action == 2)
            {
                EnterShort(size, tag);
                if (useStop) SetStopLoss(tag, CalculationMode.Price, stop, false);
                if (useTarget) SetProfitTarget(tag, CalculationMode.Price, target);
            }

            Print($"Signal processed: Action={action}, Size={size}, Confidence={confidence:F2}");
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
