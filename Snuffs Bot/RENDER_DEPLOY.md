# Deploying Snuffs Bot to Render

## Quick Deploy

### Option 1: Blueprint (Recommended)

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **New** → **Blueprint**
4. Connect your GitHub repo
5. Render will detect the `render.yaml` and create the services
6. Set your environment variables (see below)

### Option 2: Manual Setup

1. **Create a Background Worker** for the trading bot:
   - Runtime: Docker
   - Dockerfile Path: `./docker/Dockerfile`
   - Plan: Starter ($7/month)
   - Add a 1GB disk mounted at `/app/data`

2. **Create a Web Service** for the dashboard (optional):
   - Runtime: Docker
   - Dockerfile Path: `./docker/Dockerfile`
   - Docker Command: `python -m snuffs_bot.main dashboard`
   - Plan: Starter ($7/month)

---

## Environment Variables

Set these in the Render dashboard under **Environment**:

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `TASTYTRADE_USERNAME` | Your Tastytrade login email | `user@email.com` |
| `TASTYTRADE_PASSWORD` | Your Tastytrade password | `your-password` |
| `DEFAULT_ACCOUNT` | Your account number | `5WV12345` |

### Trading Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TASTYTRADE_ENVIRONMENT` | `sandbox` | `sandbox` for paper, `production` for live |
| `PAPER_TRADING` | `true` | Enable paper trading mode |
| `LIVE_TRADING_ENABLED` | `false` | **Keep false unless you're ready for real money** |
| `STARTING_CAPITAL` | `1000.0` | Paper trading starting balance |
| `RISK_PER_TRADE_PERCENT` | `0.04` | Risk 4% per trade |
| `MAX_DAILY_LOSS` | `100.0` | Stop trading after $100 daily loss |

### System

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DATA_DIR` | `/app/data` | Data storage path (use disk mount) |

---

## Persistent Storage

The bot stores data in SQLite databases. To persist data between deployments:

1. Add a **Disk** to your worker service
2. Mount path: `/app/data`
3. Size: 1 GB is sufficient

This preserves:
- Trade history
- ML model learnings
- Market snapshots

---

## Monitoring

### View Logs
- Go to your service → **Logs** tab
- Look for:
  - `Trading engine started [PAPER_ONLY]` - Bot is running
  - `[TASTYTRADE] SPY: $XXX.XX` - Market data streaming
  - `[VIX FALLBACK] Yahoo Finance: XX.XX` - VIX data working

### Health Indicators
- SPY price updates every ~30 seconds
- VIX updates every ~60 seconds
- No repeated ERROR messages

---

## Cost Estimate

| Service | Plan | Cost |
|---------|------|------|
| Trading Bot (Worker) | Starter | $7/month |
| Dashboard (Web) | Starter | $7/month |
| Disk Storage | 1 GB | Included |
| **Total** | | **$7-14/month** |

---

## Troubleshooting

### Bot not starting
- Check logs for authentication errors
- Verify `TASTYTRADE_USERNAME` and `TASTYTRADE_PASSWORD` are correct

### No trades executing
- Market must be open (9:30 AM - 4:00 PM ET)
- Check that `PAPER_TRADING=true`
- Verify VIX and SPY data are streaming

### Data not persisting
- Ensure disk is mounted at `/app/data`
- Check `DATA_DIR=/app/data` is set

---

## Going Live (Real Money)

**Only do this when you're confident in the bot's performance:**

1. Change `TASTYTRADE_ENVIRONMENT` to `production`
2. Set `LIVE_TRADING_ENABLED=true`
3. Set `PAPER_TRADING=false`
4. Start with small `STARTING_CAPITAL`
5. Monitor closely for the first few days
