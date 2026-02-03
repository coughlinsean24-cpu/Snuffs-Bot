#!/usr/bin/env python3
"""
Add Market Context

Quick command-line tool to add market context that the AI will learn from.
Use this when you know about major events the news APIs might miss or for real-time updates.

Usage:
    python add_context.py "War with Iran escalating, markets down"
    python add_context.py --type GEOPOLITICAL "Iran tensions causing sell-off"
    python add_context.py --type FED --sentiment -0.5 "Fed hawkish, rates staying higher"
    
Event Types:
    GEOPOLITICAL - War, sanctions, trade tensions
    FED - Federal Reserve, interest rates, FOMC
    EARNINGS - Company earnings reports
    ECONOMIC - GDP, jobs, CPI, economic data
    OTHER - General news

Sentiment:
    -1.0 = Very bearish
    -0.5 = Bearish
     0.0 = Neutral
    +0.5 = Bullish
    +1.0 = Very bullish
    
    If not specified, sentiment is auto-detected from your description.
"""

import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from snuffs_bot.local_ai.news_collector import NewsCollector


def main():
    parser = argparse.ArgumentParser(
        description="Add market context for AI learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "description",
        help="Description of the market event (e.g., 'Iran tensions causing sell-off')"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["GEOPOLITICAL", "FED", "EARNINGS", "ECONOMIC", "OTHER"],
        default="OTHER",
        help="Event type category"
    )
    parser.add_argument(
        "--sentiment", "-s",
        type=float,
        default=None,
        help="Sentiment score (-1.0 to +1.0). Auto-detected if not specified."
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show current market context after adding"
    )
    
    args = parser.parse_args()
    
    nc = NewsCollector()
    
    # Auto-detect sentiment if not provided
    if args.sentiment is None:
        sentiment, keywords = nc.analyze_sentiment(args.description)
        print(f"Auto-detected sentiment: {sentiment:+.2f} (keywords: {keywords})")
    else:
        sentiment = args.sentiment
    
    # Add the context
    nc.add_manual_context(
        event_type=args.type,
        description=args.description,
        sentiment=sentiment,
    )
    
    print(f"\n✅ Added context: {args.type}")
    print(f"   Description: {args.description}")
    print(f"   Sentiment: {sentiment:+.2f}")
    
    if args.show:
        context = nc.get_current_context()
        print(f"\n=== Current Market Context ===")
        print(f"Overall Sentiment: {context.overall_sentiment:+.2f}")
        print(f"War Tensions: {context.war_tensions}")
        print(f"Top Themes: {context.top_themes}")
        print(f"\n{context.context_summary}")


if __name__ == "__main__":
    main()
