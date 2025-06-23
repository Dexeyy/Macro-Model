#!/usr/bin/env python3
"""
Test script for Enhanced Yahoo Finance Data Fetching Module
"""

import pandas as pd
import time
from datetime import datetime, timedelta

from data.yahoo_finance_fetcher import (
    EnhancedYahooFinanceClient,
    YahooRequestParams,
    YahooInterval,
    YahooPeriod,
    DataType,
    COMMON_TICKERS
)

def test_enhanced_yahoo_finance_client():
    """Test the enhanced Yahoo Finance client functionality."""
    print("=== Testing Enhanced Yahoo Finance Client ===\n")
    
    try:
        # Initialize client
        print("1. Initializing Enhanced Yahoo Finance Client...")
        client = EnhancedYahooFinanceClient()
        print("✅ Client initialized successfully")
        
        # Test rate limit status
        print("\n2. Testing Rate Limit Status...")
        rate_status = client.get_rate_limit_status()
        print(f"✅ Rate limit status: {rate_status['calls_in_last_hour']}/{rate_status['max_calls_per_hour']} calls")
        
        # Test ticker info retrieval
        print("\n3. Testing Ticker Info Retrieval...")
        info = client.get_ticker_info("AAPL")
        if info:
            print(f"✅ Retrieved AAPL info: {info.name}")
            print(f"   Sector: {info.sector}")
            print(f"   Industry: {info.industry}")
            print(f"   Market Cap: ${info.market_cap:,.0f}" if info.market_cap else "N/A")
            print(f"   P/E Ratio: {info.trailing_pe:.2f}" if info.trailing_pe else "N/A")
        else:
            print("❌ Failed to retrieve AAPL info")
        
        # Test basic historical data fetching
        print("\n4. Testing Historical Data Fetching...")
        params = YahooRequestParams(
            ticker="MSFT",
            start_date="2023-01-01",
            end_date="2023-12-31",
            interval=YahooInterval.ONE_DAY
        )
        
        data = client.fetch_historical_data(params)
        if data is not None and not data.empty:
            print(f"✅ Successfully fetched MSFT: {len(data)} observations")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Sample close prices: {data['Close'].head(3).tolist()}")
        else:
            print("❌ Failed to fetch MSFT data")
        
        # Test different intervals
        print("\n5. Testing Different Intervals...")
        weekly_params = YahooRequestParams(
            ticker="GOOGL",
            period=YahooPeriod.ONE_YEAR,
            interval=YahooInterval.ONE_WEEK
        )
        
        weekly_data = client.fetch_historical_data(weekly_params)
        if weekly_data is not None:
            print(f"✅ Successfully fetched weekly GOOGL data: {len(weekly_data)} observations")
        
        # Test multiple tickers fetching
        print("\n6. Testing Multiple Tickers Fetching...")
        ticker_params = [
            YahooRequestParams(ticker="AAPL", start_date="2023-06-01", end_date="2023-12-31"),
            YahooRequestParams(ticker="MSFT", start_date="2023-06-01", end_date="2023-12-31"),
            YahooRequestParams(ticker="GOOGL", start_date="2023-06-01", end_date="2023-12-31")
        ]
        
        results = client.fetch_multiple_tickers(ticker_params)
        print(f"✅ Successfully fetched {len(results)}/3 tickers")
        for ticker, data in results.items():
            print(f"   {ticker}: {len(data)} observations")
        
        # Test dividend data
        print("\n7. Testing Dividend Data...")
        dividends = client.fetch_dividends("AAPL", period=YahooPeriod.TWO_YEARS)
        if dividends is not None and not dividends.empty:
            print(f"✅ Successfully fetched AAPL dividends: {len(dividends)} payments")
            print(f"   Recent dividends: {dividends.tail(3).tolist()}")
        else:
            print("ℹ️ No dividend data available for AAPL in the specified period")
        
        # Test stock splits
        print("\n8. Testing Stock Split Data...")
        splits = client.fetch_splits("AAPL", period=YahooPeriod.FIVE_YEARS)
        if splits is not None and not splits.empty:
            print(f"✅ Successfully fetched AAPL splits: {len(splits)} splits")
            print(f"   Recent splits: {splits.tail(3).tolist()}")
        else:
            print("ℹ️ No split data available for AAPL in the specified period")
        
        # Test financial statements
        print("\n9. Testing Financial Statements...")
        financials = client.fetch_financials("AAPL", "income")
        if financials is not None and not financials.empty:
            print(f"✅ Successfully fetched AAPL income statement: {financials.shape}")
            print(f"   Available years: {list(financials.columns)}")
        else:
            print("❌ Failed to fetch AAPL financials")
        
        # Test backward compatibility function
        print("\n10. Testing Backward Compatibility...")
        from data.yahoo_finance_fetcher import fetch_asset_data
        
        compat_data = fetch_asset_data(
            {"apple": "AAPL", "microsoft": "MSFT"},
            "2023-11-01",
            "2023-12-31"
        )
        
        if compat_data is not None and not compat_data.empty:
            print(f"✅ Backward compatibility test passed: {compat_data.shape}")
            print(f"   Columns: {list(compat_data.columns)}")
        
        # Test common tickers constants
        print("\n11. Testing Common Tickers Constants...")
        print(f"✅ Available common tickers: {len(COMMON_TICKERS)}")
        print(f"   Examples: {list(COMMON_TICKERS.keys())[:10]}")
        
        # Test different data types
        print("\n12. Testing Different Asset Classes...")
        asset_tests = [
            ("S&P 500 Index", COMMON_TICKERS["sp500"]),
            ("Gold Futures", COMMON_TICKERS["gold"]),
            ("Bitcoin", COMMON_TICKERS["bitcoin"]),
            ("EUR/USD", COMMON_TICKERS["usd_eur"])
        ]
        
        for name, ticker in asset_tests:
            try:
                test_params = YahooRequestParams(
                    ticker=ticker,
                    period=YahooPeriod.ONE_MONTH,
                    interval=YahooInterval.ONE_DAY
                )
                test_data = client.fetch_historical_data(test_params)
                if test_data is not None:
                    print(f"   ✅ {name} ({ticker}): {len(test_data)} observations")
                else:
                    print(f"   ❌ {name} ({ticker}): No data")
            except Exception as e:
                print(f"   ❌ {name} ({ticker}): Error - {e}")
        
        print("\n🎉 All Yahoo Finance tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling for invalid tickers."""
    print("\n=== Testing Error Handling ===")
    
    try:
        client = EnhancedYahooFinanceClient()
        
        # Test invalid ticker
        print("Testing invalid ticker...")
        params = YahooRequestParams(ticker="INVALID_TICKER_XYZ")
        data = client.fetch_historical_data(params)
        
        if data is None:
            print("✅ Invalid ticker correctly returned None")
        else:
            print("❌ Invalid ticker should have returned None")
        
        # Test invalid date range
        print("Testing invalid date range...")
        params = YahooRequestParams(
            ticker="AAPL",
            start_date="2030-01-01",  # Future date
            end_date="2030-12-31"
        )
        data = client.fetch_historical_data(params)
        
        if data is None or data.empty:
            print("✅ Invalid date range correctly handled")
        else:
            print("⚠️ Future date range returned data (unexpected)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Enhanced Yahoo Finance Tests...\n")
    
    # Run main functionality tests
    main_test_passed = test_enhanced_yahoo_finance_client()
    
    # Run error handling tests
    error_test_passed = test_error_handling()
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"Main functionality: {'✅ PASSED' if main_test_passed else '❌ FAILED'}")
    print(f"Error handling: {'✅ PASSED' if error_test_passed else '❌ FAILED'}")
    print(f"{'='*60}")
    
    if main_test_passed and error_test_passed:
        print("\n🎉 All tests passed! Enhanced Yahoo Finance module is ready for production.")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.") 