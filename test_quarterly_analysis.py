"""
Test script for quarterly aggregation and Borda Count functions.
Tests the implementation without requiring trained models.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import the functions we want to test
from forecast import (
    aggregate_to_quarterly,
    borda_count_ranking,
    plot_yearly_top5,
    plot_quarterly_top5,
    plot_borda_process,
    run_quarterly_analysis,
    get_quarter,
    PLOTS_DIR_QUARTERLY
)


def generate_mock_forecast_data() -> pd.DataFrame:
    """
    Generate mock forecast data for testing.
    Creates 24 months of data (May 2025 - April 2027) with 10 products.
    """
    np.random.seed(42)
    
    products = [
        ("kursi auditorium ll 516 tb tg", "Auditorium Series"),
        ("kursi auditorium ll 526 tb", "Auditorium Series"),
        ("kursi auditorium ll 184", "Auditorium Series"),
        ("kursi kerja dp 301 tb", "Stacking Series"),
        ("kursi siswa ll 507 tc", "Kursi Siswa"),
        ("meja siswa ll 507 tc", "Meja Siswa"),
        ("kursi br 213 har mesh", "Secretary Series"),
        ("kursi stool st 02", "Stacking Series"),
        ("meja komputer cd 06", "Meja Komputer"),
        ("meja siswa ll 508 tc", "Meja Siswa"),
    ]
    
    # Generate 24 months starting from May 2025
    dates = pd.date_range(start="2025-05-01", periods=24, freq="MS")
    
    rows = []
    for product, category in products:
        # Base value varies by product
        base = np.random.uniform(100, 1000)
        
        for date in dates:
            # Add some variation
            trend = 1.0 + 0.02 * (date.month - 5)  # small trend
            seasonality = 1.0 + 0.1 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(1.0, 0.15)
            
            mean_val = base * trend * seasonality * noise
            
            rows.append({
                "product": product,
                "category": category,
                "date": date.strftime("%Y-%m-%d"),
                "mean": max(0.1, mean_val),
                "p10": max(0.1, mean_val * 0.8),
                "p50": max(0.1, mean_val),
                "p90": mean_val * 1.2
            })
    
    return pd.DataFrame(rows)


def test_get_quarter():
    """Test the quarter mapping function."""
    print("\n=== Testing get_quarter() ===")
    
    test_cases = [
        (1, "Q1"), (2, "Q1"), (3, "Q1"),
        (4, "Q2"), (5, "Q2"), (6, "Q2"),
        (7, "Q3"), (8, "Q3"), (9, "Q3"),
        (10, "Q4"), (11, "Q4"), (12, "Q4"),
    ]
    
    all_passed = True
    for month, expected in test_cases:
        result = get_quarter(month)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  Month {month:2d} -> {result} (expected {expected}) [{status}]")
    
    return all_passed


def test_aggregate_to_quarterly():
    """Test quarterly aggregation."""
    print("\n=== Testing aggregate_to_quarterly() ===")
    
    # Generate mock data
    df = generate_mock_forecast_data()
    print(f"Generated {len(df)} rows of mock data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Run aggregation
    result = aggregate_to_quarterly(df, top_n=5)
    
    # Verify structure
    assert isinstance(result, dict), "Result should be a dictionary"
    
    for year in result:
        print(f"\nYear {year}:")
        for quarter, qdf in result[year].items():
            print(f"  {quarter}: {len(qdf)} products")
            assert 'quarterly_sum' in qdf.columns, "Should have quarterly_sum column"
            assert 'rank' in qdf.columns, "Should have rank column"
            assert len(qdf) <= 5, "Should have at most 5 products"
    
    return True


def test_borda_count_ranking():
    """Test Borda Count ranking."""
    print("\n=== Testing borda_count_ranking() ===")
    
    # Generate mock data and quarterly rankings
    df = generate_mock_forecast_data()
    quarterly_data = aggregate_to_quarterly(df, top_n=5)
    
    # Test for each year
    for year in quarterly_data:
        print(f"\nTesting year {year}:")
        borda_df = borda_count_ranking(quarterly_data[year], year, top_n=5)
        
        # Verify structure
        required_cols = ['product', 'total_score', 'rank', 
                        'Q1_score', 'Q2_score', 'Q3_score', 'Q4_score']
        for col in required_cols:
            assert col in borda_df.columns, f"Missing column: {col}"
        
        # Verify score calculation
        for _, row in borda_df.iterrows():
            calculated_total = (row['Q1_score'] + row['Q2_score'] + 
                              row['Q3_score'] + row['Q4_score'])
            assert row['total_score'] == calculated_total, \
                f"Score mismatch for {row['product']}: {row['total_score']} != {calculated_total}"
        
        # Verify ranking order
        scores = borda_df['total_score'].tolist()
        assert scores == sorted(scores, reverse=True), "Should be sorted by score descending"
        
        print(f"  Top 3 products:")
        for _, row in borda_df.head(3).iterrows():
            print(f"    {row['product'][:30]}... Score: {row['total_score']}")
    
    return True


def test_visualizations():
    """Test visualization functions."""
    print("\n=== Testing Visualization Functions ===")
    
    # Generate mock data
    df = generate_mock_forecast_data()
    quarterly_data = aggregate_to_quarterly(df, top_n=5)
    
    # Create output directory
    os.makedirs(PLOTS_DIR_QUARTERLY, exist_ok=True)
    
    for year in quarterly_data:
        print(f"\nGenerating plots for year {year}:")
        
        # Get Borda ranking
        borda_df = borda_count_ranking(quarterly_data[year], year, top_n=5)
        
        # Test yearly plot
        try:
            yearly_path = plot_yearly_top5(borda_df, year, PLOTS_DIR_QUARTERLY, top_n=5)
            assert os.path.exists(yearly_path), f"Yearly plot not created: {yearly_path}"
            print(f"  Yearly plot: OK")
        except Exception as e:
            print(f"  Yearly plot: FAILED - {e}")
            return False
        
        # Test quarterly plot
        try:
            quarterly_path = plot_quarterly_top5(quarterly_data[year], year, PLOTS_DIR_QUARTERLY, top_n=5)
            assert os.path.exists(quarterly_path), f"Quarterly plot not created: {quarterly_path}"
            print(f"  Quarterly plot: OK")
        except Exception as e:
            print(f"  Quarterly plot: FAILED - {e}")
            return False
        
        # Test Borda process plot
        try:
            borda_path = plot_borda_process(borda_df, year, PLOTS_DIR_QUARTERLY, top_n=5)
            assert os.path.exists(borda_path), f"Borda plot not created: {borda_path}"
            print(f"  Borda process plot: OK")
        except Exception as e:
            print(f"  Borda process plot: FAILED - {e}")
            return False
    
    return True


def test_run_quarterly_analysis():
    """Test the complete analysis pipeline."""
    print("\n=== Testing run_quarterly_analysis() ===")
    
    # Generate mock data
    df = generate_mock_forecast_data()
    
    # Run complete analysis
    results = run_quarterly_analysis(df, top_n=5)
    
    # Verify outputs
    assert 'quarterly_csv' in results, "Should have quarterly_csv"
    assert 'yearly_csv' in results, "Should have yearly_csv"
    assert 'plots' in results, "Should have plots"
    
    # Check files exist
    for csv_path in results['quarterly_csv']:
        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
        print(f"  Created: {csv_path}")
    
    for csv_path in results['yearly_csv']:
        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
        print(f"  Created: {csv_path}")
    
    for plot_path in results['plots']:
        assert os.path.exists(plot_path), f"Plot not found: {plot_path}"
        print(f"  Created: {plot_path}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING QUARTERLY ANALYSIS TESTS")
    print("="*60)
    
    tests = [
        ("get_quarter", test_get_quarter),
        ("aggregate_to_quarterly", test_aggregate_to_quarterly),
        ("borda_count_ranking", test_borda_count_ranking),
        ("visualizations", test_visualizations),
        ("run_quarterly_analysis", test_run_quarterly_analysis),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"\nException in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "ERROR"
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"  {name}: {status}")
    
    all_passed = all(s == "PASS" for s in results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

