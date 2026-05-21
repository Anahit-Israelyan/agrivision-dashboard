"""Tests for the analytics mathematical module.

This module uses pytest to verify the correctness of yield estimation, CV 
calculation, revenue projection, and metric dictionary construction functions.
"""

import math
import pytest

from src.config import get_region, REGIONS
from src.analytics import (
    heads_per_square_meter,
    estimate_yield_tonnes_per_hectare,
    coefficient_of_variation,
    project_revenue,
    compute_field_metrics,
)


class TestHeadsPerSquareMeter:
    def test_normal_case(self):
        assert heads_per_square_meter(1000, 2.0) == pytest.approx(500.0)

    def test_zero_heads_returns_zero(self):
        assert heads_per_square_meter(0, 10.0) == pytest.approx(0.0)

    def test_zero_area_raises_value_error(self):
        with pytest.raises(ValueError):
            heads_per_square_meter(100, 0.0)

    def test_negative_area_raises_value_error(self):
        with pytest.raises(ValueError):
            heads_per_square_meter(100, -5.0)

    def test_negative_heads_raises_value_error(self):
        with pytest.raises(ValueError):
            heads_per_square_meter(-10, 10.0)


class TestEstimateYield:
    def test_realistic_armenian_case(self):
        region = get_region("armenia_ararat")
        yld = estimate_yield_tonnes_per_hectare(250000, 500.0, region)
        assert yld == pytest.approx(6.65, rel=0.01)

    def test_zero_heads_returns_zero_yield(self):
        region = get_region("armenia_ararat")
        yld = estimate_yield_tonnes_per_hectare(0, 500.0, region)
        assert yld == pytest.approx(0.0)

    def test_higher_tgw_produces_higher_yield(self):
        region_france = get_region("france_beauce")
        region_kansas = get_region("usa_kansas")
        
        yld_france = estimate_yield_tonnes_per_hectare(100000, 100.0, region_france)
        yld_kansas = estimate_yield_tonnes_per_hectare(100000, 100.0, region_kansas)
        
        assert yld_france > yld_kansas

    def test_invalid_area_raises_value_error(self):
        region = get_region("armenia_ararat")
        with pytest.raises(ValueError):
            estimate_yield_tonnes_per_hectare(1000, 0.0, region)


class TestCoefficientOfVariation:
    def test_uniform_list_returns_zero(self):
        assert coefficient_of_variation([100, 100, 100, 100]) == pytest.approx(0.0)

    def test_known_case_variation(self):
        assert coefficient_of_variation([10, 20]) == pytest.approx(33.3333, rel=0.01)

    def test_single_element_returns_zero(self):
        assert coefficient_of_variation([100]) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        assert coefficient_of_variation([]) == pytest.approx(0.0)

    def test_all_zeros_returns_zero(self):
        assert coefficient_of_variation([0, 0, 0]) == pytest.approx(0.0)

    def test_negative_count_raises_value_error(self):
        with pytest.raises(ValueError):
            coefficient_of_variation([10, -5, 20])

    def test_highly_variable_list_produces_high_cv(self):
        cv = coefficient_of_variation([10, 200, 5, 300])
        assert cv > 50.0


class TestProjectRevenue:
    def test_usd_revenue_calculation(self):
        region = get_region("armenia_ararat")
        revenue = project_revenue(5.0, 10.0, region, use_local_currency=False)
        assert revenue == pytest.approx(12000.0)

    def test_local_currency_revenue_calculation(self):
        region = get_region("armenia_ararat")
        revenue = project_revenue(5.0, 10.0, region, use_local_currency=True)
        assert revenue == pytest.approx(4680000.0)

    def test_zero_yield_returns_zero_revenue(self):
        region = get_region("armenia_ararat")
        revenue = project_revenue(0.0, 10.0, region)
        assert revenue == pytest.approx(0.0)

    def test_negative_yield_raises_value_error(self):
        region = get_region("armenia_ararat")
        with pytest.raises(ValueError):
            project_revenue(-5.0, 10.0, region)

    def test_zero_area_raises_value_error(self):
        region = get_region("armenia_ararat")
        with pytest.raises(ValueError):
            project_revenue(5.0, 0.0, region)


class TestComputeFieldMetrics:
    def test_returned_dict_exact_keys(self):
        region = get_region("armenia_ararat")
        metrics = compute_field_metrics(1000, 10.0, [10, 12, 10], region)
        expected_keys = {
            "num_heads", "area_m2", "area_ha", "heads_per_m2", 
            "yield_t_per_ha", "cv_percent", "revenue_usd", 
            "revenue_local", "currency_code", "region_name"
        }
        assert set(metrics.keys()) == expected_keys

    def test_area_ha_conversion(self):
        region = get_region("armenia_ararat")
        metrics = compute_field_metrics(1000, 500.0, [10, 12, 10], region)
        assert metrics["area_ha"] == pytest.approx(metrics["area_m2"] / 10000.0)

    def test_currency_code_matches_region(self):
        region = get_region("armenia_ararat")
        metrics = compute_field_metrics(1000, 10.0, [10, 12, 10], region)
        assert metrics["currency_code"] == region.currency_code

    def test_sane_values_for_realistic_inputs(self):
        region = get_region("armenia_ararat")
        metrics = compute_field_metrics(250000, 500.0, [100, 100], region)
        
        assert metrics["yield_t_per_ha"] > 0.0
        assert metrics["cv_percent"] == pytest.approx(0.0)
        assert metrics["revenue_usd"] > 0.0
        assert metrics["revenue_local"] > 0.0
