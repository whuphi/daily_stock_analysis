# -*- coding: utf-8 -*-
"""Regression tests for A-share volume unit normalization (股 vs 手)."""

import sys
from types import SimpleNamespace

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import is_star_market_code, lots_to_shares
from data_provider.efinance_fetcher import EfinanceFetcher


class _DummyCircuitBreaker:
    def is_available(self, source: str) -> bool:
        return True

    def record_success(self, source: str) -> None:
        pass

    def record_failure(self, source: str, error=None) -> None:
        pass


def _history_frame(**columns) -> pd.DataFrame:
    base = {
        "日期": ["2026-05-25", "2026-05-26"],
        "开盘": [10.0, 10.2],
        "收盘": [10.2, 10.4],
        "最高": [10.5, 10.6],
        "最低": [9.8, 10.0],
    }
    base.update(columns)
    return pd.DataFrame(base)


def test_base_volume_helpers_normalize_lots_and_star_market_with_prefixes() -> None:
    assert is_star_market_code("sh688111")
    assert is_star_market_code("688111.SH")
    assert is_star_market_code("689009")
    assert not is_star_market_code("600000.SH")
    assert lots_to_shares(1000, "600000") == 100000.0
    assert lots_to_shares("1000", "sh600000") == 100000.0
    assert lots_to_shares(1000, "688111") == 1000.0
    assert lots_to_shares(1000, "688111.SH") == 1000.0
    assert lots_to_shares("n/a", "600000") == "n/a"
    assert lots_to_shares(None, "600000") is None


def test_akshare_eastmoney_history_volume_is_converted_to_shares() -> None:
    fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)
    em_df = _history_frame(成交量=[1000, 2000], volume=[1000, 2000])
    fetcher._fetch_stock_data_em = lambda *args: em_df
    fetcher._fetch_stock_data_sina = lambda *args: pd.DataFrame()
    fetcher._fetch_stock_data_tx = lambda *args: pd.DataFrame()

    result = fetcher._fetch_stock_data("600519", "2026-05-01", "2026-05-26")

    assert result["成交量"].tolist() == [100000.0, 200000.0]
    assert result["volume"].tolist() == [100000.0, 200000.0]


def test_akshare_tencent_history_volume_uses_lots_except_star_market() -> None:
    fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)
    tx_df = _history_frame(成交量=[1000, 2000])
    fetcher._fetch_stock_data_em = lambda *args: pd.DataFrame()
    fetcher._fetch_stock_data_sina = lambda *args: pd.DataFrame()
    fetcher._fetch_stock_data_tx = lambda *args: tx_df

    normal = fetcher._fetch_stock_data("600519", "2026-05-01", "2026-05-26")
    assert normal["成交量"].tolist() == [100000.0, 200000.0]

    star = fetcher._fetch_stock_data("688111", "2026-05-01", "2026-05-26")
    assert star["成交量"].tolist() == [1000.0, 2000.0]


def test_akshare_sina_history_volume_is_not_multiplied(monkeypatch) -> None:
    fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)
    sina_df = _history_frame(成交量=[1000, 2000])
    fetcher._fetch_stock_data_em = lambda *args: pd.DataFrame()
    fetcher._fetch_stock_data_sina = lambda *args: sina_df
    fetcher._fetch_stock_data_tx = lambda *args: pd.DataFrame()

    result = fetcher._fetch_stock_data("600519", "2026-05-01", "2026-05-26")

    assert result["成交量"].tolist() == [1000, 2000]


def test_akshare_eastmoney_realtime_volume_is_converted_to_shares(monkeypatch) -> None:
    spot_df = pd.DataFrame(
        [
            {
                "代码": "600519",
                "名称": "贵州茅台",
                "最新价": 1500.0,
                "涨跌幅": 1.5,
                "成交量": 1000,
                "成交额": 1500000000.0,
            }
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "akshare",
        SimpleNamespace(stock_zh_a_spot_em=lambda: spot_df),
    )
    monkeypatch.setattr(
        "data_provider.akshare_fetcher.get_realtime_circuit_breaker",
        lambda: _DummyCircuitBreaker(),
    )
    monkeypatch.setattr(
        "data_provider.akshare_fetcher._realtime_cache",
        {"data": None, "timestamp": 0, "ttl": 1200},
    )

    fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)
    monkeypatch.setattr(fetcher, "_enforce_rate_limit", lambda: None)
    monkeypatch.setattr(fetcher, "_set_random_user_agent", lambda: None)

    quote = fetcher._get_stock_realtime_quote_em("600519")

    assert quote is not None
    assert quote.volume == 100000


def test_efinance_normalize_converts_a_share_history_volume() -> None:
    fetcher = EfinanceFetcher(sleep_min=0, sleep_max=0)
    raw = pd.DataFrame(
        {
            "日期": ["2026-05-25"],
            "开盘": [10.0],
            "收盘": [10.2],
            "最高": [10.5],
            "最低": [9.8],
            "成交量": [1000],
            "成交额": [1000000],
        }
    )

    result = fetcher._normalize_data(raw, "600519")

    assert result["volume"].tolist() == [100000]


def test_efinance_normalize_leaves_etf_volume_unchanged() -> None:
    fetcher = EfinanceFetcher(sleep_min=0, sleep_max=0)
    raw = pd.DataFrame(
        {
            "日期": ["2026-05-25"],
            "开盘": [10.0],
            "收盘": [10.2],
            "最高": [10.5],
            "最低": [9.8],
            "成交量": [1000],
            "成交额": [1000000],
        }
    )

    result = fetcher._normalize_data(raw, "512400")

    assert result["volume"].tolist() == [1000]
