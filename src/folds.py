"""Chronological walk-forward fold construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_year: int
    estimation: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex


def make_walk_forward_folds(
    index: pd.DatetimeIndex,
    estimation_days: int,
    validation_days: int,
    test_days: int,
    first_test_year: int,
    last_test_year: int,
) -> list[Fold]:
    index = pd.DatetimeIndex(index).sort_values().unique()
    folds: list[Fold] = []
    for year in range(first_test_year, last_test_year + 1):
        test = index[index.year == year][:test_days]
        if len(test) < test_days:
            continue
        before_test = index[index < test[0]]
        validation = before_test[-validation_days:]
        before_validation = before_test[before_test < validation[0]]
        estimation = before_validation[-estimation_days:]
        if len(estimation) == estimation_days and len(validation) == validation_days:
            folds.append(Fold(year, estimation, validation, test))
    return folds
