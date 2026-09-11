"""Coin-counter and fuel-gauge readers: declared fields, state gates, unknown stays unknown."""

import hashlib

import cv2
import numpy as np
import pytest

from gradientclimb.perception.fields import (
    COIN_FIELD_ROI,
    FUEL_GAUGE_BOX,
    FUEL_ICON_BOX,
    CoinCounterReader,
    FuelGaugeReader,
    crop_field,
)
from gradientclimb.perception.hud import HUDDigitReader

WIDTH, HEIGHT = 1034, 581


def digit_bank():
    reader = HUDDigitReader(expected_size=(WIDTH, HEIGHT), roi=COIN_FIELD_ROI, alignment_pixels=1)
    for digit in "0123456789":
        frame = np.full((HEIGHT, WIDTH, 3), 30, np.uint8)
        cv2.putText(frame, digit, (140, 118), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        reader.add_labeled_region(
            frame,
            digit,
            COIN_FIELD_ROI,
            source_sha256=hashlib.sha256(digit.encode()).hexdigest(),
            annotation_id=f"synthetic-{digit}",
        )
    return reader


def playing_frame(coins="16245", fill=0.4):
    frame = np.full((HEIGHT, WIDTH, 3), (120, 180, 230), np.uint8)
    cv2.putText(frame, coins, (125, 118), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    left, top, right, bottom = FUEL_GAUGE_BOX
    cv2.rectangle(frame, (left - 4, top - 3), (right + 3, bottom + 2), (20, 20, 20), -1)
    frame[top:bottom, left:right] = (60, 60, 60)
    filled = round((right - left) * fill)
    if filled:
        frame[top:bottom, left : left + filled] = (240, 220, 30)
    il, it, ir, ib = FUEL_ICON_BOX
    frame[it + 4 : ib - 4, il + 6 : ir - 6] = (220, 30, 30)
    return frame


def test_coin_reader_reads_declared_field_only_when_playing_confirmed():
    reader = CoinCounterReader(digit_bank())
    frame = playing_frame("16245")
    unconfirmed = reader.read(frame)
    assert not unconfirmed["valid"] and unconfirmed["coins"] is None
    assert reader.field == "hud_coin_counter"
    result = reader.read(frame, playing_state_confirmed=True)
    assert result["valid"] and result["coins"] == 16245
    assert "lifetime" in result["semantics"]
    assert reader.read(frame, playing_state_confirmed=1)["valid"] is False
    with pytest.raises(ValueError, match="1034x581"):
        CoinCounterReader(HUDDigitReader(expected_size=(600, 240), roi=(0, 0, 1, 1)))


def test_coin_reader_unknown_on_missing_or_ambiguous_digits():
    reader = CoinCounterReader(digit_bank())
    blank = playing_frame("16245")
    crop = crop_field(blank, COIN_FIELD_ROI)
    assert crop.shape[0] > 10 and crop.shape[1] > 10
    blank[80:130, 115:255] = (120, 180, 230)
    result = reader.read(blank, playing_state_confirmed=True)
    assert not result["valid"] and result["coins"] is None
    assert "segmentation" in result["reason"] or "Unrecognized" in result["reason"]


@pytest.mark.parametrize("fill", [0.0, 0.25, 0.7, 1.0])
def test_fuel_gauge_reads_fill_fraction_from_origin(fill):
    gauge = FuelGaugeReader()
    result = gauge.read(playing_frame(fill=fill), playing_state_confirmed=True)
    assert result["valid"], result
    assert result["fill_fraction"] == pytest.approx(fill, abs=0.02)
    assert result["field"] == "hud_fuel_gauge" and "not fuel units" in result["semantics"]
    assert not gauge.read(playing_frame(fill=fill))["valid"]


def test_fuel_gauge_unknown_without_icon_frame_or_contiguous_fill():
    gauge = FuelGaugeReader()
    no_icon = playing_frame(fill=0.5)
    il, it, ir, ib = FUEL_ICON_BOX
    no_icon[it:ib, il:ir] = (120, 180, 230)
    assert "icon" in gauge.read(no_icon, playing_state_confirmed=True)["reason"]
    no_frame = playing_frame(fill=0.5)
    left, top, right, bottom = FUEL_GAUGE_BOX
    no_frame[top:bottom, right - 2 : right + 4] = (120, 180, 230)
    assert "edges" in gauge.read(no_frame, playing_state_confirmed=True)["reason"]
    gap = playing_frame(fill=0.6)
    gap[top:bottom, left + 20 : left + 30] = (60, 60, 60)
    assert "contiguous" in gauge.read(gap, playing_state_confirmed=True)["reason"]
    floating = playing_frame(fill=0.0)
    floating[top:bottom, left + 40 : left + 70] = (240, 220, 30)
    assert "origin" in gauge.read(floating, playing_state_confirmed=True)["reason"]
    assert not gauge.read(playing_frame()[:300], playing_state_confirmed=True)["valid"]


def test_fuel_gauge_constructor_bounds():
    with pytest.raises(ValueError, match="inside the frame"):
        FuelGaugeReader(box=(0, 0, 2000, 10))
    with pytest.raises(ValueError, match="positive integer"):
        FuelGaugeReader(minimum_rows=0)
    with pytest.raises(ValueError, match="0..255"):
        FuelGaugeReader(saturation=300)
