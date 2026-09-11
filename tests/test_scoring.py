"""Synthetic field-isolation and bounded glyph-matching tests; no real accuracy claim."""

import hashlib

import cv2
import numpy as np

from gradientclimb.perception.hud import HUDDigitReader
from gradientclimb.perception.scoring import PausedDistanceReader, ResultDistanceReader, _white_text


def fixture():
    image = np.full((240, 600, 3), 30, np.uint8)
    reader = HUDDigitReader(
        expected_size=(600, 240), roi=(0.65, 0.35, 0.99, 0.6), alignment_pixels=1
    )
    for digit in "0123456789":
        template = image.copy()
        cv2.putText(template, digit, (400, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        reader.add_labeled_region(
            template,
            digit,
            reader.roi,
            source_sha256=hashlib.sha256(digit.encode()).hexdigest(),
            annotation_id=digit,
        )
    cv2.putText(image, "313", (400, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, "DISTANCE:", (270, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    roi = (260 / 600, 100 / 240, 385 / 600, 140 / 240)
    anchor = _white_text(image[100:140, 260:385])
    return image, ResultDistanceReader(reader, anchor, roi)


def test_result_confirmation_and_right_field_isolation():
    image, reader = fixture()
    assert not reader.read(image)["valid"]
    cv2.putText(image, "BEST 9999", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    result = reader.read(image, result_state_confirmed=True)
    assert result["valid"] and result["distance_meters"] == 313
    image[100:140, 260:385] = 30
    assert not reader.read(image, result_state_confirmed=True)["valid"]
    assert not reader.read(image[:100], result_state_confirmed=True)["valid"]


def test_bounded_alignment_preserves_no_shift_legacy_and_cache_growth():
    glyph = np.zeros((32, 20), np.uint8)
    glyph[8:22, 6:9] = 255
    displaced = np.zeros_like(glyph)
    displaced[9:23, 7:10] = 255
    reader = HUDDigitReader(expected_size=(600, 240), roi=(0, 0, 1, 1), alignment_pixels=1)
    reader.glyphs.append({"digit": "1", "pixels": glyph})
    assert reader.glyph_scores(displaced)["1"] == 1
    reader.glyphs.append({"digit": "2", "pixels": displaced})
    assert set(reader.glyph_scores(displaced)) == {"1", "2"}
    strict = HUDDigitReader(expected_size=(600, 240), roi=(0, 0, 1, 1))
    assert strict.glyph_scores(displaced) == {}
    strict.glyphs = reader.glyphs[:1]
    assert strict.glyph_scores(displaced)["1"] < 1


def test_reader_manifest_records_alignment_and_roundtrips(tmp_path):
    image, result_reader = fixture()
    path = result_reader.reader.save(tmp_path / "glyphs")
    restored = HUDDigitReader.from_manifest(path)
    assert restored.alignment_pixels == 1
    assert restored.read(image)["hud_displayed_progress_meters"] == 313


def test_paused_reader_excludes_bright_modal_and_requires_explicit_state():
    reader = HUDDigitReader(
        expected_size=(1034, 581), roi=(300 / 1034, 68 / 581, 470 / 1034, 110 / 581)
    )
    blank = np.full((581, 1034, 3), 40, np.uint8)
    for digit in "0123456789":
        image = blank.copy()
        cv2.putText(image, digit, (350, 101), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        reader.add_labeled_region(
            image,
            digit,
            reader.roi,
            source_sha256=hashlib.sha256(digit.encode()).hexdigest(),
            annotation_id=digit,
        )
    image = blank.copy()
    cv2.putText(image, "49", (350, 101), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    image = np.uint8(image * 0.3)
    cv2.rectangle(image, (419, 74), (681, 407), (255, 255, 255), 3)
    paused = PausedDistanceReader(reader)
    assert not paused.read(image)["valid"]
    assert not paused.read(image, paused_state_confirmed=1)["valid"]
    result = paused.read(image, paused_state_confirmed=True)
    assert result["valid"] and result["distance_meters"] == 49
    assert result["score_boundary"] == "verified_paused_frame"
