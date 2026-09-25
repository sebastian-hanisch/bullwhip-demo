"""Zahlenformate mit deutschem Dezimalkomma und Tausenderpunkt (bw_format): Standardwerte und Grenzfälle."""
import bw_format as F


def test_fmt_num_defaults_to_one_digit_without_sign():
    assert F.fmt_num(1.25) == "1,2" and F.fmt_num(1.26) == "1,3" and F.fmt_num(-1.26) == "-1,3" and F.fmt_num(3.0) == "3,0"
    assert F.fmt_num(1.2345, 3) == "1,234" and F.fmt_num(0.0, 0) == "0" and F.fmt_num(7.5, 0) == "8"
    assert F.fmt_num(1.26, 1, True) == "+1,3" and F.fmt_num(-1.26, 1, True) == "-1,3" and F.fmt_num(0.0, 1, True) == "+0,0"
    assert F.fmt_num(5.0, digits=2) == "5,00" and F.fmt_num(5.0, signed=True) == "+5,0"


def test_fmt_level_shows_one_decimal_or_the_exact_short_number():
    assert F.fmt_level(0) == "0,0" and F.fmt_level(0.1) == "0,1" and F.fmt_level(0.25) == "0,25" and F.fmt_level(0.5) == "0,5" and F.fmt_level(1.0) == "1,0" and F.fmt_level(0.8) == "0,8"
    assert F.fmt_level(0.05) == "0,05" and F.fmt_level(0.15) == "0,15" and F.fmt_level(0.30000000000000004) == "0,3" and F.fmt_level(2) == "2,0"


def test_fmt_share_is_a_whole_percent_and_none_is_a_dash():
    assert F.fmt_share(0.0) == "0 %" and F.fmt_share(0.456) == "46 %" and F.fmt_share(1.0) == "100 %" and F.fmt_share(0.005) == "0 %" and F.fmt_share(None) == "–"


def test_fmt_pct_scales_by_one_hundred():
    assert F.fmt_pct(0.9512) == "95,1 %" and F.fmt_pct(0.9512, 2) == "95,12 %" and F.fmt_pct(0.05, 1, True) == "+5,0 %" and F.fmt_pct(-0.05, 0, True) == "-5 %" and F.fmt_pct(1.0, 0) == "100 %"


def test_fmt_band_is_signed_mean_and_unsigned_error():
    assert F.fmt_band(-15.7, 0.21) == "-15,7 ± 0,2" and F.fmt_band(6.94, 0.36) == "+6,9 ± 0,4" and F.fmt_band(6.94, 0.36, signed=False) == "6,9 ± 0,4"
    assert F.fmt_band(1.234, 0.056, 2) == "+1,23 ± 0,06" and F.fmt_band(1.234, 0.056, 2, False) == "1,23 ± 0,06" and F.fmt_band(-1.0, 0.5, 0) == "-1 ± 0"


def test_fmt_int_uses_the_thousands_point():
    assert F.fmt_int(0) == "0" and F.fmt_int(999) == "999" and F.fmt_int(1234567) == "1.234.567" and F.fmt_int(1234.5) == "1.234"
    assert F.fmt_int(12345, True) == "+12.345" and F.fmt_int(-12345, True) == "-12.345" and F.fmt_int(-0.4) == "-0"
