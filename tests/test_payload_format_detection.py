from __future__ import annotations

from runtime.jquants_cache_builder import _looks_like_csv_text, detect_payload_format


def test_csv_sniffer_survives_multibyte_boundary_cut() -> None:
    """A fixed-size sample that cuts a CP932 character in half must still be
    recognized as CSV text.

    Bulk monthly CSV files (e.g. short-sale-report) contain Japanese reporter
    names; when the sniffer sample boundary lands inside a multibyte character,
    strict decoding fails for every candidate encoding and the payload used to
    be misclassified as unknown binary, aborting the download.
    """

    header = b"DiscDate,Code,SSName\n"
    row = "2014-04-01,13010,野村アセットマネジメント\n".encode("cp932")
    body = header + row * 200
    for cut in range(len(body) - 1, len(header), -1):
        sample = body[:cut]
        try:
            sample.decode("cp932")
        except UnicodeDecodeError:
            break
    else:
        raise AssertionError("could not construct a boundary-cut sample")

    assert _looks_like_csv_text(sample) is True
    assert detect_payload_format(sample) == "csv_or_text"


def test_csv_sniffer_still_rejects_non_csv_payloads() -> None:
    assert _looks_like_csv_text(b"") is False
    assert _looks_like_csv_text(b"\x00\x01\x02\x03binarypayload") is False
    assert detect_payload_format(b"\x00\x01\x02\x03binarypayload") == "binary_unknown"
    assert detect_payload_format(b"\x1f\x8b\x08\x08rest-of-gzip") == "gzip"
