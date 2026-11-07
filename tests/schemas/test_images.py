import pytest

from gpustack.schemas.images import (
    ImageRequestParams,
    compute_image_usage,
    finished_image_indexes,
    parse_image_size,
)
from gpustack.schemas.model_usage import USAGE_QUANTITY_MAX


@pytest.mark.parametrize(
    "size, expected",
    [
        ("1024x1024", (1024, 1024)),
        ("1792X1024", (1792, 1024)),
        (" 512 x 768 ", (512, 768)),
        ("512*512", (512, 512)),
        ("auto", None),
        ("", None),
        (None, None),
        ("0x1024", None),
        ("1024", None),
        ("-1x5", None),
        ("1024x1024x3", None),
    ],
)
def test_parse_image_size(size, expected):
    assert parse_image_size(size) == expected


@pytest.mark.parametrize(
    "params, expected",
    [
        (None, ImageRequestParams(n=1, size=None)),
        ({}, ImageRequestParams(n=1, size=None)),
        ({"n": 4, "size": "512x512"}, ImageRequestParams(n=4, size="512x512")),
        # Multipart forms carry every field as a string.
        ({"n": "2", "size": "256x256"}, ImageRequestParams(n=2, size="256x256")),
        ({"n": "many"}, ImageRequestParams(n=1)),
        ({"n": 0}, ImageRequestParams(n=1)),
        ({"n": -3}, ImageRequestParams(n=1)),
        ({"size": 1024}, ImageRequestParams(n=1, size=None)),
    ],
)
def test_image_request_params_from_mapping(params, expected):
    assert ImageRequestParams.from_mapping(params) == expected


def test_usage_uses_requested_n_without_a_response_count():
    params = ImageRequestParams(n=3, size="1024x1024")

    assert compute_image_usage(params) == (3, 3 * 1024 * 1024)


def test_returned_image_count_takes_precedence_over_requested_n():
    params = ImageRequestParams(n=4, size="512x512")

    assert compute_image_usage(params, generated_count=2) == (2, 2 * 512 * 512)


def test_symbolic_size_counts_images_but_not_pixels():
    params = ImageRequestParams(n=2, size="auto")

    assert compute_image_usage(params) == (2, 0)


def test_missing_params_default_to_one_image():
    assert compute_image_usage(None) == (1, 0)


def test_client_supplied_n_cannot_overflow_the_usage_columns():
    params = ImageRequestParams(n=10**30, size="999999x999999")

    assert compute_image_usage(params) == (USAGE_QUANTITY_MAX, USAGE_QUANTITY_MAX)


@pytest.mark.parametrize(
    "chunk, expected",
    [
        ({"data": [{"index": 0, "finish_reason": "stop"}]}, [0]),
        ({"data": [{"index": 1, "finish_reason": None}]}, []),
        (
            {
                "data": [
                    {"index": 0, "finish_reason": "stop"},
                    {"index": 2, "finish_reason": "stop"},
                ]
            },
            [0, 2],
        ),
        ({"data": "nope"}, []),
        ({}, []),
    ],
)
def test_finished_image_indexes(chunk, expected):
    assert finished_image_indexes(chunk) == expected


def test_empty_response_records_no_images():
    params = ImageRequestParams(n=2, size="512x512")

    assert compute_image_usage(params, generated_count=0) == (0, 0)
