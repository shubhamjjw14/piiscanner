"""Nested path flattening -- the part of the Mongo connector that needs no server."""

from __future__ import annotations

import pytest
from bson import ObjectId

from piiscanner.connectors.base import ReadOnlyViolation
from piiscanner.connectors.mongodb import (
    assert_read_only_pipeline,
    flatten_document,
    infer_bson_type,
)


def test_flattens_nested_objects_into_dotted_paths() -> None:
    doc = {"user": {"contact": {"phone": "9876543210", "email": "a@b.com"}}}
    assert flatten_document(doc) == {
        "user.contact.phone": ["9876543210"],
        "user.contact.email": ["a@b.com"],
    }


def test_flattens_three_levels_deep() -> None:
    doc = {"account": {"owner": {"identity": {"aadhaar": "234567890123"}}}}
    assert flatten_document(doc) == {
        "account.owner.identity.aadhaar": ["234567890123"]
    }


def test_array_indices_are_merged_into_one_path() -> None:
    """The important one: two orders are one field, not two schemas."""
    doc = {
        "orders": [
            {"shipping": {"pincode": "400001"}},
            {"shipping": {"pincode": "560001"}},
            {"shipping": {"pincode": "110001"}},
        ]
    }
    assert flatten_document(doc) == {
        "orders[].shipping.pincode": ["400001", "560001", "110001"]
    }


def test_arrays_of_scalars() -> None:
    doc = {"tags": ["a", "b"], "phones": ["9876543210", "9123456780"]}
    assert flatten_document(doc) == {
        "tags[]": ["a", "b"],
        "phones[]": ["9876543210", "9123456780"],
    }


def test_nested_arrays() -> None:
    doc = {"a": [[{"b": 1}], [{"b": 2}]]}
    assert flatten_document(doc) == {"a[][].b": [1, 2]}


def test_pii_inside_an_array_of_objects_three_levels_deep() -> None:
    doc = {
        "customer": {
            "addresses": [
                {"contact": {"mobile": "9876543210"}},
                {"contact": {"mobile": "9123456780"}},
            ]
        }
    }
    assert flatten_document(doc) == {
        "customer.addresses[].contact.mobile": ["9876543210", "9123456780"]
    }


def test_empty_containers_yield_no_paths() -> None:
    assert flatten_document({"a": {}, "b": [], "c": {"d": {}}}) == {}


def test_scalar_and_null_handling() -> None:
    doc = {"a": None, "b": 0, "c": "", "d": False}
    flat = flatten_document(doc)
    assert flat == {"a": [None], "b": [0], "c": [""], "d": [False]}


def test_depth_limit_stops_runaway_nesting() -> None:
    doc: dict = {"v": "leaf"}
    for _ in range(50):
        doc = {"n": doc}
    flat = flatten_document(doc, max_depth=5)
    assert all(p.count(".") <= 5 for p in flat)


def test_long_arrays_are_capped() -> None:
    doc = {"items": [{"v": i} for i in range(500)]}
    flat = flatten_document(doc)
    assert len(flat["items[].v"]) == 20


def test_documents_with_differing_shapes_merge_by_path() -> None:
    """Sparse fields: not every document has every path."""
    docs = [
        {"user": {"email": "a@b.com"}},
        {"user": {"email": "c@d.com", "phone": "9876543210"}},
        {"user": {}},
    ]
    flattened = [flatten_document(d) for d in docs]
    assert flattened[0] == {"user.email": ["a@b.com"]}
    assert set(flattened[1]) == {"user.email", "user.phone"}
    assert flattened[2] == {}


@pytest.mark.parametrize(
    "values,expected",
    [
        (["a", "b"], "str"),
        ([1, 2, 3], "int"),
        ([None, None], "null"),
        ([ObjectId(), ObjectId()], "ObjectId"),
    ],
)
def test_infer_bson_type(values: list, expected: str) -> None:
    assert infer_bson_type(values) == expected


def test_infer_bson_type_reports_mixed_columns() -> None:
    assert infer_bson_type(["a", 1, 2]).startswith("mixed(")


# --------------------------------------------------------------------------
# Read-only enforcement
# --------------------------------------------------------------------------


def test_read_only_pipeline_accepts_sampling() -> None:
    assert_read_only_pipeline([{"$sample": {"size": 500}}, {"$limit": 10}])


@pytest.mark.parametrize(
    "stage",
    [
        {"$out": "copy"},
        {"$merge": {"into": "copy"}},
        {"$merge": {"into": {"db": "x", "coll": "y"}}},
    ],
)
def test_read_only_pipeline_rejects_write_stages(stage: dict) -> None:
    """`$out` and `$merge` are the only aggregation stages that can write."""
    with pytest.raises(ReadOnlyViolation):
        assert_read_only_pipeline([{"$sample": {"size": 10}}, stage])
