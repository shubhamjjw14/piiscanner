"""How MySQL identifiers are quoted."""

from __future__ import annotations

import pytest


class TestQuotingIdentifiers:
    """Names are quoted the way MySQL documents, not filtered to a safe few."""

    @pytest.mark.parametrize("name, quoted", [
        ("users", "`users`"),
        ("Database Version", "`Database Version`"),     # AdventureWorks
        ("issued-invoices", "`issued-invoices`"),       # a Czech accounting schema
        ("příjmení", "`příjmení`"),
        ("2B", "`2B`"),                                 # baseball doubles
    ])
    def test_real_names_are_quoted(self, name, quoted):
        from piiscanner.connectors.mysql import quote_identifier

        assert quote_identifier(name) == quoted

    def test_a_backtick_cannot_end_the_quoting(self):
        from piiscanner.connectors.mysql import quote_identifier

        # Written twice, so it stays part of the one name.
        assert quote_identifier("a` ; DROP TABLE users; --") == \
            "`a`` ; DROP TABLE users; --`"

    @pytest.mark.parametrize("name", ["", "bad\x00name", "line\nbreak", "x" * 65])
    def test_control_characters_and_overlong_names_are_refused(self, name):
        from piiscanner.connectors.mysql import quote_identifier

        with pytest.raises(ValueError):
            quote_identifier(name)
