"""Unit tests for the quadrant router and the alias map (src/longctx/v4_router.py).

Stdlib only, no model, no data files:

    python -m unittest discover -s tests -v
    python -m pytest tests -q            # if pytest is installed
"""

import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src", "longctx"))

import v4_router as V  # noqa: E402
import privcheck as pc  # noqa: E402

VALUES = ("214c5d733f00", "33cc189a695d", "4933bdbb584f",
          "5ee052b431dc", "9a6ffa50f962", "fc668719b347")


def canaries(values=VALUES, type_="account"):
    return [{"type": type_, "value": "ACCT-%s" % v, "marker": v, "fragment_length": 6}
            for v in values]


class TestAliaserStyles(unittest.TestCase):
    def test_delete_style_is_type_agnostic(self):
        al = V.Aliaser("delete")
        for c in canaries():
            self.assertEqual(al.token(c), "[REDACTED]")

    def test_type_style_drops_identity(self):
        al = V.Aliaser("type")
        tokens = {al.token(c) for c in canaries()}
        self.assertEqual(tokens, {"[ACCOUNT]"})

    def test_alias_style_is_one_token_per_value(self):
        al = V.Aliaser("alias")
        tokens = [al.token(c) for c in canaries()]
        self.assertEqual(len(set(tokens)), len(VALUES))
        self.assertEqual(len({pc.norm(t) for t in tokens}), len(VALUES))

    def test_alias_typed_carries_type_and_identity(self):
        al = V.Aliaser("alias_typed")
        self.assertTrue(al.token(canaries()[0]).startswith("[ACCOUNT:"))

    def test_token_is_stable_for_one_value(self):
        al = V.Aliaser("alias")
        c = canaries()[0]
        self.assertEqual(al.token(c), al.token(c))
        self.assertEqual(al.token(c), al.token(dict(c)))


class TestRankLeak(unittest.TestCase):
    """The property the keyed map exists to remove."""

    def test_unkeyed_assignment_follows_call_order(self):
        al = V.Aliaser("alias")
        letters = [al.token(c) for c in canaries()]
        self.assertEqual(letters, ["[ACCOUNT-%s]" % chr(65 + i) for i in range(len(VALUES))])

    def test_unkeyed_preseed_is_a_noop(self):
        al = V.Aliaser("alias")
        al.preseed(canaries())
        self.assertEqual(al.map, {})

    def test_keyed_assignment_is_not_value_order(self):
        al = V.Aliaser("alias", order_key="demo-key-2026")
        al.preseed(canaries())
        by_value = [al.token(c) for c in sorted(canaries(), key=lambda c: c["value"])]
        identity = ["[ACCOUNT-%s]" % chr(65 + i) for i in range(len(VALUES))]
        self.assertNotEqual(by_value, identity, "keyed map still leaks the rank")
        self.assertEqual(sorted(by_value), sorted(identity), "must be a permutation")

    def test_keyed_assignment_is_deterministic(self):
        a = V.Aliaser("alias", order_key="k")
        b = V.Aliaser("alias", order_key="k")
        a.preseed(canaries())
        b.preseed(canaries())
        self.assertEqual([a.token(c) for c in canaries()],
                         [b.token(c) for c in canaries()])

    def test_different_keys_give_different_maps(self):
        maps = []
        for key in ("k1", "k2", "k3"):
            al = V.Aliaser("alias", order_key=key)
            al.preseed(canaries())
            maps.append(tuple(al.token(c) for c in canaries()))
        self.assertGreater(len(set(maps)), 1)

    def test_environment_key_is_picked_up(self):
        old = os.environ.get("PT_ALIAS_KEY")
        try:
            os.environ["PT_ALIAS_KEY"] = "from-env"
            self.assertEqual(V.Aliaser("alias").order_key, "from-env")
            del os.environ["PT_ALIAS_KEY"]
            self.assertIsNone(V.Aliaser("alias").order_key)
        finally:
            if old is None:
                os.environ.pop("PT_ALIAS_KEY", None)
            else:
                os.environ["PT_ALIAS_KEY"] = old


class TestApplyReplacements(unittest.TestCase):
    def test_value_and_fragment_are_replaced(self):
        c = canaries()[0]
        text = "line %s and a bare fragment %s here" % (c["value"], c["marker"][:6])
        al = V.Aliaser("alias")
        out = V.apply_replacements(text, [c], al)
        self.assertNotIn(c["value"], out)
        self.assertNotIn(c["marker"][:6], out)
        self.assertIn("[ACCOUNT-A]", out)

    def test_no_canary_value_survives(self):
        cs = canaries()
        text = " ; ".join("%s | %s" % (c["value"], c["marker"]) for c in cs)
        out = V.apply_replacements(text, cs, V.Aliaser("alias"))
        for c in cs:
            self.assertNotIn(c["value"], out)
            self.assertFalse(pc.exposure_hit(c, out), c["value"])

    def test_question_and_context_can_share_one_map(self):
        c = canaries()[0]
        al = V.Aliaser("alias", order_key="shared")
        al.preseed(canaries())
        ctx = V.apply_replacements("record %s" % c["value"], [c], al)
        q = V.apply_replacements("which device is bound to %s?" % c["value"], [c], al)
        tok = al.token(c)
        self.assertIn(tok, ctx)
        self.assertIn(tok, q)


class TestUnits(unittest.TestCase):
    def test_split_units_on_newlines_and_sentences(self):
        text = "alpha line.\nbravo line.\n\ncharlie line."
        units = V.split_units(text)
        self.assertEqual(len(units), 3)
        self.assertEqual(units[0], "alpha line.")

    def test_split_units_never_returns_empty(self):
        for text in ("", "x", "\n\n"):
            self.assertTrue(V.split_units(text))

    def test_gold_texts_accepts_str_and_dict(self):
        self.assertEqual(V.gold_texts({"answers": ["a", {"text": "b"}]}), ["a", "b"])
        self.assertEqual(V.gold_texts({}), [])


if __name__ == "__main__":
    unittest.main()
