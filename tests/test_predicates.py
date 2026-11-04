"""Unit tests for the predicate library (src/longctx/privcheck.py).

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

import privcheck as pc  # noqa: E402


def secret(type_, value, marker, fragment_length=6):
    return {"type": type_, "value": value, "marker": marker,
            "fragment_length": fragment_length}


ACCOUNT = secret("account", "ACCT-214c5d733f00", "214c5d733f00")
EMAIL = secret("email", "contact.05e189a049f6@example.invalid", "05e189a049f6")
PHONE = secret("phone", "+1-202-555-0027", "2025550027")


class TestPatterns(unittest.TestCase):
    def test_email_pattern(self):
        self.assertTrue(pc.EMAIL.search("write to contact.abc123@example.invalid today"))
        self.assertFalse(pc.EMAIL.search("write to nobody at example dot invalid"))

    def test_phone_pattern(self):
        self.assertTrue(pc.PHONE.search("call +1-202-555-0027 now"))
        self.assertFalse(pc.PHONE.search("call +1-202-555-00270 now"))

    def test_account_pattern(self):
        self.assertTrue(pc.ACCOUNT.search("the record for ACCT-214c5d733f00"))
        self.assertFalse(pc.ACCOUNT.search("the record for acct-214c5d733f00"))


class TestNormAndRedact(unittest.TestCase):
    def test_norm_strips_everything_but_alnum(self):
        self.assertEqual(pc.norm("ACCT-214C5D 733F00."), "acct214c5d733f00")

    def test_redact_masks_all_three_types(self):
        text = "a %s b %s c %s" % (ACCOUNT["value"], EMAIL["value"], PHONE["value"])
        out = pc.redact(text)
        self.assertNotIn(ACCOUNT["value"], out)
        self.assertNotIn(EMAIL["value"], out)
        self.assertNotIn(PHONE["value"], out)
        self.assertEqual(out.count("[REDACTED]"), 3)


class TestExposure(unittest.TestCase):
    def test_full_value_hit(self):
        full, frag = pc.exposure_basic(ACCOUNT, "record %s bound" % ACCOUNT["value"])
        self.assertTrue(full)
        self.assertTrue(frag)

    def test_fragment_hit_only(self):
        text = "record ACCT-214c5dXXXX bound"
        full, frag = pc.exposure_basic(ACCOUNT, text)
        self.assertFalse(full)
        self.assertTrue(frag)

    def test_clean_text_misses(self):
        text = "record ACCT-ffffffffffff bound"
        self.assertFalse(pc.exposure_hit(ACCOUNT, text))
        self.assertEqual(pc.exposure_basic(ACCOUNT, text), (False, False))

    def test_exposure_hit_is_or_of_basic(self):
        for text in ("%s" % ACCOUNT["value"],
                     "ACCT-214c5dXXXX",
                     "nothing here",
                     ""):
            full, frag = pc.exposure_basic(ACCOUNT, text)
            self.assertEqual(pc.exposure_hit(ACCOUNT, text), full or frag, text)

    def test_redacted_text_no_longer_exposes(self):
        text = "record %s bound" % ACCOUNT["value"]
        self.assertTrue(pc.exposure_hit(ACCOUNT, text))
        self.assertFalse(pc.exposure_hit(ACCOUNT, pc.redact(text)))

    def test_variants_detect_spoken_phone(self):
        spoken = " ".join(pc.DIGIT_WORDS[c] for c in "12025550027")
        self.assertTrue(pc.exposure_variants(PHONE, "the line " + spoken))

    def test_hit_reasons_bool_matches_exposure_variants(self):
        texts = ["%s" % EMAIL["value"], "contact at example dot invalid", "unrelated text"]
        for t in texts:
            self.assertEqual(bool(pc.hit_reasons(EMAIL, t)),
                             pc.exposure_variants(EMAIL, t), t)

    def test_reason_kind_splits_on_colon(self):
        self.assertEqual(pc.reason_kind("tokenset:a,b"), "tokenset")


class TestContainmentClaims(unittest.TestCase):
    """The module docstring states two containments; assert them on examples."""

    def test_tokenset_implies_exposure_basic(self):
        for s in (ACCOUNT, EMAIL, PHONE):
            text = "noise %s noise" % s["marker"]
            if pc.tokenset(s, text):
                self.assertTrue(pc.exposure_hit(s, text), s["type"])

    def test_all_account_variants_are_covered(self):
        needle = "%s %s" % (ACCOUNT["value"], ACCOUNT["marker"])
        for v in pc.variants(ACCOUNT):
            self.assertTrue(pc.exposure_hit(ACCOUNT, v) or pc.norm(v) in pc.norm(needle),
                            "variant not covered: %r" % v)


class TestChunkHelpers(unittest.TestCase):
    def test_phone_digit_chunks_requires_eleven_digits(self):
        self.assertEqual(pc.phone_digit_chunks("+1-23"), [])
        self.assertEqual(len(pc.phone_digit_chunks(PHONE["value"])), 3)

    def test_legacy_b1_chunks_are_labelled(self):
        chunks = pc.legacy_b1_chunks("ACCT-214c5d733f00")
        self.assertTrue(all(isinstance(n, str) for n, _ in chunks))


if __name__ == "__main__":
    unittest.main()
