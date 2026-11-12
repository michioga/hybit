import unittest

from benchmarks.scripts.P3A_gate import EXPECTED, EXPECTED_SCALE, parse


def record(name, expected, observed, residual="1e-14"):
    return (f"P3A_CASE|name={name}|expected={expected}|observed={observed}|"
            f"relative_residual={residual}|relative_x_error=1e-15|factor_bytes={64 if observed == 'accept' else 0}|"
            f"mathematical_spd={'true' if expected == 'accept' else 'false'}|scale={EXPECTED_SCALE[name]}\n")


def good_data():
    return "".join(record(name, expected, expected) for name, expected in EXPECTED.items()) + \
        "P3A_SUMMARY|cases=7|contract_discrepancies=0\n"


class P3AGateTests(unittest.TestCase):
    def test_good_all_mathematically_correct(self):
        self.assertEqual(len(parse(good_data())), 7)

    def test_discrepancy_is_not_hidden(self):
        text = good_data().replace(
            record("spd_small_1e16", "accept", "accept"),
            record("spd_small_1e16", "accept", "reject"),
        ).replace("contract_discrepancies=0", "contract_discrepancies=1")
        self.assertEqual(parse(text)["spd_small_1e16"]["contract_discrepancy"], 1)

    def test_rejects_nonfinite_residual(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            parse(good_data().replace("relative_residual=1e-14", "relative_residual=nan", 1))

    def test_rejects_mislabeled_scale(self):
        with self.assertRaisesRegex(ValueError, "declared configuration"):
            parse(good_data().replace("scale=1e-16", "scale=1e-12", 1))

    def test_rejects_wrong_summary(self):
        with self.assertRaisesRegex(ValueError, "summary"):
            parse(good_data().replace("contract_discrepancies=0", "contract_discrepancies=2"))

    def test_rejects_duplicate_case(self):
        duplicate = record("spd_unit", "accept", "accept")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse(good_data().replace("P3A_SUMMARY", duplicate + "P3A_SUMMARY"))


if __name__ == "__main__":
    unittest.main()
