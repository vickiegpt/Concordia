import importlib.util
from pathlib import Path
import unittest


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


alias = module('patch-xm-isa-alias')
profile = module('profile-jobd-phases')


class DiagnosticsTests(unittest.TestCase):
    def test_only_existing_matrix_aliases_change(self):
        for sep in (b'_', b'\0'):
            original = sep.join([b'v', b'xlxvfnrclipxfqf', *alias.ALIASES]) + b'\0'
            expected = sep.join([b'v', b'xlxvfnrclipxfqf', *alias.ALIASES.values()]) + b'\0'
            self.assertEqual(alias.normalize(original, sep), expected)
            self.assertEqual(len(original), len(expected))

    def test_missing_duplicate_and_mixed_aliases_fail(self):
        original = b'_'.join(alias.ALIASES) + b'\0'
        for invalid in (original[:-1], original.replace(b'xlxmmbase_', b''),
                        b'xlxmmbase_'+original, b'xsfmmbase_'+original):
            with self.assertRaises(ValueError):
                alias.normalize(invalid, b'_')

    def test_nearest_rank_does_not_hide_small_sample_max(self):
        self.assertEqual(profile.percentile([5]*7+[4195], .99), 4195)
        self.assertEqual(profile.percentile(list(range(1, 101)), .99), 99)

    def test_expected_count_is_a_gate_without_index_gaps(self):
        self.assertFalse(profile.valid_capture(0, False, 7, {}, [], 8, True))
        self.assertTrue(profile.valid_capture(0, False, 8, {}, [], 8, True))

    def test_timeout_and_wrong_identity_fail_closed(self):
        self.assertFalse(profile.valid_capture(0, True, 8, {}, [], 8, True))
        self.assertFalse(profile.valid_capture(0, False, 8, {}, [], 8, False))


if __name__ == '__main__':
    unittest.main()
