"""Scheduling integrity: process identity and one owner per scientific arm."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('budget_parallel', Path(__file__).parents[1]/'tools/run_history_budget_parallel.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class SchedulingIntegrity(unittest.TestCase):
    def test_linux_stat_with_parentheses_and_spaces(self):
        fields = ['S', '1', '44', '44'] + ['0']*15 + ['12345']
        stat = '123 (loader (queue)) ' + ' '.join(fields)
        self.assertEqual(p.parse_stat(stat), dict(state='S', ppid=1, pgid=44, sid=44, starttime=12345))

    def test_reused_pid_is_not_adopted(self):
        identity = dict(pid=12, starttime=100, uid=1, cwd='/science', command='python train.py')
        self.assertTrue(p.same_process(dict(identity, state='R'), identity))
        for current in [dict(identity, starttime=101, state='R'),
                        dict(identity, command='another task', state='R'),
                        dict(identity, state='Z'), dict(pid=12, alive=False)]:
            self.assertFalse(p.same_process(current, identity))

    def test_disjoint_authorized_assignments_and_claim_once(self):
        self.assertTrue(set(p.ASSIGNMENTS[0]).isdisjoint(p.ASSIGNMENTS[1]))
        self.assertEqual(set(p.ASSIGNMENTS[0]+p.ASSIGNMENTS[1]), {'h8-velocity','h8-geometry','h2-velocity','h2-geometry'})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p.claim(root, 'h8-geometry', 1, 'scheduler')
            before = (root/'h8-geometry_claim.json').read_bytes()
            with self.assertRaises(FileExistsError):
                p.claim(root, 'h8-geometry', 1, 'another')
            self.assertEqual((root/'h8-geometry_claim.json').read_bytes(), before)
            self.assertEqual(json.loads(before)['git_revision'], p.REVISION)
            with self.assertRaises(ValueError):
                p.claim(root, 'h8-velocity', 1, 'scheduler')

    def test_adopted_exit_requires_complete_artifacts_and_no_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); log = root/'train.log';log.write_text('normal training\n')
            with self.assertRaisesRegex(RuntimeError, 'without final'):
                p.adopted_completion(root, root, log)
            (root/'epoch_10.pth').touch();(root/'validation_epoch_10_full.json').touch()
            self.assertIsNone(p.adopted_completion(root, root, log)['returncode'])
            log.write_text('Traceback (most recent call last):\n')
            with self.assertRaisesRegex(RuntimeError, 'error'):
                p.adopted_completion(root, root, log)


if __name__ == '__main__':
    unittest.main()
