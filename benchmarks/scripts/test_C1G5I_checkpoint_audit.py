"""C1-G5I evidence checkpoint safety and correctness regression tests."""
from __future__ import annotations
import io
from pathlib import Path
import tempfile
import unittest
import C1G5I_checkpoint as c

class Tests(unittest.TestCase):
    def test_exact_scoped_paths(self):
        self.assertEqual(len(c.TRACKED),4)
        self.assertEqual(len(c.NEW),14)
        self.assertEqual(len(c.ALL),18)
        self.assertEqual(len(set(c.ALL)),18)
        self.assertFalse(any(s.startswith('crates/hybit-auto/') for s in c.ALL))
        self.assertFalse(any(s.startswith('crates/hybit-precond/') for s in c.ALL))

    def test_source_pins(self):
        for name,pin in c.SOURCE_PIN.items():
            self.assertEqual(c.sha((c.PKG/name).read_bytes()),pin)

    def test_registry_preserves_historical_order(self):
        old=b'MANIFEST.txt\nSOURCE_SHA256.txt\nz.rs\na.rs\n'
        led=(f'{"1"*64}  z.rs\n'+f'{"2"*64}  a.rs\n').encode()
        m,order,h=c.parse_registry(old,led)
        self.assertEqual(m,['MANIFEST.txt','SOURCE_SHA256.txt','z.rs','a.rs'])
        self.assertEqual(order,['z.rs','a.rs'])
        baseline={'MANIFEST.txt':old,'SOURCE_SHA256.txt':led,
                  'docs/SESSION_HANDOFF.md':b'head\n','docs/SUMMARY.md':b'summary\n'}
        additions={p:(p+'\n').encode() for p in c.NEW}
        out=c.build_plan(baseline,additions)
        self.assertEqual(out['MANIFEST.txt'].splitlines()[2:4],[b'z.rs',b'a.rs'])
        self.assertIn(f'{"1"*64}  z.rs\n'.encode(),out['SOURCE_SHA256.txt'])
        self.assertEqual(len(out),18)
        self.assertIn(b'production',out['docs/SESSION_HANDOFF.md'].lower())

    def test_registry_rejects_duplicate(self):
        with self.assertRaises(ValueError):
            c.parse_registry(b'MANIFEST.txt\nSOURCE_SHA256.txt\na.rs\na.rs\n',f'{"1"*64}  a.rs\n'.encode())

    def test_registry_rejects_untracked(self):
        with self.assertRaises(ValueError):
            c.parse_registry(b'MANIFEST.txt\nSOURCE_SHA256.txt\na.rs\n',b'')

    def test_registry_rejects_unsafe(self):
        with self.assertRaises(ValueError):
            c.parse_registry(b'MANIFEST.txt\nSOURCE_SHA256.txt\n../x.rs\n',f'{"1"*64}  ../x.rs\n'.encode())

    def test_compare_csv_is_full_grid(self):
        rows=[{k:(False if k=='local_on' else 0) for k in ('case','arm','off_status','off_iters','on_status','on_iters','off_true','on_true','on_kahan','local_on')}]
        data=c.compare_csv(rows)
        self.assertEqual(data.count(b'\n'),2)
        self.assertIn(b'false',data)

    def test_raw_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            (a/'fake').symlink_to(root)
            with self.assertRaises(ValueError):c.raw_index(a,b)

    def test_no_production_promotion_text(self):
        self.assertIn('No P2 feature default change',c.HANDOFF)
        self.assertIn('Breakdown 124',c.HANDOFF)
        self.assertIn('P2 on certified 430',c.HANDOFF)

    def test_gate_uses_discovery(self):
        self.assertTrue(any('discover' in command for command in c.commands()))
        self.assertTrue(all('benchmarks.scripts.test_C1G5I_' not in ' '.join(command) for command in c.commands()))

    def test_source_bytes_reproduce_git_copy(self):
        self.assertEqual(len(c.ORIGINAL_TO_REPO),6)
        for p in c.ORIGINAL_TO_REPO:
            self.assertIn('benchmarks/scripts/'+p,c.NEW)

    def test_apply_rolls_back_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for t in c.TRACKED:
                q=root/t;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b'original')
            # Fail after some tracked files were written because one new path is a symlink.
            bad=root/c.NEW[3];bad.parent.mkdir(parents=True,exist_ok=True);bad.symlink_to(root)
            before={p:(root/p).read_bytes() for p in c.TRACKED}
            planned={p:b'new' for p in c.ALL}
            with self.assertRaises(ValueError):c.apply(root,planned)
            for p in c.TRACKED:self.assertEqual((root/p).read_bytes(),before[p])
            for p in c.NEW:
                if p!=c.NEW[3]:self.assertFalse((root/p).exists())

if __name__=='__main__':unittest.main()
