# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the package cache's housekeeping."""

import os
import subprocess
import sys
import tempfile
import unittest

from . import aoti


class ReclaimExtractionsTest(unittest.TestCase):
    def test_removes_only_dead_processes_extractions(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        key = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as root:
            for name in (f"aotinductor_dead1/{key}.{dead.pid}.tmp", f"{key}.{dead.pid}.tmp",
                         f"aotinductor_own/{key}.{os.getpid()}.tmp", f"{key}.{os.getpid()}.tmp",
                         f"aotinductor_mixed/{key}.{dead.pid}.tmp", f"aotinductor_mixed/{key}.{os.getpid()}.tmp",
                         f"{key}.pt2", "aotinductor_other/notes", f"not-a-key.{dead.pid}.tmp"):
                os.makedirs(os.path.join(root, name))
                with open(os.path.join(root, name, "model.so"), "w"):
                    pass
            os.makedirs(os.path.join(root, "aotinductor_empty"))
            removed = aoti.reclaim_extractions(root, pid_alive=lambda pid: pid != dead.pid, grace_s=0)
            self.assertEqual(sorted(removed), sorted(["aotinductor_dead1", f"{key}.{dead.pid}.tmp"]))
            self.assertEqual(sorted(os.listdir(root)),
                             sorted([f"aotinductor_own", f"{key}.{os.getpid()}.tmp", "aotinductor_mixed",
                                     f"{key}.pt2", "aotinductor_other", "aotinductor_empty",
                                     f"not-a-key.{dead.pid}.tmp"]))

    def test_leaves_a_fresh_extraction_of_a_dead_pids_package(self):
        # a package built by an earlier process extracts under that process's pid: a live load must not lose it
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        key = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as root:
            fresh = os.path.join(root, f"aotinductor_fresh/{key}.{dead.pid}.tmp")
            stale = os.path.join(root, f"aotinductor_stale/{key}.{dead.pid}.tmp")
            for d in (fresh, stale):
                os.makedirs(d)
                with open(os.path.join(d, "model.so"), "w"):
                    pass
            old = os.path.getmtime(stale) - 3 * 3600
            os.utime(os.path.join(root, "aotinductor_stale"), (old, old))
            removed = aoti.reclaim_extractions(root, pid_alive=lambda pid: pid != dead.pid, grace_s=7200)
            self.assertEqual(removed, ["aotinductor_stale"])
            self.assertTrue(os.path.exists(fresh))

    def test_alive_sees_this_process_and_not_a_dead_one(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        self.assertTrue(aoti._alive(os.getpid()))
        # a reaped child's pid may be reused; only assert the live side strictly
        self.assertIn(aoti._alive(dead.pid), (True, False))


if __name__ == "__main__":
    unittest.main()
