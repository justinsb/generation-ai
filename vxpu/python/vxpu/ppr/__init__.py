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

"""One process per card (#539): the tensor-parallel step with one executor process per rank.

Today one Python process drives every rank: four rank graphs on four threads, with the host
work between programs (the drafter's output, the acceptance, the layout, the launches) done
once by that process. This package moves each rank into a process of its own and the small
control plane between them into shared memory:

  mailbox    fixed-size slots with sequence numbers and a barrier over shared memory, with no
             torch at the protocol level (slice 1)
  control    the step's control messages: the step, the ranks' choices, the acceptance and the
             layout, as fixed-size records (slice 1)
  world      a parallel.World whose collectives cross processes through the mailboxes, for the
             eager CPU step (slice 2)
  executor   the process that owns one rank: it loads that rank's state through the engine and
             runs its step graph on every step message (slice 2)
  coordinator  spawns the executors, distributes the plan and runs the step loop (slice 2)
"""
