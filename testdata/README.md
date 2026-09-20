# SYNTHETIC — a Slurm `sacct` export shaped like a real one, and never one

Nothing in `sacct_sample.txt` was measured. It is not a cluster, not a site,
not a workload, and no number taken off it describes anything but itself. It
exists so that this crate's tests have a file to be tested against.

The user names are `u01`–`u09` for the same reason: a real export carries the
column, and a test that the parser carries it through is worth more than an
empty field.

## What it is

Thirty rows on the documented `--format`, over three local days from
`2026-03-02T00:00:00-06:00`:

| Rows | What they are |
| --- | --- |
| 20 | finished jobs, among them one `TIMEOUT`, one `FAILED`, one `CANCELLED by 1001`, and one with `TimelimitRaw = UNLIMITED` |
| 5 | job steps (`1041.batch`, `1044.0`, …) — still rows, and `is_step()` says so |
| 3 | jobs still running at export time, with a real `End` reading written in |
| 2 | jobs that asked for no GPU |

None of the thirty is a rejection: every row parses, and the test that reads
this file asserts all thirty come back.

Several rows carry both a `gres/gpu` total and its `gres/gpu:a100` breakdown,
which is what Slurm writes and what `gpu_count()` must not add up twice.
