# SYNTHETIC — a Slurm `sacct` export shaped like a partner's, and never one

Nothing in `sacct_sample.txt` was measured. It is not a cluster, not a
partner, not a workload, and no number taken off it describes anything but
itself. It exists so that `loadshift ingest slurm` has a file to be tested
against, and so that the mapping rules in `docs/ingest-slurm.md` have
something to be read beside.

The user names are `u01`–`u09` for the same reason: a real export carries the
column and LoadShift drops it, and a test of that is worth more than an empty
field.

## What it is

Thirty rows on the documented `--format`, over three local days from
`2026-03-02T00:00:00-06:00` — a Monday, six days before the clock springs
forward, so the export crosses nothing.

| Rows | What they are | What ingest does |
| --- | --- | --- |
| 20 | finished GPU jobs | kept |
| 5 | job steps (`1041.batch`, `1044.0`, …) | skipped: the job's own row is in the file |
| 3 | jobs still running at export time | skipped: `ElapsedRaw` is a stopwatch still going |
| 2 | finished jobs that asked for no GPU | skipped: not work this product schedules |

None of the thirty is a rejection. Every rejection has a unit test of its own
in `crates/loadshift-ingest/src/slurm.rs`, where a five-row fixture can carry
one fault per row without making the committed sample unreadable.

Among the twenty kept: `1047` carries `TimelimitRaw = UNLIMITED`, `1051` is
`CANCELLED by 1001` and still finished, `1045` is `TIMEOUT` and `1048` is
`FAILED` — all three ran, so all three are measurements. Several rows carry
both a `gres/gpu` total and its `gres/gpu:a100` breakdown, which is what
Slurm writes and what the mapper must not add up twice.

## How it is read

```bash
loadshift ingest slurm testdata/slurm/sacct_sample.txt \
    --map testdata/slurm/tiers.toml \
    --t0 2026-03-02T00:00:00-06:00 --tz America/Chicago \
    --out out/jobs.csv
```

`crates/loadshift-cli/tests/ingest_slurm.rs` runs exactly that, replays the
result, and reads it back through the jobs contract.
