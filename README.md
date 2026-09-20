# sacct-parse

## What this is

A parser for one Slurm `sacct` export row: pipe-separated `--parsable2` text in, one typed `Record` out, with a named error for every field it cannot read.

## Who it is for

A Slurm administrator or tooling author who has `sacct` output and wants typed records from it, without taking on anyone's scheduling opinions.

## How to use it

Run `sacct` with this documented format:

```
sacct --parsable2 --noheader \
  --format=JobIDRaw,Submit,Start,End,ElapsedRaw,AllocTRES,TimelimitRaw,QOS,Partition,State,User
```

Add the crate to a Rust project:

```sh
cargo add sacct-parse
```

That writes this line into `Cargo.toml`:

```toml
sacct-parse = "0.1.0"
```

Minimal example:

```rust
use sacct_parse::parse_file;

// Synthetic example data; the offset is UTC-6 in seconds.
let text = "101|2026-03-02T08:15:00|2026-03-02T08:00:00|2026-03-02T09:00:00|7200|cpu=8,gres/gpu=1|180|normal|batch|COMPLETED|u01\n";
let records = parse_file(text, -21_600).expect("the row parses");
assert_eq!(records.len(), 1);
```

`sacct` prints clock readings with no zone on them, so the caller supplies `utc_offset_seconds`: `-21_600` for UTC-6, `3_600` for UTC+1. A printed reading is interpreted as civil time in no zone, and the record's timestamp is `civil_interpreted_as_utc − utc_offset_seconds`.

## What it deliberately does not do

It does not run `sacct`, apply time zones or a zone database, decide which jobs matter, or write any file.

Version 0.x: the API may change before 1.0.
