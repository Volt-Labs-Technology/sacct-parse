//! Parse Slurm `sacct --parsable2` rows into typed records.
//!
//! [`sacct`](https://slurm.schedmd.com/sacct.html) prints one pipe-separated
//! line per accounting row. This crate reads one line at a time and turns it
//! into a [`Record`]: timestamps as Unix seconds, runtimes as seconds, wall
//! limits as [`Limit`], job states as a [`State`] enum, and allocated
//! resources as a count per resource name. A row that cannot be read earns an
//! [`Error`] that names the row and the field, never a quiet default.
//!
//! # The clock is the caller's
//!
//! `sacct` prints local clock readings with no zone on them. This crate does
//! not carry a timezone database and does not guess one: every parsing
//! function takes `utc_offset_seconds`, the zone's offset at the time of the
//! export, and computes `unix = civil_interpreted_as_utc − utc_offset_seconds`.
//! A site whose clock changes mid-export makes two calls with two offsets;
//! the crate stays arithmetic.
//!
//! # Parsing, not deciding
//!
//! Every row parses. Job steps are still records — [`Record::is_step`] says
//! which they are, and the caller decides what to do with them. An
//! unfinished job parses like a finished one. Nothing here filters by GPU,
//! renumbers ids, applies tiers, or writes a file.

use std::collections::BTreeMap;

/// The number of `--format` fields the documented `sacct` invocation has.
const FORMAT_FIELDS: usize = 11;

/// `TimelimitRaw` for a job with no wall-clock limit.
const UNLIMITED_TOKEN: &str = "UNLIMITED";

/// The `AllocTRES` key for a job's whole GPU allocation.
const GPU_TOTAL_KEY: &str = "gres/gpu";

/// The `AllocTRES` key prefix for one GPU type of an allocation.
const GPU_TYPE_PREFIX: &str = "gres/gpu:";

/// One parsed `sacct` row. **Data.**
///
/// Every field is the row's own, typed and unfiltered: a step, a running job
/// and a finished job all parse, and what to keep is the caller's decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// `JobIDRaw` as printed: `1041`, or `1041.batch` for a job step.
    pub job_id: String,
    /// `Submit` as Unix seconds, from a civil reading and the caller's offset.
    pub submit: i64,
    /// `Start` as Unix seconds.
    pub start: i64,
    /// `End` as Unix seconds.
    pub end: i64,
    /// `ElapsedRaw` in seconds.
    pub elapsed_seconds: u64,
    /// `AllocTRES` as a count per resource name, in name order.
    pub alloc_tres: BTreeMap<String, u64>,
    /// `TimelimitRaw` in minutes, or no limit at all.
    pub timelimit: Limit,
    /// `QOS` as printed.
    pub qos: String,
    /// `Partition` as printed.
    pub partition: String,
    /// `State`, matched on its first whitespace-separated word.
    pub state: State,
    /// `User` as printed.
    pub user: String,
}

impl Record {
    /// Whether this row is a job step rather than a job. **Calculation.**
    ///
    /// Slurm writes a step's `JobIDRaw` with a dot — `1041.batch`, `1044.0` —
    /// and the step is part of the job whose own row is in the same export.
    #[must_use]
    pub fn is_step(&self) -> bool {
        self.job_id.contains('.')
    }

    /// The GPUs this record's `AllocTRES` allocates. **Calculation.**
    ///
    /// Slurm writes both a total and a per-type breakdown when a job asked
    /// for a typed GPU — `gres/gpu=3,gres/gpu:a100=3` — so the total wins
    /// where it is present, and the types are summed only where it is not.
    /// Adding both would count the job twice.
    #[must_use]
    pub fn gpu_count(&self) -> u64 {
        if let Some(total) = self.alloc_tres.get(GPU_TOTAL_KEY) {
            return *total;
        }
        self.alloc_tres
            .iter()
            .filter(|(key, _)| key.starts_with(GPU_TYPE_PREFIX))
            .map(|(_, count)| *count)
            .sum()
    }
}

/// `TimelimitRaw`: the minutes a job asked for, or no limit. **Data.**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    /// A wall-clock limit, in minutes.
    Minutes(u32),
    /// The row printed `UNLIMITED`.
    Unlimited,
}

/// `State`, matched on the first whitespace-separated word of the field.
/// **Data.**
///
/// `sacct` appends a clause to some states — `CANCELLED by 1001` — and the
/// clause is not part of the state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum State {
    /// The job ran and finished within its limit.
    Completed,
    /// The job hit its wall-clock limit.
    Timeout,
    /// The job ran and exited with a failure.
    Failed,
    /// The job was cancelled, by a user or the scheduler.
    Cancelled,
    /// The job is running now; its runtime is not final.
    Running,
    /// The job is queued and has not started.
    Pending,
    /// Any other state, carried as printed.
    Other(String),
}

impl State {
    /// The state one `State` field carries. **Calculation.**
    fn parse(field: &str) -> Self {
        match field.split_whitespace().next().unwrap_or_default() {
            "COMPLETED" => Self::Completed,
            "TIMEOUT" => Self::Timeout,
            "FAILED" => Self::Failed,
            "CANCELLED" => Self::Cancelled,
            "RUNNING" => Self::Running,
            "PENDING" => Self::Pending,
            other => Self::Other(other.to_owned()),
        }
    }
}

/// Why a row could not be parsed. **Data.**
///
/// Every variant names the row, counted from 1, and the field that failed,
/// because "row 12 is bad" sends a reader back to the file and "row 12,
/// `ElapsedRaw`" sends them to the field.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The row does not have the 11 documented `--format` fields.
    #[error("row {row}: has {actual} fields, and the documented sacct format has {expected}")]
    Arity {
        /// The row's line number in the export, counted from 1.
        row: usize,
        /// The number of fields the documented format has.
        expected: usize,
        /// The number of fields the row has.
        actual: usize,
    },

    /// A field holds a value this crate cannot read.
    #[error("row {row}: field `{field}` holds `{value}`, which is not {expected}")]
    Field {
        /// The row's line number in the export, counted from 1.
        row: usize,
        /// The field that failed, named as the `--format` names it.
        field: &'static str,
        /// The value as printed.
        value: String,
        /// What the field was expected to hold.
        expected: &'static str,
    },
}

/// One `sacct` row, split into its fields. **Data.**
struct Fields<'a> {
    job_id: &'a str,
    submit: &'a str,
    start: &'a str,
    end: &'a str,
    elapsed: &'a str,
    tres: &'a str,
    timelimit: &'a str,
    qos: &'a str,
    partition: &'a str,
    state: &'a str,
    user: &'a str,
}

impl<'a> Fields<'a> {
    /// One `|`-separated line, split into the documented fields.
    /// **Calculation.**
    fn parse(row: usize, line: &'a str) -> Result<Self, Error> {
        let fields: Vec<&str> = line.split('|').map(str::trim).collect();
        let actual = fields.len();
        let [job_id, submit, start, end, elapsed, tres, timelimit, qos, partition, state, user] =
            fields[..]
        else {
            return Err(Error::Arity {
                row,
                expected: FORMAT_FIELDS,
                actual,
            });
        };

        Ok(Self {
            job_id,
            submit,
            start,
            end,
            elapsed,
            tres,
            timelimit,
            qos,
            partition,
            state,
            user,
        })
    }

    /// A timestamp field, read as a civil clock reading and turned into Unix
    /// seconds with the caller's offset. **Calculation.**
    ///
    /// `sacct` writes `Unknown` for a stamp the job has not earned yet; a
    /// reading without the expected shape is a field error, not a zero.
    fn timestamp(
        &self,
        row: usize,
        field: &'static str,
        text: &'a str,
        utc_offset_seconds: i64,
    ) -> Result<i64, Error> {
        let bad = |value: &str, expected: &'static str| Error::Field {
            row,
            field,
            value: value.to_owned(),
            expected,
        };

        if text.is_empty() || text == "Unknown" {
            return Err(bad(
                text,
                "a clock reading like 2026-03-02T08:15:00, not Unknown",
            ));
        }
        let Some((date, time)) = text.split_once('T') else {
            return Err(bad(text, "a clock reading like 2026-03-02T08:15:00"));
        };
        let days = civil_days(row, field, date)?;
        let Some((hour, rest)) = time.split_once(':') else {
            return Err(bad(text, "a clock reading like 2026-03-02T08:15:00"));
        };
        let (minute, second) = match rest.split_once(':') {
            Some((minute, second)) => (minute, second),
            None => (rest, "00"),
        };
        let hour: i64 = whole(row, field, hour, "an hour of the day")?;
        let minute: i64 = whole(row, field, minute, "a minute of the hour")?;
        let second: i64 = whole(row, field, second, "a second of the minute")?;
        if !(0..24).contains(&hour) || !(0..60).contains(&minute) || !(0..60).contains(&second) {
            return Err(bad(
                text,
                "a clock reading whose time is a real time of day",
            ));
        }
        Ok(days * 86_400 + hour * 3_600 + minute * 60 + second - utc_offset_seconds)
    }

    /// `ElapsedRaw` as seconds. **Calculation.**
    fn elapsed(&self, row: usize) -> Result<u64, Error> {
        self.elapsed.parse().map_err(|_| Error::Field {
            row,
            field: "ElapsedRaw",
            value: self.elapsed.to_owned(),
            expected: "a count of seconds",
        })
    }

    /// `AllocTRES` as a count per resource name. **Calculation.**
    ///
    /// An entry without `=` carries no count and is carried as `0`. A count
    /// that carries a Slurm size suffix — `mem=256G` — is carried as the
    /// number Slurm printed, with the suffix dropped; the suffix is not
    /// converted, because a count of resources and a quantity of memory are
    /// different things and this map holds counts.
    fn tres(&self, row: usize) -> Result<BTreeMap<String, u64>, Error> {
        let mut map = BTreeMap::new();
        for entry in self.tres.split(',').filter(|entry| !entry.is_empty()) {
            let (name, count) = match entry.split_once('=') {
                Some((name, count)) => (name, count),
                None => (entry, "0"),
            };
            let count: u64 = count_of(row, count)?;
            map.insert(name.to_owned(), count);
        }
        Ok(map)
    }

    /// `TimelimitRaw` as a limit. **Calculation.**
    fn timelimit(&self, row: usize) -> Result<Limit, Error> {
        if self.timelimit == UNLIMITED_TOKEN {
            return Ok(Limit::Unlimited);
        }
        self.timelimit
            .parse()
            .map(Limit::Minutes)
            .map_err(|_| Error::Field {
                row,
                field: "TimelimitRaw",
                value: self.timelimit.to_owned(),
                expected: "a count of minutes, or UNLIMITED",
            })
    }
}

/// A whole number a timestamp part carries. **Calculation.**
fn whole(
    row: usize,
    field: &'static str,
    text: &str,
    expected: &'static str,
) -> Result<i64, Error> {
    text.parse().map_err(|_| Error::Field {
        row,
        field,
        value: text.to_owned(),
        expected,
    })
}

/// The days since the Unix epoch a `YYYY-MM-DD` date carries, as a civil
/// calculation with no zone and no table. **Calculation.**
///
/// This is Howard Hinnant's `days_from_civil`, reduced to what a date needs:
/// a month of 1–12, a day valid for that month and year, and arithmetic that
/// treats year 0 like any other.
fn civil_days(row: usize, field: &'static str, date: &str) -> Result<i64, Error> {
    let bad = |value: &str, expected: &'static str| Error::Field {
        row,
        field,
        value: value.to_owned(),
        expected,
    };

    let parts: Vec<&str> = date.split('-').collect();
    let [year, month, day] = parts[..] else {
        return Err(bad(date, "a date like 2026-03-02"));
    };
    let year: i64 = whole(row, field, year, "a year")?;
    let month: i64 = whole(row, field, month, "a month from 1 to 12")?;
    let day: i64 = whole(row, field, day, "a day from 1 to 31")?;
    if !(1..=12).contains(&month) {
        return Err(bad(date, "a date whose month and day are real"));
    }

    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year.rem_euclid(400) == 0
            || (year.rem_euclid(4) == 0 && year.rem_euclid(100) != 0) =>
        {
            29
        }
        2 => 28,
        _ => return Err(bad(date, "a date whose month and day are real")),
    };
    if !(1..=days_in_month).contains(&day) {
        return Err(bad(date, "a date whose month and day are real"));
    }

    let jan_or_feb = month <= 2;
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if jan_or_feb { 9 } else { -3 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Ok(era * 146_097 + day_of_era - 719_468)
}

/// The count one `AllocTRES` entry carries. **Calculation.**
///
/// A count may carry a Slurm size suffix — `256G` — and any such suffix is
/// dropped without conversion: the map holds the number Slurm printed. A
/// value with no number in it at all is a fault.
fn count_of(row: usize, text: &str) -> Result<u64, Error> {
    let bad = Error::Field {
        row,
        field: "AllocTRES",
        value: text.to_owned(),
        expected: "a count for a resource in the AllocTRES list",
    };

    let digits: &str = text
        .split(|c: char| !c.is_ascii_digit() && c != '.')
        .next()
        .unwrap_or_default();
    if let Ok(count) = digits.parse() {
        return Ok(count);
    }
    digits
        .parse::<f64>()
        .ok()
        .filter(|count| count.is_finite() && *count >= 0.0)
        .map(|count| count as u64)
        .ok_or(bad)
}

/// Parse one `sacct --parsable2 --noheader` row. **Calculation.**
///
/// The row's fields are, in order: `JobIDRaw, Submit, Start, End,
/// ElapsedRaw, AllocTRES, TimelimitRaw, QOS, Partition, State, User`. The
/// clock readings are civil time with no zone; `utc_offset_seconds` is the
/// zone's offset, so `-21_600` for UTC-6, and the record's stamps are
/// `civil_interpreted_as_utc − utc_offset_seconds`.
///
/// # Errors
/// [`Error::Arity`] when the row has the wrong number of fields, and
/// [`Error::Field`] naming the row and the field for any value that does not
/// parse.
pub fn parse_row(line: &str, utc_offset_seconds: i64) -> Result<Record, Error> {
    parse_row_numbered(1, line, utc_offset_seconds)
}

/// [`parse_row`] with the row's own number, counted from 1. **Calculation.**
fn parse_row_numbered(row: usize, line: &str, utc_offset_seconds: i64) -> Result<Record, Error> {
    let fields = Fields::parse(row, line)?;
    Ok(Record {
        job_id: fields.job_id.to_owned(),
        submit: fields.timestamp(row, "Submit", fields.submit, utc_offset_seconds)?,
        start: fields.timestamp(row, "Start", fields.start, utc_offset_seconds)?,
        end: fields.timestamp(row, "End", fields.end, utc_offset_seconds)?,
        elapsed_seconds: fields.elapsed(row)?,
        alloc_tres: fields.tres(row)?,
        timelimit: fields.timelimit(row)?,
        qos: fields.qos.to_owned(),
        partition: fields.partition.to_owned(),
        state: State::parse(fields.state),
        user: fields.user.to_owned(),
    })
}

/// Parse a whole `sacct --parsable2 --noheader` export. **Calculation.**
///
/// Blank lines are skipped. Every remaining line is parsed in order, and the
/// first row that cannot be read fails the whole call with an error naming
/// that row and field.
///
/// # Errors
/// The first [`Error`] any row earns.
pub fn parse_file(text: &str, utc_offset_seconds: i64) -> Result<Vec<Record>, Error> {
    text.lines()
        .enumerate()
        .map(|(offset, line)| (offset + 1, line))
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(row, line)| parse_row_numbered(row, line, utc_offset_seconds))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFSET: i64 = -21_600;

    /// One row on the documented format, with every field a given test does
    /// not care about already filled in: a finished one-GPU job.
    ///
    /// Synthetic data; nothing in these fixtures was measured.
    fn row(job_id: &str, submit: &str, elapsed: &str, tres: &str, timelimit: &str) -> String {
        [
            job_id,
            submit,
            "2026-03-02T08:00:00",
            "2026-03-02T09:00:00",
            elapsed,
            tres,
            timelimit,
            "normal",
            "batch",
            "COMPLETED",
            "u01",
        ]
        .join("|")
    }

    fn parsed(line: &str) -> Record {
        parse_row(line, OFFSET).expect("the row parses")
    }

    fn fault(line: &str) -> Error {
        parse_row(line, OFFSET).expect_err("the row is rejected")
    }

    #[test]
    fn a_finished_row_parses_into_a_record() {
        let record = parsed(&row(
            "101",
            "2026-03-02T08:15:00",
            "7200",
            "cpu=8,gres/gpu=1",
            "180",
        ));

        assert_eq!(record.job_id, "101");
        assert!(!record.is_step());
        assert_eq!(record.submit, 1_772_460_900);
        assert_eq!(record.start, 1_772_460_000);
        assert_eq!(record.end, 1_772_463_600);
        assert_eq!(record.elapsed_seconds, 7_200);
        assert_eq!(record.timelimit, Limit::Minutes(180));
        assert_eq!(record.qos, "normal");
        assert_eq!(record.partition, "batch");
        assert_eq!(record.state, State::Completed);
        assert_eq!(record.user, "u01");
    }

    #[test]
    fn a_timestamp_is_the_civil_reading_minus_the_callers_offset() {
        // `2026-03-02T08:15:00` read at UTC-6 is 14:15 UTC, so 1_772_460_900.
        let record = parsed(&row("101", "2026-03-02T08:15:00", "7200", "cpu=8", "180"));
        assert_eq!(record.submit, 1_772_460_900);

        // The same reading at UTC+1 is 07:15 UTC, seven hours earlier.
        let east = parse_row(
            &row("101", "2026-03-02T08:15:00", "7200", "cpu=8", "180"),
            3_600,
        )
        .expect("the row parses");
        assert_eq!(east.submit, record.submit - 7 * 3_600);
    }

    #[test]
    fn a_state_is_its_first_word_and_the_clause_is_not_part_of_it() {
        let cancelled = State::parse("CANCELLED by 1001");
        assert_eq!(cancelled, State::Cancelled);
        assert_eq!(State::parse("RUNNING"), State::Running);
        assert_eq!(State::parse("PENDING"), State::Pending);
        assert_eq!(
            State::parse("REQUEUED"),
            State::Other("REQUEUED".to_owned())
        );
        assert_eq!(State::parse(""), State::Other(String::new()));
    }

    #[test]
    fn a_row_with_the_wrong_field_count_names_the_row_and_the_counts() {
        let short = "101|2026-03-02T08:15:00|7200|COMPLETED";

        match fault(short) {
            Error::Arity {
                row,
                expected,
                actual,
            } => {
                assert_eq!(row, 1);
                assert_eq!(expected, 11);
                assert_eq!(actual, 4);
            }
            other => panic!("expected an arity error, got {other}"),
        }
    }

    #[test]
    fn an_unreadable_elapsed_names_the_row_and_elapsedraw() {
        let line = row("101", "2026-03-02T08:15:00", "02:00:00", "cpu=8", "180");

        match fault(&line) {
            Error::Field {
                row,
                field,
                value,
                expected,
            } => {
                assert_eq!(row, 1);
                assert_eq!(field, "ElapsedRaw");
                assert_eq!(value, "02:00:00");
                assert_eq!(expected, "a count of seconds");
            }
            other => panic!("expected a field error, got {other}"),
        }
    }

    #[test]
    fn an_unreadable_timelimit_names_the_row_and_timelimitraw() {
        let line = row("101", "2026-03-02T08:15:00", "7200", "cpu=8", "3:00:00");

        assert!(matches!(
            fault(&line),
            Error::Field {
                field: "TimelimitRaw",
                ..
            }
        ));
    }

    #[test]
    fn unlimited_is_a_limit_and_not_a_fault() {
        let line = row("101", "2026-03-02T08:15:00", "7200", "cpu=8", "UNLIMITED");

        assert_eq!(parsed(&line).timelimit, Limit::Unlimited);
    }

    #[test]
    fn an_unreadable_tres_count_names_the_row_and_alloctres() {
        let line = row("101", "2026-03-02T08:15:00", "7200", "gres/gpu=many", "180");

        assert!(matches!(
            fault(&line),
            Error::Field {
                field: "AllocTRES",
                ..
            }
        ));
    }

    #[test]
    fn an_unknown_stamp_names_the_row_and_the_field() {
        let line = [
            "101",
            "2026-03-02T08:15:00",
            "Unknown",
            "2026-03-02T09:00:00",
            "7200",
            "cpu=8",
            "180",
            "normal",
            "batch",
            "COMPLETED",
            "u01",
        ]
        .join("|");

        assert!(matches!(fault(&line), Error::Field { field: "Start", .. }));
    }

    #[test]
    fn an_impossible_date_names_the_row_and_the_field() {
        let line = row("101", "2026-13-02T08:15:00", "7200", "cpu=8", "180");

        assert!(matches!(
            fault(&line),
            Error::Field {
                field: "Submit",
                ..
            }
        ));
    }

    #[test]
    fn rejects_days_that_their_month_cannot_hold() {
        for date in ["2026-02-31", "2025-02-29", "2026-04-31", "1900-02-29"] {
            let stamp = format!("{date}T08:15:00");
            let line = row("101", &stamp, "7200", "cpu=8", "180");

            match fault(&line) {
                Error::Field {
                    row, field, value, ..
                } => {
                    assert_eq!(row, 1, "date {date}");
                    assert_eq!(field, "Submit", "date {date}");
                    assert_eq!(value, date, "date {date}");
                }
                other => panic!("expected a field error for {date}, got {other}"),
            }
        }
    }

    #[test]
    fn accepts_february_29_in_a_leap_year() {
        // Synthetic dates; 2024 and 2000 are leap years, 2000 also divisible by 400.
        for stamp in ["2024-02-29T08:15:00", "2000-02-29T08:15:00"] {
            let line = row("101", stamp, "7200", "cpu=8", "180");
            assert!(parse_row(&line, OFFSET).is_ok(), "{stamp}");
        }
    }

    #[test]
    fn an_impossible_time_of_day_names_the_row_and_the_field() {
        let line = row("101", "2026-03-02T25:15:00", "7200", "cpu=8", "180");

        assert!(matches!(
            fault(&line),
            Error::Field {
                field: "Submit",
                ..
            }
        ));
    }

    #[test]
    fn alloc_tres_carries_a_count_per_resource_and_drops_size_suffixes() {
        let record = parsed(&row(
            "101",
            "2026-03-02T08:15:00",
            "7200",
            "billing=32,cpu=32,gres/gpu=4,mem=256G,node=2",
            "180",
        ));

        let keys: Vec<&str> = record.alloc_tres.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["billing", "cpu", "gres/gpu", "mem", "node"]);
        assert_eq!(record.alloc_tres["mem"], 256);
        assert_eq!(record.alloc_tres["node"], 2);
    }

    #[test]
    fn the_gpu_total_wins_and_is_never_added_to_its_own_breakdown() {
        let both = parsed(&row(
            "101",
            "2026-03-02T08:15:00",
            "7200",
            "cpu=8,gres/gpu=3,gres/gpu:a100=3",
            "180",
        ));
        assert_eq!(both.gpu_count(), 3);
    }

    #[test]
    fn typed_gpus_are_summed_when_the_total_is_absent() {
        let typed = parsed(&row(
            "101",
            "2026-03-02T08:15:00",
            "7200",
            "cpu=8,gres/gpu:a100=2,gres/gpu:h100=1",
            "180",
        ));
        assert_eq!(typed.gpu_count(), 3);
    }

    #[test]
    fn a_job_step_is_a_step() {
        for step in ["1041.batch", "1041.0"] {
            let record = parsed(&row(step, "2026-03-02T08:15:00", "7200", "cpu=8", "180"));
            assert!(record.is_step(), "{step} is a step");
            assert_eq!(record.job_id, step);
        }
        let job = parsed(&row("1041", "2026-03-02T08:15:00", "7200", "cpu=8", "180"));
        assert!(!job.is_step());
    }

    #[test]
    fn parse_file_skips_blank_lines_and_names_the_rows_it_reads() {
        let text = [
            row("101", "2026-03-02T08:15:00", "7200", "cpu=8", "180"),
            String::new(),
            row("102", "2026-03-02T09:15:00", "600", "cpu=4", "60"),
            row("103", "2026-03-02T10:15:00", "900", "cpu=4", "60"),
        ]
        .join("\n");

        let records = parse_file(&text, OFFSET).expect("every row parses");
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].job_id, "101");
        assert_eq!(records[2].job_id, "103");
    }

    #[test]
    fn parse_file_fails_on_the_first_rejection_and_names_its_row() {
        let text = [
            row("101", "2026-03-02T08:15:00", "7200", "cpu=8", "180"),
            String::new(),
            row("102", "2026-03-02T09:15:00", "600", "cpu=4", "60"),
            row("103", "2026-03-02T10:15:00", "not-a-number", "cpu=4", "60"),
        ]
        .join("\n");

        match parse_file(&text, OFFSET) {
            Err(Error::Field { row, field, .. }) => {
                assert_eq!(row, 4);
                assert_eq!(field, "ElapsedRaw");
            }
            other => panic!("expected a field error on row 4, got {other:?}"),
        }
    }

    #[test]
    fn the_committed_sample_parses_with_zero_rejections_and_thirty_rows() {
        let sample = include_str!("../testdata/sacct_sample.txt");
        let records = parse_file(sample, OFFSET).expect("every row of the sample parses");

        assert_eq!(records.len(), 30);

        let steps = records.iter().filter(|record| record.is_step()).count();
        assert_eq!(steps, 5);

        let with_gpus = records
            .iter()
            .filter(|record| record.gpu_count() > 0)
            .count();
        assert_eq!(with_gpus, 28);

        let unlimited = records
            .iter()
            .filter(|record| record.timelimit == Limit::Unlimited)
            .count();
        assert_eq!(unlimited, 1);

        let states: Vec<&State> = records.iter().map(|record| &record.state).collect();
        assert!(states.contains(&&State::Timeout));
        assert!(states.contains(&&State::Failed));
        assert!(states.contains(&&State::Cancelled));
        assert!(states.contains(&&State::Running));
    }
}
