//! Slurm's `sacct` export, turned into the jobs contract.
//!
//! Three quarters of the design partner's customers run Slurm, and none of
//! them has a file on the jobs contract. They have `sacct`. Everything in this
//! module is the rule that turns one into the other, written once, as a
//! calculation: text in, [`Job`] values out, and a named refusal for every row
//! that cannot become one. Reading the file and writing the result are the
//! caller's actions; nothing here touches a disk.
//!
//! # A row is kept, skipped, or rejected
//!
//! Those are three different things and the summary counts them separately.
//! A **skipped** row is not a fault: a job step, a job still running at export
//! time, a job that asked for no GPU. A **rejected** row is a fault in the
//! export — a field this module cannot read — and it is reported with its row
//! number and the field that failed, never dropped quietly. The tally is what
//! the operator reads to see that the two numbers add up to the file they sent.
//!
//! # No offset arithmetic lives here
//!
//! `sacct` prints local clock readings with no zone on them, and turning one
//! into an hour index is the site clock's job, not this module's. So
//! [`LocalHours`] asks [`SiteClock::local_iso`] what the site's clock read at
//! each hour of the export and indexes the answers: every reading this module
//! resolves is one the clock itself produced. The DST rule is
//! [`SiteClock::crossing_report`], applied here and defined there.
//!
//! # `Submit` is the only stamp converted
//!
//! `arrival_h` comes from `Submit` and `act_hours` from `ElapsedRaw`, so
//! `Start` and `End` have no reader. They stay in the documented `sacct`
//! format because the operator's own diagnostics need them, and converting
//! them here would be arithmetic nobody consumes. `docs/ingest-slurm.md` says
//! so out loud.

use std::collections::BTreeMap;
use std::ops::Range;

use loadshift_domain::{
    DstCrossing, Gpus, HourIndex, Hours, Job, JobId, SiteClock, Tier, UtcSeconds,
};

use crate::error::IngestError;

/// The `--format` fields of the documented `sacct` invocation, in order.
///
/// `docs/ingest-slurm.md` carries the command itself; this is the same list in
/// the one place the parser can be held to it.
const FORMAT: [&str; 11] = [
    "JobIDRaw",
    "Submit",
    "Start",
    "End",
    "ElapsedRaw",
    "AllocTRES",
    "TimelimitRaw",
    "QOS",
    "Partition",
    "State",
    "User",
];

/// States that mean the job ran and is over, so its runtime is final.
///
/// Everything else — `RUNNING`, `PENDING`, `REQUEUED`, `SUSPENDED` — is a job
/// whose `ElapsedRaw` is a stopwatch still going, and a runtime that will
/// change is not a measurement. Those rows are skipped and counted.
const FINISHED: [&str; 4] = ["COMPLETED", "TIMEOUT", "FAILED", "CANCELLED"];

/// `TimelimitRaw` for a job with no wall-clock limit.
const UNLIMITED: &str = "UNLIMITED";

/// The `AllocTRES` key for a job's whole GPU allocation.
const GPU_KEY: &str = "gres/gpu";

/// The `AllocTRES` key prefix for one GPU type of an allocation.
const GPU_TYPE_PREFIX: &str = "gres/gpu:";

const SECONDS_PER_HOUR: f64 = 3_600.0;
const MINUTES_PER_HOUR: u32 = 60;

/// v1's windows and pads, the constants the jobs contract carries.
///
/// Tier 1 is a 24-hour window with `pad=7`, Tier 2 a 7-day window with
/// `pad=4`. Both are fixed by AVRIL and neither is inverted here.
const LATENCY_WINDOW_HOURS: u32 = 24;
const FLEXIBLE_WINDOW_HOURS: u32 = 7 * 24;
const LATENCY_PAD_HOURS: u32 = 7;
const FLEXIBLE_PAD_HOURS: u32 = 4;

/// The longest window a mapping file may set, so that a deadline cannot be
/// pushed past the reach of an hour index by a typo in a rule.
const MAX_WINDOW_HOURS: u32 = 366 * 24;

/// The longest export this reads: two years of hours, which is past the end of
/// the committed zone tables and far past a pilot's thirty-day export.
const MAX_EXPORT_HOURS: u32 = 2 * 366 * 24;

/// Width of `YYYY-MM-DDTHH`, the part of a local reading that names its hour.
const READING_HOUR_WIDTH: usize = 13;

/// What an export turned into. **Data.**
#[derive(Debug, Clone, PartialEq)]
pub struct Conversion {
    /// The jobs, in `(arrival_h, id)` order, like every other jobs file.
    pub jobs: Vec<Job>,
    /// `(id, JobIDRaw)`: the side file that can turn an id back into a Slurm
    /// job, and the one the anonymiser may drop.
    pub id_map: Vec<(JobId, String)>,
    /// What became of every row, so the counts can be checked against the
    /// file the operator sent.
    pub tally: Tally,
    /// Every faulty row, in file order.
    pub rejected: Vec<Rejected>,
}

/// How an export's rows were accounted for. **Data.**
///
/// `read` is every non-blank line; the other five partition it, so an operator
/// can add them up and get their own file back.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Tally {
    pub read: usize,
    pub kept: usize,
    pub steps: usize,
    pub unfinished: usize,
    pub non_gpu: usize,
    pub rejected: usize,
}

/// One row that could not become a job, and why. **Data.**
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejected {
    /// The row's line number in the export, counted from 1.
    pub row: usize,
    pub cause: RejectCause,
}

impl Rejected {
    /// The refusal as one line, naming the row and the field. **Calculation.**
    #[must_use]
    pub fn line(&self) -> String {
        format!("row {}: {}", self.row, self.cause)
    }
}

/// Why a row is a fault in the export rather than a job.
///
/// Every variant names the field it failed on, because "row 12 is bad" sends
/// an operator to read a row and "row 12, `AllocTRES`" sends them to read a
/// field.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RejectCause {
    #[error("the row has {actual} fields, and the documented sacct format has {expected}")]
    Arity { expected: usize, actual: usize },

    #[error("field `{field}` holds `{value}`, which is not {expected}")]
    Field {
        field: &'static str,
        value: String,
        expected: &'static str,
    },

    #[error("field `QOS` holds `{qos}` and `Partition` holds `{partition}`, which no rule matches")]
    Unmapped { qos: String, partition: String },

    #[error("field `Submit` holds `{value}`, which is not an hour this export covers ({covers})")]
    Outside { value: String, covers: String },

    #[error("field `Submit` puts the deadline {window} hours later than an hour index reaches")]
    Deadline { window: u32 },
}

impl RejectCause {
    /// The field the row failed on. **Calculation.**
    #[must_use]
    pub const fn field(&self) -> &'static str {
        match self {
            Self::Arity { .. } => "the row",
            Self::Field { field, .. } => field,
            Self::Unmapped { .. } => "QOS",
            Self::Outside { .. } | Self::Deadline { .. } => "Submit",
        }
    }
}

/// Why a row is not a job, without being a fault. **Data.**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Skip {
    /// A job step, `1234.batch`, which is part of a job already counted.
    Step,
    /// Still running, or never ran: its runtime is not final.
    Unfinished,
    /// Asked for no GPU, so it is not work this product schedules.
    NoGpu,
}

/// One `sacct` line, split but not yet interpreted. **Data.**
///
/// `Start`, `End` and `User` are not fields of this type: nothing maps them,
/// and `User` is the column the pilot terms say never leaves the site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SacctRow<'a> {
    job_id: &'a str,
    submit: &'a str,
    elapsed: &'a str,
    tres: &'a str,
    timelimit: &'a str,
    qos: &'a str,
    partition: &'a str,
    state: &'a str,
}

/// A row that will be a job once the export's hours are known. **Data.**
///
/// Everything but `arrival_h`, `deadline_h` and `id`, which cannot be settled
/// row by row: the hour index needs the export's range, and the id is the
/// row's position in `(Submit, JobIDRaw)` order.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate<'a> {
    row: usize,
    job_id: &'a str,
    /// `JobIDRaw` as a number, which is what `(Submit, JobIDRaw)` orders by:
    /// `9` precedes `10` numerically and follows it alphabetically.
    job_number: u64,
    submit: &'a str,
    tier: Tier,
    gpus: f64,
    actual_hours: f64,
    requested_hours: u32,
    window_hours: u32,
    pad_hours: u32,
}

/// One rule of the mapping file. **Data.**
///
/// A rule with neither matcher is the default rule, and matches every row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TierRule {
    qos: Option<String>,
    partition: Option<String>,
    tier: Tier,
    /// The window this rule's jobs get, when the site's own SLA is not v1's.
    window_hours: Option<u32>,
}

/// The mapping file: rules matched in the order they are written. **Data.**
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierMap(Vec<TierRule>);

/// What can be wrong with a line of the mapping file.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TierMapFault {
    #[error("`{line}` is not a [[rule]] header, a `key = value` line, or a comment")]
    NotALine { line: String },

    #[error("`{key}` comes before any [[rule]] header")]
    Homeless { key: String },

    #[error("`{key}` is not a rule key: expected qos, partition, tier or window_hours")]
    UnknownKey { key: String },

    #[error("`{key}` wants {expected}, and `{value}` is not one")]
    Value {
        key: String,
        expected: &'static str,
        value: String,
    },

    #[error("tier {tier} has no window and no pad in v1: a rule names tier 1 or tier 2")]
    Tier { tier: i64 },

    #[error("a window of {hours} hours is not between 1 and the {MAX_WINDOW_HOURS} a rule may set")]
    Window { hours: i64 },

    #[error("this [[rule]] sets no tier")]
    NoTier,
}

impl TierMap {
    /// The mapping file's rules, in the order it writes them.
    /// **Calculation.**
    ///
    /// The grammar is the subset the issue specifies — `[[rule]]` tables whose
    /// keys take a quoted string or a whole number — parsed here rather than
    /// by a TOML crate, because the workspace has no TOML dependency and one
    /// file shape does not earn one.
    ///
    /// # Errors
    /// [`IngestError::TierMap`] naming the line and what is wrong with it, and
    /// [`IngestError::NoTierRules`] for a file that declares none.
    pub fn parse(text: &str) -> Result<Self, IngestError> {
        let rules = text
            .lines()
            .enumerate()
            .map(|(offset, line)| (offset + 1, line.trim()))
            .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'))
            .try_fold(Vec::new(), read_map_line)?;

        if rules.is_empty() {
            return Err(IngestError::NoTierRules);
        }
        Ok(Self(rules))
    }

    /// The first rule that matches this row's QOS and partition.
    /// **Calculation.**
    #[must_use]
    fn rule_for(&self, qos: &str, partition: &str) -> Option<&TierRule> {
        self.0.iter().find(|rule| rule.matches(qos, partition))
    }
}

impl TierRule {
    /// Whether this rule covers that QOS and partition. **Calculation.**
    fn matches(&self, qos: &str, partition: &str) -> bool {
        let same = |matcher: &Option<String>, value: &str| {
            matcher.as_ref().is_none_or(|wanted| wanted == value)
        };
        same(&self.qos, qos) && same(&self.partition, partition)
    }

    /// The window this rule's jobs get: its own, or the tier's v1 default.
    /// **Calculation.**
    fn window_hours(&self) -> u32 {
        self.window_hours.unwrap_or(match self.tier {
            Tier::Latency => LATENCY_WINDOW_HOURS,
            Tier::Flexible | Tier::Fixed => FLEXIBLE_WINDOW_HOURS,
        })
    }

    /// The pad the tier carries. v1's constants, carried into the file and
    /// unused by the engine. **Calculation.**
    fn pad_hours(&self) -> u32 {
        match self.tier {
            Tier::Latency => LATENCY_PAD_HOURS,
            Tier::Flexible | Tier::Fixed => FLEXIBLE_PAD_HOURS,
        }
    }
}

/// One line of the mapping file, folded into the rules so far.
/// **Calculation.**
fn read_map_line(
    mut rules: Vec<TierRule>,
    (line_number, line): (usize, &str),
) -> Result<Vec<TierRule>, IngestError> {
    let fault = |fault| IngestError::TierMap {
        line: line_number,
        source: fault,
    };

    if line == "[[rule]]" {
        rules.push(TierRule {
            qos: None,
            partition: None,
            tier: Tier::Flexible,
            window_hours: None,
        });
        return Ok(rules);
    }

    let (key, value) = line.split_once('=').ok_or_else(|| {
        fault(TierMapFault::NotALine {
            line: line.to_owned(),
        })
    })?;
    let (key, value) = (key.trim().to_owned(), value.trim());
    let rule = rules
        .last_mut()
        .ok_or_else(|| fault(TierMapFault::Homeless { key: key.clone() }))?;

    set_rule_key(rule, &key, value).map_err(fault)?;
    Ok(rules)
}

/// One `key = value` line applied to the rule it belongs to. **Calculation.**
fn set_rule_key(rule: &mut TierRule, key: &str, value: &str) -> Result<(), TierMapFault> {
    match key {
        "qos" => rule.qos = Some(quoted(key, value)?),
        "partition" => rule.partition = Some(quoted(key, value)?),
        "tier" => rule.tier = tier_of(whole_number(key, value)?)?,
        "window_hours" => rule.window_hours = Some(window_of(whole_number(key, value)?)?),
        _ => {
            return Err(TierMapFault::UnknownKey {
                key: key.to_owned(),
            });
        }
    }
    Ok(())
}

/// The text a `"quoted"` value carries. **Calculation.**
fn quoted(key: &str, value: &str) -> Result<String, TierMapFault> {
    value
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .filter(|text| !text.contains('"'))
        .map(str::to_owned)
        .ok_or_else(|| TierMapFault::Value {
            key: key.to_owned(),
            expected: "a quoted string",
            value: value.to_owned(),
        })
}

/// The number a bare value carries. **Calculation.**
fn whole_number(key: &str, value: &str) -> Result<i64, TierMapFault> {
    value.parse().map_err(|_| TierMapFault::Value {
        key: key.to_owned(),
        expected: "a whole number",
        value: value.to_owned(),
    })
}

/// The tier a rule's `tier = n` names. **Calculation.**
///
/// Tier 0 is refused although the domain has it: `Fixed` is work already
/// running, for which v1 fixes neither a window nor a pad, and inventing a
/// pair for it would be inventing a figure.
fn tier_of(code: i64) -> Result<Tier, TierMapFault> {
    match Tier::from_code(code) {
        Some(tier @ (Tier::Latency | Tier::Flexible)) => Ok(tier),
        _ => Err(TierMapFault::Tier { tier: code }),
    }
}

/// The window a rule's `window_hours = n` names. **Calculation.**
fn window_of(hours: i64) -> Result<u32, TierMapFault> {
    u32::try_from(hours)
        .ok()
        .filter(|hours| (1..=MAX_WINDOW_HOURS).contains(hours))
        .ok_or(TierMapFault::Window { hours })
}

/// The site's own local clock readings, hour by hour. **Data.**
///
/// Keyed by `YYYY-MM-DDTHH`, which is what the site's clock read at the start
/// of each hour of the export. Both readings of a doubled hour share a key and
/// the earlier one wins; a missing local hour has no key at all, so a row
/// claiming it is refused rather than placed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalHours {
    by_reading: BTreeMap<String, HourIndex>,
    /// The first hour past the export: the one whose reading is later than
    /// every `Submit` in it, and so the exclusive end of its range.
    past_end: HourIndex,
}

impl LocalHours {
    /// The clock's readings, far enough to cover `until`. **Calculation.**
    ///
    /// # Errors
    /// [`IngestError::ExportTooLong`] when `until` is further from `t0` than
    /// this reads, which is a `--t0` from the wrong year rather than an export
    /// that long.
    fn of(clock: &SiteClock, until: &str) -> Result<Self, IngestError> {
        let mut by_reading = BTreeMap::new();

        for hour in (0..MAX_EXPORT_HOURS).map(HourIndex::new) {
            let key = hour_key(&clock.local_iso(hour)).to_owned();
            let past = key.as_str() > until;
            by_reading.entry(key).or_insert(hour);
            if past {
                return Ok(Self {
                    by_reading,
                    past_end: hour,
                });
            }
        }

        Err(IngestError::ExportTooLong {
            hours: MAX_EXPORT_HOURS,
        })
    }

    /// The readings of an export with nothing to place. **Calculation.**
    ///
    /// An export whose every row was skipped or rejected has no range, and
    /// asking the clock to render one would be inventing one.
    fn none() -> Self {
        Self {
            by_reading: BTreeMap::new(),
            past_end: HourIndex::new(0),
        }
    }

    /// The hour index a local reading falls in. **Calculation.**
    fn hour_of(&self, reading: &str) -> Option<HourIndex> {
        self.by_reading.get(hour_key(reading)).copied()
    }

    /// The export's hours, as the range a crossing is looked for over.
    /// **Calculation.**
    fn range(&self) -> Range<HourIndex> {
        HourIndex::new(0)..self.past_end
    }

    /// The span this covers, as the refusal names it. **Calculation.**
    fn covers(&self, clock: &SiteClock) -> String {
        let last = HourIndex::new(self.past_end.get().saturating_sub(1));
        format!(
            "{} to {}",
            clock.local_iso(HourIndex::new(0)),
            clock.local_iso(last)
        )
    }
}

/// The `YYYY-MM-DDTHH` a local reading starts with. **Calculation.**
fn hour_key(reading: &str) -> &str {
    reading.get(..READING_HOUR_WIDTH).unwrap_or(reading)
}

/// Turn a whole `sacct` export into jobs on the contract. **Calculation.**
///
/// `allow_dst_crossing` is the operator saying they know the export spans a
/// clock change and want it read anyway; without it a crossing is refused by
/// name, because an hour that happened twice makes two rows' order a guess.
///
/// An export that yields no jobs is not an error here: the tally and the
/// refusals are exactly what the operator needs to see, and a caller that
/// wanted a jobs file decides for itself that an empty one is no use. Only a
/// file with no rows at all has nothing to report.
///
/// # Errors
/// [`IngestError::NoRows`] for a file with no rows,
/// [`IngestError::DstCrossing`] for a span the operator has not allowed, and
/// [`IngestError::ExportTooLong`] for a `Submit` past this module's reach.
pub fn map_export(
    text: &str,
    clock: &SiteClock,
    map: &TierMap,
    allow_dst_crossing: bool,
) -> Result<Conversion, IngestError> {
    let read = Read::of(text, map);
    if read.rows == 0 {
        return Err(IngestError::NoRows);
    }

    let Some(latest) = read.candidates.iter().map(|job| hour_key(job.submit)).max() else {
        return Ok(read.placed(clock, &LocalHours::none()));
    };

    let hours = LocalHours::of(clock, latest)?;
    match clock.crossing_report(hours.range()) {
        Some(crossing) if !allow_dst_crossing => Err(crossing_refused(clock, crossing)),
        Some(_) | None => Ok(read.placed(clock, &hours)),
    }
}

/// Every row of an export, classified but not yet placed in time. **Data.**
///
/// `skipped` counts the rows that are not faults; `kept` and `rejected` are
/// not counted here, because a candidate can still fail to be placed and the
/// tally must describe what the conversion produced, not what it hoped for.
#[derive(Debug, Clone, Default, PartialEq)]
struct Read<'a> {
    candidates: Vec<Candidate<'a>>,
    rejected: Vec<Rejected>,
    /// Every non-blank line the export had.
    rows: usize,
    skipped: Skips,
}

/// The rows that are not jobs, by what made them so. **Data.**
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Skips {
    steps: usize,
    unfinished: usize,
    non_gpu: usize,
}

impl Skips {
    /// This skip, counted. **Calculation.**
    fn and(self, skip: Skip) -> Self {
        match skip {
            Skip::Step => Self {
                steps: self.steps + 1,
                ..self
            },
            Skip::Unfinished => Self {
                unfinished: self.unfinished + 1,
                ..self
            },
            Skip::NoGpu => Self {
                non_gpu: self.non_gpu + 1,
                ..self
            },
        }
    }
}

impl<'a> Read<'a> {
    /// Classify every row, then order the candidates by `(Submit, JobIDRaw)`.
    /// **Calculation.**
    fn of(text: &'a str, map: &TierMap) -> Self {
        let mut read = text
            .lines()
            .enumerate()
            .map(|(offset, line)| (offset + 1, line.trim_end()))
            .filter(|(_, line)| !line.is_empty())
            .fold(Self::default(), |read, (row, line)| {
                read.and(row, classify(row, line, map))
            });

        read.candidates.sort_by(|left, right| {
            (left.submit, left.job_number).cmp(&(right.submit, right.job_number))
        });
        read
    }

    /// One classified row, in whichever pile it belongs in. **Calculation.**
    fn and(mut self, row: usize, outcome: Result<Classified<'a>, RejectCause>) -> Self {
        match outcome {
            Ok(Classified::Kept(candidate)) => self.candidates.push(candidate),
            Ok(Classified::Skipped(skip)) => self.skipped = self.skipped.and(skip),
            Err(cause) => self.rejected.push(Rejected { row, cause }),
        }
        Self {
            rows: self.rows + 1,
            ..self
        }
    }

    /// Place every candidate in the export's hours, and number them.
    /// **Calculation.**
    ///
    /// Ids are dense and follow `(Submit, JobIDRaw)`; the file is then sorted
    /// by `(arrival_h, id)`, which is the order every other jobs file is in.
    fn placed(self, clock: &SiteClock, hours: &LocalHours) -> Conversion {
        let Self {
            candidates,
            mut rejected,
            rows,
            skipped,
        } = self;

        let mut jobs: Vec<Job> = Vec::with_capacity(candidates.len());
        let mut id_map = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            match candidate.place(clock, hours, next_id(&jobs)) {
                Ok((job, raw)) => {
                    jobs.push(job);
                    id_map.push((job.id, raw));
                }
                Err(cause) => rejected.push(Rejected {
                    row: candidate.row,
                    cause,
                }),
            }
        }

        jobs.sort_by_key(|job| (job.arrival, job.id));
        rejected.sort_by_key(|refusal| refusal.row);
        Conversion {
            tally: Tally {
                read: rows,
                kept: jobs.len(),
                steps: skipped.steps,
                unfinished: skipped.unfinished,
                non_gpu: skipped.non_gpu,
                rejected: rejected.len(),
            },
            jobs,
            id_map,
            rejected,
        }
    }
}

/// What one row turned out to be. **Data.**
#[derive(Debug, Clone, Copy, PartialEq)]
enum Classified<'a> {
    Kept(Candidate<'a>),
    Skipped(Skip),
}

/// The id the next job takes: dense, from 0. **Calculation.**
///
/// The saturation is unreachable: a conversion holds every kept row in
/// memory, so a job count past `u32::MAX` would have exhausted the machine
/// long before it exhausted the id.
fn next_id(jobs: &[Job]) -> JobId {
    JobId::new(u32::try_from(jobs.len()).unwrap_or(u32::MAX))
}

/// The refusal a clock crossing earns. **Calculation.**
fn crossing_refused(clock: &SiteClock, crossing: DstCrossing) -> IngestError {
    let (kind, at) = match crossing {
        DstCrossing::Missing { at } => ("never happens", at),
        DstCrossing::Doubled { at } => ("happens twice", at),
    };
    IngestError::DstCrossing {
        local: clock.local_iso(at),
        kind,
    }
}

impl Candidate<'_> {
    /// This candidate as a job at its arrival hour. **Calculation.**
    fn place(
        &self,
        clock: &SiteClock,
        hours: &LocalHours,
        id: JobId,
    ) -> Result<(Job, String), RejectCause> {
        let arrival = hours
            .hour_of(self.submit)
            .ok_or_else(|| RejectCause::Outside {
                value: self.submit.to_owned(),
                covers: hours.covers(clock),
            })?;
        let deadline = arrival
            .offset_by(self.window_hours)
            .ok_or(RejectCause::Deadline {
                window: self.window_hours,
            })?;

        Ok((
            Job {
                id,
                tier: self.tier,
                gpus: Gpus::new(self.gpus),
                actual_hours: Hours::new(self.actual_hours),
                arrival,
                requested_hours: self.requested_hours,
                deadline,
                pad_hours: self.pad_hours,
            },
            self.job_id.to_owned(),
        ))
    }
}

/// One row, as whatever it turns out to be. **Calculation.**
fn classify<'a>(row: usize, line: &'a str, map: &TierMap) -> Result<Classified<'a>, RejectCause> {
    let fields = SacctRow::parse(line)?;

    let Some(job_number) = fields.job_number() else {
        return Ok(Classified::Skipped(Skip::Step));
    };
    if !fields.is_finished() {
        return Ok(Classified::Skipped(Skip::Unfinished));
    }
    let gpus = gpus_of(fields.tres)?;
    if gpus == 0 {
        return Ok(Classified::Skipped(Skip::NoGpu));
    }

    let rule = map
        .rule_for(fields.qos, fields.partition)
        .ok_or_else(|| RejectCause::Unmapped {
            qos: fields.qos.to_owned(),
            partition: fields.partition.to_owned(),
        })?;

    Ok(Classified::Kept(Candidate {
        row,
        job_id: fields.job_id,
        job_number,
        submit: local_reading(fields.submit)?,
        tier: rule.tier,
        gpus: f64::from(gpus),
        actual_hours: actual_hours_of(fields.elapsed)?,
        requested_hours: requested_hours_of(fields.timelimit)?,
        window_hours: rule.window_hours(),
        pad_hours: rule.pad_hours(),
    }))
}

impl<'a> SacctRow<'a> {
    /// One `|`-separated line, split into the fields anything reads.
    /// **Calculation.**
    fn parse(line: &'a str) -> Result<Self, RejectCause> {
        let fields: Vec<&str> = line.split('|').collect();
        let [
            job_id,
            submit,
            _start,
            _end,
            elapsed,
            tres,
            timelimit,
            qos,
            partition,
            state,
            _user,
        ] = fields[..]
        else {
            return Err(RejectCause::Arity {
                expected: FORMAT.len(),
                actual: fields.len(),
            });
        };

        Ok(Self {
            job_id: job_id.trim(),
            submit: submit.trim(),
            elapsed: elapsed.trim(),
            tres: tres.trim(),
            timelimit: timelimit.trim(),
            qos: qos.trim(),
            partition: partition.trim(),
            state: state.trim(),
        })
    }

    /// `JobIDRaw` as a number, or `None` for a job step. **Calculation.**
    ///
    /// A step — `1234.batch`, `1234.0` — is part of a job whose own row is in
    /// the same export, so counting it would count the job twice. A `JobIDRaw`
    /// that is neither is still skipped rather than rejected: Slurm writes
    /// `1234+0` for a heterogeneous job, which is a shape this module does not
    /// read and a fault nobody in the export made.
    fn job_number(&self) -> Option<u64> {
        self.job_id.parse().ok()
    }

    /// Whether the job ran and is over. **Calculation.**
    ///
    /// `State` can carry a trailing clause — `CANCELLED by 1001` — so the
    /// first word is what is matched.
    fn is_finished(&self) -> bool {
        self.state
            .split_whitespace()
            .next()
            .is_some_and(|state| FINISHED.contains(&state))
    }
}

/// GPUs allocated, out of `AllocTRES`. **Calculation.**
///
/// Slurm writes both a total and a per-type breakdown when a job asked for a
/// typed GPU — `gres/gpu=3,gres/gpu:a100=3` — so the total wins where it is
/// present and the types are summed only where it is not. Adding both would
/// bill the job twice.
fn gpus_of(tres: &str) -> Result<u32, RejectCause> {
    let entries: Vec<(&str, &str)> = tres
        .split(',')
        .filter(|entry| !entry.is_empty())
        .map(|entry| entry.split_once('=').unwrap_or((entry, "")))
        .collect();

    let total = gpu_sum(&entries, |key| key == GPU_KEY)?;
    if total > 0 {
        return Ok(total);
    }
    gpu_sum(&entries, |key| key.starts_with(GPU_TYPE_PREFIX))
}

/// The GPU counts of the `AllocTRES` entries a predicate picks.
/// **Calculation.**
fn gpu_sum(entries: &[(&str, &str)], wanted: impl Fn(&str) -> bool) -> Result<u32, RejectCause> {
    entries
        .iter()
        .filter(|(key, _)| wanted(key))
        .map(|(_, count)| {
            count.parse::<u32>().map_err(|_| RejectCause::Field {
                field: "AllocTRES",
                value: (*count).to_owned(),
                expected: "a GPU count",
            })
        })
        .sum()
}

/// `Submit`, once it is a reading a clock could have shown. **Calculation.**
///
/// The shape is checked here, with the other fields, rather than left to the
/// lookup: `sacct` writes `Unknown` for a job that never started, and a value
/// that is not a reading at all must not become the bound the export's hours
/// are built out to.
///
/// A local reading is an instant as soon as a zone is added, and *which* zone
/// is irrelevant to the question "is this a reading at all" — so the domain's
/// one timestamp parser is asked with `Z` and its answer is thrown away. That
/// keeps this module from carrying a second opinion about what a date is,
/// including about 30 February.
fn local_reading(submit: &str) -> Result<&str, RejectCause> {
    if UtcSeconds::from_iso(&format!("{submit}Z")).is_err() {
        return Err(RejectCause::Field {
            field: "Submit",
            value: submit.to_owned(),
            expected: "a local clock reading, like 2026-03-02T08:15:00",
        });
    }
    Ok(submit)
}

/// The job's real runtime in hours, out of `ElapsedRaw` seconds.
/// **Calculation.**
fn actual_hours_of(elapsed: &str) -> Result<f64, RejectCause> {
    let field = |expected| RejectCause::Field {
        field: "ElapsedRaw",
        value: elapsed.to_owned(),
        expected,
    };
    let seconds: u32 = elapsed.parse().map_err(|_| field("a count of seconds"))?;
    if seconds == 0 {
        return Err(field("a runtime: a job that ran no seconds ran nothing"));
    }
    Ok(f64::from(seconds) / SECONDS_PER_HOUR)
}

/// The hours the operator asked for, out of `TimelimitRaw` minutes.
/// **Calculation.**
///
/// Rounded up: a 90-minute limit asks for two hours of the cluster. A job with
/// no limit carries 0, which the duration belief reads as "unknown" and gives
/// its cold-start value.
fn requested_hours_of(timelimit: &str) -> Result<u32, RejectCause> {
    if timelimit == UNLIMITED {
        return Ok(0);
    }
    let minutes: u32 = timelimit.parse().map_err(|_| RejectCause::Field {
        field: "TimelimitRaw",
        value: timelimit.to_owned(),
        expected: "a count of minutes, or UNLIMITED",
    })?;
    Ok(minutes.div_ceil(MINUTES_PER_HOUR))
}

#[cfg(test)]
mod tests {
    use loadshift_domain::{LocalMidnight, OffsetMinutes, SiteClock, Transition, UtcSeconds};

    use super::*;

    /// `2026-03-02T00:00:00-06:00`, a Monday six days before the clock springs
    /// forward, and the `t0` the committed sample is exported against.
    const MARCH_MONDAY: i64 = 1_772_431_200;
    /// `2026-03-08T02:00:00-06:00`, where the local clock jumps to 03:00.
    const SPRING_FORWARD: i64 = 1_772_956_800;
    /// `2025-11-02T02:00:00-05:00`, the transition before the pilot's range.
    const PREVIOUS_FALL_BACK: i64 = 1_762_066_800;

    /// The mapping the committed sample is converted with.
    const RULES: &str = r#"
# urgent work is Tier 1
[[rule]]
qos = "urgent"
tier = 1

[[rule]]
partition = "batch"
tier = 2

[[rule]]
tier = 2
"#;

    fn central(t0: i64) -> SiteClock {
        let transition = |at_utc, minutes| Transition {
            at_utc: UtcSeconds::new(at_utc),
            offset_after: OffsetMinutes::new(minutes).expect("a Central offset"),
        };
        SiteClock::new(
            LocalMidnight::at_utc(UtcSeconds::new(t0)),
            vec![
                transition(PREVIOUS_FALL_BACK, -360),
                transition(SPRING_FORWARD, -300),
            ],
        )
        .expect("a Central local midnight is a clock")
    }

    fn rules() -> TierMap {
        TierMap::parse(RULES).expect("the committed mapping parses")
    }

    /// One `sacct` line on the documented format, with every field a given
    /// test does not care about already filled in: a finished one-GPU job.
    ///
    /// `Start`, `End` and `User` are constant, because nothing maps them.
    #[derive(Debug, Clone, Copy)]
    struct Row<'a> {
        job_id: &'a str,
        submit: &'a str,
        elapsed: &'a str,
        tres: &'a str,
        timelimit: &'a str,
        qos: &'a str,
        partition: &'a str,
        state: &'a str,
    }

    impl Default for Row<'_> {
        fn default() -> Self {
            Self {
                job_id: "101",
                submit: "2026-03-02T08:15:00",
                elapsed: "7200",
                tres: "cpu=8,gres/gpu=1",
                timelimit: "180",
                qos: "urgent",
                partition: "gpu",
                state: "COMPLETED",
            }
        }
    }

    impl Row<'_> {
        fn line(self) -> String {
            [
                self.job_id,
                self.submit,
                "2026-03-02T08:00:00",
                "2026-03-02T09:00:00",
                self.elapsed,
                self.tres,
                self.timelimit,
                self.qos,
                self.partition,
                self.state,
                "ada",
            ]
            .join("|")
        }
    }

    /// A finished one-GPU job that differs only in the four fields a mapping
    /// rule looks at.
    fn job_line(job_id: &str, submit: &str, qos: &str, partition: &str) -> String {
        Row {
            job_id,
            submit,
            qos,
            partition,
            ..Row::default()
        }
        .line()
    }

    fn convert(lines: &[String]) -> Conversion {
        map_export(&lines.join("\n"), &central(MARCH_MONDAY), &rules(), false)
            .expect("the export converts")
    }

    fn refusal(lines: &[String]) -> Rejected {
        convert(lines)
            .rejected
            .first()
            .cloned()
            .expect("the row is rejected")
    }

    #[test]
    fn a_finished_gpu_job_becomes_a_row_on_the_jobs_contract() {
        let converted = convert(&[job_line("101", "2026-03-02T08:15:00", "urgent", "gpu")]);

        let job = converted.jobs.first().copied().expect("one job");
        assert_eq!(job.id, JobId::new(0));
        assert_eq!(job.tier, Tier::Latency);
        assert_eq!(job.gpus, Gpus::new(1.0));
        assert_eq!(job.actual_hours, Hours::new(2.0));
        assert_eq!(job.arrival, HourIndex::new(8));
        assert_eq!(job.requested_hours, 3);
        assert_eq!(job.deadline, HourIndex::new(8 + LATENCY_WINDOW_HOURS));
        assert_eq!(job.pad_hours, LATENCY_PAD_HOURS);
        assert_eq!(converted.id_map, vec![(JobId::new(0), "101".to_owned())]);
    }

    #[test]
    fn tier_two_carries_the_seven_day_window_and_the_pad_that_goes_with_it() {
        let converted = convert(&[job_line("101", "2026-03-02T08:15:00", "normal", "batch")]);

        let job = converted.jobs.first().copied().expect("one job");
        assert_eq!(job.tier, Tier::Flexible);
        assert_eq!(job.deadline, HourIndex::new(8 + FLEXIBLE_WINDOW_HOURS));
        assert_eq!(job.pad_hours, FLEXIBLE_PAD_HOURS);
    }

    #[test]
    fn a_rule_may_set_the_window_the_site_actually_sells() {
        let map = TierMap::parse("[[rule]]\ntier = 2\nwindow_hours = 48\n").expect("a rule");
        let export = job_line("101", "2026-03-02T08:15:00", "normal", "batch");

        let converted = map_export(&export, &central(MARCH_MONDAY), &map, false).expect("converts");

        assert_eq!(
            converted.jobs.first().expect("one job").deadline,
            HourIndex::new(8 + 48)
        );
    }

    #[test]
    fn ids_are_dense_in_submit_order_and_the_file_is_sorted_by_arrival() {
        let converted = convert(&[
            job_line("12", "2026-03-02T09:30:00", "urgent", "gpu"),
            job_line("9", "2026-03-02T08:00:00", "urgent", "gpu"),
            job_line("10", "2026-03-02T08:00:00", "urgent", "gpu"),
        ]);

        // `9` precedes `10` because JobIDRaw orders numerically, not as text.
        assert_eq!(
            converted.id_map,
            vec![
                (JobId::new(0), "9".to_owned()),
                (JobId::new(1), "10".to_owned()),
                (JobId::new(2), "12".to_owned()),
            ]
        );
        let order: Vec<(u32, u32)> = converted
            .jobs
            .iter()
            .map(|job| (job.arrival.get(), job.id.get()))
            .collect();
        assert_eq!(order, vec![(8, 0), (8, 1), (9, 2)]);
    }

    #[test]
    fn a_job_step_is_skipped_rather_than_counted_twice() {
        let converted = convert(&[
            job_line("101", "2026-03-02T08:15:00", "urgent", "gpu"),
            job_line("101.batch", "2026-03-02T08:15:00", "urgent", "gpu"),
        ]);

        assert_eq!(converted.tally.steps, 1);
        assert_eq!(converted.tally.kept, 1);
        assert!(converted.rejected.is_empty(), "a step is not a fault");
    }

    #[test]
    fn a_job_still_running_at_export_time_is_dropped_and_counted() {
        let running = Row {
            elapsed: "600",
            state: "RUNNING",
            ..Row::default()
        }
        .line();

        let converted = convert(&[
            job_line("100", "2026-03-02T08:15:00", "urgent", "gpu"),
            running,
        ]);

        assert_eq!(converted.tally.unfinished, 1);
        assert_eq!(converted.tally.kept, 1);
    }

    #[test]
    fn a_cancelled_job_carries_its_clause_and_is_still_finished() {
        let cancelled = Row {
            state: "CANCELLED by 1001",
            ..Row::default()
        }
        .line();

        assert_eq!(convert(&[cancelled]).tally.kept, 1);
    }

    #[test]
    fn a_job_that_asked_for_no_gpu_is_skipped_and_counted() {
        let cpu_only = Row {
            tres: "cpu=8,mem=64G",
            ..Row::default()
        }
        .line();

        let converted = convert(&[
            job_line("100", "2026-03-02T08:15:00", "urgent", "gpu"),
            cpu_only,
        ]);

        assert_eq!(converted.tally.non_gpu, 1);
        assert_eq!(converted.tally.kept, 1);
    }

    #[test]
    fn typed_gpus_are_summed_and_a_stated_total_is_not_added_to_its_own_breakdown() {
        assert_eq!(gpus_of("cpu=8,gres/gpu:a100=2,gres/gpu:v100=1"), Ok(3));
        assert_eq!(gpus_of("cpu=8,gres/gpu=3,gres/gpu:a100=3"), Ok(3));
        assert_eq!(gpus_of("cpu=8,mem=64G"), Ok(0));
    }

    #[test]
    fn a_row_with_the_wrong_field_count_names_the_row_and_the_counts() {
        let short = "101|2026-03-02T08:15:00|7200|COMPLETED".to_owned();

        let rejected = refusal(&[
            job_line("100", "2026-03-02T08:15:00", "urgent", "gpu"),
            short,
        ]);

        assert_eq!(rejected.row, 2);
        assert_eq!(rejected.cause.field(), "the row");
        assert_eq!(
            rejected.line(),
            "row 2: the row has 4 fields, and the documented sacct format has 11"
        );
    }

    #[test]
    fn an_unreadable_gpu_count_names_the_row_and_alloctres() {
        let rejected = refusal(&[Row {
            tres: "gres/gpu=many",
            ..Row::default()
        }
        .line()]);

        assert_eq!(rejected.row, 1);
        assert_eq!(rejected.cause.field(), "AllocTRES");
        assert!(rejected.line().starts_with("row 1: field `AllocTRES`"));
    }

    #[test]
    fn a_job_that_ran_no_seconds_names_the_row_and_elapsedraw() {
        let rejected = refusal(&[Row {
            elapsed: "0",
            ..Row::default()
        }
        .line()]);

        assert_eq!(rejected.row, 1);
        assert_eq!(rejected.cause.field(), "ElapsedRaw");
    }

    #[test]
    fn an_unreadable_runtime_names_the_row_and_elapsedraw() {
        let rejected = refusal(&[Row {
            elapsed: "02:00:00",
            ..Row::default()
        }
        .line()]);

        assert_eq!(rejected.cause.field(), "ElapsedRaw");
        assert!(
            rejected.line().contains("a count of seconds"),
            "got {}",
            rejected.line()
        );
    }

    #[test]
    fn an_unreadable_time_limit_names_the_row_and_timelimitraw() {
        let rejected = refusal(&[Row {
            timelimit: "3:00:00",
            ..Row::default()
        }
        .line()]);

        assert_eq!(rejected.cause.field(), "TimelimitRaw");
    }

    #[test]
    fn a_time_limit_is_rounded_up_to_whole_hours_and_unlimited_is_unknown() {
        assert_eq!(requested_hours_of("180"), Ok(3));
        assert_eq!(requested_hours_of("90"), Ok(2));
        assert_eq!(requested_hours_of("1"), Ok(1));
        assert_eq!(requested_hours_of(UNLIMITED), Ok(0));
    }

    #[test]
    fn a_row_no_rule_matches_names_the_row_and_the_qos() {
        let map = TierMap::parse("[[rule]]\nqos = \"urgent\"\ntier = 1\n").expect("one rule");
        let export = job_line("101", "2026-03-02T08:15:00", "normal", "batch");

        let converted =
            map_export(&export, &central(MARCH_MONDAY), &map, false).expect("the export converts");

        let rejected = converted.rejected.first().expect("the row is rejected");
        assert_eq!(rejected.cause.field(), "QOS");
        assert_eq!(
            rejected.line(),
            "row 1: field `QOS` holds `normal` and `Partition` holds `batch`, \
             which no rule matches"
        );
    }

    #[test]
    fn a_submit_before_t0_names_the_row_the_submit_and_the_span_covered() {
        let rejected = refusal(&[
            job_line("100", "2026-03-02T08:15:00", "urgent", "gpu"),
            job_line("101", "2026-03-01T23:00:00", "urgent", "gpu"),
        ]);

        assert_eq!(rejected.row, 2);
        assert_eq!(rejected.cause.field(), "Submit");
        assert!(
            rejected.line().contains("2026-03-02T00:00:00-06:00"),
            "the refusal names what is covered: {}",
            rejected.line()
        );
    }

    #[test]
    fn an_unreadable_submit_is_rejected_rather_than_placed_at_hour_zero() {
        let rejected = refusal(&[
            job_line("100", "2026-03-02T08:15:00", "urgent", "gpu"),
            job_line("101", "Unknown", "urgent", "gpu"),
        ]);

        assert_eq!(rejected.cause.field(), "Submit");
    }

    #[test]
    fn the_tally_adds_up_to_the_file_the_operator_sent() {
        let converted = convert(&[
            job_line("100", "2026-03-02T08:15:00", "urgent", "gpu"),
            job_line("100.batch", "2026-03-02T08:15:00", "urgent", "gpu"),
            Row {
                job_id: "101",
                submit: "2026-03-02T09:00:00",
                tres: "cpu=8",
                ..Row::default()
            }
            .line(),
            Row {
                job_id: "102",
                submit: "2026-03-02T09:00:00",
                state: "RUNNING",
                ..Row::default()
            }
            .line(),
            "103|2026-03-02T09:00:00|broken".to_owned(),
        ]);

        let Tally {
            read,
            kept,
            steps,
            unfinished,
            non_gpu,
            rejected,
        } = converted.tally;
        assert_eq!(read, 5);
        assert_eq!(kept + steps + unfinished + non_gpu + rejected, read);
        assert_eq!(
            (kept, steps, unfinished, non_gpu, rejected),
            (1, 1, 1, 1, 1)
        );
    }

    #[test]
    fn an_export_that_spans_a_clock_change_is_refused_by_name() {
        let export = [
            job_line("100", "2026-03-07T08:00:00", "urgent", "gpu"),
            job_line("101", "2026-03-09T08:00:00", "urgent", "gpu"),
        ]
        .join("\n");

        let error = map_export(&export, &central(MARCH_MONDAY), &rules(), false)
            .expect_err("a crossing is refused");

        assert!(
            matches!(&error, IngestError::DstCrossing { local, kind }
                if local.starts_with("2026-03-08T03:00:00") && *kind == "never happens"),
            "got {error}"
        );
    }

    #[test]
    fn an_operator_who_knows_about_the_clock_change_may_read_it_anyway() {
        let export = [
            job_line("100", "2026-03-07T08:00:00", "urgent", "gpu"),
            job_line("101", "2026-03-09T08:00:00", "urgent", "gpu"),
        ]
        .join("\n");

        let converted =
            map_export(&export, &central(MARCH_MONDAY), &rules(), true).expect("allowed");

        // The 9th is seven days on, less the hour the local clock skipped.
        assert_eq!(
            converted.jobs.last().expect("two jobs").arrival,
            HourIndex::new(7 * 24 + 8 - 1)
        );
    }

    #[test]
    fn a_window_that_would_push_a_deadline_past_the_index_is_rejected_not_wrapped() {
        // `window_of` caps a rule at a year, which is what keeps this
        // unreachable from a mapping file; the refusal is still the one a
        // candidate carrying such a window would get, so it is tested here
        // rather than left to be discovered.
        let clock = central(MARCH_MONDAY);
        let hours = LocalHours::of(&clock, "2026-03-02T08").expect("the range is short");
        let candidate = Candidate {
            row: 1,
            job_id: "101",
            job_number: 101,
            submit: "2026-03-02T08:15:00",
            tier: Tier::Flexible,
            gpus: 1.0,
            actual_hours: 2.0,
            requested_hours: 3,
            window_hours: u32::MAX,
            pad_hours: FLEXIBLE_PAD_HOURS,
        };

        let refused = candidate
            .place(&clock, &hours, JobId::new(0))
            .expect_err("hour 8 plus u32::MAX is no hour");

        assert_eq!(refused.field(), "Submit");
        assert!(
            matches!(refused, RejectCause::Deadline { window } if window == u32::MAX),
            "got {refused}"
        );
    }

    #[test]
    fn a_file_with_no_rows_at_all_is_an_error_rather_than_an_empty_conversion() {
        let error = map_export("\n\n", &central(MARCH_MONDAY), &rules(), false)
            .expect_err("nothing was sent");

        assert!(matches!(error, IngestError::NoRows), "got {error}");
    }

    #[test]
    fn an_export_whose_every_row_fails_reports_the_failures_rather_than_nothing() {
        let steps = job_line("100.batch", "2026-03-02T08:15:00", "urgent", "gpu");
        let broken = "103|2026-03-02T09:00:00|broken".to_owned();

        let converted = convert(&[steps, broken]);

        assert!(converted.jobs.is_empty());
        assert_eq!(converted.tally.steps, 1);
        assert_eq!(
            converted.rejected.first().map(Rejected::line),
            Some("row 2: the row has 3 fields, and the documented sacct format has 11".to_owned())
        );
    }

    #[test]
    fn a_mapping_file_is_read_in_order_with_comments_and_blank_lines_ignored() {
        let map = rules();

        assert_eq!(
            map.rule_for("urgent", "gpu").map(|rule| rule.tier),
            Some(Tier::Latency)
        );
        assert_eq!(
            map.rule_for("normal", "batch").map(|rule| rule.tier),
            Some(Tier::Flexible)
        );
        assert_eq!(
            map.rule_for("anything", "anywhere").map(|rule| rule.tier),
            Some(Tier::Flexible)
        );
    }

    #[test]
    fn a_mapping_file_that_maps_nothing_is_refused_rather_than_rejecting_every_row() {
        assert!(matches!(
            TierMap::parse("# nothing but a comment\n"),
            Err(IngestError::NoTierRules)
        ));
    }

    #[test]
    fn a_mapping_file_names_the_line_and_the_fault() {
        let faults = [
            ("[[rule]]\nqos = urgent\ntier = 1\n", 2, "a quoted string"),
            ("[[rule]]\ntier = two\n", 2, "a whole number"),
            ("[[rule]]\nqueue = \"gpu\"\ntier = 1\n", 2, "not a rule key"),
            ("tier = 1\n", 1, "before any [[rule]]"),
            ("[[rule]]\ntier = 0\n", 2, "tier 1 or tier 2"),
            ("[[rule]]\ntier = 2\nwindow_hours = 0\n", 3, "a window of"),
            ("[[rule]]\nnonsense\n", 2, "is not a [[rule]] header"),
        ];

        for (text, line, wanted) in faults {
            let error = TierMap::parse(text).expect_err("a fault");

            assert!(
                matches!(&error, IngestError::TierMap { line: at, .. } if *at == line)
                    && error.to_string().contains(wanted),
                "{text:?} should fail on line {line} with {wanted}: got {error}"
            );
        }
    }
}
