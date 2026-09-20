//! Consumer contract: the public row shape, field names, and sample counts.
//!
//! Pins named fields, variants, and behaviour. New `Record` fields, enum
//! variants, and `Error` fields are allowed; renaming or removing a named
//! item is not.
//!
//! Synthetic fixtures; nothing here was measured.

use std::fs;
use std::path::{Path, PathBuf};

use sacct_parse::{parse_file, parse_row, Error, Limit, Record, State};

const OFFSET: i64 = -21_600;

const COMPLETED_ROW: &str = "101|2026-03-02T08:15:00|2026-03-02T08:00:00|2026-03-02T09:00:00|7200|cpu=8,gres/gpu=1|180|normal|batch|COMPLETED|u01";

/// One synthetic `--parsable2` row. Fields not under test stay fixed.
fn row(job_id: &str, elapsed: &str, tres: &str, timelimit: &str, state: &str) -> String {
    [
        job_id,
        "2026-03-02T08:15:00",
        "2026-03-02T08:00:00",
        "2026-03-02T09:00:00",
        elapsed,
        tres,
        timelimit,
        "normal",
        "batch",
        state,
        "u01",
    ]
    .join("|")
}

fn parsed(line: &str) -> Record {
    parse_row(line, OFFSET).expect("the row parses")
}

#[test]
fn completed_row_pins_record_fields() {
    let record = parsed(COMPLETED_ROW);

    assert_eq!(record.gpu_count(), 1);
    assert!(!record.is_step());
    assert_eq!(record.job_id, "101");
    assert_eq!(record.submit, 1_772_460_900);
    assert_eq!(record.start, 1_772_460_000);
    assert_eq!(record.end, 1_772_463_600);
    assert_eq!(record.elapsed_seconds, 7200);
    assert_eq!(record.alloc_tres.get("gres/gpu"), Some(&1));
    assert_eq!(record.timelimit, Limit::Minutes(180));
    assert_eq!(record.qos, "normal");
    assert_eq!(record.partition, "batch");
    assert_eq!(record.state, State::Completed);
    assert_eq!(record.user, "u01");
}

#[test]
fn unlimited_timelimit_is_limit_unlimited() {
    let record = parsed(&row(
        "102",
        "7200",
        "cpu=8,gres/gpu=1",
        "UNLIMITED",
        "COMPLETED",
    ));

    assert_eq!(record.timelimit, Limit::Unlimited);
}

#[test]
fn cancelled_by_clause_is_state_cancelled() {
    let record = parsed(&row(
        "103",
        "7200",
        "cpu=8,gres/gpu=1",
        "180",
        "CANCELLED by 1001",
    ));

    assert_eq!(record.state, State::Cancelled);
}

#[test]
fn typed_gpus_without_total_sum_to_three() {
    let record = parsed(&row(
        "104",
        "7200",
        "cpu=8,gres/gpu:a100=2,gres/gpu:h100=1",
        "180",
        "COMPLETED",
    ));

    assert_eq!(record.gpu_count(), 3);
}

#[test]
fn gpu_total_wins_over_typed_breakdown() {
    let record = parsed(&row(
        "105",
        "7200",
        "cpu=8,gres/gpu=3,gres/gpu:a100=3",
        "180",
        "COMPLETED",
    ));

    assert_eq!(record.gpu_count(), 3);
}

#[test]
fn cpu_and_mem_only_has_zero_gpus() {
    let record = parsed(&row("106", "7200", "cpu=8,mem=64G", "180", "COMPLETED"));

    assert_eq!(record.gpu_count(), 0);
}

#[test]
fn dotted_job_ids_are_steps() {
    assert!(parsed(&row("1041.batch", "7200", "cpu=8", "180", "COMPLETED")).is_step());
    assert!(parsed(&row("1044.0", "7200", "cpu=8", "180", "COMPLETED")).is_step());
    assert!(!parsed(&row("1041", "7200", "cpu=8", "180", "COMPLETED")).is_step());
}

#[test]
fn ten_fields_is_arity_error_on_row_one() {
    let line = "101|2026-03-02T08:15:00|2026-03-02T08:00:00|2026-03-02T09:00:00|7200|cpu=8,gres/gpu=1|180|normal|batch|COMPLETED";

    match parse_row(line, OFFSET) {
        Err(Error::Arity {
            row,
            expected,
            actual,
            ..
        }) => {
            assert_eq!(row, 1);
            assert_eq!(expected, 11);
            assert_eq!(actual, 10);
        }
        other => panic!("expected Error::Arity on row 1, got {other:?}"),
    }
}

#[test]
fn bad_elapsed_raw_display_names_row_and_field() {
    let err = parse_row(&row("101", "abc", "cpu=8", "180", "COMPLETED"), OFFSET)
        .expect_err("ElapsedRaw=abc is rejected");
    let display = err.to_string();

    assert!(display.contains("row 1"), "{display}");
    assert!(display.contains("ElapsedRaw"), "{display}");
}

#[test]
fn bad_timelimit_raw_names_that_field() {
    match parse_row(
        &row("101", "7200", "cpu=8", "not-a-limit", "COMPLETED"),
        OFFSET,
    ) {
        Err(Error::Field {
            field: "TimelimitRaw",
            row: 1,
            ..
        }) => {}
        other => panic!("expected TimelimitRaw field error on row 1, got {other:?}"),
    }
}

#[test]
fn unknown_end_is_typed_field_error_naming_end() {
    let line = "101|2026-03-02T08:15:00|2026-03-02T08:00:00|Unknown|7200|cpu=8,gres/gpu=1|180|normal|batch|COMPLETED|u01";

    match parse_row(line, OFFSET) {
        Err(Error::Field {
            field: "End",
            row: 1,
            ..
        }) => {}
        other => panic!("expected End field error on row 1, got {other:?}"),
    }
}

#[test]
fn sample_file_parses_thirty_records_with_five_steps() {
    let text = include_str!("../testdata/sacct_sample.txt");
    let records = parse_file(text, OFFSET).expect("every sample row parses");

    assert_eq!(records.len(), 30);
    assert_eq!(records.iter().filter(|record| record.is_step()).count(), 5);
    assert!(records
        .iter()
        .any(|record| record.timelimit == Limit::Unlimited));
}

#[test]
fn named_limit_and_state_variants_still_exist() {
    assert!(matches!(Limit::Minutes(1), Limit::Minutes(_)));
    assert!(matches!(Limit::Unlimited, Limit::Unlimited));
    assert!(matches!(State::Completed, State::Completed));
    assert!(matches!(State::Timeout, State::Timeout));
    assert!(matches!(State::Failed, State::Failed));
    assert!(matches!(State::Cancelled, State::Cancelled));
    assert!(matches!(State::Running, State::Running));
    assert!(matches!(State::Pending, State::Pending));
    assert!(matches!(State::Other(String::new()), State::Other(_)));
}

#[test]
fn cargo_toml_runtime_dependencies_are_only_thiserror() {
    let names = dependency_names(include_str!("../Cargo.toml"));
    assert_eq!(names, ["thiserror"]);
}

#[test]
fn source_does_not_contain_forbidden_identifiers() {
    let forbidden = [
        "JobId",
        concat!("Ti", "er"),
        concat!("Site", "Clock"),
        "HashMap",
        "chrono",
        "Command::new(\"sacct\")",
    ];
    let files = rust_source_files();
    assert!(!files.is_empty(), "src must contain at least one .rs file");

    for path in files {
        let text = fs::read_to_string(&path).expect("source file is readable");
        for needle in forbidden {
            assert!(
                !text.contains(needle),
                "{} contains forbidden identifier {needle}",
                path.display()
            );
        }
    }
}

#[test]
fn testdata_readme_first_line_says_synthetic() {
    let first = include_str!("../testdata/README.md")
        .lines()
        .next()
        .expect("testdata README has a first line");
    assert!(
        first.to_ascii_lowercase().contains("synthetic"),
        "first line must say the sample is synthetic, got: {first}"
    );
}

fn dependency_names(toml: &str) -> Vec<&str> {
    let mut in_deps = false;
    let mut names = Vec::new();
    for line in toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_deps = trimmed == "[dependencies]";
            continue;
        }
        if in_deps && !trimmed.is_empty() && !trimmed.starts_with('#') {
            if let Some(name) = trimmed.split(['=', '.']).next() {
                let name = name.trim();
                if !name.is_empty() {
                    names.push(name);
                }
            }
        }
    }
    names
}

fn rust_source_files() -> Vec<PathBuf> {
    let mut pending = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).expect("src is readable") {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }
    files
}
