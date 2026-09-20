# Contributing

## Checks

Run these locally before opening a pull request:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all
```

A warning is an error.

## Pull requests

Rebase the branch onto `main` and merge by rebase. Do not squash. Do not add a merge commit.

## Writing standard

* Write for a reader who has never heard of Volt Labs. Define each term the first time it appears.
* No internal code names, customer names, site names, ticket numbers or meeting references.
* Example data is synthetic and says so.
* No performance, savings or financial figures.
* Short sentences. Say what the code does and what it refuses to do.
