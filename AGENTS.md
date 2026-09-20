# Rules for this repository

This crate is a Rust library. It does no network I/O, no file I/O, and does not read a clock, unless the README names a feature that turns that behaviour on.

Do not write customer names, site names, internal project names, ticket numbers, or meeting references anywhere in this repository, including commit messages.

Example and test data is synthetic. Say so in a comment or a README line next to the data.

Do not write performance, savings, or financial figures.

Every public function has a rustdoc comment and at least one test. Compiler and clippy warnings are errors.
