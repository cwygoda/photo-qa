# Photo QA - Justfile
# Run `just` to see all available commands

set dotenv-load

default:
    @just --list

# === Setup ===

# Bootstrap development environment
bootstrap:
    mise trust
    mise install
    lefthook install
    cargo check

# === Development ===

# Build the project
build:
    cargo build

# Build release version
build-release:
    cargo build --release

# Check the project without building
check:
    cargo check --all-targets

# Run the project
run *ARGS:
    cargo run -- {{ARGS}}

# === Testing ===

# Run all tests
test:
    cargo test

# Run all tests including accuracy benchmarks (requires datasets)
test-all: test bench-accuracy

# Run tests with output
test-verbose:
    cargo test -- --nocapture

# Run integration tests only
test-integration:
    cargo test --test '*'

# Run tests with coverage (requires cargo-llvm-cov)
test-coverage:
    cargo llvm-cov --html

# === Quality ===

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt --check

# Run clippy linter
lint:
    cargo clippy --all-targets -- -D warnings

# Fix clippy warnings automatically
lint-fix:
    cargo clippy --all-targets --fix --allow-dirty

# Run all quality checks
quality: fmt-check lint test

# === CI ===

# Run full CI pipeline
ci: fmt-check lint test build

# === Models ===

# Fetch ML models
fetch-models:
    cargo run -- models fetch

# List installed models
list-models:
    cargo run -- models list

# Print models directory
models-path:
    cargo run -- models path

# === Test Data ===

# Fetch test datasets
fetch-test-data:
    ./scripts/fetch-test-data.sh

# Fetch specific test dataset
fetch-test-data-set SET:
    ./scripts/fetch-test-data.sh {{SET}}

# === Release ===

# Dry run release build
release-dry-run:
    cargo build --release

# === Utilities ===

# Clean build artifacts
clean:
    cargo clean

# Update dependencies
update:
    cargo update

# Check for outdated dependencies (requires cargo-outdated)
outdated:
    cargo outdated

# Generate documentation
doc:
    cargo doc --no-deps --open

# Watch for changes and run tests
watch:
    cargo watch -x test

# Watch for changes and run check
watch-check:
    cargo watch -x check

# === Benchmarks ===

# Run accuracy benchmarks (requires datasets)
bench-accuracy:
    cargo test --test accuracy -- --ignored --nocapture

# Run accuracy benchmarks for specific module
bench-accuracy-blur:
    cargo test --test accuracy blur -- --ignored --nocapture

bench-accuracy-exposure:
    cargo test --test accuracy exposure -- --ignored --nocapture

bench-accuracy-eyes:
    cargo test --test accuracy eyes -- --ignored --nocapture
