# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
just bootstrap    # First-time setup: mise trust, install tools, lefthook
just build        # cargo build
just test         # cargo test
just lint         # cargo clippy --all-targets -- -D warnings
just fmt          # cargo fmt
just ci           # fmt-check + lint + test + build

# Run single test
cargo test test_name

# Run tests in specific crate
cargo test -p photo-qa-core

# Run CLI
cargo run -- /path/to/images
cargo run -- --help
```

## Architecture

Hexagonal architecture with three crates:

```
photo-qa-core       Domain logic, ports (traits), QA modules
    ├── domain/     Core types: Issue, AnalysisResult, QaModule trait
    ├── modules/    BlurModule, ExposureModule, EyesModule
    ├── ports/      ImageSource, ResultOutput, ProgressSink traits
    └── inference/  Candle ML wrappers

photo-qa-cli        CLI binary (clap)
    ├── commands/   check, models subcommands
    └── output/     JSON output, progress bar adapters

photo-qa-adapters   External adapters
    ├── fs.rs       Filesystem ImageSource implementation
    └── models.rs   Model download/cache management
```

**Key pattern**: Domain core defines port traits; adapters implement them. QA modules implement `QaModule` trait for extensibility.

## Code Style

- MSRV: 1.82
- Strict clippy: `pedantic`, `nursery`, `unwrap_used`, `expect_used` warnings enabled
- `unsafe_code` denied
- Conventional commits required (semantic-release on main)
- Pre-commit hooks: fmt + clippy via lefthook

## ML Inference

Uses Candle (pure Rust) with safetensors models. Models auto-download to `~/.local/share/photo-qa/models/` on first run. GPU (Metal/CUDA) auto-detected.

## Task Tracking

Use `TASKS.md` for implementation progress:

- Before starting work: find relevant task in TASKS.md
- Mark `[ ]` → `[x]` when completing a task
- Commit TASKS.md updates alongside code changes
- Keep tasks granular; split if needed
