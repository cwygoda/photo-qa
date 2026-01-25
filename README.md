# photo-qa

Rust CLI for automated photo quality assessment.

## Features

- **Blur detection** — identifies motion blur vs defocus blur
- **Exposure analysis** — flags underexposed and overexposed images
- **Closed eyes detection** — per-face analysis for portraits and group shots

## Installation

```bash
# From source
cargo install --path crates/photo-qa-cli

# Or build locally
cargo build --release
```

## Usage

```bash
# Analyze images (all checks enabled by default)
photo-qa /path/to/photos

# Analyze recursively
photo-qa -r /path/to/photos

# Specific checks only
photo-qa --no-eyes /path/to/photos

# Adjust thresholds
photo-qa --blur-threshold 0.7 --under-threshold 0.4 /path/to/photos

# Include EXIF metadata
photo-qa --exif /path/to/photos

# Show progress bar
photo-qa --progress /path/to/photos
```

### Output

JSON Lines format (one object per image):

```json
{"path":"/photos/IMG_001.jpg","timestamp":"2024-01-15T10:30:00Z","dimensions":{"width":4000,"height":3000},"issues":[{"type":"blur","score":0.72,"details":{"blur_type":"motion"}}]}
```

### Exit Codes

- `0` — success, no issues found
- `1` — success, issues found
- `2` — runtime error

## Development

### Bootstrap

```bash
just bootstrap
```

This installs Rust via mise and sets up git hooks.

### Commands

```bash
just              # list all commands
just build        # compile
just test         # run tests
just lint         # clippy
just fmt          # format code
just ci           # full CI pipeline
```

### Project Structure

```
crates/
├── photo-qa-core/      # Domain logic, QA modules, ports
├── photo-qa-cli/       # CLI binary
└── photo-qa-adapters/  # Filesystem, model download
```

## License

MIT
