# Code Review: Configuration Layering Implementation (Final)

**Reviewed:** 2026-02-01
**Files:** `crates/photo-qa-cli/src/config.rs`, `crates/photo-qa-cli/src/commands/check.rs`, `crates/photo-qa-cli/src/main.rs`, `crates/photo-qa-cli/tests/config_layering.rs`
**Scope:** Phase 7 config file support - final review after addressing all previous feedback

## Summary

All previous review items addressed. CLI validation now uses custom `parse_threshold()` parser. Unused `ModelsConfig` removed from schema. Integration tests added in `config_layering.rs`. Implementation is clean, well-tested, and ready to ship.

## Critical Issues

None.

## Improvements

### 1. Duplicate Default Values Remain

**Location:** `check.rs:defaults` module vs core module `Default` impls

**Current:**
```rust
// check.rs
mod defaults {
    pub const BLUR_THRESHOLD: f32 = 0.5;
    pub const EAR_THRESHOLD: f32 = 0.2;
    // ...
}
```

Core crates also define defaults in their `Default` impls.

**Suggested:** Extract to core crate constants if maintaining consistency is important.

**Why:** Low risk - values are stable and unlikely to drift. Acceptable trade-off for avoiding cross-crate coupling on CLI defaults.

---

### 2. Consider `#[must_use]` on `with_config`

**Location:** `check.rs:119`

```rust
#[must_use]
pub fn with_config(mut args: Self, config: &AppConfig) -> Self {
```

**Why:** Prevents accidentally discarding the merged args. Not critical since current usage is correct.

---

### 3. `quiet` flag not wired to config

**Location:** `check.rs:97-98`

The `--quiet` CLI flag exists but has no corresponding `output.quiet` config option. Either add it or document as CLI-only.

## Minor/Style

- `config.rs:225`: The `or_else(|| self.output.format.take())` pattern for `String` differs from other fields using `or()` - works but slightly inconsistent
- Consider doc comment on `defaults` module explaining these are CLI fallbacks distinct from core defaults

## Positive Notes

- `parse_threshold()` custom parser provides clear error messages: "'2.0' is not in 0.0..=1.0"
- Integration tests in `config_layering.rs` cover CLI > config > default priority chain
- Removed unused `ModelsConfig` - schema matches implementation
- `Box<check::CheckArgs>` in enum variant avoids large-enum-variant lint
- `#[serde(default)]` on all config structs makes partial TOML files seamless
- Validation after merge catches combined config issues
- All 55 tests pass, clippy clean

## Action Items

- [ ] (Optional) Add `#[must_use]` to `with_config`
- [ ] (Optional) Add `output.quiet` to config or document CLI-only
- [ ] (Optional) Extract default constants to single location

All items are polish - implementation is solid. Ready to commit.
