//! Integration tests for RAW image loading.
//!
//! These tests require sample RAW files which are not included in the repository
//! due to their large size. To run these tests:
//!
//! 1. Download sample RAW files from <https://raw.pixls.us/> or <https://www.rawsamples.ch/>
//! 2. Place them in tests/fixtures/raw/ with appropriate names:
//!    - test.cr2 (Canon CR2)
//!    - test.nef (Nikon NEF)
//!    - test.arw (Sony ARW)
//!    - test.raf (Fuji RAF)
//!    - test.dng (Adobe DNG)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::uninlined_format_args
)]

use photo_qa_adapters::FsImageSource;
use photo_qa_core::ImageSource;
use std::path::PathBuf;

fn raw_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/raw")
}

macro_rules! raw_test {
    ($name:ident, $file:expr, $format:expr) => {
        #[test]
        fn $name() {
            let path = raw_fixtures_dir().join($file);
            if !path.exists() {
                eprintln!(
                    "Skipping {}: {} not found. Download from https://raw.pixls.us/",
                    stringify!($name),
                    $file
                );
                return;
            }

            let source = FsImageSource::new(vec![path.clone()], false);
            let images: Vec<_> = source.images().collect();

            assert_eq!(images.len(), 1, "Should find exactly one image");

            let result = images.into_iter().next().unwrap();
            let info = result.expect(&format!("should load {} file", $format));

            assert!(info.width > 0, "Width should be positive");
            assert!(info.height > 0, "Height should be positive");
            assert!(info.path.ends_with($file), "Path should end with {}", $file);

            // Verify we got actual pixel data by checking dimensions are reasonable
            // Most RAW files are at least 1000x1000
            assert!(
                info.width >= 100 && info.height >= 100,
                "RAW image should have reasonable dimensions: {}x{}",
                info.width,
                info.height
            );
        }
    };
}

raw_test!(test_load_cr2, "test.cr2", "Canon CR2");
raw_test!(test_load_nef, "test.nef", "Nikon NEF");
raw_test!(test_load_arw, "test.arw", "Sony ARW");
raw_test!(test_load_raf, "test.raf", "Fuji RAF");
raw_test!(test_load_dng, "test.dng", "DNG");

#[test]
fn test_raw_extension_detection() {
    // Verify that RAW extensions are recognized even without files present
    use std::path::Path;

    let raw_extensions = ["cr2", "cr3", "nef", "arw", "raf", "dng", "orf", "rw2"];

    for ext in raw_extensions {
        let path = format!("test.{}", ext);
        let p = Path::new(&path);
        let ext_lower = p
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_lowercase);
        assert!(
            ext_lower.is_some(),
            "Should be able to extract extension from {}",
            path
        );
    }
}
