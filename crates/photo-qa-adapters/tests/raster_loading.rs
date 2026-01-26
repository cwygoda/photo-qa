//! Integration tests for raster image loading.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use photo_qa_adapters::FsImageSource;
use photo_qa_core::{ImageInfo, ImageSource};
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn test_load_jpeg() {
    let path = fixtures_dir().join("test.jpg");
    let source = FsImageSource::new(vec![path], false);

    let images: Vec<_> = source.images().collect();
    assert_eq!(images.len(), 1);

    let info = images
        .into_iter()
        .next()
        .unwrap()
        .expect("should load JPEG");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert!(info.path.ends_with("test.jpg"));
}

#[test]
fn test_load_png() {
    let path = fixtures_dir().join("test.png");
    let source = FsImageSource::new(vec![path], false);

    let images: Vec<_> = source.images().collect();
    assert_eq!(images.len(), 1);

    let info = images.into_iter().next().unwrap().expect("should load PNG");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert!(info.path.ends_with("test.png"));
}

#[test]
fn test_load_tiff() {
    let path = fixtures_dir().join("test.tiff");
    let source = FsImageSource::new(vec![path], false);

    let images: Vec<_> = source.images().collect();
    assert_eq!(images.len(), 1);

    let info = images
        .into_iter()
        .next()
        .unwrap()
        .expect("should load TIFF");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert!(info.path.ends_with("test.tiff"));
}

#[test]
fn test_load_webp() {
    let path = fixtures_dir().join("test.webp");
    let source = FsImageSource::new(vec![path], false);

    let images: Vec<_> = source.images().collect();
    assert_eq!(images.len(), 1);

    let info = images
        .into_iter()
        .next()
        .unwrap()
        .expect("should load WebP");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert!(info.path.ends_with("test.webp"));
}

#[test]
fn test_load_bmp() {
    let path = fixtures_dir().join("test.bmp");
    let source = FsImageSource::new(vec![path], false);

    let images: Vec<_> = source.images().collect();
    assert_eq!(images.len(), 1);

    let info = images.into_iter().next().unwrap().expect("should load BMP");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert!(info.path.ends_with("test.bmp"));
}

#[test]
fn test_load_gif() {
    let path = fixtures_dir().join("test.gif");
    let source = FsImageSource::new(vec![path], false);

    let images: Vec<_> = source.images().collect();
    assert_eq!(images.len(), 1);

    let info = images.into_iter().next().unwrap().expect("should load GIF");
    // GIF has 2 frames but we load first frame which is 8x8
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert!(info.path.ends_with("test.gif"));
}

#[test]
fn test_load_directory() {
    let dir = fixtures_dir();
    let source = FsImageSource::new(vec![dir], false);

    let images: Vec<_> = source.images().collect();
    // Should find all 6 test images
    assert_eq!(images.len(), 6);

    // All should load successfully
    for result in images {
        let info: ImageInfo = result.expect("all fixtures should load");
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
    }
}

#[test]
fn test_count_hint() {
    let dir = fixtures_dir();
    let source = FsImageSource::new(vec![dir], false);

    assert_eq!(source.count_hint(), Some(6));
}
