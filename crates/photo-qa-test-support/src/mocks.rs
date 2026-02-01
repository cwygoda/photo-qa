//! Mock implementations of core port traits.

use std::sync::{Arc, Mutex, PoisonError};

use photo_qa_core::domain::{AnalysisResult, ImageInfo};
use photo_qa_core::ports::{ImageSource, ProgressEvent, ProgressSink, ResultOutput};

/// Mock implementation of `ImageSource` for testing.
///
/// Yields pre-built images and tracks iteration for assertions.
pub struct MockImageSource {
    images: Vec<ImageInfo>,
    iteration_count: Arc<Mutex<usize>>,
}

impl MockImageSource {
    /// Creates a new mock source with the given images.
    #[must_use]
    pub fn new(images: Vec<ImageInfo>) -> Self {
        Self {
            images,
            iteration_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Creates an empty mock source.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(vec![])
    }

    /// Returns the number of times the source has been iterated.
    #[must_use]
    pub fn iteration_count(&self) -> usize {
        *self
            .iteration_count
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl ImageSource for MockImageSource {
    fn images(&self) -> Box<dyn Iterator<Item = anyhow::Result<ImageInfo>> + Send + '_> {
        let count = Arc::clone(&self.iteration_count);
        if let Ok(mut c) = count.lock() {
            *c += 1;
        }
        Box::new(self.images.iter().cloned().map(Ok))
    }

    fn count_hint(&self) -> Option<usize> {
        Some(self.images.len())
    }
}

/// Mock implementation of `ResultOutput` for testing.
///
/// Captures results for later assertions.
pub struct MockResultOutput {
    results: Arc<Mutex<Vec<AnalysisResult>>>,
    flush_count: Arc<Mutex<usize>>,
}

impl MockResultOutput {
    /// Creates a new mock output.
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: Arc::new(Mutex::new(Vec::new())),
            flush_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Returns all captured results.
    #[must_use]
    pub fn results(&self) -> Vec<AnalysisResult> {
        self.results
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Returns the number of times `flush()` was called.
    #[must_use]
    pub fn flush_count(&self) -> usize {
        *self
            .flush_count
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl Default for MockResultOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultOutput for MockResultOutput {
    fn write(&self, result: &AnalysisResult) -> anyhow::Result<()> {
        self.results
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(result.clone());
        Ok(())
    }

    fn flush(&self) -> anyhow::Result<()> {
        if let Ok(mut c) = self.flush_count.lock() {
            *c += 1;
        }
        Ok(())
    }
}

/// Mock implementation of `ProgressSink` for testing.
///
/// Captures events for later assertions.
pub struct MockProgressSink {
    events: Arc<Mutex<Vec<ProgressEvent>>>,
}

impl MockProgressSink {
    /// Creates a new mock progress sink.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns all captured events.
    #[must_use]
    pub fn events(&self) -> Vec<ProgressEvent> {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Returns the number of `Started` events.
    #[must_use]
    pub fn started_count(&self) -> usize {
        self.events()
            .iter()
            .filter(|e| matches!(e, ProgressEvent::Started { .. }))
            .count()
    }

    /// Returns the number of `Completed` events.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.events()
            .iter()
            .filter(|e| matches!(e, ProgressEvent::Completed { .. }))
            .count()
    }

    /// Returns the number of `Skipped` events.
    #[must_use]
    pub fn skipped_count(&self) -> usize {
        self.events()
            .iter()
            .filter(|e| matches!(e, ProgressEvent::Skipped { .. }))
            .count()
    }

    /// Returns whether a `Finished` event was received.
    #[must_use]
    pub fn has_finished(&self) -> bool {
        self.events()
            .iter()
            .any(|e| matches!(e, ProgressEvent::Finished { .. }))
    }

    /// Returns the final counts from the `Finished` event, if any.
    #[must_use]
    pub fn finished_counts(&self) -> Option<(usize, usize)> {
        self.events().iter().find_map(|e| match e {
            ProgressEvent::Finished { processed, skipped } => Some((*processed, *skipped)),
            _ => None,
        })
    }
}

impl Default for MockProgressSink {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressSink for MockProgressSink {
    fn on_event(&self, event: ProgressEvent) {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use photo_qa_core::domain::ImageDimensions;

    #[test]
    fn test_mock_image_source_empty() {
        let source = MockImageSource::empty();
        assert_eq!(source.count_hint(), Some(0));
        assert_eq!(source.images().count(), 0);
        assert_eq!(source.iteration_count(), 1);
    }

    #[test]
    fn test_mock_image_source_with_images() {
        let img = image::DynamicImage::new_rgb8(100, 100);
        let info = ImageInfo::new("test.jpg", img);
        let source = MockImageSource::new(vec![info]);

        assert_eq!(source.count_hint(), Some(1));
        assert_eq!(source.images().count(), 1);
    }

    #[test]
    fn test_mock_result_output() {
        let output = MockResultOutput::new();

        let result = AnalysisResult {
            path: "test.jpg".into(),
            timestamp: "2024-01-01T00:00:00Z".into(),
            dimensions: ImageDimensions::new(100, 100),
            issues: vec![],
            exif: None,
        };

        output.write(&result).unwrap();
        output.flush().unwrap();

        assert_eq!(output.results().len(), 1);
        assert_eq!(output.results()[0].path, "test.jpg");
        assert_eq!(output.flush_count(), 1);
    }

    #[test]
    fn test_mock_progress_sink() {
        let sink = MockProgressSink::new();

        sink.on_event(ProgressEvent::Started {
            path: "test.jpg".into(),
            index: 0,
            total: Some(1),
        });

        sink.on_event(ProgressEvent::Finished {
            processed: 1,
            skipped: 0,
        });

        assert_eq!(sink.started_count(), 1);
        assert!(sink.has_finished());
        assert_eq!(sink.finished_counts(), Some((1, 0)));
    }
}
