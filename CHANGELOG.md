## [0.1.1](https://github.com/cwygoda/photo-qa/compare/v0.1.0...v0.1.1) (2026-02-04)


### Bug Fixes

* **cli:** improve error message visibility and actionability ([102b54b](https://github.com/cwygoda/photo-qa/commit/102b54b2c7958b1031e7d94aee506d01c7c004e3))

# [0.1.0](https://github.com/cwygoda/photo-qa/compare/v0.0.0...v0.1.0) (2026-02-04)


### Bug Fixes

* **ci:** enable Git LFS for test job ([b0db7ff](https://github.com/cwygoda/photo-qa/commit/b0db7ff715677959cda0f7148d9d2549fa1af152))
* format and clippy warnings in blazeface.rs ([1ed98ef](https://github.com/cwygoda/photo-qa/commit/1ed98efd08af4c8e40bb8f1c47a2603b1c40b8f6))
* **models:** use PEP 723 inline deps, fix verify_models.py ([a6f3418](https://github.com/cwygoda/photo-qa/commit/a6f3418ff9af1f5ebdfbecc82622713a7649faa0))


### Features

* **adapters:** add progress callback support for model downloads ([839a67c](https://github.com/cwygoda/photo-qa/commit/839a67c1b7de74e21179687679e3353dd27424c3))
* **adapters:** add raster format loading with LFS test fixtures ([959c645](https://github.com/cwygoda/photo-qa/commit/959c645417ff2599e6d83030280bef517384819a))
* **adapters:** implement RAW format demosaicing with bilinear interpolation ([17ceac2](https://github.com/cwygoda/photo-qa/commit/17ceac258df890d281c3e2dbf80b7fbe07f07d77))
* add project scaffold with bootstrap, lint, and test infrastructure ([86a0637](https://github.com/cwygoda/photo-qa/commit/86a0637e9349b2968d42db4c89969c4ac3530858))
* **cli:** add --models-dir option for custom model path ([5804f22](https://github.com/cwygoda/photo-qa/commit/5804f22d48a89d9f523b955e85c8b170858f170f))
* **cli:** implement check command with full analysis pipeline ([3d9dff1](https://github.com/cwygoda/photo-qa/commit/3d9dff11b81de41fcbe96515cef5cec93fd45153))
* **cli:** implement models fetch/list commands with progress bar ([9fc1d18](https://github.com/cwygoda/photo-qa/commit/9fc1d18feca39a94652763ae131dd958c13d3f95))
* **cli:** implement TOML configuration file support ([ff768a4](https://github.com/cwygoda/photo-qa/commit/ff768a4796f956e70f967a78a704bceac4de2866))
* **core:** add BoundingBox type and enhance ImageDimensions ([eab28f9](https://github.com/cwygoda/photo-qa/commit/eab28f96b9848793303359b1ef30d5ccc4249dd2))
* **core:** add ImageInfo methods for pixel access and metadata ([22f85b2](https://github.com/cwygoda/photo-qa/commit/22f85b234d06f9642c6419e72c579ca87a108dd3))
* **core:** export module config types for QaModule pattern ([3a2c20d](https://github.com/cwygoda/photo-qa/commit/3a2c20dea768b35e365507aa2e8e78f1450c4892))
* **core:** implement blur detection module ([e18f72f](https://github.com/cwygoda/photo-qa/commit/e18f72fb76991a2b87be2f8ea9d154b3c2b3e802))
* **core:** implement closed eyes detection module ([4e2f5f9](https://github.com/cwygoda/photo-qa/commit/4e2f5f97dd7c739ee57c2360932b1671f488ef4e))
* **core:** implement exposure analysis module ([553ac8e](https://github.com/cwygoda/photo-qa/commit/553ac8e7dbe766cab133eb41a05f07657918c651))
* **inference:** add safetensors loading utility with lazy model support ([3e02783](https://github.com/cwygoda/photo-qa/commit/3e02783107d12ae7fa483424b1fd4184df5a0d29))
* **justfile:** add bootstrap target ([1f1f38f](https://github.com/cwygoda/photo-qa/commit/1f1f38f48c953c395e96c538665c446d8d397cdf))
* **models:** add model conversion and release infrastructure ([5841f98](https://github.com/cwygoda/photo-qa/commit/5841f984be6bf71141ec707bafc0cef475f77555))
* **models:** add pretrained BlazeFace from hollance/BlazeFace-PyTorch ([d7898d7](https://github.com/cwygoda/photo-qa/commit/d7898d7c922bbae317d2470ad54a4b21467d2f75))
* **models:** add production model weights and checksums ([57e4f46](https://github.com/cwygoda/photo-qa/commit/57e4f4645310512305015e1969f342f6868518ec))
* **models:** make conversion scripts one-shot, add justfile targets ([50e4395](https://github.com/cwygoda/photo-qa/commit/50e4395da804692efeae9937f21b4d7f37031318))
* **test:** add comprehensive testing infrastructure ([4231683](https://github.com/cwygoda/photo-qa/commit/423168338f753204dd2e9d4613f22e12fabe8c7e))

# [0.1.0](https://github.com/cwygoda/photo-qa/compare/v0.0.0...v0.1.0) (2026-01-26)


### Features

* **adapters:** add raster format loading with LFS test fixtures ([959c645](https://github.com/cwygoda/photo-qa/commit/959c645417ff2599e6d83030280bef517384819a))
* **adapters:** implement RAW format demosaicing with bilinear interpolation ([17ceac2](https://github.com/cwygoda/photo-qa/commit/17ceac258df890d281c3e2dbf80b7fbe07f07d77))
* add project scaffold with bootstrap, lint, and test infrastructure ([86a0637](https://github.com/cwygoda/photo-qa/commit/86a0637e9349b2968d42db4c89969c4ac3530858))
* **core:** add BoundingBox type and enhance ImageDimensions ([eab28f9](https://github.com/cwygoda/photo-qa/commit/eab28f96b9848793303359b1ef30d5ccc4249dd2))
* **core:** add ImageInfo methods for pixel access and metadata ([22f85b2](https://github.com/cwygoda/photo-qa/commit/22f85b234d06f9642c6419e72c579ca87a108dd3))
* **core:** export module config types for QaModule pattern ([3a2c20d](https://github.com/cwygoda/photo-qa/commit/3a2c20dea768b35e365507aa2e8e78f1450c4892))
* **justfile:** add bootstrap target ([1f1f38f](https://github.com/cwygoda/photo-qa/commit/1f1f38f48c953c395e96c538665c446d8d397cdf))
