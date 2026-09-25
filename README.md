# Jpegli-redux: an improved JPEG encoder and decoder implementation

This repository is a fork of Jpegli (a JPEG encoder and decoder implementation that is API and ABI compatible with libjpeg62).
The goal of `jpegli-redux` is to merge in various pull requests fixing issues, add features for tuning the encoding/decoding process, and provide a repository for binary releases.

## New Features and Fixes

This repository includes several improvements and fixes merged from upstream pull requests:

*   **Build lib on Windows** by elnoir ([PR 112](https://github.com/google/jpegli/pull/112)): Enabled building the `libjpeg` shared library on Windows.
*   **Fix APP14 markers** by jonnyawsom3 ([PR 135](https://github.com/google/jpegli/pull/135)): Correctly write Adobe APP14 markers for RGB, XYB, and CMYK/YCCK images.
*   **Default to 444 and fix XYB Subsampling** by jonnyawsom3 ([PR 136](https://github.com/google/jpegli/pull/136)): Default to 4:4:4 chroma subsampling and correct subsampling handling for XYB color space.
*   **Change settings based on distance** by jonnyawsom3 ([PR 137](https://github.com/google/jpegli/pull/137)): Dynamically select subsampling and disable adaptive quantization at high qualities, and auto-select RGB at quality 100.
*   **Added sharpyuv encoding** by Galaxy4594 ([PR 190](https://github.com/google/jpegli/pull/190)): Added Sharp YUV chroma downsampling for 4:2:0 subsampling.

## Usage

When [building the project](doc/building_and_testing.md), two binaries,
`tools/cjpegli` and `tools/djpegli` will be built, as well as a
`lib/jpegli/libjpeg.so.62.3.0` shared library that can be used as a drop-in
replacement for the system library with the same name.

## Development process

*   [More information on testing/build options](doc/building_and_testing.md)
*   [Git guide for jpegli](doc/developing_in_github.md) - for developers

## Blog post
For more information check out the 
[blog post](https://opensource.googleblog.com/2024/04/introducing-jpegli-new-jpeg-coding-library.html) 
on the Google Open Source blog.

## Contact

If you encounter a bug or other issue with the software, please open an Issue here.
