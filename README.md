# Jpegli-redux: an improved JPEG encoder and decoder implementation

This repository is a fork of Jpegli (a JPEG encoder and decoder implementation that is API and ABI compatible with libjpeg62).
The goal of `jpegli-redux` is to merge in various pull requests fixing issues, add features for tuning the encoding/decoding process, and provide a repository for binary releases.

## New Features and Fixes

This repository includes several improvements and fixes merged from upstream pull requests:

*   **Build lib on Windows** by elnoir ([PR 112](https://github.com/google/jpegli/pull/112)): Enabled building the `libjpeg` shared library on Windows.
*   **Fix APP14 markers** by jonnyawsom3 ([PR 135](https://github.com/google/jpegli/pull/135)): Correctly write Adobe APP14 markers for RGB, XYB, and CMYK/YCCK images.
*   **Default to 444 and fix XYB Subsampling** by jonnyawsom3 ([PR 136](https://github.com/google/jpegli/pull/136)): Default to 4:4:4 chroma subsampling and correct subsampling handling for XYB color space.
*   **Change settings based on distance** by jonnyawsom3 ([PR 137](https://github.com/google/jpegli/pull/137)): Dynamically select subsampling based on quality, and auto-select RGB at quality 100.
*   **Added sharpyuv encoding** by Galaxy4594 ([PR 190](https://github.com/google/jpegli/pull/190)): Added Sharp YUV chroma downsampling for 4:2:0 subsampling.
*   **Allow empty DHT marker** by kleisauke ([PR 222](https://github.com/google/jpegli/pull/222)): Allow encoding and decoding empty DHT markers found in real-world progressive JPEGs.
*   **Enhanced Adaptive Quantization (AQ) & Chroma Preservation**:
    *   **Active at high qualities**: Upstream jpegli always had AQ enabled no matter the quality level. In `jpegli-redux`, AQ is enabled by default, but its strength is scaled with the compression distance.
    *   **Smooth quality-based strength tapering**: AQ strength scales smoothly with compression distance:
        *   Distance 0 (Quality 100): 0% AQ (turns off cleanly, byte-identical to `--noadaptive_quantization`).
        *   Distance 1.0 (Quality ~90): 50% AQ strength.
        *   Distance >= 2.0: 100% full AQ strength.
    *   **Manual AQ scaling (`--aq_scale=FLOAT`)**: Manually controls the AQ scale. If omitted, automatically scales with distance.
    *   **Visual energy correction (`--aq_mode=0|1|2`)**: Mode 2 (default) balances RDOQ deadzoning across channels to prevent double-dipping chroma loss. Mode 1 retains legacy AQ behavior, and mode 0 disables AQ. Can also be toggled with `--visual_energy_correction` and `--novisual_energy_correction`.
    *   **Color shift correction (`--color_shift_correction=FLOAT`, `--nocolor_shift_correction`)**: Aligns Cb quantization tables and deadzoning parameters with Cr to prevent color shifts and desaturation in fine warm/yellow details. Enabled by default with automatic warm-pixel detection, or manually adjustable via `[0.0, 1.0]` and toggleable with `--nocolor_shift_correction`.

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
