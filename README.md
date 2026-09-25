# Jpegli-redux: an improved JPEG encoder and decoder implementation

This repository is a fork of Jpegli (a JPEG encoder and decoder implementation that is API and ABI compatible with libjpeg62).
The goal of `jpegli-redux` is to merge in various pull requests fixing issues, add features for tuning the encoding/decoding process, and provide a repository for binary releases.

## Planned Features and Fixes

The following features, fixes, and PRs from the original repository are planned to be merged:

*   **Prominent PRs to be merged:**
    *   PR 112
    *   PR 135
    *   PR 136
    *   PR 137
    *   PR 190
*   **Various new features** for tuning the encoding and decoding processes.
*   **Prominent bug fixes** to improve stability and performance.

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
