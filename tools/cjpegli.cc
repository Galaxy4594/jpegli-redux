// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "lib/base/common.h"
#include "lib/base/printf_macros.h"
#include "lib/base/span.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/jpegli.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/time.h"
#include "lib/jpegli/encode.h"
#include "tools/args.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"
#include "tools/speed_stats.h"

namespace jpegli_tools {
namespace {

struct Args {
  void AddCommandLineOptions(CommandLineParser* cmdline) {
    std::string input_help("the input can be ");
    input_help.append(jpegli::extras::ListOfDecodeCodecs());
    cmdline->AddPositionalOption("INPUT", /* required = */ true, input_help,
                                 &file_in);
    cmdline->AddPositionalOption("OUTPUT", /* required = */ true,
                                 "the compressed JPEG output file", &file_out);

    cmdline->AddOptionFlag('\0', "disable_output",
                           "No output file will be written (for benchmarking)",
                           &disable_output, &SetBooleanTrue, 1);

    cmdline->AddOptionValue(
        'x', "dec-hints", "key=value",
        "color_space indicates the ColorEncoding, see Description();\n"
        "    icc_pathname refers to a binary file containing an ICC profile.",
        &color_hints_proxy, &ParseAndAppendKeyValue<ColorHintsProxy>, 1);

    opt_distance_id = cmdline->AddOptionValue(
        'd', "distance", "maxError",
        "Max. butteraugli distance, lower = higher quality.\n"
        "    1.0 = visually lossless (default).\n"
        "    Recommended range: 0.5 .. 3.0. Allowed range: 0.0 ... 25.0.\n"
        "    Mutually exclusive with --quality and --target_size.",
        &settings.distance, &ParseFloat);

    opt_quality_id = cmdline->AddOptionValue(
        'q', "quality", "QUALITY",
        "Quality setting (is remapped to --distance)."
        "    Default is quality 90.\n"
        "    Quality values roughly match libjpeg quality.\n"
        "    Recommended range: 68 .. 96. Allowed range: 1 .. 100.\n"
        "    Mutually exclusive with --distance and --target_size.",
        &quality, &ParseSigned);

    cmdline->AddOptionValue('\0', "chroma_subsampling", "444|440|422|420",
                            "Chroma subsampling setting.",
                            &settings.chroma_subsampling, &ParseString);

    cmdline->AddOptionValue(
        'p', "progressive_level", "N",
        "Progressive level setting. Range: 0 .. 2.\n"
        "    Default: 2. Higher number is more scans, 0 means sequential.",
        &settings.progressive_level, &ParseSigned);

    cmdline->AddOptionFlag('\0', "xyb", "Convert to XYB colorspace",
                           &settings.xyb, &SetBooleanTrue, 1);

    cmdline->AddOptionFlag('\0', "sharp_yuv",
                           "Use true linear-light sharp YUV downsampling for 4:2:0",
                           &settings.use_sharpyuv, &SetBooleanTrue, 1);

    cmdline->AddOptionValue(
        '\0', "aq_mode", "0|1|2",
        "Adaptive quantization mode: 0=off, 1=old behavior, 2=AQ+visual_energy_correction (default: 2).",
        &settings.adaptive_quantization_mode, &ParseSigned, 1);

    cmdline->AddOptionValue(
        '\0', "aq_scale", "FLOAT",
        "Adaptive quantization scale factor. (If omitted or negative, auto).",
        &settings.aq_scale, &ParseFloat, 1);

    cmdline->AddOptionFlag(
        '\0', "visual_energy_correction",
        "Enable visual energy correction (AQ mode 2)",
        &settings.visual_energy_correction, &SetBooleanTrue, 1);

    cmdline->AddOptionFlag(
        '\0', "novisual_energy_correction",
        "Disable visual energy correction (use AQ mode 1)",
        &settings.visual_energy_correction, &SetBooleanFalse, 1);

    cmdline->AddOptionFlag(
        '\0', "std_quant",
        "Use quantization tables based on Annex K of the JPEG standard.",
        &settings.use_std_quant_tables, &SetBooleanTrue, 1);

    cmdline->AddOptionFlag(
        '\0', "noadaptive_quantization", "Disable adaptive quantization.",
        &settings.use_adaptive_quantization, &SetBooleanFalse, 1);

    cmdline->AddOptionFlag(
        '\0', "fixed_code",
        "Disable Huffman code optimization. Must be used together with -p 0.",
        &settings.optimize_coding, &SetBooleanFalse, 1);

    cmdline->AddOptionValue(
        '\0', "target_size", "N",
        "If non-zero, set target size in bytes. This is useful for image \n"
        "    quality comparisons, but makes encoding speed up to 20x slower.\n"
        "    Mutually exclusive with --distance and --quality.",
        &settings.target_size, &ParseUnsigned, 2);

    cmdline->AddOptionValue('\0', "num_reps", "N",
                            "How many times to compress. (For benchmarking).",
                            &num_reps, &ParseUnsigned, 1);

    cmdline->AddOptionFlag('\0', "quiet", "Suppress informative output", &quiet,
                           &SetBooleanTrue, 1);

    cmdline->AddOptionValue(
        '\0', "color_shift_correction", "FLOAT",
        "Color shift correction factor (0.0 to 1.0) to align Cb quantization\n"
        "    and deadzoning with Cr. If negative or omitted, auto-detected.",
        &color_shift_correction, &ParseFloat, 1);

    cmdline->AddOptionFlag(
        '\0', "nocolor_shift_correction",
        "Disable color shift correction.",
        &nocolor_shift_correction, &SetBooleanTrue, 1);

    cmdline->AddOptionFlag(
        'v', "verbose",
        "Verbose output; can be repeated, also applies to help (!).", &verbose,
        &SetBooleanTrue);
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  bool disable_output = false;
  ColorHintsProxy color_hints_proxy;
  jpegli::extras::JpegSettings settings;
  int quality = 90;
  size_t num_reps = 1;
  bool quiet = false;
  bool verbose = false;
  float color_shift_correction = -1.0f;
  bool nocolor_shift_correction = false;
  // References (ids) of specific options to check if they were matched.
  CommandLineParser::OptionId opt_distance_id = -1;
  CommandLineParser::OptionId opt_quality_id = -1;
};

bool ValidateArgs(const Args& args) {
  const jpegli::extras::JpegSettings& settings = args.settings;
  if (settings.distance < 0.0 || settings.distance > 25.0) {
    fprintf(stderr, "Invalid --distance argument\n");
    return false;
  }
  if (args.quality <= 0 || args.quality > 100) {
    fprintf(stderr, "Invalid --quality argument\n");
    return false;
  }
  std::string cs = settings.chroma_subsampling;
  if (!cs.empty() && cs != "444" && cs != "440" && cs != "422" && cs != "420") {
    fprintf(stderr, "Invalid --chroma_subsampling argument\n");
    return false;
  }
  if (settings.progressive_level < 0 || settings.progressive_level > 2) {
    fprintf(stderr, "Invalid --progressive_level argument\n");
    return false;
  }
  if (settings.progressive_level > 0 && !settings.optimize_coding) {
    fprintf(stderr, "--fixed_code must be used together with -p 0\n");
    return false;
  }
  return true;
}

bool SetDistance(const Args& args, const CommandLineParser& cmdline,
                 jpegli::extras::JpegSettings* settings) {
  bool distance_set = cmdline.GetOption(args.opt_distance_id)->matched();
  bool quality_set = cmdline.GetOption(args.opt_quality_id)->matched();
  int num_quality_settings = (distance_set ? 1 : 0) + (quality_set ? 1 : 0) +
                             (args.settings.target_size > 0 ? 1 : 0);
  if (num_quality_settings > 1) {
    fprintf(
        stderr,
        "Only one of --distance, --quality, or --target_size can be set.\n");
    return false;
  }
  if (quality_set) {
    settings->quality = args.quality;
  }
  return true;
}

int CJpegliMain(int argc, const char* argv[]) {
  Args args;
  CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, const_cast<const char**>(argv))) {
    // Parse already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information.\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (cmdline.HelpFlagPassed() || !args.file_in) {
    cmdline.PrintHelp();
    return EXIT_SUCCESS;
  }

  if (!args.file_out && !args.disable_output) {
    fprintf(stderr,
            "No output file specified and --disable_output flag not passed.\n");
    return EXIT_FAILURE;
  }

  if (args.disable_output && !args.quiet) {
    fprintf(stderr,
            "Encoding will be performed, but the result will be discarded.\n");
  }

  std::vector<uint8_t> input_bytes;
  if (!ReadFile(args.file_in, &input_bytes)) {
    fprintf(stderr, "Failed to read input image %s\n", args.file_in);
    return EXIT_FAILURE;
  }

  jpegli::extras::PackedPixelFile ppf;
  if (!jpegli::extras::DecodeBytes(jpegli::Bytes(input_bytes),
                                   args.color_hints_proxy.target, &ppf)) {
    fprintf(stderr, "Failed to decode input image %s\n", args.file_in);
    return EXIT_FAILURE;
  }

  if (!args.quiet) {
    fprintf(stderr, "Read %ux%u image, %" PRIuS " bytes.\n", ppf.info.xsize,
            ppf.info.ysize, input_bytes.size());
  }

  {
    float correction = args.color_shift_correction;
    if (args.nocolor_shift_correction) {
      correction = 0.0f;
    } else if (correction < 0.0f) {
      // Auto-detect warm yellow pixels in the image.
      // In warm tones, asymmetric Cb/Cr quantization and deadzoning causes 
      // Cb coefficients to round to 0 earlier than Cr, shifting color hue
      // and desaturating fine yellow details.
      float warm_pixels = 0.0f;
      float total_pixels = 0.0f;
      for (const auto& img : ppf.frames) {
        if (img.color.format.num_channels < 3 || args.settings.xyb) continue;
        for (size_t y = 0; y < img.color.ysize; ++y) {
          for (size_t x = 0; x < img.color.xsize; ++x) {
            float r = img.color.GetPixelValue(y, x, 0);
            float g = img.color.GetPixelValue(y, x, 1);
            float b = img.color.GetPixelValue(y, x, 2);
            if (r > g && g > b && r < 0.8f && r > 0.2f && (r - g) > 0.05f) {
              warm_pixels += 1.0f;
            }
            total_pixels += 1.0f;
          }
        }
      }
      if (total_pixels > 0.0f) {
        correction = std::min(1.0f, (warm_pixels / total_pixels) * 10.0f);
      } else {
        correction = 0.0f;
      }
    } else {
      correction = std::min(1.0f, std::max(0.0f, correction));
    }
    args.settings.color_shift_correction = correction;
    if (!args.quiet && correction > 0.0f) {
      fprintf(stderr, "Color shift correction factor: %.3f\n", correction);
    }
  }

  if (!ValidateArgs(args) || !SetDistance(args, cmdline, &args.settings)) {
    return EXIT_FAILURE;
  }

  if (!args.settings.use_adaptive_quantization) {
    args.settings.adaptive_quantization_mode = 0;
  } else if (!args.settings.visual_energy_correction &&
             args.settings.adaptive_quantization_mode == 2) {
    args.settings.adaptive_quantization_mode = 1;
  } else if (args.settings.adaptive_quantization_mode == 0) {
    args.settings.use_adaptive_quantization = false;
  }

  if (!args.quiet) {
    const jpegli::extras::JpegSettings& s = args.settings;
    float calculated_distance =
        cmdline.GetOption(args.opt_quality_id)->matched()
            ? jpegli_quality_to_distance(s.quality)
            : s.distance;
    const char* aq_str = "noAQ";
    if (s.use_adaptive_quantization && s.adaptive_quantization_mode > 0) {
      aq_str = s.adaptive_quantization_mode == 2 ? "AQ2" : "AQ1";
    }
    fprintf(stderr, "Encoding [%s%s d%.3f%s %s p%d %s]\n",
            s.xyb ? "XYB" : "YUV", s.chroma_subsampling.c_str(),
            calculated_distance, s.use_std_quant_tables ? " StdQuant" : "",
            aq_str, s.progressive_level,
            s.optimize_coding ? "OPT" : "FIX");
  }

  jpegli_tools::SpeedStats stats;
  std::vector<uint8_t> jpeg_bytes;
  for (size_t num_rep = 0; num_rep < args.num_reps; ++num_rep) {
    const double t0 = jpegli::Now();
    if (!jpegli::extras::EncodeJpeg(ppf, args.settings, nullptr, &jpeg_bytes)) {
      fprintf(stderr, "jpegli encoding failed\n");
      return EXIT_FAILURE;
    }
    const double t1 = jpegli::Now();
    stats.NotifyElapsed(t1 - t0);
    stats.SetImageSize(ppf.info.xsize, ppf.info.ysize);
  }

  if (args.file_out && !args.disable_output) {
    if (!WriteFile(args.file_out, jpeg_bytes)) {
      fprintf(stderr, "Could not write jpeg to %s\n", args.file_out);
      return EXIT_FAILURE;
    }
  }
  if (!args.quiet) {
    fprintf(stderr, "Compressed to %" PRIuS " bytes ", jpeg_bytes.size());
    const double num_pixels =
        static_cast<double>(ppf.info.xsize) * ppf.info.ysize;
    const double bpp =
        static_cast<double>(jpeg_bytes.size() * jpegli::kBitsPerByte) /
        num_pixels;
    fprintf(stderr, "(%.3f bpp).\n", bpp);
    stats.Print(1);
  }
  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace jpegli_tools

int main(int argc, const char** argv) {
  return jpegli_tools::CJpegliMain(argc, argv);
}
