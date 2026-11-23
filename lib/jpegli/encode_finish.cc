// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/encode_finish.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "lib/jpegli/common.h"
#include "lib/jpegli/encode_internal.h"
#include "lib/jpegli/huffman.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jpegli/quant.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jpegli/encode_finish.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jpegli/dct-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::GetLane;

using D = HWY_FULL(float);
using DI = HWY_FULL(int32_t);
using DI16 = Rebind<int16_t, HWY_FULL(int32_t)>;

void ReQuantizeBlock(int16_t* block, const float* qmc, float aq_strength,
                     const float* zero_bias_offset,
                     const float* zero_bias_mul) {
  D d;
  DI di;
  DI16 di16;
  const auto aq_mul = Set(d, aq_strength);
  for (size_t k = 0; k < DCTSIZE2; k += Lanes(d)) {
    const auto in = Load(di16, block + k);
    const auto val = ConvertTo(d, PromoteTo(di, in));
    const auto q = Load(d, qmc + k);
    const auto qval = Mul(val, q);
    const auto zb_offset = Load(d, zero_bias_offset + k);
    const auto zb_mul = Load(d, zero_bias_mul + k);
    const auto threshold = Add(zb_offset, Mul(zb_mul, aq_mul));
    const auto nzero_mask = Ge(Abs(qval), threshold);
    const auto iqval = IfThenElseZero(nzero_mask, Round(qval));
    Store(DemoteTo(di16, ConvertTo(di, iqval)), di16, block + k);
  }
}

float BlockError(const int16_t* block, const float* qmc, const float* iqmc,
                 const float aq_strength, const float* zero_bias_offset,
                 const float* zero_bias_mul) {
  D d;
  DI di;
  DI16 di16;
  auto err = Zero(d);
  const auto scale = Set(d, 1.0 / 16);
  const auto aq_mul = Set(d, aq_strength);
  for (size_t k = 0; k < DCTSIZE2; k += Lanes(d)) {
    const auto in = Load(di16, block + k);
    const auto val = ConvertTo(d, PromoteTo(di, in));
    const auto q = Load(d, qmc + k);
    const auto qval = Mul(val, q);
    const auto zb_offset = Load(d, zero_bias_offset + k);
    const auto zb_mul = Load(d, zero_bias_mul + k);
    const auto threshold = Add(zb_offset, Mul(zb_mul, aq_mul));
    const auto nzero_mask = Ge(Abs(qval), threshold);
    const auto iqval = IfThenElseZero(nzero_mask, Round(qval));
    const auto invq = Load(d, iqmc + k);
    const auto rval = Mul(iqval, invq);
    const auto diff = Mul(Sub(val, rval), scale);
    err = Add(err, Mul(diff, diff));
  }
  return GetLane(SumOfLanes(d, err));
}

void ComputeInverseWeights(const float* qmc, float* iqmc) {
  for (int k = 0; k < 64; ++k) {
    iqmc[k] = 1.0f / qmc[k];
  }
}

float ComputePSNR(j_compress_ptr cinfo, int sampling) {
  jpeg_comp_master* m = cinfo->master;
  InitQuantizer(cinfo, QuantPass::SEARCH_SECOND_PASS);
  double error = 0.0;
  size_t num = 0;
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    const float* qmc = m->quant_mul[c];
    const size_t h_factor = m->h_factor[c];
    const size_t v_factor = m->v_factor[c];
    const float* zero_bias_offset = m->zero_bias_offset[c];
    const float* zero_bias_mul = m->zero_bias_mul[c];
    HWY_ALIGN float iqmc[64];
    ComputeInverseWeights(qmc, iqmc);
    for (JDIMENSION by = 0; by < comp->height_in_blocks; by += sampling) {
      JBLOCKARRAY blocks = GetBlockRow(cinfo, c, by);
      const float* qf = m->quant_field.Row(by * v_factor);
      for (JDIMENSION bx = 0; bx < comp->width_in_blocks; bx += sampling) {
        error += BlockError(&blocks[0][bx][0], qmc, iqmc, qf[bx * h_factor],
                            zero_bias_offset, zero_bias_mul);
        num += DCTSIZE2;
      }
    }
  }
  return 4.3429448f * log(num / (error / 255. / 255.));
}

void ReQuantizeCoeffs(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  InitQuantizer(cinfo, QuantPass::SEARCH_SECOND_PASS);
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    const float* qmc = m->quant_mul[c];
    const size_t h_factor = m->h_factor[c];
    const size_t v_factor = m->v_factor[c];
    const float* zero_bias_offset = m->zero_bias_offset[c];
    const float* zero_bias_mul = m->zero_bias_mul[c];
    for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
      JBLOCKARRAY block = GetBlockRow(cinfo, c, by);
      const float* qf = m->quant_field.Row(by * v_factor);
      for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
        ReQuantizeBlock(&block[0][bx][0], qmc, qf[bx * h_factor],
                        zero_bias_offset, zero_bias_mul);
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jpegli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jpegli {
namespace {
HWY_EXPORT(ComputePSNR);
HWY_EXPORT(ReQuantizeCoeffs);

void ReQuantizeCoeffs(j_compress_ptr cinfo) {
  HWY_DYNAMIC_DISPATCH(ReQuantizeCoeffs)(cinfo);
}

float ComputePSNR(j_compress_ptr cinfo, int sampling) {
  return HWY_DYNAMIC_DISPATCH(ComputePSNR)(cinfo, sampling);
}

void UpdateDistance(j_compress_ptr cinfo, float distance) {
  cinfo->master->distance = distance;
  float distances[NUM_QUANT_TBLS] = {distance, distance, distance};
  SetQuantMatrices(cinfo, distances, /*add_two_chroma_tables=*/true);
}

float Clamp(float val, float minval, float maxval) {
  return std::max(minval, std::min(maxval, val));
}

#define PSNR_SEARCH_DBG 0

float FindDistanceForPSNR(j_compress_ptr cinfo) {
  constexpr int kMaxIters = 20;
  const float psnr_target = cinfo->master->psnr_target;
  const float tolerance = cinfo->master->psnr_tolerance;
  const float min_dist = cinfo->master->min_distance;
  const float max_dist = cinfo->master->max_distance;
  float d = Clamp(1.0f, min_dist, max_dist);
  for (int sampling : {4, 1}) {
    float best_diff = std::numeric_limits<float>::max();
    float best_distance = 0.0f;
    float best_psnr = 0.0;
    float dmin = min_dist;
    float dmax = max_dist;
    bool found_lower_bound = false;
    bool found_upper_bound = false;
    for (int i = 0; i < kMaxIters; ++i) {
      UpdateDistance(cinfo, d);
      float psnr = ComputePSNR(cinfo, sampling);
      if (psnr > psnr_target) {
        dmin = d;
        found_lower_bound = true;
      } else {
        dmax = d;
        found_upper_bound = true;
      }
#if (PSNR_SEARCH_DBG > 1)
      printf("sampling %d iter %2d d %7.4f psnr %.2f", sampling, i, d, psnr);
      if (found_upper_bound && found_lower_bound) {
        printf("    d-interval: [ %7.4f .. %7.4f ]", dmin, dmax);
      }
      printf("\n");
#endif
      float diff = std::abs(psnr - psnr_target);
      if (diff < best_diff) {
        best_diff = diff;
        best_distance = d;
        best_psnr = psnr;
      }
      if (diff < tolerance * psnr_target || dmin == dmax) {
        break;
      }
      if (!found_lower_bound || !found_upper_bound) {
        d *= std::exp(0.15f * (psnr - psnr_target));
      } else {
        d = 0.5f * (dmin + dmax);
      }
      d = Clamp(d, min_dist, max_dist);
    }
    d = best_distance;
    if (sampling == 1 && PSNR_SEARCH_DBG) {
      printf("Final PSNR %.2f at distance %.4f\n", best_psnr, d);
    } else {
      (void)best_psnr;
    }
  }
  return d;
}

}  // namespace

void QuantizetoPSNR(j_compress_ptr cinfo) {
  float distance = FindDistanceForPSNR(cinfo);
  UpdateDistance(cinfo, distance);
  ReQuantizeCoeffs(cinfo);
}

namespace {

inline int Ctz64(uint64_t x) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
  unsigned long idx = 0;
  _BitScanForward64(&idx, x);
  return static_cast<int>(idx);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(x);
#else
  int n = 0;
  while ((x & 1ULL) == 0) {
    x >>= 1;
    ++n;
  }
  return n;
#endif
}

inline int SizeOfLevel(int level) {
  // JPEG magnitude category for |level| > 0, clamped to 1..15.
  uint32_t v = static_cast<uint32_t>(level);
#if defined(_MSC_VER)
  unsigned long idx = 0;
  _BitScanReverse(&idx, v);
  int s = static_cast<int>(idx) + 1;
#elif defined(__GNUC__) || defined(__clang__)
  int s = 32 - __builtin_clz(v);
#else
  int s = 0;
  while (v > 0) {
    v >>= 1;
    ++s;
  }
#endif
  return std::min(s, 15);
}

struct AcPrices {
  float symbol[256];

  void BuildFromFrequencies(const uint32_t freq[257]) {
    uint32_t smoothed[257];
    for (int s = 0; s < 256; ++s) {
      smoothed[s] = std::max<uint32_t>(freq[s], 1u);
    }
    smoothed[256] = 1u;  // Pseudo-symbol for all-1s code.
    uint8_t depth[257] = {};
    CreateHuffmanTree(smoothed, 257, 16, depth);
    for (int s = 0; s < 256; ++s) {
      symbol[s] = static_cast<float>(depth[s] > 0 ? depth[s] : 16);
    }
  }

  inline float Eob() const { return symbol[0x00]; }
  inline float Zrl() const { return symbol[0xF0]; }

  inline float Nonzero(int run, int size) const {
    return static_cast<float>(run >> 4) * Zrl() +
           symbol[((run & 15) << 4) | size] + static_cast<float>(size);
  }
};

// Accumulate JPEG AC symbol frequencies for a plain-rounded block (zigzag).
void AccumulatePlainAcHistogram(const int16_t* cblock, const float* qmc,
                                uint32_t freq[257]) {
  int last = 0;
  int levels[DCTSIZE2];
  for (int z = 1; z < DCTSIZE2; ++z) {
    float x = static_cast<float>(cblock[z]) * qmc[z];
    int q = static_cast<int>(std::fabs(x) + 0.5f);
    levels[z] = q;
    if (q != 0) last = z;
  }
  if (last == 0) {
    ++freq[0x00];
    return;
  }
  int run = 0;
  for (int z = 1; z <= last; ++z) {
    int q = levels[z];
    if (q == 0) {
      ++run;
      continue;
    }
    while (run >= 16) {
      ++freq[0xF0];
      run -= 16;
    }
    int size = SizeOfLevel(q);
    ++freq[(run << 4) | size];
    run = 0;
  }
  if (last < 63) {
    ++freq[0x00];
  }
}

// Sparse-bitmask Viterbi trellis RDOQ for one 8x8 block stored in zigzag order
// as Q4 unquantized DCT coefficients (cblock[z] * qmc[z] == ideal level).
void TrellisQuantizeBlock(int16_t* cblock, const float* qmc,
                          const AcPrices& prices, const float weights[DCTSIZE2],
                          float lambda, float dc_threshold,
                          int16_t* last_dc_coeff) {
  // Quantize DC with deadzone protection relative to previous block's DC.
  const float dc_ideal = static_cast<float>(cblock[0]) * qmc[0];
  int16_t dc;
  if (std::fabs(dc_ideal - static_cast<float>(*last_dc_coeff)) < dc_threshold) {
    dc = *last_dc_coeff;
  } else {
    dc = static_cast<int16_t>(std::lround(dc_ideal));
  }
  *last_dc_coeff = dc;

  float ideal[DCTSIZE2];
  float prefix[DCTSIZE2];
  prefix[0] = 0.0f;
  uint64_t nonzero = 0;
  for (int k = 1; k < DCTSIZE2; ++k) {
    float v = static_cast<float>(cblock[k]) * qmc[k];
    ideal[k] = v;
    prefix[k] = prefix[k - 1] + weights[k] * v * v;
    if (static_cast<int>(std::fabs(v) + 0.5f) != 0) {
      nonzero |= (1ULL << k);
    }
  }

  if (nonzero == 0) {
    memset(cblock, 0, DCTSIZE2 * sizeof(int16_t));
    cblock[0] = dc;
    return;
  }

  struct State {
    uint8_t position;
    uint8_t from;
    int16_t level;
    float prefix;
    float cost;
  };
  State reachable[DCTSIZE2];
  reachable[0] = {0, 0, 0, 0.0f, 0.0f};
  int reachable_len = 1;

  uint64_t remaining = nonzero;
  while (remaining != 0) {
    int k = Ctz64(remaining);
    remaining &= remaining - 1;
    float x = ideal[k];
    float ax = std::fabs(x);
    int rounded = static_cast<int>(ax + 0.5f);
    int size = SizeOfLevel(rounded);
    float d = ax - static_cast<float>(rounded);
    float distortion = weights[k] * d * d;
    float zeros = prefix[k - 1];

    float best = std::numeric_limits<float>::infinity();
    int best_from = 0;
    int chosen = rounded;

    // Power-of-2 category boundary (2->1, 4->3, 8->7, ...):
    // Evaluate both rounded and rounded-1 in a single backwards pass with
    // monotonic prefix early-break.
    if (rounded >= 2 && (rounded & (rounded - 1)) == 0) {
      int lower = rounded - 1;
      int lower_size = size - 1;
      float dl = ax - static_cast<float>(lower);
      float lower_distortion = weights[k] * dl * dl;
      float lower_best = std::numeric_limits<float>::infinity();
      int lower_from = 0;

      for (int i = reachable_len - 1; i >= 0; --i) {
        const State& pred = reachable[i];
        float tail = zeros - pred.prefix;
        bool keep_rounded = (tail + distortion) < best;
        bool keep_lower = (tail + lower_distortion) < std::min(best, lower_best);
        if (!keep_rounded && !keep_lower) {
          break;
        }
        int run = k - static_cast<int>(pred.position) - 1;
        if (keep_rounded) {
          float cost = pred.cost + tail + distortion +
                       lambda * prices.Nonzero(run, size);
          if (cost < best) {
            best = cost;
            best_from = i;
          }
        }
        if (keep_lower) {
          float cost = pred.cost + tail + lower_distortion +
                       lambda * prices.Nonzero(run, lower_size);
          if (cost < lower_best) {
            lower_best = cost;
            lower_from = i;
          }
        }
      }
      if (lower_best < best) {
        best = lower_best;
        best_from = lower_from;
        chosen = lower;
      }
    } else {
      for (int i = reachable_len - 1; i >= 0; --i) {
        const State& pred = reachable[i];
        float tail = zeros - pred.prefix;
        if (tail + distortion >= best) {
          break;
        }
        int run = k - static_cast<int>(pred.position) - 1;
        float cost =
            pred.cost + tail + distortion + lambda * prices.Nonzero(run, size);
        if (cost < best) {
          best = cost;
          best_from = i;
        }
      }
    }

    int signed_level = (x < 0.0f) ? -chosen : chosen;
    signed_level = std::max(-32767, std::min(32767, signed_level));
    reachable[reachable_len++] = {
        static_cast<uint8_t>(k),
        static_cast<uint8_t>(best_from),
        static_cast<int16_t>(signed_level),
        prefix[k],
        best,
    };
  }

  // Close the block: trailing zeros + EOB symbol if last nonzero < 63.
  int end = 0;
  float end_cost = std::numeric_limits<float>::infinity();
  for (int i = 0; i < reachable_len; ++i) {
    const State& st = reachable[i];
    if (!std::isfinite(st.cost)) continue;
    float tail = prefix[63] - st.prefix;
    float eob = (st.position < 63) ? (lambda * prices.Eob()) : 0.0f;
    float cost = st.cost + tail + eob;
    if (cost < end_cost) {
      end_cost = cost;
      end = i;
    }
  }

  memset(cblock, 0, DCTSIZE2 * sizeof(int16_t));
  cblock[0] = dc;
  int idx = end;
  while (idx > 0) {
    const State& st = reachable[idx];
    cblock[st.position] = st.level;
    idx = st.from;
  }
}

}  // namespace

void TrellisQuantizeCoeffs(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  InitQuantizer(cinfo, QuantPass::SEARCH_SECOND_PASS);

  // Pass 1: Build real AC Huffman bit prices per AC Huffman table slot from
  // plain-rounded coefficients (heap-allocated to keep stack frame small).
  std::vector<uint32_t> ac_freq_buf(NUM_HUFF_TBLS * 257, 0);
  bool table_used[NUM_HUFF_TBLS] = {};
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info* comp = &cinfo->comp_info[c];
    const int tbl = std::max(0, std::min(comp->ac_tbl_no, NUM_HUFF_TBLS - 1));
    table_used[tbl] = true;
    const float* qmc = m->quant_mul[c];
    uint32_t* freq = &ac_freq_buf[tbl * 257];
    for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
      JBLOCKARRAY blocks = GetBlockRow(cinfo, c, by);
      for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
        AccumulatePlainAcHistogram(&blocks[0][bx][0], qmc, freq);
      }
    }
  }

  std::vector<AcPrices> prices(NUM_HUFF_TBLS);
  for (int t = 0; t < NUM_HUFF_TBLS; ++t) {
    if (table_used[t]) {
      prices[t].BuildFromFrequencies(&ac_freq_buf[t * 257]);
    }
  }

  const bool is_ycbcr = (cinfo->jpeg_color_space == JCS_YCbCr);
  const bool is_yuv420 =
      is_ycbcr && cinfo->num_components >= 3 &&
      cinfo->comp_info[0].h_samp_factor == 2 &&
      cinfo->comp_info[0].v_samp_factor == 2 &&
      cinfo->comp_info[1].h_samp_factor == 1 &&
      cinfo->comp_info[1].v_samp_factor == 1 &&
      cinfo->comp_info[2].h_samp_factor == 1 &&
      cinfo->comp_info[2].v_samp_factor == 1;

  // Distance-adaptive base rate-distortion lambda (quantizer-unit^2 per bit):
  // At high quality (distance <= 1.2), 0.045 saves substantial bitrate cleanly;
  // at low quality (large distance / large DQT steps), decay lambda smoothly so
  // the trellis does not wipe out +/-1 AC coefficients into flat DC-only blocks.
  const float distance = (m->distance >= 0.0f) ? m->distance : 1.0f;
  const float auto_lambda =
      0.045f / (1.0f + 0.55f * std::max(0.0f, distance - 1.2f));
  const float base_lambda =
      (m->trellis_lambda >= 0.0f) ? m->trellis_lambda : auto_lambda;
  // Tighten the upper AQ lambda_scale ceiling (1.50 -> 1.25 at high distance)
  // to prevent lone over-quantized flat blocks inside detailed regions.
  const float aq_hi_cap =
      Clamp(1.50f - 0.10f * std::max(0.0f, distance - 1.5f), 1.25f, 1.50f);
  // Gentle pixel-domain distortion tilt tuned for jpegli's steeper kExponent
  // DQT curve (clamped step_ratio prevents starving LF at low quality).
  static constexpr float kDistortionPower = 0.15f;

  // Precomputed Sobolev H^-1 / 2-Wasserstein radial transport weights
  // in natural order: 1 / sqrt(1 + 0.10 * (u^2 + v^2)).
  static constexpr float kTransportW[DCTSIZE2] = {
      1.0000000f, 0.9534626f, 0.8451543f, 0.7254763f, 0.6201737f, 0.5345225f, 0.4662524f, 0.4116935f,
      0.9534626f, 0.9128709f, 0.8164966f, 0.7071068f, 0.6085806f, 0.5270463f, 0.4612656f, 0.4082483f,
      0.8451543f, 0.8164966f, 0.7453560f, 0.6593805f, 0.5773503f, 0.5063697f, 0.4472136f, 0.3984095f,
      0.7254763f, 0.7071068f, 0.6593805f, 0.5976143f, 0.5345225f, 0.4767313f, 0.4264014f, 0.3834825f,
      0.6201737f, 0.6085806f, 0.5773503f, 0.5345225f, 0.4879500f, 0.4428074f, 0.4016097f, 0.3651484f,
      0.5345225f, 0.5270463f, 0.5063697f, 0.4767313f, 0.4428074f, 0.4082483f, 0.3752933f, 0.3450328f,
      0.4662524f, 0.4612656f, 0.4472136f, 0.4264014f, 0.4016097f, 0.3752933f, 0.3492151f, 0.3244428f,
      0.4116935f, 0.4082483f, 0.3984095f, 0.3834825f, 0.3651484f, 0.3450328f, 0.3244428f, 0.3042903f,
  };

  // Pass 2: Run sparse-bitmask Viterbi trellis per block.
  // Note: brightness modulation is already handled once by GammaModulation
  // inside ComputeAdaptiveQuantField (quant_field), so no duplicate luma
  // brightness tilt is applied here.
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info* comp = &cinfo->comp_info[c];
    const int tbl = std::max(0, std::min(comp->ac_tbl_no, NUM_HUFF_TBLS - 1));
    const JQUANT_TBL* qtable = cinfo->quant_tbl_ptrs[comp->quant_tbl_no];
    const float* qmc = m->quant_mul[c];
    const size_t h_factor = m->h_factor[c];
    const size_t v_factor = m->v_factor[c];
    const bool is_luma = (c == 0);
    const bool is_subsampled_chroma =
        (is_ycbcr && c > 0 && (h_factor > 1 || v_factor > 1));

    // Per-channel distortion weight (protect 4:2:0 chroma, especially Cr).
    float channel_weight = 1.0f;
    if (is_subsampled_chroma) {
      channel_weight = 3.0f;
      if (m->wasserstein_quant && c == 2) {
        channel_weight = 3.5f;
      }
    }

    float mean_step = 0.0f;
    for (int k = 1; k < DCTSIZE2; ++k) {
      mean_step += static_cast<float>(qtable->quantval[k]);
    }
    mean_step /= 63.0f;

    float weights[DCTSIZE2];
    weights[0] = channel_weight;
    for (int z = 1; z < DCTSIZE2; ++z) {
      int k = kJPEGNaturalOrder[z];
      float step_ratio = Clamp(
          static_cast<float>(qtable->quantval[k]) / std::max(1.0f, mean_step),
          0.60f, 1.65f);
      float w = channel_weight * std::pow(step_ratio, 2.0f * kDistortionPower);
      if (m->wasserstein_quant && is_luma) {
        // Internal Wasserstein / Sobolev H^-1 radial transport distortion metric.
        w *= (0.90f + 0.25f * kTransportW[k]);
      }
      weights[z] = w;
    }

    int16_t last_dc = 0;
    const float dc_zb_offset = m->zero_bias_offset[c][0];
    const float dc_zb_mul = m->zero_bias_mul[c][0];
    const bool use_aq = m->use_adaptive_quantization;
    const bool sharpyuv_luma = (m->use_sharpyuv && is_yuv420 && is_luma);

    // Pre-pass: 3-class DCT structure / maskability classifier with
    // content-only median normalization (ignoring flat blocks < 1.0 variance).
    const size_t num_blocks =
        static_cast<size_t>(comp->height_in_blocks) * comp->width_in_blocks;
    std::vector<float> block_ac_energy;
    std::vector<float> block_raw_mask;
    float median_mask = 1.0f;
    if (use_aq && num_blocks > 0) {
      block_ac_energy.resize(num_blocks);
      block_raw_mask.resize(num_blocks);
      std::vector<float> content_masks;
      content_masks.reserve(num_blocks);
      static constexpr float kInvQ4Scale = 1.0f / 128.0f;
      static constexpr float kMinContentVariance = 1.0f;

      for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
        JBLOCKARRAY blocks = GetBlockRow(cinfo, c, by);
        const float* qf = m->quant_field.Row(by * v_factor);
        for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
          const int16_t* cblock = &blocks[0][bx][0];
          float ac_energy = 0.0f;
          float peak_energy = 0.0f;
          float diag_energy = 0.0f;
          for (int z = 1; z < DCTSIZE2; ++z) {
            float v = static_cast<float>(cblock[z]) * kInvQ4Scale;
            float e = v * v;
            ac_energy += e;
            if (e > peak_energy) peak_energy = e;
            int k = kJPEGNaturalOrder[z];
            if ((k >> 3) >= 2 && (k & 7) >= 2) {
              diag_energy += e;
            }
          }
          float spread =
              1.0f - Clamp((peak_energy / (ac_energy + 1e-6f)) * 3.0f, 0.0f, 1.0f);
          float isotropy =
              Clamp((diag_energy / (ac_energy + 1e-6f)) * 2.5f, 0.0f, 1.0f);
          float maskability = spread * (0.4f + 0.6f * isotropy);
          float raw_mask = qf[bx * h_factor] * maskability;

          size_t bidx = static_cast<size_t>(by) * comp->width_in_blocks + bx;
          block_ac_energy[bidx] = ac_energy;
          block_raw_mask[bidx] = raw_mask;
        }
      }
      if (comp->width_in_blocks >= 2 && comp->height_in_blocks >= 2) {
        // 2D 5-point cross smoothing of block_raw_mask to eliminate lone
        // over-quantized flat blocks inside detailed regions.
        std::vector<float> smoothed(num_blocks);
        const size_t w_blk = comp->width_in_blocks;
        const size_t h_blk = comp->height_in_blocks;
        for (size_t by = 0; by < h_blk; ++by) {
          size_t y_up = (by > 0) ? (by - 1) : 0;
          size_t y_dn = (by + 1 < h_blk) ? (by + 1) : (h_blk - 1);
          for (size_t bx = 0; bx < w_blk; ++bx) {
            size_t x_l = (bx > 0) ? (bx - 1) : 0;
            size_t x_r = (bx + 1 < w_blk) ? (bx + 1) : (w_blk - 1);
            float center = block_raw_mask[by * w_blk + bx];
            float l = block_raw_mask[by * w_blk + x_l];
            float r = block_raw_mask[by * w_blk + x_r];
            float t = block_raw_mask[y_up * w_blk + bx];
            float b = block_raw_mask[y_dn * w_blk + bx];
            smoothed[by * w_blk + bx] = 0.50f * center + 0.125f * (l + r + t + b);
          }
        }
        block_raw_mask.swap(smoothed);
      }
      for (size_t bidx = 0; bidx < num_blocks; ++bidx) {
        if (block_ac_energy[bidx] >= kMinContentVariance) {
          content_masks.push_back(block_raw_mask[bidx]);
        }
      }
      if (!content_masks.empty()) {
        size_t mid = content_masks.size() / 2;
        std::nth_element(content_masks.begin(), content_masks.begin() + mid,
                         content_masks.end());
        median_mask = std::max(content_masks[mid], 0.10f);
      }
    }

    for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
      JBLOCKARRAY blocks = GetBlockRow(cinfo, c, by);
      const float* qf = use_aq ? m->quant_field.Row(by * v_factor) : nullptr;
      for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
        int16_t* cblock = &blocks[0][bx][0];
        const float aq_strength = use_aq ? qf[bx * h_factor] : 0.0f;

        // Recover approximate mean luma (0..255) from Q4 centered DC.
        const float dct0 = static_cast<float>(cblock[0]) / 128.0f + 128.0f;

        float lambda_scale = 1.0f;
        if (use_aq && !block_ac_energy.empty()) {
          size_t bidx = static_cast<size_t>(by) * comp->width_in_blocks + bx;
          if (block_ac_energy[bidx] < 1.0f) {
            // Smooth / flat gradient block: protect strongly against banding.
            lambda_scale *= 0.45f;
          } else {
            // Content block normalized against content-only median:
            // coherent fine detail stays ~0.72..0.95, noisy texture goes up to aq_hi_cap.
            float norm_mask = block_raw_mask[bidx] / median_mask;
            lambda_scale *= Clamp(0.65f + 0.35f * norm_mask, 0.45f, aq_hi_cap);
          }
        }
        if (sharpyuv_luma) {
          lambda_scale *= 0.88f;
        }

        float dc_threshold = dc_zb_offset + aq_strength * dc_zb_mul;
        if (m->wasserstein_quant && is_luma && dct0 < 64.0f) {
          float darkness = (64.0f - std::max(0.0f, dct0)) / 64.0f;
          dc_threshold *= (1.0f - 0.25f * darkness);
        }

        TrellisQuantizeBlock(cblock, qmc, prices[tbl], weights,
                             base_lambda * lambda_scale, dc_threshold,
                             &last_dc);
      }
    }
  }
}

}  // namespace jpegli
#endif  // HWY_ONCE
