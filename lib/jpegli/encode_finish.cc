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
                          float lambda, float dc_only_penalty) {
  // Plain-round DC (matches jhm17 and avoids DC banding/blocking).
  const int16_t dc = static_cast<int16_t>(
      std::lround(static_cast<float>(cblock[0]) * qmc[0]));

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
    if (i == 0 && dc_only_penalty > 0.0f) {
      cost += dc_only_penalty * prefix[63];
    }
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

// Separable 8x8 IDCT basis matching TransformFromPixels in dct-inl.h.
static constexpr float kSpatialIdctBasis[8][8] = {
    {1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f},
    {1.387039f,  1.175876f,  0.785695f,  0.275899f, -0.275899f, -0.785695f, -1.175876f, -1.387039f},
    {1.306563f,  0.541196f, -0.541196f, -1.306563f, -1.306563f, -0.541196f,  0.541196f,  1.306563f},
    {1.175876f, -0.275899f, -1.387039f, -0.785695f,  0.785695f,  1.387039f,  0.275899f, -1.175876f},
    {1.000000f, -1.000000f, -1.000000f,  1.000000f,  1.000000f, -1.000000f, -1.000000f,  1.000000f},
    {0.785695f, -1.387039f,  0.275899f,  1.175876f, -1.175876f, -0.275899f,  1.387039f, -0.785695f},
    {0.541196f, -1.306563f,  1.306563f, -0.541196f, -0.541196f,  1.306563f, -1.306563f,  0.541196f},
    {0.275899f, -0.785695f,  1.175876f, -1.387039f,  1.387039f, -1.175876f,  0.785695f, -0.275899f},
};

struct SpatialBasisTable {
  // basis2d[z][y*8 + x] for zigzag index z = 0..63.
  float basis[DCTSIZE2][DCTSIZE2];
  SpatialBasisTable() {
    for (int z = 0; z < DCTSIZE2; ++z) {
      int k = kJPEGNaturalOrder[z];
      int v = k >> 3;
      int u = k & 7;
      for (int y = 0; y < 8; ++y) {
        float bv = kSpatialIdctBasis[v][y];
        for (int x = 0; x < 8; ++x) {
          basis[z][y * 8 + x] = bv * kSpatialIdctBasis[u][x];
        }
      }
    }
  }
};

const SpatialBasisTable& GetSpatialBasis() {
  static const SpatialBasisTable kTable;
  return kTable;
}

void SeparableIdctFromNatural(const float* dct, float* pixels) {
  float tmp[DCTSIZE2];
  for (int v = 0; v < 8; ++v) {
    for (int x = 0; x < 8; ++x) {
      float sum = 0.0f;
      for (int u = 0; u < 8; ++u) {
        sum += dct[v * 8 + u] * kSpatialIdctBasis[u][x];
      }
      tmp[v * 8 + x] = sum;
    }
  }
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      float sum = 0.0f;
      for (int v = 0; v < 8; ++v) {
        sum += tmp[v * 8 + x] * kSpatialIdctBasis[v][y];
      }
      pixels[y * 8 + x] = sum;
    }
  }
}

// O(1) exact AC Huffman bit delta for changing cblock[z] from old_q to new_q,
// given the nearest nonzero neighbors prev_nz (< z, or 0) and next_nz (> z, or 64).
JPEGLI_INLINE float DeltaAcBits(int z, int old_q, int new_q, int prev_nz,
                                int next_nz, int next_size,
                                const AcPrices& prices) {
  auto seg_bits = [&](int qv) -> float {
    if (qv != 0) {
      float b = prices.Nonzero(z - prev_nz - 1, SizeOfLevel(std::abs(qv)));
      if (next_nz < 64) {
        b += prices.Nonzero(next_nz - z - 1, next_size);
      } else if (z < 63) {
        b += prices.Eob();
      }
      return b;
    } else {
      if (next_nz < 64) {
        return prices.Nonzero(next_nz - prev_nz - 1, next_size);
      } else if (prev_nz < 63) {
        return prices.Eob();
      }
      return 0.0f;
    }
  };
  return seg_bits(new_q) - seg_bits(old_q);
}

// Spatial-domain masked & clipped coordinate-descent R-D refinement for
// sharp text / graphic / clipped-boundary blocks.
// Minimizes: sum_{i=0..63} W[i] * (clip(rec[i]) - orig[i])^2 + lambda_z * bits
// where lambda_z = trellis_lambda * (Q[k]^2 / weights[z]) * avg_w matches
// Parseval's exact scaling at every frequency, and flat background pixels next
// to sharp text strokes receive elevated weight W_bg so mosquito noise, chroma
// speckles, and salt-and-pepper are suppressed cleanly without blurring text.
void RefineTextBlockSpatial(const int16_t* raw_cblock, int16_t* cblock,
                            const UINT16* quantval, const AcPrices& prices,
                            const float* weights, float trellis_lambda,
                            float text_score, bool is_luma, bool has_clip) {
  const SpatialBasisTable& tbl = GetSpatialBasis();

  // Reconstruct original pixels (1/128 accuracy) and current quantized recon.
  float raw_dct[DCTSIZE2];
  float cur_dct[DCTSIZE2];
  raw_dct[0] = 128.0f + static_cast<float>(raw_cblock[0]) * (1.0f / 128.0f);
  cur_dct[0] =
      128.0f + static_cast<float>(cblock[0]) * (static_cast<float>(quantval[0]) * 0.125f);
  for (int z = 1; z < DCTSIZE2; ++z) {
    int k = kJPEGNaturalOrder[z];
    raw_dct[k] = static_cast<float>(raw_cblock[z]) * (1.0f / 128.0f);
    cur_dct[k] =
        static_cast<float>(cblock[z]) * (static_cast<float>(quantval[k]) * 0.125f);
  }
  HWY_ALIGN float orig[DCTSIZE2];
  HWY_ALIGN float rec[DCTSIZE2];
  SeparableIdctFromNatural(raw_dct, orig);
  SeparableIdctFromNatural(cur_dct, rec);

  if (is_luma) {
    for (int i = 0; i < DCTSIZE2; ++i) {
      orig[i] = Clamp(orig[i], 0.0f, 255.0f);
    }
  }

  // Build per-pixel spatial masking weights: flat background pixels in text
  // blocks get elevated weight W_bg so ringing on clean backgrounds is penalized
  // much more heavily than minor stroke softening.
  HWY_ALIGN float W[DCTSIZE2];
  const float eps = is_luma ? 2.5f : 2.0f;
  const float w_bg_max = is_luma ? 2.5f : 3.5f;
  const float w_bg = 1.0f + text_score * (w_bg_max - 1.0f);
  float sum_w = 0.0f;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      int i = y * 8 + x;
      float v = orig[i];
      bool flat = true;
      if (x > 0 && std::abs(orig[i - 1] - v) > eps) flat = false;
      if (flat && x < 7 && std::abs(orig[i + 1] - v) > eps) flat = false;
      if (flat && y > 0 && std::abs(orig[i - 8] - v) > eps) flat = false;
      if (flat && y < 7 && std::abs(orig[i + 8] - v) > eps) flat = false;
      float wi = flat ? w_bg : 1.0f;
      W[i] = wi;
      sum_w += wi;
    }
  }
  const float avg_w = sum_w * (1.0f / 64.0f);
  const bool use_clip = is_luma && (has_clip || text_score > 0.0f);

  auto eval_dist_delta = [&](const float* b, float delta) -> float {
    float d = 0.0f;
    if (use_clip) {
      for (int i = 0; i < DCTSIZE2; ++i) {
        float rv = rec[i] + delta * b[i];
        rv = rv < 0.0f ? 0.0f : (rv > 255.0f ? 255.0f : rv);
        float e = rv - orig[i];
        d += W[i] * e * e;
      }
    } else {
      for (int i = 0; i < DCTSIZE2; ++i) {
        float e = (rec[i] + delta * b[i]) - orig[i];
        d += W[i] * e * e;
      }
    }
    return d;
  };

  float cur_dist = eval_dist_delta(tbl.basis[0], 0.0f);

  // Precompute ideal unquantized levels in zigzag order to skip irrelevant HF.
  float ideal_abs[DCTSIZE2];
  for (int z = 0; z < DCTSIZE2; ++z) {
    int k = kJPEGNaturalOrder[z];
    ideal_abs[z] = std::abs(static_cast<float>(raw_cblock[z])) /
                   (16.0f * static_cast<float>(quantval[k]));
  }

  const int max_iters = (text_score > 0.25f) ? 2 : 1;
  for (int iter = 0; iter < max_iters; ++iter) {
    bool changed = false;
    for (int z = 0; z < DCTSIZE2; ++z) {
      int16_t old_q = cblock[z];
      if (z > 20 && old_q == 0 && ideal_abs[z] < 0.35f) continue;

      int k = kJPEGNaturalOrder[z];
      const float qk = static_cast<float>(quantval[k]);
      const float step = qk * 0.125f;
      // Exact Parseval-matched spatial lambda for coefficient z.
      const float lam_z = trellis_lambda * (qk * qk / weights[z]) * avg_w;
      const float* b = tbl.basis[z];

      // Locate nearest nonzero neighbors once for O(1) exact bit deltas.
      int prev_nz = 0;
      int next_nz = 64;
      int next_size = 0;
      if (z > 0) {
        for (int p = z - 1; p >= 1; --p) {
          if (cblock[p] != 0) {
            prev_nz = p;
            break;
          }
        }
        for (int n = z + 1; n < DCTSIZE2; ++n) {
          if (cblock[n] != 0) {
            next_nz = n;
            next_size = SizeOfLevel(std::abs(static_cast<int>(cblock[n])));
            break;
          }
        }
      }

      int16_t best_q = old_q;
      float best_gain = 0.0f;  // positive = improvement (reduction in D + lam*R)
      float best_dist = cur_dist;

      auto try_val = [&](int32_t new_q_i32) {
        if (new_q_i32 == old_q) return;
        if (new_q_i32 < -1023 || new_q_i32 > 1023) return;
        int16_t new_q = static_cast<int16_t>(new_q_i32);
        float delta = static_cast<float>(new_q - old_q) * step;
        float nd = eval_dist_delta(b, delta);
        float dbits =
            (z == 0) ? 0.0f
                     : DeltaAcBits(z, old_q, new_q, prev_nz, next_nz, next_size,
                                   prices);
        float gain = (cur_dist - nd) - lam_z * dbits;
        if (gain > best_gain + 1e-3f) {
          best_gain = gain;
          best_q = new_q;
          best_dist = nd;
        }
      };

      try_val(old_q - 1);
      try_val(old_q + 1);
      if (z == 0 && has_clip) {
        try_val(old_q - 2);
        try_val(old_q + 2);
        try_val(old_q - 4);
        try_val(old_q + 4);
      } else if (z > 0 && std::abs(old_q) >= 2) {
        try_val(0);
      }

      if (best_q != old_q) {
        float delta = static_cast<float>(best_q - old_q) * step;
        for (int i = 0; i < DCTSIZE2; ++i) {
          rec[i] += delta * b[i];
        }
        cblock[z] = best_q;
        cur_dist = best_dist;
        changed = true;
      }
    }
    if (!changed) break;
  }
}

// Causal inter-block seam-continuity R-D refinement for smooth gradient blocks.
// Optimizes only the 6 low-frequency gradient coefficients (DC, (0,1), (1,0),
// (2,0), (1,1), (0,2)) to minimize in-block L2 distortion plus cross-seam
// discontinuity against already-finalized smooth left/top neighbors, plus
// Parseval-matched exact AC Huffman bit deltas.
void RefineGradientSeamBlock(const float* orig, float* rec, int16_t* cblock,
                             const UINT16* quantval, const AcPrices& prices,
                             const float* weights, float trellis_lambda,
                             bool has_left, const float* left_rec,
                             const float* left_orig, bool has_top,
                             const float* top_rec, const float* top_orig,
                             float seam_weight) {
  // Per-pixel smoothness gate across the left and top 8x8 boundaries so we
  // never smooth across a true sharp edge aligned with an 8x8 block seam.
  float wL[8] = {};
  float wT[8] = {};
  float targetL[8] = {};
  float targetT[8] = {};
  int active_seam = 0;
  static constexpr float kMaxSeamJump = 14.0f;
  if (has_left) {
    for (int y = 0; y < 8; ++y) {
      float diff = orig[y * 8] - left_orig[y];
      if (std::abs(diff) <= kMaxSeamJump) {
        wL[y] = seam_weight;
        targetL[y] = (0.75f * left_rec[y] + 0.25f * left_orig[y]) + diff;
        ++active_seam;
      }
    }
  }
  if (has_top) {
    for (int x = 0; x < 8; ++x) {
      float diff = orig[x] - top_orig[x];
      if (std::abs(diff) <= kMaxSeamJump) {
        wT[x] = seam_weight;
        targetT[x] = (0.75f * top_rec[x] + 0.25f * top_orig[x]) + diff;
        ++active_seam;
      }
    }
  }
  if (active_seam < 3) return;

  // Only refine blocks with a visible cross-seam discontinuity.
  float seam_err2 = 0.0f;
  if (has_left) {
    for (int y = 0; y < 8; ++y) {
      if (wL[y] > 0.0f) {
        float e = rec[y * 8] - targetL[y];
        seam_err2 += e * e;
      }
    }
  }
  if (has_top) {
    for (int x = 0; x < 8; ++x) {
      if (wT[x] > 0.0f) {
        float e = rec[x] - targetT[x];
        seam_err2 += e * e;
      }
    }
  }
  if (seam_err2 < 0.50f * static_cast<float>(active_seam)) return;

  float orig_mean = 0.0f;
  for (int i = 0; i < DCTSIZE2; ++i) orig_mean += orig[i];
  orig_mean *= (1.0f / 64.0f);
  const float ideal_dc =
      (orig_mean - 128.0f) * (8.0f / static_cast<float>(quantval[0]));

  const SpatialBasisTable& tbl = GetSpatialBasis();
  const float avg_w =
      1.0f + (static_cast<float>(active_seam) * seam_weight) * (1.0f / 64.0f);

  auto eval_dist_delta = [&](const float* b, float delta) -> float {
    float d = 0.0f;
    for (int i = 0; i < DCTSIZE2; ++i) {
      float e = (rec[i] + delta * b[i]) - orig[i];
      d += e * e;
    }
    if (has_left) {
      for (int y = 0; y < 8; ++y) {
        if (wL[y] > 0.0f) {
          float e = (rec[y * 8] + delta * b[y * 8]) - targetL[y];
          d += wL[y] * e * e;
        }
      }
    }
    if (has_top) {
      for (int x = 0; x < 8; ++x) {
        if (wT[x] > 0.0f) {
          float e = (rec[x] + delta * b[x]) - targetT[x];
          d += wT[x] * e * e;
        }
      }
    }
    return d;
  };

  float cur_dist = eval_dist_delta(tbl.basis[0], 0.0f);

  for (int iter = 0; iter < 2; ++iter) {
    bool changed = false;
    for (int z = 0; z < 6; ++z) {
      int16_t old_q = cblock[z];
      int k = kJPEGNaturalOrder[z];
      const float qk = static_cast<float>(quantval[k]);
      const float step = qk * 0.125f;
      const float lam_z = trellis_lambda * (qk * qk / weights[z]) * avg_w;
      const float* b = tbl.basis[z];

      int prev_nz = 0;
      int next_nz = 64;
      int next_size = 0;
      if (z > 0) {
        for (int p = z - 1; p >= 1; --p) {
          if (cblock[p] != 0) {
            prev_nz = p;
            break;
          }
        }
        for (int n = z + 1; n < DCTSIZE2; ++n) {
          if (cblock[n] != 0) {
            next_nz = n;
            next_size = SizeOfLevel(std::abs(static_cast<int>(cblock[n])));
            break;
          }
        }
      }

      int16_t best_q = old_q;
      float best_gain = 0.0f;
      float best_dist = cur_dist;

      for (int delta_q : {-1, 1}) {
        int32_t new_q_i32 = static_cast<int32_t>(old_q) + delta_q;
        if (new_q_i32 < -1023 || new_q_i32 > 1023) continue;
        // Bound DC adjustment near the half-step rounding boundary to guarantee
        // zero macro-scale drift across smooth gradients.
        if (z == 0 && std::abs(static_cast<float>(new_q_i32) - ideal_dc) > 0.82f) {
          continue;
        }
        int16_t new_q = static_cast<int16_t>(new_q_i32);
        float delta = static_cast<float>(delta_q) * step;
        float nd = eval_dist_delta(b, delta);
        float dbits =
            (z == 0) ? 0.0f
                     : DeltaAcBits(z, old_q, new_q, prev_nz, next_nz, next_size,
                                   prices);
        float gain = (cur_dist - nd) - lam_z * dbits;
        if (gain > best_gain + 1e-3f) {
          best_gain = gain;
          best_q = new_q;
          best_dist = nd;
        }
      }

      if (best_q != old_q) {
        float delta = static_cast<float>(best_q - old_q) * step;
        for (int i = 0; i < DCTSIZE2; ++i) {
          rec[i] += delta * b[i];
        }
        cblock[z] = best_q;
        cur_dist = best_dist;
        changed = true;
      }
    }
    if (!changed) break;
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

  // Distance- and subsampling-adaptive base rate-distortion lambda
  // (quantizer-unit^2 per bit):
  // - 4:4:4 starts at 0.030 (preserving crisp photographic detail and BA max)
  //   and decays smoothly toward ~0.012 at Q30.
  // - 4:2:0 starts at 0.015 (gentler on Luma since Chroma is already protected)
  //   and decays smoothly toward ~0.006 at Q30.
  const float distance = (m->distance >= 0.0f) ? m->distance : 1.0f;
  const float lambda_hq = is_yuv420 ? 0.015f : 0.030f;
  const float auto_lambda =
      lambda_hq / (1.0f + 0.30f * std::max(0.0f, distance - 1.2f));
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
      } else if (m->wasserstein_quant && !is_luma) {
        // Gentle Sobolev transport rolloff on Chroma to prefer dropping HF
        // chroma speckles over low-frequency color accuracy.
        w *= (0.85f + 0.30f * kTransportW[k]);
      }
      weights[z] = w;
    }

    const bool use_aq = m->use_adaptive_quantization;
    const bool sharpyuv_luma = (m->use_sharpyuv && is_yuv420 && is_luma);

    // Pre-pass: 3-class DCT structure / maskability classifier with
    // content-only median normalization (ignoring flat blocks < 1.0 variance).
    const size_t num_blocks =
        static_cast<size_t>(comp->height_in_blocks) * comp->width_in_blocks;
    std::vector<float> block_ac_energy;
    std::vector<float> block_raw_mask;
    std::vector<uint8_t> block_is_smooth(num_blocks, 0);
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
          float hf_energy = 0.0f;
          float peak_energy = 0.0f;
          float diag_energy = 0.0f;
          for (int z = 1; z < DCTSIZE2; ++z) {
            float v = static_cast<float>(cblock[z]) * kInvQ4Scale;
            float e = v * v;
            ac_energy += e;
            if (z >= 6) hf_energy += e;
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
          // Smooth flat or pure low-frequency gradient block (negligible HF
          // energy beyond the 5 lowest linear/quadratic basis functions).
          if (ac_energy < 1.0f || (hf_energy < 1.5f && ac_energy < 150.0f)) {
            block_is_smooth[bidx] = 1;
          }
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

    // 1-row causal seam buffers for smooth gradient continuity refinement.
    const size_t w_blk_c = comp->width_in_blocks;
    std::vector<float> prev_row_rec_bot(w_blk_c * 8, 0.0f);
    std::vector<float> prev_row_orig_bot(w_blk_c * 8, 0.0f);
    std::vector<uint8_t> prev_row_valid(w_blk_c, 0);
    std::vector<float> cur_row_rec_bot(w_blk_c * 8, 0.0f);
    std::vector<float> cur_row_orig_bot(w_blk_c * 8, 0.0f);
    std::vector<uint8_t> cur_row_valid(w_blk_c, 0);

    for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
      JBLOCKARRAY blocks = GetBlockRow(cinfo, c, by);
      float prev_col_rec_right[8] = {};
      float prev_col_orig_right[8] = {};
      bool prev_col_valid = false;
      std::fill(cur_row_valid.begin(), cur_row_valid.end(), 0);
      for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
        int16_t* cblock = &blocks[0][bx][0];
        const size_t bidx = static_cast<size_t>(by) * w_blk_c + bx;

        // Look up pixel-domain text/sharp-graphic confidence and clip flags.
        float text_score = 0.0f;
        bool has_clip = false;
        if (m->text_block_map[c] != nullptr) {
          uint8_t flags = m->text_block_map[c][bidx];
          has_clip = ((flags & 0x80) != 0);
          text_score = static_cast<float>(flags & 0x7F) * (1.0f / 127.0f);
        }
        if (!is_luma && m->text_block_map[0] != nullptr) {
          const jpeg_component_info* ycomp = &cinfo->comp_info[0];
          uint8_t max_yts = 0;
          for (size_t dy = 0; dy < v_factor; ++dy) {
            size_t yby =
                std::min<size_t>(by * v_factor + dy, ycomp->height_in_blocks - 1);
            for (size_t dx = 0; dx < h_factor; ++dx) {
              size_t ybx =
                  std::min<size_t>(bx * h_factor + dx, ycomp->width_in_blocks - 1);
              uint8_t yts =
                  m->text_block_map[0][yby * ycomp->width_in_blocks + ybx] & 0x7F;
              if (yts > max_yts) max_yts = yts;
            }
          }
          text_score = std::max(text_score,
                                static_cast<float>(max_yts) * (1.0f / 127.0f));
        }

        const bool is_smooth_grad =
            (!block_is_smooth.empty() && block_is_smooth[bidx] != 0 &&
             text_score <= 0.05f);
        const bool do_spatial_refine =
            (text_score > 0.05f || (is_luma && has_clip));
        // Apply causal seam-continuity refinement on smooth Chroma gradients
        // (where quantization steps are coarser and 8x8 hue blocking appears)
        // while keeping Luma drift-free on natural photography.
        const bool do_seam_refine =
            (!is_luma && !do_spatial_refine && is_smooth_grad);
        int16_t raw_cblock[DCTSIZE2];
        if (do_spatial_refine || do_seam_refine) {
          memcpy(raw_cblock, cblock, DCTSIZE2 * sizeof(int16_t));
        }

        float lambda_scale = 1.0f;
        if (use_aq && !block_ac_energy.empty()) {
          if (block_ac_energy[bidx] < 1.0f || is_smooth_grad) {
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
        if (text_score > 0.0f) {
          // Sharp text/graphics on flat backgrounds have broadband DCT spectra
          // that mimic noisy texture; prevent AQ from over-quantizing them.
          float target_ls = is_luma ? 0.85f : 0.95f;
          if (lambda_scale > target_ls) {
            lambda_scale += text_score * (target_ls - lambda_scale);
          }
        }

        float dc_only_penalty = 0.0f;
        if (m->wasserstein_quant && is_luma && !block_ac_energy.empty() &&
            comp->width_in_blocks >= 2 && comp->height_in_blocks >= 2) {
          const size_t w_blk = comp->width_in_blocks;
          const size_t h_blk = comp->height_in_blocks;
          size_t y_up = (by > 0) ? (by - 1) : 0;
          size_t y_dn = (by + 1 < h_blk) ? (by + 1) : (h_blk - 1);
          size_t x_l = (bx > 0) ? (bx - 1) : 0;
          size_t x_r = (bx + 1 < w_blk) ? (bx + 1) : (w_blk - 1);
          float nbr_e = 0.25f * (block_ac_energy[by * w_blk + x_l] +
                                 block_ac_energy[by * w_blk + x_r] +
                                 block_ac_energy[y_up * w_blk + bx] +
                                 block_ac_energy[y_dn * w_blk + bx]);
          if (nbr_e > 4.0f) {
            dc_only_penalty = 0.5f * Clamp((nbr_e - 4.0f) / 20.0f, 0.0f, 1.0f);
          }
        }

        TrellisQuantizeBlock(cblock, qmc, prices[tbl], weights,
                             base_lambda * lambda_scale, dc_only_penalty);

        if (do_spatial_refine) {
          RefineTextBlockSpatial(raw_cblock, cblock, qtable->quantval,
                                 prices[tbl], weights,
                                 base_lambda * lambda_scale, text_score,
                                 is_luma, has_clip);
          prev_col_valid = false;
        } else if (do_seam_refine) {
          float raw_dct[DCTSIZE2];
          float cur_dct[DCTSIZE2];
          raw_dct[0] =
              128.0f + static_cast<float>(raw_cblock[0]) * (1.0f / 128.0f);
          cur_dct[0] =
              128.0f + static_cast<float>(cblock[0]) *
                           (static_cast<float>(qtable->quantval[0]) * 0.125f);
          for (int z = 1; z < DCTSIZE2; ++z) {
            int k = kJPEGNaturalOrder[z];
            raw_dct[k] = static_cast<float>(raw_cblock[z]) * (1.0f / 128.0f);
            cur_dct[k] = static_cast<float>(cblock[z]) *
                         (static_cast<float>(qtable->quantval[k]) * 0.125f);
          }
          HWY_ALIGN float orig_px[DCTSIZE2];
          HWY_ALIGN float rec_px[DCTSIZE2];
          SeparableIdctFromNatural(raw_dct, orig_px);
          SeparableIdctFromNatural(cur_dct, rec_px);

          bool has_left = (bx > 0 && prev_col_valid);
          bool has_top = (by > 0 && prev_row_valid[bx] != 0);
          if (has_left || has_top) {
            RefineGradientSeamBlock(
                orig_px, rec_px, cblock, qtable->quantval, prices[tbl], weights,
                base_lambda * lambda_scale, has_left, prev_col_rec_right,
                prev_col_orig_right, has_top, &prev_row_rec_bot[bx * 8],
                &prev_row_orig_bot[bx * 8], /*seam_weight=*/3.5f);
          }
          for (int y = 0; y < 8; ++y) {
            prev_col_rec_right[y] = rec_px[y * 8 + 7];
            prev_col_orig_right[y] = orig_px[y * 8 + 7];
          }
          for (int x = 0; x < 8; ++x) {
            cur_row_rec_bot[bx * 8 + x] = rec_px[7 * 8 + x];
            cur_row_orig_bot[bx * 8 + x] = orig_px[7 * 8 + x];
          }
          prev_col_valid = true;
          cur_row_valid[bx] = 1;
        } else {
          prev_col_valid = false;
        }
      }
      prev_row_rec_bot.swap(cur_row_rec_bot);
      prev_row_orig_bot.swap(cur_row_orig_bot);
      prev_row_valid.swap(cur_row_valid);
    }
  }
}

}  // namespace jpegli
#endif  // HWY_ONCE
