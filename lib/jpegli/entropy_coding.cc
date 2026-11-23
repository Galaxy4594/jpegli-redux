// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/entropy_coding.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "lib/base/bits.h"
#include "lib/base/status.h"
#include "lib/base/types.h"
#include "lib/jpegli/common.h"
#include "lib/jpegli/common_internal.h"
#include "lib/jpegli/encode_internal.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/huffman.h"
#include "lib/jpegli/memory_manager.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jpegli/entropy_coding.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jpegli/entropy_coding-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {

void ComputeTokensSequential(const coeff_t* block, int last_dc, int dc_ctx,
                             int ac_ctx, Token** tokens_ptr) {
  ComputeTokensForBlock<coeff_t, true>(block, last_dc, dc_ctx, ac_ctx,
                                       tokens_ptr);
}

void ComputeACMasksSIMD(const coeff_t* block, int al, uint64_t* nz_any_out,
                        uint64_t* nz_cur_out) {
  const HWY_CAPPED(int16_t, 16) d16;
  const size_t N = Lanes(d16);
  const auto zero = Zero(d16);
  const auto thresh = Set(d16, static_cast<int16_t>(1 << al));
  uint64_t nz_any = 0;
  uint64_t nz_cur = 0;
  for (size_t k = 0; k < DCTSIZE2; k += N) {
    const auto v = LoadU(d16, block + k);
    const auto m_any = Ne(v, zero);
    uint8_t b_any[2] = {0, 0};
    StoreMaskBits(d16, m_any, b_any);
    uint64_t bits_any =
        static_cast<uint64_t>(b_any[0]) | (static_cast<uint64_t>(b_any[1]) << 8);
    nz_any |= bits_any << k;
    if (al > 0) {
      const auto m_cur = Ge(Abs(v), thresh);
      uint8_t b_cur[2] = {0, 0};
      StoreMaskBits(d16, m_cur, b_cur);
      uint64_t bits_cur = static_cast<uint64_t>(b_cur[0]) |
                          (static_cast<uint64_t>(b_cur[1]) << 8);
      nz_cur |= bits_cur << k;
    }
  }
  *nz_any_out = nz_any;
  *nz_cur_out = (al > 0) ? nz_cur : nz_any;
}

void ComputeACMultiMasksSIMD(const coeff_t* block, uint64_t nz_out[4]) {
  const HWY_CAPPED(int16_t, 16) d16;
  const size_t N = Lanes(d16);
  const auto v1 = Set(d16, 1);
  const auto v2 = Set(d16, 2);
  const auto v4 = Set(d16, 4);
  const auto v8 = Set(d16, 8);
  uint64_t nz0 = 0, nz1 = 0, nz2 = 0, nz3 = 0;
  for (size_t k = 0; k < DCTSIZE2; k += N) {
    const auto a = Abs(LoadU(d16, block + k));
    uint8_t b[2] = {0, 0};
    StoreMaskBits(d16, Ge(a, v1), b);
    nz0 |= (static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 8)) << k;
    b[0] = b[1] = 0;
    StoreMaskBits(d16, Ge(a, v2), b);
    nz1 |= (static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 8)) << k;
    b[0] = b[1] = 0;
    StoreMaskBits(d16, Ge(a, v4), b);
    nz2 |= (static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 8)) << k;
    b[0] = b[1] = 0;
    StoreMaskBits(d16, Ge(a, v8), b);
    nz3 |= (static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 8)) << k;
  }
  nz_out[0] = nz0 & ~uint64_t{1};
  nz_out[1] = nz1 & ~uint64_t{1};
  nz_out[2] = nz2 & ~uint64_t{1};
  nz_out[3] = nz3 & ~uint64_t{1};
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jpegli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jpegli {

size_t MaxNumTokensPerMCURow(j_compress_ptr cinfo) {
  int MCUs_per_row = DivCeil(cinfo->image_width, 8 * cinfo->max_h_samp_factor);
  size_t blocks_per_mcu = 0;
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    blocks_per_mcu +=
        static_cast<size_t>(comp->h_samp_factor) * comp->v_samp_factor;
  }
  return kDCTBlockSize * blocks_per_mcu * MCUs_per_row;
}

size_t EstimateNumTokens(j_compress_ptr cinfo, size_t mcu_y, size_t ysize_mcus,
                         size_t num_tokens, size_t max_per_row) {
  size_t estimate;
  if (mcu_y == 0) {
    estimate = 4 * max_per_row;
  } else {
    estimate = (4 * ysize_mcus * num_tokens) / (3 * mcu_y);
  }
  size_t mcus_left = ysize_mcus - mcu_y;
  return std::min(mcus_left * max_per_row,
                  std::max(max_per_row, estimate > num_tokens ? estimate - num_tokens : max_per_row));
}

namespace {
HWY_EXPORT(ComputeTokensSequential);
HWY_EXPORT(ComputeACMasksSIMD);
HWY_EXPORT(ComputeACMultiMasksSIMD);

inline int TrailingZeros64Fast(uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
  unsigned long idx;
  _BitScanForward64(&idx, v);
  return static_cast<int>(idx);
#else
  return __builtin_ctzll(v);
#endif
}

inline int PopCount64Fast(uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
  return static_cast<int>(__popcnt64(v));
#else
  return __builtin_popcountll(v);
#endif
}

void TokenizeProgressiveDC(const coeff_t* coeffs, int context, int Al,
                           coeff_t* last_dc_coeff, Token** next_token) {
  coeff_t temp2;
  coeff_t temp;
  temp2 = coeffs[0] >> Al;
  temp = temp2 - *last_dc_coeff;
  *last_dc_coeff = temp2;
  temp2 = temp;
  if (temp < 0) {
    temp = -temp;
    temp2--;
  }
  int nbits = (temp == 0) ? 0 : (jpegli::FloorLog2Nonzero<uint32_t>(temp) + 1);
  int bits = temp2 & ((1 << nbits) - 1);
  *(*next_token)++ = Token(context, nbits, bits);
}

void TokenizeACProgressiveScan(j_compress_ptr cinfo, int scan_index,
                               int context, ScanTokenInfo* sti) {
  jpeg_comp_master* m = cinfo->master;
  const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
  const int comp_idx = scan_info->component_index[0];
  const jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
  const int Al = scan_info->Al;
  const int Ss = scan_info->Ss;
  const int Se = scan_info->Se;
  const size_t restart_interval = sti->restart_interval;
  int restarts_to_go = restart_interval;
  size_t num_blocks =
      static_cast<size_t>(comp->height_in_blocks) * comp->width_in_blocks;
  size_t num_restarts =
      restart_interval > 0 ? DivCeil(num_blocks, restart_interval) : 1;
  size_t restart_idx = 0;
  int eob_run = 0;
  TokenArray* ta = &m->token_arrays[m->cur_token_array];
  sti->token_offset = m->total_num_tokens + ta->num_tokens;
  sti->restarts = Allocate<size_t>(cinfo, num_restarts, JPOOL_IMAGE);
  const auto emit_eob_run = [&]() {
    int nbits = jpegli::FloorLog2Nonzero<uint32_t>(eob_run);
    int symbol = nbits << 4u;
    *m->next_token++ = Token(context, symbol, eob_run & ((1 << nbits) - 1));
    eob_run = 0;
  };
  for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
    JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(
        reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[comp_idx], by,
        1, FALSE);
    // Each coefficient can appear in at most one token, but we have to reserve
    // one extra EOBrun token that was rolled over from the previous block-row
    // and has to be flushed at the end.
    int max_tokens_per_row = 1 + comp->width_in_blocks * (Se - Ss + 1);
    if (ta->num_tokens + max_tokens_per_row > m->num_tokens) {
      if (ta->tokens) {
        m->total_num_tokens += ta->num_tokens;
        ++m->cur_token_array;
        ta = &m->token_arrays[m->cur_token_array];
      }
      size_t scan_tokens = m->total_num_tokens - sti->token_offset;
      m->num_tokens =
          EstimateNumTokens(cinfo, by, comp->height_in_blocks,
                            scan_tokens, max_tokens_per_row);
      ta->tokens = Allocate<Token>(cinfo, m->num_tokens, JPOOL_IMAGE);
      m->next_token = ta->tokens;
    }
    for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
      if (restart_interval > 0 && restarts_to_go == 0) {
        if (eob_run > 0) emit_eob_run();
        ta->num_tokens = m->next_token - ta->tokens;
        sti->restarts[restart_idx++] = m->total_num_tokens + ta->num_tokens;
        restarts_to_go = restart_interval;
      }
      const coeff_t* block = &blocks[0][bx][0];
      uint64_t nz_any = 0, nz_cur = 0;
      HWY_DYNAMIC_DISPATCH(ComputeACMasksSIMD)(block, Al, &nz_any, &nz_cur);
      const uint64_t band =
          (~uint64_t{0} << Ss) & (~uint64_t{0} >> (63 - Se));
      uint64_t nz = nz_cur & band;
      int num_future_nzeros =
          (Al > 0) ? PopCount64Fast((nz_any & ~nz_cur) & band) : 0;
      int num_nzeros = 0;
      int next_k = Ss;
      while (nz != 0) {
        int k = TrailingZeros64Fast(nz);
        nz &= nz - 1;
        int r = k - next_k;
        coeff_t temp = block[k];
        coeff_t temp2;
        if (temp < 0) {
          temp = -temp;
          temp >>= Al;
          temp2 = ~temp;
        } else {
          temp >>= Al;
          temp2 = temp;
        }
        if (eob_run > 0) emit_eob_run();
        while (r > 15) {
          *m->next_token++ = Token(context, 0xf0, 0);
          r -= 16;
        }
        int nbits = jpegli::FloorLog2Nonzero<uint32_t>(temp) + 1;
        int symbol = (r << 4u) + nbits;
        *m->next_token++ = Token(context, symbol, temp2 & ((1 << nbits) - 1));
        ++num_nzeros;
        next_k = k + 1;
      }
      if (next_k <= Se) {
        ++eob_run;
        if (eob_run == 0x7FFF) emit_eob_run();
      }
      sti->num_nonzeros += num_nzeros;
      sti->num_future_nonzeros += num_future_nzeros;
      --restarts_to_go;
    }
    ta->num_tokens = m->next_token - ta->tokens;
  }
  if (eob_run > 0) {
    emit_eob_run();
    ++ta->num_tokens;
  }
  sti->num_tokens = m->total_num_tokens + ta->num_tokens - sti->token_offset;
  sti->restarts[restart_idx++] = m->total_num_tokens + ta->num_tokens;
}

void TokenizeACRefinementScan(j_compress_ptr cinfo, int scan_index,
                              ScanTokenInfo* sti) {
  jpeg_comp_master* m = cinfo->master;
  const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
  const int comp_idx = scan_info->component_index[0];
  const jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
  const int Al = scan_info->Al;
  const int Ss = scan_info->Ss;
  const int Se = scan_info->Se;
  const size_t restart_interval = sti->restart_interval;
  int restarts_to_go = restart_interval;
  RefToken token;
  int eob_run = 0;
  int eob_refbits = 0;
  size_t num_blocks =
      static_cast<size_t>(comp->height_in_blocks) * comp->width_in_blocks;
  size_t num_restarts =
      restart_interval > 0 ? DivCeil(num_blocks, restart_interval) : 1;
  sti->tokens = m->next_refinement_token;
  sti->refbits = m->next_refinement_bit;
  sti->eobruns = Allocate<uint16_t>(cinfo, num_blocks / 2, JPOOL_IMAGE);
  sti->restarts = Allocate<size_t>(cinfo, num_restarts, JPOOL_IMAGE);
  RefToken* next_token = sti->tokens;
  RefToken* next_eob_token = next_token;
  uint8_t* next_ref_bit = sti->refbits;
  uint16_t* next_eobrun = sti->eobruns;
  size_t restart_idx = 0;
  for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
    JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(
        reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[comp_idx], by,
        1, FALSE);
    for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
      if (restart_interval > 0 && restarts_to_go == 0) {
        sti->restarts[restart_idx++] = next_token - sti->tokens;
        restarts_to_go = restart_interval;
        next_eob_token = next_token;
        eob_run = eob_refbits = 0;
      }
      const coeff_t* block = &blocks[0][bx][0];
      uint64_t nz_any = 0, nz_cur = 0;
      HWY_DYNAMIC_DISPATCH(ComputeACMasksSIMD)(block, Al, &nz_any, &nz_cur);
      const uint64_t band =
          (~uint64_t{0} << Ss) & (~uint64_t{0} >> (63 - Se));
      uint64_t nz = nz_cur & band;
      int num_eob_refinement_bits = 0;
      int num_refinement_bits = 0;
      int num_nzeros = 0;
      int r = 0;
      int next_k = Ss;
      while (nz != 0) {
        int k = TrailingZeros64Fast(nz);
        nz &= nz - 1;
        r += k - next_k;
        next_k = k + 1;
        int absval = block[k];
        const int mask = absval >> (8 * sizeof(int) - 1);
        absval += mask;
        absval ^= mask;
        absval >>= Al;
        while (r > 15) {
          token.symbol = 0xf0;
          token.refbits = num_refinement_bits;
          *next_token++ = token;
          r -= 16;
          num_eob_refinement_bits += num_refinement_bits;
          num_refinement_bits = 0;
        }
        if (absval > 1) {
          *next_ref_bit++ = absval & 1u;
          ++num_refinement_bits;
          continue;
        }
        int symbol = (r << 4u) + 1 + ((mask + 1) << 1);
        token.symbol = symbol;
        token.refbits = num_refinement_bits;
        *next_token++ = token;
        ++num_nzeros;
        num_refinement_bits = 0;
        num_eob_refinement_bits = 0;
        r = 0;
        next_eob_token = next_token;
        eob_run = eob_refbits = 0;
      }
      r += (Se + 1) - next_k;
      if (r > 0 || num_eob_refinement_bits + num_refinement_bits > 0) {
        ++eob_run;
        eob_refbits += num_eob_refinement_bits + num_refinement_bits;
        if (eob_refbits > 255) {
          ++next_eob_token;
          eob_refbits = num_eob_refinement_bits + num_refinement_bits;
          eob_run = 1;
        }
        next_token = next_eob_token;
        next_token->refbits = eob_refbits;
        if (eob_run == 1) {
          next_token->symbol = 0;
        } else if (eob_run == 2) {
          next_token->symbol = 16;
          *next_eobrun++ = 0;
        } else if ((eob_run & (eob_run - 1)) == 0) {
          next_token->symbol += 16;
          next_eobrun[-1] = 0;
        } else {
          ++next_eobrun[-1];
        }
        ++next_token;
        if (eob_run == 0x7fff) {
          next_eob_token = next_token;
          eob_run = eob_refbits = 0;
        }
      }
      sti->num_nonzeros += num_nzeros;
      --restarts_to_go;
    }
  }
  sti->num_tokens = next_token - sti->tokens;
  sti->restarts[restart_idx++] = sti->num_tokens;
  m->next_refinement_token = next_token;
  m->next_refinement_bit = next_ref_bit;
}

void TokenizeScan(j_compress_ptr cinfo, size_t scan_index, int ac_ctx_offset,
                  ScanTokenInfo* sti) {
  const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
  if (scan_info->Ss > 0) {
    if (scan_info->Ah == 0) {
      TokenizeACProgressiveScan(cinfo, scan_index, ac_ctx_offset, sti);
    } else {
      TokenizeACRefinementScan(cinfo, scan_index, sti);
    }
    return;
  }

  jpeg_comp_master* m = cinfo->master;
  size_t restart_interval = sti->restart_interval;
  int restarts_to_go = restart_interval;
  coeff_t last_dc_coeff[MAX_COMPS_IN_SCAN] = {0};

  // "Non-interleaved" means color data comes in separate scans, in other words
  // each scan can contain only one color component.
  const bool is_interleaved = (scan_info->comps_in_scan > 1);
  const bool is_progressive = FROM_JPEGLI_BOOL(cinfo->progressive_mode);
  const int Ah = scan_info->Ah;
  const int Al = scan_info->Al;
  HWY_ALIGN constexpr coeff_t kSinkBlock[DCTSIZE2] = {0};

  size_t restart_idx = 0;
  TokenArray* ta = &m->token_arrays[m->cur_token_array];
  sti->token_offset = Ah > 0 ? 0 : m->total_num_tokens + ta->num_tokens;

  if (Ah > 0) {
    sti->refbits = Allocate<uint8_t>(cinfo, sti->num_blocks, JPOOL_IMAGE);
  } else if (cinfo->progressive_mode) {
    if (ta->num_tokens + sti->num_blocks > m->num_tokens) {
      if (ta->tokens) {
        m->total_num_tokens += ta->num_tokens;
        ++m->cur_token_array;
        ta = &m->token_arrays[m->cur_token_array];
      }
      m->num_tokens = sti->num_blocks;
      ta->tokens = Allocate<Token>(cinfo, m->num_tokens, JPOOL_IMAGE);
      m->next_token = ta->tokens;
    }
  }

  JBLOCKARRAY blocks[MAX_COMPS_IN_SCAN];
  size_t block_idx = 0;
  for (size_t mcu_y = 0; mcu_y < sti->MCU_rows_in_scan; ++mcu_y) {
    for (int i = 0; i < scan_info->comps_in_scan; ++i) {
      int comp_idx = scan_info->component_index[i];
      jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
      int n_blocks_y = is_interleaved ? comp->v_samp_factor : 1;
      int by0 = mcu_y * n_blocks_y;
      int block_rows_left = comp->height_in_blocks - by0;
      int max_block_rows = std::min(n_blocks_y, block_rows_left);
      blocks[i] = (*cinfo->mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[comp_idx],
          by0, max_block_rows, FALSE);
    }
    if (!cinfo->progressive_mode) {
      int max_tokens_per_mcu_row = MaxNumTokensPerMCURow(cinfo);
      if (ta->num_tokens + max_tokens_per_mcu_row > m->num_tokens) {
        if (ta->tokens) {
          m->total_num_tokens += ta->num_tokens;
          ++m->cur_token_array;
          ta = &m->token_arrays[m->cur_token_array];
        }
        m->num_tokens =
            EstimateNumTokens(cinfo, mcu_y, sti->MCU_rows_in_scan,
                              m->total_num_tokens, max_tokens_per_mcu_row);
        ta->tokens = Allocate<Token>(cinfo, m->num_tokens, JPOOL_IMAGE);
        m->next_token = ta->tokens;
      }
    }
    for (size_t mcu_x = 0; mcu_x < sti->MCUs_per_row; ++mcu_x) {
      // Possibly emit a restart marker.
      if (restart_interval > 0 && restarts_to_go == 0) {
        restarts_to_go = restart_interval;
        memset(last_dc_coeff, 0, sizeof(last_dc_coeff));
        ta->num_tokens = m->next_token - ta->tokens;
        sti->restarts[restart_idx++] =
            Ah > 0 ? block_idx : m->total_num_tokens + ta->num_tokens;
      }
      // Encode one MCU
      for (int i = 0; i < scan_info->comps_in_scan; ++i) {
        int comp_idx = scan_info->component_index[i];
        jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
        int n_blocks_y = is_interleaved ? comp->v_samp_factor : 1;
        int n_blocks_x = is_interleaved ? comp->h_samp_factor : 1;
        for (int iy = 0; iy < n_blocks_y; ++iy) {
          for (int ix = 0; ix < n_blocks_x; ++ix) {
            size_t block_y = mcu_y * n_blocks_y + iy;
            size_t block_x = mcu_x * n_blocks_x + ix;
            const coeff_t* block;
            if (block_x >= comp->width_in_blocks ||
                block_y >= comp->height_in_blocks) {
              block = kSinkBlock;
            } else {
              block = &blocks[i][iy][block_x][0];
            }
            if (!is_progressive) {
              HWY_DYNAMIC_DISPATCH(ComputeTokensSequential)
              (block, last_dc_coeff[i], comp_idx, ac_ctx_offset + i,
               &m->next_token);
              last_dc_coeff[i] = block[0];
            } else {
              if (Ah == 0) {
                TokenizeProgressiveDC(block, comp_idx, Al, last_dc_coeff + i,
                                      &m->next_token);
              } else {
                sti->refbits[block_idx] = (block[0] >> Al) & 1;
              }
            }
            ++block_idx;
          }
        }
      }
      --restarts_to_go;
    }
    ta->num_tokens = m->next_token - ta->tokens;
  }
  JPEGLI_DASSERT(block_idx == sti->num_blocks);
  sti->num_tokens =
      Ah > 0 ? sti->num_blocks
             : m->total_num_tokens + ta->num_tokens - sti->token_offset;
  sti->restarts[restart_idx++] =
      Ah > 0 ? sti->num_blocks : m->total_num_tokens + ta->num_tokens;
  if (Ah == 0 && cinfo->progressive_mode) {
    JPEGLI_DASSERT(sti->num_blocks == sti->num_tokens);
  }
}

}  // namespace

void TokenizeJpeg(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  std::vector<int> processed(cinfo->num_scans);
  size_t max_refinement_tokens = 0;
  size_t num_refinement_bits = 0;
  int num_refinement_scans[kMaxComponents][DCTSIZE2] = {};
  int max_num_refinement_scans = 0;
  for (int i = 0; i < cinfo->num_scans; ++i) {
    const jpeg_scan_info* si = &cinfo->scan_info[i];
    ScanTokenInfo* sti = &m->scan_token_info[i];
    if (si->Ss > 0 && si->Ah == 0 && si->Al > 0) {
      int offset = m->ac_ctx_offset[i];
      int comp_idx = si->component_index[0];
      TokenizeScan(cinfo, i, offset, sti);
      processed[i] = 1;
      max_refinement_tokens += sti->num_future_nonzeros;
      for (int k = si->Ss; k <= si->Se; ++k) {
        num_refinement_scans[comp_idx][k] = si->Al;
      }
      max_num_refinement_scans = std::max(max_num_refinement_scans, si->Al);
      num_refinement_bits += sti->num_nonzeros;
    }
    if (si->Ss > 0 && si->Ah > 0) {
      int comp_idx = si->component_index[0];
      const jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
      size_t num_blocks =
          static_cast<size_t>(comp->width_in_blocks) * comp->height_in_blocks;
      max_refinement_tokens += (1 + (si->Se - si->Ss) / 16) * num_blocks;
    }
  }
  if (max_refinement_tokens > 0) {
    m->next_refinement_token =
        Allocate<RefToken>(cinfo, max_refinement_tokens, JPOOL_IMAGE);
  }
  for (int j = 0; j < max_num_refinement_scans; ++j) {
    uint8_t* refinement_bits =
        Allocate<uint8_t>(cinfo, num_refinement_bits, JPOOL_IMAGE);
    m->next_refinement_bit = refinement_bits;
    size_t new_refinement_bits = 0;
    for (int i = 0; i < cinfo->num_scans; ++i) {
      const jpeg_scan_info* si = &cinfo->scan_info[i];
      int comp_idx = si->component_index[0];
      ScanTokenInfo* sti = &m->scan_token_info[i];
      if (si->Ss > 0 && si->Ah > 0 &&
          si->Ah == num_refinement_scans[comp_idx][si->Ss] - j) {
        int offset = m->ac_ctx_offset[i];
        TokenizeScan(cinfo, i, offset, sti);
        processed[i] = 1;
        new_refinement_bits += sti->num_nonzeros;
      }
    }
    JPEGLI_DASSERT(m->next_refinement_bit <=
                   refinement_bits + num_refinement_bits);
    num_refinement_bits += new_refinement_bits;
  }
  for (int i = 0; i < cinfo->num_scans; ++i) {
    if (processed[i]) {
      continue;
    }
    int offset = m->ac_ctx_offset[i];
    TokenizeScan(cinfo, i, offset, &m->scan_token_info[i]);
    processed[i] = 1;
  }
}

namespace {

struct Histogram {
  int count[kJpegHuffmanAlphabetSize];
  Histogram() { memset(count, 0, sizeof(count)); }
};

void BuildHistograms(j_compress_ptr cinfo, Histogram* histograms) {
  jpeg_comp_master* m = cinfo->master;
  size_t num_token_arrays = m->cur_token_array + 1;
  for (size_t i = 0; i < num_token_arrays; ++i) {
    Token* tokens = m->token_arrays[i].tokens;
    size_t num_tokens = m->token_arrays[i].num_tokens;
    for (size_t j = 0; j < num_tokens; ++j) {
      Token t = tokens[j];
      ++histograms[t.context].count[t.symbol];
    }
  }
  for (int i = 0; i < cinfo->num_scans; ++i) {
    const jpeg_scan_info& si = cinfo->scan_info[i];
    const ScanTokenInfo& sti = m->scan_token_info[i];
    if (si.Ss > 0 && si.Ah > 0) {
      int context = m->ac_ctx_offset[i];
      int* ac_histo = &histograms[context].count[0];
      for (size_t j = 0; j < sti.num_tokens; ++j) {
        ++ac_histo[sti.tokens[j].symbol & 253];
      }
    }
  }
}

struct JpegClusteredHistograms {
  std::vector<Histogram> histograms;
  std::vector<uint32_t> histogram_indexes;
  std::vector<uint32_t> slot_ids;
};

float HistogramCost(const Histogram& histo) {
  std::vector<uint32_t> counts(kJpegHuffmanAlphabetSize + 1);
  std::vector<uint8_t> depths(kJpegHuffmanAlphabetSize + 1);
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    counts[i] = histo.count[i];
  }
  counts[kJpegHuffmanAlphabetSize] = 1;
  CreateHuffmanTree(counts.data(), counts.size(), kJpegHuffmanMaxBitLength,
                    depths.data());
  size_t header_bits = (1 + kJpegHuffmanMaxBitLength) * 8;
  size_t data_bits = 0;
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    if (depths[i] > 0) {
      header_bits += 8;
      data_bits += static_cast<size_t>(counts[i]) * depths[i];
    }
  }
  return header_bits + data_bits;
}

void AddHistograms(const Histogram& a, const Histogram& b, Histogram* c) {
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    c->count[i] = a.count[i] + b.count[i];
  }
}

bool IsEmptyHistogram(const Histogram& histo) {
  for (int count : histo.count) {
    if (count) return false;
  }
  return true;
}

void ClusterJpegHistograms(j_compress_ptr cinfo, const Histogram* histograms,
                           size_t num, JpegClusteredHistograms* clusters) {
  clusters->histogram_indexes.resize(num);
  std::vector<uint32_t> slot_histograms;
  std::vector<float> slot_costs;
  // Since not all jpeg decoders support the extended sequential mode, i.e. the
  // 0xff 0xc1 SOF marker, we will limit the number of clusters to 2 in
  // sequential mode, unless the quantization tables already require the
  // extended sequential mode.
  const bool force_baseline =
      !cinfo->progressive_mode && cinfo->master->force_baseline;

  for (size_t i = 0; i < num; ++i) {
    const Histogram& cur = histograms[i];
    if (IsEmptyHistogram(cur)) {
      continue;
    }
    size_t best_slot = slot_histograms.size();
    float best_cost = force_baseline && best_slot > 1
                          ? std::numeric_limits<float>::max()
                          : HistogramCost(cur);
    for (size_t j = 0; j < slot_histograms.size(); ++j) {
      size_t prev_idx = slot_histograms[j];
      const Histogram& prev = clusters->histograms[prev_idx];
      Histogram combined;
      AddHistograms(prev, cur, &combined);
      float combined_cost = HistogramCost(combined);
      float cost = combined_cost - slot_costs[j];
      if (cost < best_cost) {
        best_cost = cost;
        best_slot = j;
      }
    }
    if (best_slot == slot_histograms.size()) {
      // Create new histogram.
      size_t histogram_index = clusters->histograms.size();
      clusters->histograms.push_back(cur);
      clusters->histogram_indexes[i] = histogram_index;
      if (best_slot < 4) {
        // We have a free slot, so we put the new histogram there.
        slot_histograms.push_back(histogram_index);
        slot_costs.push_back(best_cost);
      } else {
        // TODO(szabadka) Find the best histogram to replace.
        best_slot = (clusters->slot_ids.back() + 1) % 4;
      }
      slot_histograms[best_slot] = histogram_index;
      slot_costs[best_slot] = best_cost;
      clusters->slot_ids.push_back(best_slot);
    } else {
      // Merge this histogram with a previous one.
      size_t histogram_index = slot_histograms[best_slot];
      const Histogram& prev = clusters->histograms[histogram_index];
      AddHistograms(prev, cur, &clusters->histograms[histogram_index]);
      clusters->histogram_indexes[i] = histogram_index;
      JPEGLI_CHECK(clusters->slot_ids[histogram_index] == best_slot);
      slot_costs[best_slot] += best_cost;
    }
  }
}

void CopyHuffmanTable(j_compress_ptr cinfo, int index, bool is_dc,
                      int* inv_slot_map, uint8_t* slot_id_map,
                      JHUFF_TBL* huffman_tables, size_t* num_huffman_tables) {
  const char* type = is_dc ? "DC" : "AC";
  if (index < 0 || index >= NUM_HUFF_TBLS) {
    JPEGLI_ERROR("Invalid %s Huffman table index %d", type, index);
  }
  // Check if we have already copied this Huffman table.
  int slot_idx = index + (is_dc ? 0 : NUM_HUFF_TBLS);
  if (inv_slot_map[slot_idx] != -1) {
    return;
  }
  inv_slot_map[slot_idx] = *num_huffman_tables;
  // Look up and validate Huffman table.
  JHUFF_TBL* table =
      is_dc ? cinfo->dc_huff_tbl_ptrs[index] : cinfo->ac_huff_tbl_ptrs[index];
  if (table == nullptr) {
    JPEGLI_ERROR("Missing %s Huffman table %d", type, index);
  }
  ValidateHuffmanTable(reinterpret_cast<j_common_ptr>(cinfo), table, is_dc);
  // Copy Huffman table to the end of the list and save slot id.
  slot_id_map[*num_huffman_tables] = index + (is_dc ? 0 : 0x10);
  memcpy(&huffman_tables[*num_huffman_tables], table, sizeof(JHUFF_TBL));
  ++(*num_huffman_tables);
}

void BuildJpegHuffmanTable(const Histogram& histo, JHUFF_TBL* table) {
  std::vector<uint32_t> counts(kJpegHuffmanAlphabetSize + 1);
  std::vector<uint8_t> depths(kJpegHuffmanAlphabetSize + 1);
  for (size_t j = 0; j < kJpegHuffmanAlphabetSize; ++j) {
    counts[j] = histo.count[j];
  }
  counts[kJpegHuffmanAlphabetSize] = 1;
  CreateHuffmanTree(counts.data(), counts.size(), kJpegHuffmanMaxBitLength,
                    depths.data());
  memset(table, 0, sizeof(JHUFF_TBL));
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    if (depths[i] > 0) {
      ++table->bits[depths[i]];
    }
  }
  int offset[kJpegHuffmanMaxBitLength + 1] = {0};
  for (size_t i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
    offset[i] = offset[i - 1] + table->bits[i - 1];
  }
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    if (depths[i] > 0) {
      table->huffval[offset[depths[i]]++] = i;
    }
  }
}

}  // namespace

void CopyHuffmanTables(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  size_t max_huff_tables = 2 * cinfo->num_components;
  // Copy Huffman tables and save slot ids.
  m->huffman_tables = Allocate<JHUFF_TBL>(cinfo, max_huff_tables, JPOOL_IMAGE);
  m->slot_id_map = Allocate<uint8_t>(cinfo, max_huff_tables, JPOOL_IMAGE);
  m->num_huffman_tables = 0;
  int inv_slot_map[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    CopyHuffmanTable(cinfo, comp->dc_tbl_no, /*is_dc=*/true, &inv_slot_map[0],
                     m->slot_id_map, m->huffman_tables, &m->num_huffman_tables);
    CopyHuffmanTable(cinfo, comp->ac_tbl_no, /*is_dc=*/false, &inv_slot_map[0],
                     m->slot_id_map, m->huffman_tables, &m->num_huffman_tables);
  }
  // Compute context map.
  m->context_map = Allocate<uint8_t>(cinfo, 8, JPOOL_IMAGE);
  memset(m->context_map, 0, 8);
  for (int c = 0; c < cinfo->num_components; ++c) {
    m->context_map[c] = inv_slot_map[cinfo->comp_info[c].dc_tbl_no];
  }
  int ac_ctx = 4;
  for (int i = 0; i < cinfo->num_scans; ++i) {
    const jpeg_scan_info* si = &cinfo->scan_info[i];
    if (si->Se > 0) {
      for (int j = 0; j < si->comps_in_scan; ++j) {
        int c = si->component_index[j];
        jpeg_component_info* comp = &cinfo->comp_info[c];
        m->context_map[ac_ctx++] = inv_slot_map[comp->ac_tbl_no + 4];
      }
    }
  }
}

void OptimizeHuffmanCodes(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  // Build DC and AC histograms.
  std::vector<Histogram> histograms(m->num_contexts);
  BuildHistograms(cinfo, histograms.data());

  // Cluster DC histograms.
  JpegClusteredHistograms dc_clusters;
  ClusterJpegHistograms(cinfo, histograms.data(), cinfo->num_components,
                        &dc_clusters);

  // Cluster AC histograms.
  JpegClusteredHistograms ac_clusters;
  ClusterJpegHistograms(cinfo, histograms.data() + 4, m->num_contexts - 4,
                        &ac_clusters);

  // Create Huffman tables and slot ids clusters.
  size_t num_dc_huff = dc_clusters.histograms.size();
  m->num_huffman_tables = num_dc_huff + ac_clusters.histograms.size();
  m->huffman_tables =
      Allocate<JHUFF_TBL>(cinfo, m->num_huffman_tables, JPOOL_IMAGE);
  m->slot_id_map = Allocate<uint8_t>(cinfo, m->num_huffman_tables, JPOOL_IMAGE);
  for (size_t i = 0; i < m->num_huffman_tables; ++i) {
    JHUFF_TBL huff_table = {};
    if (i < dc_clusters.histograms.size()) {
      m->slot_id_map[i] = i;
      BuildJpegHuffmanTable(dc_clusters.histograms[i], &huff_table);
    } else {
      m->slot_id_map[i] = 16 + ac_clusters.slot_ids[i - num_dc_huff];
      BuildJpegHuffmanTable(ac_clusters.histograms[i - num_dc_huff],
                            &huff_table);
    }
    memcpy(&m->huffman_tables[i], &huff_table, sizeof(huff_table));
  }

  // Create context map from clustered histogram indexes.
  m->context_map = Allocate<uint8_t>(cinfo, m->num_contexts, JPOOL_IMAGE);
  memset(m->context_map, 0, m->num_contexts);
  for (size_t i = 0; i < m->num_contexts; ++i) {
    if (i < static_cast<size_t>(cinfo->num_components)) {
      m->context_map[i] = dc_clusters.histogram_indexes[i];
    } else if (i >= 4) {
      m->context_map[i] = num_dc_huff + ac_clusters.histogram_indexes[i - 4];
    }
  }
}

namespace {

constexpr uint8_t kNumExtraBits[256] = {
    0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    1,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    2,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    3,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    4,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    5,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    6,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    7,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    8,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    9,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
    0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  //
};

void BuildHuffmanCodeTable(const JHUFF_TBL& table, HuffmanCodeTable* code) {
  int huff_code[kJpegHuffmanAlphabetSize];
  // +1 for a sentinel element.
  uint32_t huff_size[kJpegHuffmanAlphabetSize + 1];
  int p = 0;
  for (size_t l = 1; l <= kJpegHuffmanMaxBitLength; ++l) {
    int i = table.bits[l];
    while (i--) huff_size[p++] = l;
  }

  // Reuse sentinel element.
  int last_p = p;
  huff_size[last_p] = 0;

  int next_code = 0;
  uint32_t si = huff_size[0];
  p = 0;
  while (huff_size[p]) {
    while ((huff_size[p]) == si) {
      huff_code[p++] = next_code;
      next_code++;
    }
    next_code <<= 1;
    si++;
  }
  for (p = 0; p < last_p; p++) {
    int i = table.huffval[p];
    int nbits = kNumExtraBits[i];
    code->depth[i] = huff_size[p] + nbits;
    code->code[i] = huff_code[p] << nbits;
  }
}

}  // namespace

void InitEntropyCoder(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  m->coding_tables =
      Allocate<HuffmanCodeTable>(cinfo, m->num_huffman_tables, JPOOL_IMAGE);
  for (size_t i = 0; i < m->num_huffman_tables; ++i) {
    BuildHuffmanCodeTable(m->huffman_tables[i], &m->coding_tables[i]);
  }
}

namespace {

struct CandidateScan {
  uint8_t ss;
  uint8_t se;
  uint8_t ah;
  uint8_t al;
  bool operator==(const CandidateScan& o) const {
    return ss == o.ss && se == o.se && ah == o.ah && al == o.al;
  }
};

struct PreparedACBlock {
  uint64_t nonzero[4];  // AC bitmask (bit 0 clear) for al = 0..3
  uint8_t sizes[32];    // 4-bit packed bit-lengths (2 positions per byte) -> 64 bytes total
  inline uint8_t size_at(int k) const {
    return (sizes[k >> 1] >> ((k & 1) << 2)) & 0xF;
  }
  inline void set_size(int k, uint8_t sz) {
    sizes[k >> 1] |= static_cast<uint8_t>((sz & 0xF) << ((k & 1) << 2));
  }
};

inline int TrailingZeros64(uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
  unsigned long idx;
  _BitScanForward64(&idx, v);
  return static_cast<int>(idx);
#else
  return __builtin_ctzll(v);
#endif
}

inline int LeadingZeros64(uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
  unsigned long idx;
  _BitScanReverse64(&idx, v);
  return 63 - static_cast<int>(idx);
#else
  return __builtin_clzll(v);
#endif
}

inline int PopCount64(uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
  return static_cast<int>(__popcnt64(v));
#else
  return __builtin_popcountll(v);
#endif
}

uint64_t CostACScan(const std::vector<PreparedACBlock>& blocks,
                    const CandidateScan& scan, size_t restart_interval,
                    size_t stride) {
  uint32_t freq[257] = {};
  freq[256] = 1;  // Sentinel for all-1s code
  uint64_t raw_bits = 0;
  uint64_t restarts = 0;
  uint32_t eobrun = 0;
  size_t pending_ref = 0;

  const auto emit_eobrun = [&]() {
    if (eobrun > 0) {
      int nbits = jpegli::FloorLog2Nonzero<uint32_t>(eobrun);
      freq[nbits << 4] += 1;
      raw_bits += static_cast<uint64_t>(nbits) + pending_ref;
      eobrun = 0;
      pending_ref = 0;
    }
  };

  const uint64_t band = (~uint64_t{0} << scan.ss) & (~uint64_t{0} >> (63 - scan.se));
  const bool is_refine = (scan.ah > 0);
  const size_t nblocks = blocks.size();

  for (size_t b = 0; b < nblocks; ++b) {
    if (restart_interval > 0 && b > 0 && (b % restart_interval) == 0) {
      emit_eobrun();
      ++restarts;
    }
    const PreparedACBlock& blk = blocks[b];
    uint64_t nz = blk.nonzero[scan.al] & band;
    int next = scan.ss;
    if (!is_refine) {
      uint32_t zrl = 0;
      uint64_t rbits = 0;
      while (nz != 0) {
        int k = TrailingZeros64(nz);
        nz &= nz - 1;
        int run = k - next;
        emit_eobrun();
        zrl += static_cast<uint32_t>(run >> 4);
        int size = blk.size_at(k) - scan.al;
        freq[((run & 15) << 4) + size] += 1;
        rbits += static_cast<uint64_t>(size);
        next = k + 1;
      }
      freq[0xF0] += zrl;
      raw_bits += rbits;
      if (next <= scan.se) {
        ++eobrun;
        if (eobrun == 0x7FFF) emit_eobrun();
      }
    } else {
      uint64_t newer = nz & ~blk.nonzero[scan.al + 1];
      uint64_t older = nz & ~newer;
      uint64_t waiting = older;
      if (newer != 0) {
        emit_eobrun();
        int eob = 63 - LeadingZeros64(newer);
        uint64_t sent = older & (~uint64_t{0} >> (63 - eob));
        waiting &= ~sent;
        raw_bits += static_cast<uint64_t>(PopCount64(newer) + PopCount64(sent));
        uint32_t zrl = 0;
        uint64_t rest = newer;
        while (rest != 0) {
          int k = TrailingZeros64(rest);
          rest &= rest - 1;
          uint64_t between = (~uint64_t{0} << next) & ~(~uint64_t{0} << k);
          int zeros = k - next - PopCount64(older & between);
          zrl += static_cast<uint32_t>(zeros >> 4);
          freq[((zeros & 15) << 4) + 1] += 1;
          next = k + 1;
        }
        freq[0xF0] += zrl;
      }
      if ((newer & (uint64_t{1} << scan.se)) == 0) {
        ++eobrun;
        pending_ref += static_cast<size_t>(PopCount64(waiting));
        if (eobrun == 0x7FFF || pending_ref > 65000) emit_eobrun();
      }
    }
  }
  emit_eobrun();

  uint8_t depth[257] = {};
  uint32_t sym_count = 0;
  for (int s = 0; s < 256; ++s) {
    if (freq[s] > 0) ++sym_count;
  }
  if (sym_count == 0) {
    freq[0] = 1;
    sym_count = 1;
  }
  CreateHuffmanTree(freq, 257, kJpegHuffmanMaxBitLength, depth);

  uint64_t huff_bits = 0;
  for (int s = 0; s < 256; ++s) {
    if (freq[s] > 0) {
      huff_bits += static_cast<uint64_t>(freq[s]) * depth[s];
    }
  }
  // Scale sampled payload bits back by stride; add exact unscaled SOS + DHT header overhead.
  uint64_t payload_bits = (huff_bits + raw_bits + restarts * 23) * stride + 7;
  uint64_t header_bits = (10 + 2 + 2 + 1 + 16 + sym_count) * 8;
  return payload_bits + header_bits;
}

std::vector<std::vector<CandidateScan>> BuildCandidateACScripts(bool is_luma) {
  std::vector<std::vector<CandidateScan>> scripts;
  auto add_partition = [&](const std::vector<std::pair<uint8_t, uint8_t>>& bands,
                           uint8_t depth) {
    std::vector<CandidateScan> s;
    for (const auto& b : bands) {
      s.push_back({b.first, b.second, 0, depth});
    }
    for (int al = depth; al > 0; --al) {
      s.push_back({1, 63, static_cast<uint8_t>(al), static_cast<uint8_t>(al - 1)});
    }
    scripts.push_back(std::move(s));
  };

  if (is_luma) {
    const std::vector<std::vector<std::pair<uint8_t, uint8_t>>> parts = {
        {{1, 63}},
        {{1, 5}, {6, 63}},
        {{1, 2}, {3, 63}},
        {{1, 2}, {3, 9}, {10, 63}},
        {{1, 8}, {9, 63}},
        {{1, 5}, {6, 17}, {18, 63}},
        {{1, 3}, {4, 12}, {13, 63}},
    };
    for (uint8_t depth : {0, 1, 2}) {
      for (const auto& p : parts) add_partition(p, depth);
    }
    // Include jpegli's native -p 2 hybrid scripts so -p 3 is always <= -p 2.
    scripts.push_back({{1, 2, 0, 0}, {3, 63, 0, 2}, {3, 63, 2, 1}, {3, 63, 1, 0}});
    scripts.push_back({{1, 2, 0, 0}, {3, 63, 0, 1}, {3, 63, 1, 0}});
    scripts.push_back({{1, 5, 0, 0}, {6, 63, 0, 1}, {6, 63, 1, 0}});
    scripts.push_back({{1, 5, 0, 0}, {6, 63, 0, 2}, {6, 63, 2, 1}, {6, 63, 1, 0}});
  } else {
    const std::vector<std::vector<std::pair<uint8_t, uint8_t>>> parts = {
        {{1, 63}},
        {{1, 5}, {6, 63}},
        {{1, 2}, {3, 63}},
    };
    for (uint8_t depth : {0, 1}) {
      for (const auto& p : parts) add_partition(p, depth);
    }
    scripts.push_back({{1, 2, 0, 0}, {3, 63, 0, 2}, {3, 63, 2, 1}, {3, 63, 1, 0}});
    scripts.push_back({{1, 2, 0, 0}, {3, 63, 0, 1}, {3, 63, 1, 0}});
    scripts.push_back({{1, 5, 0, 0}, {6, 63, 0, 1}, {6, 63, 1, 0}});
  }
  return scripts;
}

}  // namespace

void OptimizeProgressiveScanScript(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  static constexpr size_t kRaceStride = 2;

  std::vector<jpeg_scan_info> new_scans;
  // DC scan(s): interleave for 4:4:4 (max_h == 1 && max_v == 1), separate otherwise.
  const bool interleave_dc =
      (cinfo->max_h_samp_factor == 1 && cinfo->max_v_samp_factor == 1);
  const int dc_comps = interleave_dc ? MAX_COMPS_IN_SCAN : 1;
  for (int c = 0; c < cinfo->num_components; c += dc_comps) {
    jpeg_scan_info si = {};
    si.Ss = 0;
    si.Se = 0;
    si.Ah = 0;
    si.Al = 0;
    si.comps_in_scan = std::min(dc_comps, cinfo->num_components - c);
    for (int j = 0; j < si.comps_in_scan; ++j) {
      si.component_index[j] = c + j;
    }
    new_scans.push_back(si);
  }

  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info* comp = &cinfo->comp_info[c];
    const size_t total_blocks =
        static_cast<size_t>(comp->height_in_blocks) * comp->width_in_blocks;
    const size_t stride = total_blocks > 256 ? kRaceStride : 1;
    std::vector<PreparedACBlock> sample;
    sample.reserve(DivCeil(total_blocks, stride));

    size_t idx = 0;
    for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
      JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[c], by, 1,
          FALSE);
      for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx, ++idx) {
        if ((idx % stride) != 0) continue;
        const coeff_t* blk = &blocks[0][bx][0];
        PreparedACBlock pb = {};
        HWY_DYNAMIC_DISPATCH(ComputeACMultiMasksSIMD)(blk, pb.nonzero);
        uint64_t nz = pb.nonzero[0];
        while (nz != 0) {
          int k = TrailingZeros64Fast(nz);
          nz &= nz - 1;
          int v = blk[k];
          uint32_t a = static_cast<uint32_t>(v < 0 ? -v : v);
          pb.set_size(k, static_cast<uint8_t>(
              jpegli::FloorLog2Nonzero<uint32_t>(a) + 1));
        }
        sample.push_back(pb);
      }
    }

    size_t restart_interval = 0;
    if (cinfo->restart_in_rows <= 0) {
      restart_interval = cinfo->restart_interval;
    } else {
      restart_interval = std::min<size_t>(
          static_cast<size_t>(comp->width_in_blocks) * cinfo->restart_in_rows,
          65535u);
    }
    size_t sampled_restart =
        restart_interval > 0 ? DivCeil(restart_interval, stride) : 0;

    const auto candidates = BuildCandidateACScripts(c == 0);
    std::vector<std::pair<CandidateScan, uint64_t>> cache;
    cache.reserve(32);
    const auto get_cost = [&](const CandidateScan& sc) -> uint64_t {
      for (const auto& entry : cache) {
        if (entry.first == sc) return entry.second;
      }
      uint64_t cost = CostACScan(sample, sc, sampled_restart, stride);
      cache.emplace_back(sc, cost);
      return cost;
    };

    uint64_t best_cost = std::numeric_limits<uint64_t>::max();
    size_t best_idx = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      uint64_t total = 0;
      for (const auto& sc : candidates[i]) {
        total += get_cost(sc);
      }
      if (total < best_cost) {
        best_cost = total;
        best_idx = i;
      }
    }

    for (const auto& sc : candidates[best_idx]) {
      jpeg_scan_info si = {};
      si.comps_in_scan = 1;
      si.component_index[0] = c;
      si.Ss = sc.ss;
      si.Se = sc.se;
      si.Ah = sc.ah;
      si.Al = sc.al;
      new_scans.push_back(si);
    }
  }

  cinfo->script_space_size = static_cast<int>(new_scans.size());
  cinfo->script_space =
      Allocate<jpeg_scan_info>(cinfo, cinfo->script_space_size, JPOOL_IMAGE);
  memcpy(cinfo->script_space, new_scans.data(),
         new_scans.size() * sizeof(jpeg_scan_info));
  cinfo->scan_info = cinfo->script_space;
  cinfo->num_scans = cinfo->script_space_size;

  // Reallocate token_arrays for the chosen scan count so small images cannot
  // overflow the initial allocation made for the default scan script.
  const int ysize_blocks = DivCeil(cinfo->image_height, DCTSIZE);
  const int num_token_arrays = cinfo->num_scans * ysize_blocks;
  m->token_arrays =
      Allocate<TokenArray>(cinfo, num_token_arrays, JPOOL_IMAGE);
  memset(m->token_arrays, 0, num_token_arrays * sizeof(TokenArray));
  m->cur_token_array = 0;
  m->num_tokens = 0;
  m->total_num_tokens = 0;

  // Rebuild scan_token_info, ac_ctx_offset, and num_contexts for the chosen script.
  m->scan_token_info =
      Allocate<ScanTokenInfo>(cinfo, cinfo->num_scans, JPOOL_IMAGE);
  memset(m->scan_token_info, 0, cinfo->num_scans * sizeof(ScanTokenInfo));
  m->ac_ctx_offset = Allocate<uint8_t>(cinfo, cinfo->num_scans, JPOOL_IMAGE);
  size_t num_ac_contexts = 0;
  for (int i = 0; i < cinfo->num_scans; ++i) {
    const jpeg_scan_info* scan_info = &cinfo->scan_info[i];
    m->ac_ctx_offset[i] = 4 + num_ac_contexts;
    if (scan_info->Se > 0) {
      num_ac_contexts += scan_info->comps_in_scan;
    }
    ScanTokenInfo* sti = &m->scan_token_info[i];
    if (scan_info->comps_in_scan == 1) {
      int comp_idx = scan_info->component_index[0];
      jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
      sti->MCUs_per_row = comp->width_in_blocks;
      sti->MCU_rows_in_scan = comp->height_in_blocks;
      sti->blocks_in_MCU = 1;
    } else {
      sti->MCUs_per_row =
          DivCeil(cinfo->image_width, DCTSIZE * cinfo->max_h_samp_factor);
      sti->MCU_rows_in_scan =
          DivCeil(cinfo->image_height, DCTSIZE * cinfo->max_v_samp_factor);
      sti->blocks_in_MCU = 0;
      for (int j = 0; j < scan_info->comps_in_scan; ++j) {
        int comp_idx = scan_info->component_index[j];
        jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
        sti->blocks_in_MCU +=
            static_cast<size_t>(comp->h_samp_factor) * comp->v_samp_factor;
      }
    }
    size_t num_MCUs = sti->MCU_rows_in_scan * sti->MCUs_per_row;
    sti->num_blocks = num_MCUs * sti->blocks_in_MCU;
    if (cinfo->restart_in_rows <= 0) {
      sti->restart_interval = cinfo->restart_interval;
    } else {
      sti->restart_interval =
          std::min<size_t>(sti->MCUs_per_row * cinfo->restart_in_rows, 65535u);
    }
    sti->num_restarts = sti->restart_interval > 0
                            ? DivCeil(num_MCUs, sti->restart_interval)
                            : 1;
    sti->restarts = Allocate<size_t>(cinfo, sti->num_restarts, JPOOL_IMAGE);
  }
  m->num_contexts = 4 + num_ac_contexts;
}

}  // namespace jpegli
#endif  // HWY_ONCE
