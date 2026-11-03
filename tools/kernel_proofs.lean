/-
  Machine-checked facts the new kernels rely on (Lean 4 core, no Mathlib; `lean kernel_proofs.lean`).

  1. muse_prefill_layer `_dequant`: the E2M1 magnitude is decoded with integer ops,
       two = mag < 4 ? mag : (2 + (mag & 1)) << ((mag >> 1) - 1),   value = two / 2,
     in place of a table lookup: one select and a shift, no gather.
  1b. muse_moe_w4a16 `_DECODE` (PTX): prmt + shift + mask puts every nibble's sign and magnitude
     at the BF16 bit positions claimed, for all 2^32 inputs, and those patterns are the E2M1
     grid times 2^-126.
  2. muse_attention_decode `_attend` / `_decode_fused`: the key range each request needs is covered by its NS
     static chunks, so no visible key is skipped, for every query row of the block.
-/

import Std.Tactic.BVDecide

namespace KernelProofs

/-! ## 1. E2M1 decode -/

/-- The E2M1 magnitudes times 2 (0, 0.5, 1, 1.5, 2, 3, 4, 6), indexed by the 3-bit code. -/
def grid2 : List Nat := [0, 1, 2, 3, 4, 6, 8, 12]

/-- What `_dequant` computes from the 3-bit magnitude code. -/
def two (mag : Nat) : Nat :=
  if mag < 4 then mag else (2 + mag % 2) <<< (mag / 2 - 1)

theorem two_eq_grid : ∀ mag : Fin 8, two mag.val = grid2[mag.val]! := by decide

/-- The kernel reads the code as `(b >> 4 * half) & 15` and the magnitude as `c & 7`, the sign
    as `c & 8`: for every byte, both nibbles decode to the grid value with the right sign. -/
theorem nibble_split (b : Nat) (hb : b < 256) :
    (b % 16) % 8 + 8 * ((b % 16) / 8) = b % 16 ∧
    ((b / 16) % 16) % 8 + 8 * (((b / 16) % 16) / 8) = b / 16 ∧
    b % 16 + 16 * ((b / 16) % 16) = b := by
  omega

/-! ## 1b. muse_moe_w4a16 `_DECODE` (PTX): four packed bytes to BF16 pairs

`prmt.b32 t01, w, 0, 0x4140` is `[b0, 0, b1, 0]`; then for the low nibbles
`((t01 << 12) & 0x80008000) | ((t01 << 6) & 0x01C001C0)` and for the high nibbles the same with
shifts 8 and 2. Each 16-bit half must be `sign << 15 | mag << 6` of its nibble. -/

abbrev prmt4140 (w : BitVec 32) : BitVec 32 := (w &&& 0xFF#32) ||| (((w >>> 8) &&& 0xFF#32) <<< 16)

abbrev place (nib : BitVec 32) : BitVec 32 := ((nib &&& 8#32) <<< 12) ||| ((nib &&& 7#32) <<< 6)

theorem decode_lo (w : BitVec 32) :
    let t := prmt4140 w
    (((t <<< 12) &&& 0x80008000#32) ||| ((t <<< 6) &&& 0x01C001C0#32)) =
      place (w &&& 0xF#32) ||| (place ((w >>> 8) &&& 0xF#32) <<< 16) := by
  simp only [prmt4140, place]
  bv_decide

theorem decode_hi (w : BitVec 32) :
    let t := prmt4140 w
    (((t <<< 8) &&& 0x80008000#32) ||| ((t <<< 2) &&& 0x01C001C0#32)) =
      place ((w >>> 4) &&& 0xF#32) ||| (place ((w >>> 12) &&& 0xF#32) <<< 16) := by
  simp only [prmt4140, place]
  bv_decide

/-- A non-negative BF16 bit pattern's value times 2^126, doubled, when that is an integer:
    exponent field E, mantissa M; E = 0 is subnormal (M / 128 * 2^-126). -/
def bf16Twice (bits : Nat) : Nat :=
  let e := (bits / 128) % 256
  let m := bits % 128
  if e = 0 then m / 64 else (128 + m) * 2 ^ (e - 1) / 64

/-- `mag << 6` is the E2M1 magnitude times 2^-126 in BF16 (then `mul.bf16x2` by 2^116 * scale). -/
theorem place_is_e2m1 : ∀ mag : Fin 8, bf16Twice (mag.val * 64) = grid2[mag.val]! := by decide

/-! ## 2. Split-KV coverage

Request with `kvLen = seq + L` keys, query row `j < L` at position `p = seq + j`. A key `t` is
visible iff `t ≤ p` and, on a sliding layer of window `W`, `t > p - W` (as integers). The
kernel starts at `lo = max(seq - W + 1, 0)` (sliding) or 0 (full) and splits `[lo, kvLen)` into
`ns` chunks of `chunk` keys from `lo`; `ns * chunk ≥ span ≥ kvLen - lo`, where span is
`W + L - 1` (sliding) or the page table's capacity (full). -/

/-- Every visible key lies in `[lo, kvLen)` (sliding layer). -/
theorem sliding_visible_in_range (seq L W j t : Int) (hj0 : 0 ≤ j) (hj : j < L)
    (ht0 : 0 ≤ t) (hle : t ≤ seq + j) (hgt : t > seq + j - W) :
    max (seq - W + 1) 0 ≤ t ∧ t < seq + L := by
  omega

/-- ...and the range is at most `W + L - 1` long, the span the chunks are sized for. -/
theorem sliding_span (seq L W : Int) :
    seq + L - max (seq - W + 1) 0 ≤ W + L - 1 := by
  omega

/-- Full layers: every visible key lies in `[0, kvLen)`. -/
theorem full_visible_in_range (seq L j t : Int) (hj : j < L) (ht0 : 0 ≤ t) (hle : t ≤ seq + j) :
    0 ≤ t ∧ t < seq + L := by
  omega

/-- `chunk = ⌈⌈span / ns0⌉ / bn⌉ * bn`, `ns = ⌈span / chunk⌉`: then `ns * chunk ≥ span`, so
    chunk `s` = `[lo + s * chunk, lo + (s + 1) * chunk)` for `s < ns` tile the whole range. -/
theorem ceil_div_mul_ge (span chunk : Nat) (hc : 0 < chunk) :
    span ≤ (span + chunk - 1) / chunk * chunk := by
  have h := Nat.div_add_mod (span + chunk - 1) chunk
  have hm := Nat.mod_lt (span + chunk - 1) hc
  have : (span + chunk - 1) / chunk * chunk = chunk * ((span + chunk - 1) / chunk) :=
    Nat.mul_comm _ _
  omega

/-- The chunk is a positive multiple of the key block, so the kernel's inner loop steps whole
    blocks and only the last one of a chunk is masked. -/
theorem chunk_pos (span ns0 bn : Nat) (hs : 0 < span) (hn : 0 < ns0) (hb : 0 < bn) :
    0 < ((span + ns0 - 1) / ns0 + bn - 1) / bn * bn := by
  have h1 : 0 < (span + ns0 - 1) / ns0 := Nat.div_pos (by omega) hn
  have h2 : 0 < ((span + ns0 - 1) / ns0 + bn - 1) / bn := Nat.div_pos (by omega) hb
  exact Nat.mul_pos h2 hb

end KernelProofs
