import Mathlib.Data.Real.Basic
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

/-!
# Corrected count and final-round statements

These declarations formalize corrected Appendix B arithmetic, the count and
variance arithmetic of Lemma 15, and the actual balanced sampling schedule in
Equation (22). They do not assume or claim the Bellman optimality statement of
Lemma 16 or the nonterminal Gaussian lower bound of Lemma 17.
-/

namespace Draft0.Counts

set_option linter.dupNamespace false

open MeasureTheory
open scoped Classical

theorem appendixB_exact_inverse (q : ℕ) (hq : 0 < q) :
    (q : ℝ)⁻¹ = 2 / (((2 * q + 1 : ℕ) : ℝ) - 1) := by
  have hqr : (q : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hq
  push_cast
  field_simp
  ring

theorem appendixB_replacement_bound (q : ℕ) (hq : 0 < q) :
    (q : ℝ)⁻¹ ≤ 3 / ((2 * q + 1 : ℕ) : ℝ) := by
  have hqr : 1 ≤ (q : ℝ) := by exact_mod_cast hq
  have hp : 0 < (q : ℝ) := by positivity
  have ht : 0 < ((2 * q + 1 : ℕ) : ℝ) := by positivity
  rw [inv_eq_one_div]
  apply (div_le_div_iff₀ hp ht).mpr
  push_cast
  linarith

/-- The index with the larger count is the argmax, as used by the proof of
Lemma 15. The action-loss inequality is a separate stochastic claim. -/
def IsTwoArmArgmax (ni nj : ℕ) : Prop := nj ≤ ni

theorem lemma15_corrected_argmax (ni nj : ℕ) (hstrict : nj < ni) :
    IsTwoArmArgmax ni nj := by
  exact Nat.le_of_lt hstrict

theorem lemma15_corrected_footnote (q ni nj : ℕ)
    (hq : 0 < q) (hsum : ni + nj = 2 * q - 1) (hstrict : nj < ni) :
    q ≤ ni ∧ nj ≤ q - 1 := by
  omega

noncomputable def varianceDecrement (n : ℕ) : ℝ :=
  1 / ((n : ℝ) * ((n : ℝ) + 1))

theorem varianceDecrement_identity (n : ℕ) (hn : 0 < n) :
    1 / (n : ℝ) - 1 / ((n : ℝ) + 1) = varianceDecrement n := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  have hn1 : (n : ℝ) + 1 ≠ 0 := by positivity
  unfold varianceDecrement
  field_simp
  ring

theorem varianceDecrement_positive (n : ℕ) (hn : 0 < n) :
    0 < varianceDecrement n := by
  unfold varianceDecrement
  positivity

theorem varianceDecrement_strict_antitone (ni nj : ℕ)
    (hj : 0 < nj) (hstrict : nj < ni) :
    varianceDecrement ni < varianceDecrement nj := by
  have hjr : 0 < (nj : ℝ) := by exact_mod_cast hj
  have hir : (nj : ℝ) < (ni : ℝ) := by exact_mod_cast hstrict
  have hirpos : 0 < (ni : ℝ) := by linarith
  unfold varianceDecrement
  apply (div_lt_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith

/-- Sampling the smaller-count arm leaves a strictly smaller sum of posterior
variances. This is a variance comparison, not an action-loss comparison. -/
theorem lemma15_sample_smaller_variance (ni nj : ℕ)
    (hj : 0 < nj) (hstrict : nj < ni) :
    1 / (ni : ℝ) + 1 / ((nj : ℝ) + 1) <
      1 / ((ni : ℝ) + 1) + 1 / (nj : ℝ) := by
  have hi : 0 < ni := by omega
  have hmono := varianceDecrement_strict_antitone ni nj hj hstrict
  rw [← varianceDecrement_identity ni hi,
    ← varianceDecrement_identity nj hj] at hmono
  linarith

structure Counts where
  n1 : ℕ
  n2 : ℕ
  n3 : ℕ
deriving DecidableEq, Repr

def balancedStep (c : Counts) : Counts :=
  if c.n1 ≤ c.n2 then ⟨c.n1 + 1, c.n2, c.n3⟩
  else ⟨c.n1, c.n2 + 1, c.n3⟩

def balancedRun : ℕ → Counts → Counts
  | 0, c => c
  | k + 1, c => balancedRun k (balancedStep c)

theorem balancedStep_sum (c : Counts) :
    (balancedStep c).n1 + (balancedStep c).n2 = c.n1 + c.n2 + 1 := by
  unfold balancedStep
  split_ifs <;> simp only <;> omega

theorem balancedStep_n3 (c : Counts) : (balancedStep c).n3 = c.n3 := by
  unfold balancedStep
  split_ifs <;> rfl

theorem balancedRun_sum (k : ℕ) (c : Counts) :
    (balancedRun k c).n1 + (balancedRun k c).n2 = c.n1 + c.n2 + k := by
  induction k generalizing c with
  | zero => simp [balancedRun]
  | succ k ih =>
      rw [balancedRun, ih, balancedStep_sum]
      omega

theorem balancedRun_n3 (k : ℕ) (c : Counts) :
    (balancedRun k c).n3 = c.n3 := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih => rw [balancedRun, ih, balancedStep_n3]

theorem balancedStep_monotone (c : Counts) :
    c.n1 ≤ (balancedStep c).n1 ∧ c.n2 ≤ (balancedStep c).n2 := by
  unfold balancedStep
  split_ifs <;> simp only <;> omega

theorem balancedRun_monotone (k : ℕ) (c : Counts) :
    c.n1 ≤ (balancedRun k c).n1 ∧ c.n2 ≤ (balancedRun k c).n2 := by
  induction k generalizing c with
  | zero => exact ⟨le_rfl, le_rfl⟩
  | succ k ih =>
      have hstep := balancedStep_monotone c
      have hrun := ih (balancedStep c)
      exact ⟨hstep.1.trans hrun.1, hstep.2.trans hrun.2⟩

noncomputable def transitionVariance (k : ℕ) (c : Counts) : ℝ :=
  (1 / (c.n1 : ℝ) - 1 / ((balancedRun k c).n1 : ℝ)) +
    (1 / (c.n2 : ℝ) - 1 / ((balancedRun k c).n2 : ℝ))

theorem inverse_difference_nonnegative (a b : ℕ) (ha : 0 < a) (hab : a ≤ b) :
    0 ≤ 1 / (a : ℝ) - 1 / (b : ℝ) := by
  have har : 0 < (a : ℝ) := by exact_mod_cast ha
  have hb : 0 < b := by omega
  have hbr : 0 < (b : ℝ) := by exact_mod_cast hb
  have hle : 1 / (b : ℝ) ≤ 1 / (a : ℝ) := by
    apply (div_le_div_iff₀ hbr har).mpr
    norm_num
    exact_mod_cast hab
  linarith

theorem inverse_difference_positive (a b : ℕ) (ha : 0 < a) (hab : a < b) :
    0 < 1 / (a : ℝ) - 1 / (b : ℝ) := by
  have har : 0 < (a : ℝ) := by exact_mod_cast ha
  have hb : 0 < b := by omega
  have hbr : 0 < (b : ℝ) := by exact_mod_cast hb
  have hlt : 1 / (b : ℝ) < 1 / (a : ℝ) := by
    apply (div_lt_div_iff₀ hbr har).mpr
    norm_num
    exact_mod_cast hab
  linarith

theorem inverse_difference_lower_bound (a b T : ℕ)
    (ha : 0 < a) (hab : a < b) (hbt : b ≤ T) :
    1 / (T : ℝ) ^ 2 ≤ 1 / (a : ℝ) - 1 / (b : ℝ) := by
  have har : 0 < (a : ℝ) := by exact_mod_cast ha
  have hbr : 0 < (b : ℝ) := by exact_mod_cast (show 0 < b by omega)
  have htr : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have habr : 1 ≤ (b : ℝ) - (a : ℝ) := by
    have hnat : a + 1 ≤ b := by omega
    have hcast : (a : ℝ) + 1 ≤ (b : ℝ) := by exact_mod_cast hnat
    linarith
  have hatr : (a : ℝ) ≤ (T : ℝ) := by exact_mod_cast (show a ≤ T by omega)
  have hbtr : (b : ℝ) ≤ (T : ℝ) := by exact_mod_cast hbt
  have hprod : (a : ℝ) * (b : ℝ) ≤ (T : ℝ) ^ 2 := by
    calc
      (a : ℝ) * (b : ℝ) ≤ (T : ℝ) * (T : ℝ) :=
        mul_le_mul hatr hbtr (le_of_lt hbr) (le_of_lt htr)
      _ = (T : ℝ) ^ 2 := by ring
  have hdelta : (T : ℝ) ^ 2 ≤ ((b : ℝ) - (a : ℝ)) * (T : ℝ) ^ 2 := by
    simpa using mul_le_mul_of_nonneg_right habr (sq_nonneg (T : ℝ))
  have hidentity : 1 / (a : ℝ) - 1 / (b : ℝ) =
      ((b : ℝ) - (a : ℝ)) / ((a : ℝ) * (b : ℝ)) := by
    field_simp
  rw [hidentity]
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  linarith

theorem transitionVariance_nonnegative (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) : 0 ≤ transitionVariance k c := by
  have hmono := balancedRun_monotone k c
  exact add_nonneg (inverse_difference_nonnegative c.n1 _ h1 hmono.1)
    (inverse_difference_nonnegative c.n2 _ h2 hmono.2)

theorem transitionVariance_positive (k : ℕ) (c : Counts)
    (hk : 0 < k) (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    0 < transitionVariance k c := by
  have hmono := balancedRun_monotone k c
  have hsum := balancedRun_sum k c
  have hpos : c.n1 < (balancedRun k c).n1 ∨ c.n2 < (balancedRun k c).n2 := by
    omega
  unfold transitionVariance
  rcases hpos with hn1 | hn2
  · exact add_pos_of_pos_of_nonneg
      (inverse_difference_positive c.n1 _ h1 hn1)
      (inverse_difference_nonnegative c.n2 _ h2 hmono.2)
  · exact add_pos_of_nonneg_of_pos
      (inverse_difference_nonnegative c.n1 _ h1 hmono.1)
      (inverse_difference_positive c.n2 _ h2 hn2)

theorem transitionVariance_terminal (c : Counts) : transitionVariance 0 c = 0 := by
  simp [transitionVariance, balancedRun]

theorem transitionVariance_ge_inverse_square_of_caps (k T : ℕ) (c : Counts)
    (hk : 0 < k) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hb1 : (balancedRun k c).n1 ≤ T) (hb2 : (balancedRun k c).n2 ≤ T) :
    1 / (T : ℝ) ^ 2 ≤ transitionVariance k c := by
  have hmono := balancedRun_monotone k c
  have hsum := balancedRun_sum k c
  have hpos : c.n1 < (balancedRun k c).n1 ∨ c.n2 < (balancedRun k c).n2 := by
    omega
  unfold transitionVariance
  rcases hpos with hn1 | hn2
  · have hbound := inverse_difference_lower_bound c.n1 _ T h1 hn1 hb1
    have hnonneg := inverse_difference_nonnegative c.n2 _ h2 hmono.2
    linarith
  · have hbound := inverse_difference_lower_bound c.n2 _ T h2 hn2 hb2
    have hnonneg := inverse_difference_nonnegative c.n1 _ h1 hmono.1
    linarith

/-- While fewer than 2q-1 samples have been taken from arms 1 and 2,
the greedy step preserves both caps and cannot newly make arm 2 reach q. -/
theorem balancedStep_preterminal_invariant (q : ℕ) (c : Counts)
    (hq : 0 < q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hsum : c.n1 + c.n2 < 2 * q - 1) :
    (balancedStep c).n1 ≤ q ∧ (balancedStep c).n2 ≤ q ∧
      ((balancedStep c).n2 = q ↔ c.n2 = q) := by
  unfold balancedStep
  split_ifs with h
  · change c.n1 + 1 ≤ q ∧ c.n2 ≤ q ∧ (c.n2 = q ↔ c.n2 = q)
    exact ⟨by omega, h2, Iff.rfl⟩
  · change c.n1 ≤ q ∧ c.n2 + 1 ≤ q ∧ (c.n2 + 1 = q ↔ c.n2 = q)
    have hsmaller : c.n2 + 1 < q := by omega
    exact ⟨h1, by omega, by constructor <;> intro <;> omega⟩

theorem balancedRun_preterminal_invariant (q k : ℕ) (c : Counts)
    (hq : 0 < q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hsum : c.n1 + c.n2 + k ≤ 2 * q - 1) :
    (balancedRun k c).n1 ≤ q ∧ (balancedRun k c).n2 ≤ q ∧
      ((balancedRun k c).n2 = q ↔ c.n2 = q) := by
  induction k generalizing c with
  | zero => exact ⟨h1, h2, Iff.rfl⟩
  | succ k ih =>
      have hbefore : c.n1 + c.n2 < 2 * q - 1 := by omega
      have hstep := balancedStep_preterminal_invariant q c hq h1 h2 hbefore
      have hnext : (balancedStep c).n1 + (balancedStep c).n2 + k ≤
          2 * q - 1 := by rw [balancedStep_sum]; omega
      have hrun := ih (balancedStep c) hstep.1 hstep.2.1 hnext
      exact ⟨hrun.1, hrun.2.1, hrun.2.2.trans hstep.2.2⟩

theorem balancedStep_preserves_caps (q : ℕ) (c : Counts)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) (hsum : c.n1 + c.n2 < 2 * q) :
    (balancedStep c).n1 ≤ q ∧ (balancedStep c).n2 ≤ q := by
  unfold balancedStep
  split_ifs <;> simp only <;> constructor <;> omega

theorem balancedRun_preserves_caps (q k : ℕ) (c : Counts)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) (hsum : c.n1 + c.n2 + k ≤ 2 * q) :
    (balancedRun k c).n1 ≤ q ∧ (balancedRun k c).n2 ≤ q := by
  induction k generalizing c with
  | zero => exact ⟨h1, h2⟩
  | succ k ih =>
      have hstep := balancedStep_preserves_caps q c h1 h2 (by omega)
      have hnext : (balancedStep c).n1 + (balancedStep c).n2 + k ≤ 2 * q := by
        rw [balancedStep_sum]
        omega
      exact ih (balancedStep c) hstep.1 hstep.2 hnext

theorem balancedRun_final_counts (q k : ℕ) (c : Counts)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) (hsum : c.n1 + c.n2 + k = 2 * q) :
    (balancedRun k c).n1 = q ∧ (balancedRun k c).n2 = q := by
  have hcaps := balancedRun_preserves_caps q k c h1 h2 hsum.le
  have htotal := balancedRun_sum k c
  constructor <;> omega

/-- Corrected Equation (22): the exceptional orientation is forced when arm 2
already has q samples. The theorem covers every permitted initial state. -/
theorem equation22_terminal_counts (q k : ℕ) (c : Counts)
    (hq : 0 < q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hsum : c.n1 + c.n2 + k = 2 * q - 1) :
    (balancedRun k c).n1 = (if c.n2 = q then q - 1 else q) ∧
    (balancedRun k c).n2 = (if c.n2 = q then q else q - 1) := by
  have hcaps := balancedRun_preterminal_invariant q k c hq h1 h2 hsum.le
  have htotal := balancedRun_sum k c
  rw [hsum] at htotal
  by_cases h2q : c.n2 = q
  · rw [if_pos h2q, if_pos h2q]
    have ht2 := hcaps.2.2.mpr h2q
    constructor <;> omega
  · rw [if_neg h2q, if_neg h2q]
    have ht2 : (balancedRun k c).n2 ≠ q := by
      exact fun h => h2q (hcaps.2.2.mp h)
    constructor <;> omega

def horizonConsistent (t : ℕ) (c : Counts) : Prop :=
  1 ≤ c.n1 ∧ 1 ≤ c.n2 ∧ c.n3 = 1 ∧ c.n1 + c.n2 + c.n3 = t - 1

/-- The paper's round convention starts round t with t-1 total samples and
uses T-t samples before the terminal decision round T. -/
theorem equation22_horizon_counts (q t : ℕ) (c : Counts)
    (hq : 0 < q) (ht : 4 ≤ t) (horizon : t ≤ 2 * q + 1)
    (hc : horizonConsistent t c) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) :
    (balancedRun (2 * q + 1 - t) c).n1 = (if c.n2 = q then q - 1 else q) ∧
    (balancedRun (2 * q + 1 - t) c).n2 = (if c.n2 = q then q else q - 1) := by
  apply equation22_terminal_counts q (2 * q + 1 - t) c hq h1 h2
  rcases hc with ⟨_, _, h3, htotal⟩
  omega

theorem equation22_corrected_increments (q t : ℕ) (c : Counts)
    (hq : 0 < q) (ht : 4 ≤ t) (horizon : t ≤ 2 * q + 1)
    (hc : horizonConsistent t c) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) :
    (balancedRun (2 * q + 1 - t) c).n1 - c.n1 =
      (if c.n2 = q then q - 1 else q) - c.n1 ∧
    (balancedRun (2 * q + 1 - t) c).n2 - c.n2 =
      (if c.n2 = q then q else q - 1) - c.n2 := by
  have heq := equation22_horizon_counts q t c hq ht horizon hc h1 h2
  rw [heq.1, heq.2]
  exact ⟨rfl, rfl⟩

theorem equation22_generic_horizon (T q t : ℕ) (c : Counts)
    (hT : T = 2 * q + 1) (hq : 0 < q) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) :
    (balancedRun (T - t) c).n1 = (if c.n2 = q then q - 1 else q) ∧
    (balancedRun (T - t) c).n2 = (if c.n2 = q then q else q - 1) := by
  subst T
  exact equation22_horizon_counts q t c hq ht horizon hc h1 h2

/-- Including the sample in round T yields equal final counts. -/
theorem balanced_final_horizon_counts (T q t : ℕ) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) :
    (balancedRun (T - t + 1) c).n1 = q ∧
    (balancedRun (T - t + 1) c).n2 = q := by
  apply balancedRun_final_counts q (T - t + 1) c h1 h2
  rcases hc with ⟨_, _, h3, htotal⟩
  omega

/-- Corrected Equation (23) has nondegenerate variance only at t<T. This
quantitative bound is obtained from the actual schedule, rather than postulated. -/
theorem lemma17_nonterminal_variance_bound (T q t : ℕ) (c : Counts)
    (hT : T = 2 * q + 1) (hq : 0 < q) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) :
    1 / (T : ℝ) ^ 2 ≤ transitionVariance (T - t) c := by
  have hcounts := equation22_generic_horizon T q t c hT hq ht hpre.le hc h1 h2
  have hb1 : (balancedRun (T - t) c).n1 ≤ T := by
    rw [hcounts.1]
    split_ifs <;> omega
  have hb2 : (balancedRun (T - t) c).n2 ≤ T := by
    rw [hcounts.2]
    split_ifs <;> omega
  exact transitionVariance_ge_inverse_square_of_caps (T - t) T c
    (by omega) (Nat.lt_of_lt_of_le (by decide) hc.1)
    (Nat.lt_of_lt_of_le (by decide) hc.2.1) hb1 hb2

theorem lemma17_terminal_variance (T : ℕ) (c : Counts) :
    transitionVariance (T - T) c = 0 := by
  simp [transitionVariance, balancedRun]

abbrev MeanPath := ℕ → ℕ → ℝ

def terminalGap (p : MeanPath) (c : Counts) : ℝ := p 1 c.n1 - p 2 c.n2

def gapBand (T : ℕ) (gap : ℝ) : Prop :=
  1 / (T : ℝ) ^ 2 ≤ gap ∧ gap ≤ 2 / (T : ℝ) ^ 2

/-- At t=T the statistic at T-1 is already determined: its probability is
the indicator of membership in the interval, for every probability measure. -/
theorem lemma17_terminal_probability_indicator {Ω : Type*}
    [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : ℕ) (p : MeanPath) (c : Counts) :
    (μ {_ω : Ω | gapBand T (terminalGap p (balancedRun (T - T) c))}).toReal =
      if gapBand T (terminalGap p c) then 1 else 0 := by
  classical
  by_cases h : gapBand T (terminalGap p c)
  · simp [balancedRun, h]
  · simp [balancedRun, h]

theorem lemma17_terminal_band_zero (T : ℕ) (hT : 0 < T) :
    ¬ gapBand T 0 := by
  have hp : 0 < 1 / (T : ℝ) ^ 2 := by positivity
  exact fun h => not_le_of_gt hp h.1

theorem lemma17_terminal_zero_gap_probability {Ω : Type*}
    [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : ℕ) (hT : 0 < T) (p : MeanPath) (c : Counts)
    (hgap : terminalGap p c = 0) :
    (μ {_ω : Ω | gapBand T (terminalGap p (balancedRun (T - T) c))}).toReal = 0 := by
  rw [lemma17_terminal_probability_indicator, hgap,
    if_neg (lemma17_terminal_band_zero T hT)]

theorem lemma17_nonterminal_has_remaining_sample (T t : ℕ) (h : t < T) :
    0 < T - t := by omega

theorem lemma17_terminal_has_no_remaining_sample (T : ℕ) : T - T = 0 := by omega

end Draft0.Counts
