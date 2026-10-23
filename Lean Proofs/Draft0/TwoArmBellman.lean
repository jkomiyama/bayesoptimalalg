import Draft0.Gaussian
import Draft0.Counts
import Mathlib.Tactic

/-!
Finite-horizon, two-arm posterior-mean Bellman model. Each action has the
Gaussian innovation law already proved in Draft0.Gaussian. The terminal reward
is the larger posterior mean. The recursive maximization is defined explicitly.
-/

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.TwoArmBellman

open Draft0.Gaussian
open Draft0.Counts

noncomputable section

def leftStep (c : Counts) : Counts := ⟨c.n1 + 1, c.n2, c.n3⟩
def rightStep (c : Counts) : Counts := ⟨c.n1, c.n2 + 1, c.n3⟩
def variance (c : Counts) : ℝ≥0 := posteriorVariance c.n1 c.n2

def cumulativeVariance : ℕ → Counts → ℝ≥0
  | 0, _ => 0
  | k + 1, c => max
      (updateVariance c.n1 + cumulativeVariance k (leftStep c))
      (updateVariance c.n2 + cumulativeVariance k (rightStep c))

def payoff : ℕ → ℝ → ℝ → Counts → ℝ
  | 0, m1, m2, _ => max m1 m2
  | k + 1, m1, m2, c => max
      (∫ z : ℝ, payoff k (m1 + z) m2 (leftStep c) ∂gaussianReal 0 (updateVariance c.n1))
      (∫ z : ℝ, payoff k m1 (m2 + z) (rightStep c) ∂gaussianReal 0 (updateVariance c.n2))

theorem hinge_smoothing_integrable (center : ℝ) (a b : ℝ≥0) :
    Integrable (fun z : ℝ => hingeLoss (center - z) b) (gaussianReal 0 a) := by
  have hconv : gaussianReal 0 a ∗ gaussianReal 0 b = gaussianReal 0 (a + b) := by
    simpa using (gaussianReal_conv_gaussianReal (m₁ := (0 : ℝ)) (m₂ := (0 : ℝ))
      (v₁ := a) (v₂ := b))
  have hi : Integrable (fun x : ℝ => max (x - center) 0)
      (gaussianReal 0 a ∗ gaussianReal 0 b) := by
    rw [hconv]
    exact hinge_integrable center (a + b)
  have h := (integrable_conv_iff hi.aestronglyMeasurable).mp hi
  have hf : (fun z : ℝ => ∫ r : ℝ, ‖max (z + r - center) 0‖ ∂gaussianReal 0 b) =
      (fun z : ℝ => hingeLoss (center - z) b) := by
    funext z
    unfold hingeLoss
    congr 1
    funext r
    rw [Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
    congr 1
    ring
  rw [hf] at h
  exact h.2

theorem hinge_smoothing (center : ℝ) (a b : ℝ≥0) :
    (∫ z : ℝ, hingeLoss (center - z) b ∂gaussianReal 0 a) =
      hingeLoss center (a + b) := by
  have hconv : gaussianReal 0 a ∗ gaussianReal 0 b = gaussianReal 0 (a + b) := by
    simpa using (gaussianReal_conv_gaussianReal (m₁ := (0 : ℝ)) (m₂ := (0 : ℝ))
      (v₁ := a) (v₂ := b))
  have hi : Integrable (fun x : ℝ => max (x - center) 0)
      (gaussianReal 0 a ∗ gaussianReal 0 b) := by
    rw [hconv]
    exact hinge_integrable center (a + b)
  unfold hingeLoss
  rw [← hconv, integral_conv hi]
  congr 1
  funext z
  congr 1
  funext r
  congr 1
  ring

theorem hinge_smoothing_plus_integrable (center : ℝ) (a b : ℝ≥0) :
    Integrable (fun z : ℝ => hingeLoss (center + z) b) (gaussianReal 0 a) := by
  have hsym : (gaussianReal 0 a).map (fun z : ℝ => -z) = gaussianReal 0 a := by
    simpa using (gaussianReal_map_neg (μ := (0 : ℝ)) (v := a))
  have hi : Integrable (fun z : ℝ => hingeLoss (center - z) b)
      ((gaussianReal 0 a).map (fun z : ℝ => -z)) := by
    rw [hsym]
    exact hinge_smoothing_integrable center a b
  have h := (integrable_map_measure hi.aestronglyMeasurable (by fun_prop)).mp hi
  change Integrable (fun z : ℝ => hingeLoss (center - -z) b) (gaussianReal 0 a) at h
  simpa only [sub_neg_eq_add] using h

theorem hinge_smoothing_plus (center : ℝ) (a b : ℝ≥0) :
    (∫ z : ℝ, hingeLoss (center + z) b ∂gaussianReal 0 a) =
      hingeLoss center (a + b) := by
  have hsym : (gaussianReal 0 a).map (fun z : ℝ => -z) = gaussianReal 0 a := by
    simpa using (gaussianReal_map_neg (μ := (0 : ℝ)) (v := a))
  have hi : AEStronglyMeasurable (fun z : ℝ => hingeLoss (center - z) b)
      ((gaussianReal 0 a).map (fun z : ℝ => -z)) := by
    rw [hsym]
    exact (hinge_smoothing_integrable center a b).aestronglyMeasurable
  have hm := integral_map (μ := gaussianReal 0 a) (by fun_prop) hi
  rw [hsym] at hm
  simp only [sub_neg_eq_add] at hm
  rw [← hm, hinge_smoothing]

theorem hingeLoss_mono_variance (center : ℝ) (a b : ℝ≥0) (hab : a ≤ b) :
    hingeLoss center a ≤ hingeLoss center b := by
  rcases hab.eq_or_lt with h | h
  · rw [h]
  · by_cases hc : 0 ≤ center
    · exact (hingeLoss_strictMono_variance center hc a b h).le
    · have hneg : 0 ≤ -center := by linarith
      have hs := hingeLoss_strictMono_variance (-center) hneg a b h
      have ha := hingeLoss_reflection center a
      have hb := hingeLoss_reflection center b
      linarith

theorem hingeLoss_max_variance (center : ℝ) (a b : ℝ≥0) :
    hingeLoss center (max a b) = max (hingeLoss center a) (hingeLoss center b) := by
  by_cases h : a ≤ b
  · rw [max_eq_right h, max_eq_right (hingeLoss_mono_variance center a b h)]
  · have h' : b ≤ a := le_of_lt (lt_of_not_ge h)
    rw [max_eq_left h', max_eq_left (hingeLoss_mono_variance center b a h')]

theorem left_formula_integrable (m1 m2 : ℝ) (a b : ℝ≥0) :
    Integrable (fun z : ℝ => m2 + hingeLoss (m2 - (m1 + z)) b) (gaussianReal 0 a) := by
  convert (integrable_const m2).add (hinge_smoothing_integrable (m2 - m1) a b) using 1
  funext z
  simp only [Pi.add_apply]
  congr 1
  congr 1
  ring

theorem right_formula_integrable (m1 m2 : ℝ) (a b : ℝ≥0) :
    Integrable (fun z : ℝ => (m2 + z) + hingeLoss ((m2 + z) - m1) b) (gaussianReal 0 a) := by
  convert ((integrable_const m2).add (gaussian_id_integrable 0 a)).add
    (hinge_smoothing_plus_integrable (m2 - m1) a b) using 1
  funext z
  simp only [Pi.add_apply]
  congr 1
  congr 1
  ring

theorem left_formula_integral (m1 m2 : ℝ) (a b : ℝ≥0) :
    (∫ z : ℝ, m2 + hingeLoss (m2 - (m1 + z)) b ∂gaussianReal 0 a) =
      m2 + hingeLoss (m2 - m1) (a + b) := by
  have hf : (fun z : ℝ => m2 + hingeLoss (m2 - (m1 + z)) b) =
      (fun z : ℝ => m2 + hingeLoss ((m2 - m1) - z) b) := by
    funext z
    congr 1
    congr 1
    ring
  rw [hf, integral_add (integrable_const _) (hinge_smoothing_integrable _ _ _),
    integral_const, hinge_smoothing]
  simp

theorem right_formula_integral (m1 m2 : ℝ) (a b : ℝ≥0) :
    (∫ z : ℝ, (m2 + z) + hingeLoss ((m2 + z) - m1) b ∂gaussianReal 0 a) =
      m2 + hingeLoss (m2 - m1) (a + b) := by
  have hf : (fun z : ℝ => (m2 + z) + hingeLoss ((m2 + z) - m1) b) =
      (fun z : ℝ => (m2 + z) + hingeLoss ((m2 - m1) + z) b) := by
    funext z
    congr 1
    congr 1
    ring
  have hiMean : Integrable (fun z : ℝ => m2 + z) (gaussianReal 0 a) :=
    (integrable_const m2).add (gaussian_id_integrable 0 a)
  have hMean : (∫ z : ℝ, m2 + z ∂gaussianReal 0 a) = m2 := by
    have h := integral_add (integrable_const m2) (gaussian_id_integrable 0 a)
    change (∫ z : ℝ, m2 + z ∂gaussianReal 0 a) = _ at h
    rw [h, integral_const, integral_id_gaussianReal]
    simp
  rw [hf, integral_add hiMean
    (hinge_smoothing_plus_integrable _ _ _),
    hMean, hinge_smoothing_plus]

/-- Exact evaluation of the actual binary Bellman recursion, for every horizon. -/
theorem payoff_eq (k : ℕ) (m1 m2 : ℝ) (c : Counts) :
    payoff k m1 m2 c = m2 + hingeLoss (m2 - m1) (cumulativeVariance k c) := by
  induction k generalizing m1 m2 c with
  | zero =>
    simp only [payoff, cumulativeVariance, hingeLoss, gaussianReal_zero_var, integral_dirac]
    by_cases h : m1 ≤ m2
    · rw [max_eq_right h, max_eq_right (by linarith)]
      ring
    · have h' : m2 ≤ m1 := le_of_lt (lt_of_not_ge h)
      rw [max_eq_left h', max_eq_left (by linarith)]
      ring
  | succ k ih =>
    simp only [payoff, ih, left_formula_integral, right_formula_integral, cumulativeVariance]
    rw [hingeLoss_max_variance, add_max]

theorem left_action_integrable (k : ℕ) (m1 m2 : ℝ) (c : Counts) :
    Integrable (fun z : ℝ => payoff k (m1 + z) m2 (leftStep c))
      (gaussianReal 0 (updateVariance c.n1)) := by
  simp only [payoff_eq]
  exact left_formula_integrable _ _ _ _

theorem right_action_integrable (k : ℕ) (m1 m2 : ℝ) (c : Counts) :
    Integrable (fun z : ℝ => payoff k m1 (m2 + z) (rightStep c))
      (gaussianReal 0 (updateVariance c.n2)) := by
  simp only [payoff_eq]
  exact right_formula_integrable _ _ _ _

theorem cumulativeVariance_horizon_mono (k : ℕ) (c : Counts) :
    cumulativeVariance k c ≤ cumulativeVariance (k + 1) c := by
  induction k generalizing c with
  | zero => exact zero_le _
  | succ k ih =>
    simp only [cumulativeVariance]
    exact max_le_max (add_le_add_left (ih (leftStep c)) _)
      (add_le_add_left (ih (rightStep c)) _)

theorem payoff_horizon_mono (k : ℕ) (m1 m2 : ℝ) (c : Counts) :
    payoff k m1 m2 c ≤ payoff (k + 1) m1 m2 c := by
  rw [payoff_eq, payoff_eq]
  exact add_le_add_left (hingeLoss_mono_variance _ _ _
    (cumulativeVariance_horizon_mono k c)) _

theorem cumulativeVariance_third_irrelevant (k n1 n2 n3 n3' : ℕ) :
    cumulativeVariance k ⟨n1, n2, n3⟩ = cumulativeVariance k ⟨n1, n2, n3'⟩ := by
  induction k generalizing n1 n2 n3 n3' with
  | zero => rfl
  | succ k ih =>
    simp only [cumulativeVariance, leftStep, rightStep]
    rw [ih (n1 + 1) n2 n3 n3', ih n1 (n2 + 1) n3 n3']

theorem payoff_third_irrelevant (k : ℕ) (m1 m2 : ℝ) (n1 n2 n3 n3' : ℕ) :
    payoff k m1 m2 ⟨n1, n2, n3⟩ = payoff k m1 m2 ⟨n1, n2, n3'⟩ := by
  rw [payoff_eq, payoff_eq, cumulativeVariance_third_irrelevant]

theorem balancedStep_max (c : Counts) :
    max (balancedStep c).n1 (balancedStep c).n2 =
      max (max c.n1 c.n2) ((c.n1 + c.n2 + 2) / 2) := by
  unfold balancedStep
  split_ifs <;> simp only <;> omega

/-- The largest final count under greedy balancing has the smallest value
allowed by the initial counts and the final total. -/
theorem balancedRun_max (k : ℕ) (c : Counts) :
    max (balancedRun k c).n1 (balancedRun k c).n2 =
      max (max c.n1 c.n2) ((c.n1 + c.n2 + k + 1) / 2) := by
  induction k generalizing c with
  | zero => simp only [balancedRun]; omega
  | succ k ih =>
    rw [balancedRun, ih, balancedStep_max, balancedStep_sum]
    omega

theorem balancedRun_max_le_allocation (k : ℕ) (c : Counts) (a b : ℕ)
    (ha : c.n1 ≤ a) (hb : c.n2 ≤ b) (hs : a + b = c.n1 + c.n2 + k) :
    max (balancedRun k c).n1 (balancedRun k c).n2 ≤ max a b := by
  rw [balancedRun_max]
  omega

theorem product_le_of_same_sum_max {a b x y : ℝ}
    (hs : a + b = x + y) (hm : max a b ≤ max x y) : x * y ≤ a * b := by
  by_cases hab : a ≤ b <;> by_cases hxy : x ≤ y
  · rw [max_eq_right hab, max_eq_right hxy] at hm
    have hxb : x ≤ b := by linarith
    have hp := mul_nonneg (sub_nonneg.mpr hm) (sub_nonneg.mpr hxb)
    nlinarith [congrArg (fun z : ℝ => z * b) hs]
  · have hyx : y ≤ x := le_of_lt (lt_of_not_ge hxy)
    rw [max_eq_right hab, max_eq_left hyx] at hm
    have hyb : y ≤ b := by linarith
    have hp := mul_nonneg (sub_nonneg.mpr hm) (sub_nonneg.mpr hyb)
    nlinarith [congrArg (fun z : ℝ => z * b) hs]
  · have hba : b ≤ a := le_of_lt (lt_of_not_ge hab)
    rw [max_eq_left hba, max_eq_right hxy] at hm
    have hxa : x ≤ a := by linarith
    have hp := mul_nonneg (sub_nonneg.mpr hm) (sub_nonneg.mpr hxa)
    nlinarith [congrArg (fun z : ℝ => z * a) hs]
  · have hba : b ≤ a := le_of_lt (lt_of_not_ge hab)
    have hyx : y ≤ x := le_of_lt (lt_of_not_ge hxy)
    rw [max_eq_left hba, max_eq_left hyx] at hm
    have hya : y ≤ a := by linarith
    have hp := mul_nonneg (sub_nonneg.mpr hm) (sub_nonneg.mpr hya)
    nlinarith [congrArg (fun z : ℝ => z * a) hs]

/-- Greedy balancing minimizes terminal posterior variance over every
allocation of the remaining observations. -/
theorem balancedRun_variance_le_allocation (k : ℕ) (c : Counts) (a b : ℕ)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) (ha : c.n1 ≤ a) (hb : c.n2 ≤ b)
    (hs : a + b = c.n1 + c.n2 + k) :
    variance (balancedRun k c) ≤ posteriorVariance a b := by
  have hmono := balancedRun_monotone k c
  have hsum := balancedRun_sum k c
  have hmax := balancedRun_max_le_allocation k c a b ha hb hs
  have hsumR : ((balancedRun k c).n1 : ℝ) + ((balancedRun k c).n2 : ℝ) =
      (a : ℝ) + b := by exact_mod_cast (hsum.trans hs.symm)
  have hmaxR : max ((balancedRun k c).n1 : ℝ) ((balancedRun k c).n2 : ℝ) ≤
      max (a : ℝ) (b : ℝ) := by exact_mod_cast hmax
  have hprod := product_le_of_same_sum_max hsumR hmaxR
  have hp1 : 0 < ((balancedRun k c).n1 : ℝ) := by
    exact_mod_cast (show 0 < (balancedRun k c).n1 by omega)
  have hp2 : 0 < ((balancedRun k c).n2 : ℝ) := by
    exact_mod_cast (show 0 < (balancedRun k c).n2 by omega)
  have hpa : 0 < (a : ℝ) := by exact_mod_cast (show 0 < a by omega)
  have hpb : 0 < (b : ℝ) := by exact_mod_cast (show 0 < b by omega)
  have hleft : ((balancedRun k c).n1 : ℝ)⁻¹ + ((balancedRun k c).n2 : ℝ)⁻¹ =
      (((balancedRun k c).n1 : ℝ) + ((balancedRun k c).n2 : ℝ)) /
        (((balancedRun k c).n1 : ℝ) * ((balancedRun k c).n2 : ℝ)) := by
    field_simp
    ring
  have hright : (a : ℝ)⁻¹ + (b : ℝ)⁻¹ = ((a : ℝ) + b) / ((a : ℝ) * b) := by
    field_simp
    ring
  change ((balancedRun k c).n1 : ℝ)⁻¹ + ((balancedRun k c).n2 : ℝ)⁻¹ ≤
    (a : ℝ)⁻¹ + (b : ℝ)⁻¹
  rw [hleft, hright, hsumR]
  exact div_le_div_of_nonneg_left (by positivity) (mul_pos hpa hpb) hprod

def balancedInnovation (c : Counts) : ℝ≥0 :=
  if c.n1 ≤ c.n2 then updateVariance c.n1 else updateVariance c.n2

def balancedCumulative : ℕ → Counts → ℝ≥0
  | 0, _ => 0
  | k + 1, c => balancedInnovation c + balancedCumulative k (balancedStep c)

theorem leftStep_variance (c : Counts) (h1 : 0 < c.n1) :
    variance c = updateVariance c.n1 + variance (leftStep c) := by
  simpa only [variance, leftStep, Nat.cast_add, Nat.cast_one] using
    posteriorVariance_after_draw (c.n1 : ℝ≥0) (c.n2 : ℝ≥0) (by exact_mod_cast h1)

theorem rightStep_variance (c : Counts) (h2 : 0 < c.n2) :
    variance c = updateVariance c.n2 + variance (rightStep c) := by
  have h := posteriorVariance_after_draw (c.n2 : ℝ≥0) (c.n1 : ℝ≥0)
    (by exact_mod_cast h2)
  rw [posteriorVariance_comm (c.n2 : ℝ≥0) (c.n1 : ℝ≥0),
    posteriorVariance_comm ((c.n2 : ℝ≥0) + 1) (c.n1 : ℝ≥0)] at h
  simpa only [variance, rightStep, Nat.cast_add, Nat.cast_one] using h

theorem balancedStep_variance (c : Counts) (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    variance c = balancedInnovation c + variance (balancedStep c) := by
  by_cases h : c.n1 ≤ c.n2
  · simpa only [balancedInnovation, balancedStep, if_pos h, leftStep] using leftStep_variance c h1
  · simpa only [balancedInnovation, balancedStep, if_neg h, rightStep] using rightStep_variance c h2

/-- Independent mean innovations telescope exactly against terminal
posterior variance along the greedy policy. -/
theorem balancedCumulative_decomposition (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    balancedCumulative k c + variance (balancedRun k c) = variance c := by
  induction k generalizing c with
  | zero => simp only [balancedCumulative, balancedRun, zero_add]
  | succ k ih =>
    have hmono := balancedStep_monotone c
    have hi := ih (balancedStep c) (by omega) (by omega)
    rw [balancedCumulative, balancedRun, add_assoc, hi]
    exact (balancedStep_variance c h1 h2).symm

theorem left_balanced_variance_le (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    variance (balancedRun (k + 1) c) ≤ variance (balancedRun k (leftStep c)) := by
  have hmono := balancedRun_monotone k (leftStep c)
  have hsum := balancedRun_sum k (leftStep c)
  apply balancedRun_variance_le_allocation (k + 1) c _ _ h1 h2
  · exact (Nat.le_succ c.n1).trans hmono.1
  · exact hmono.2
  · change (balancedRun k (leftStep c)).n1 + (balancedRun k (leftStep c)).n2 =
      (c.n1 + 1) + c.n2 + k at hsum
    omega

theorem right_balanced_variance_le (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    variance (balancedRun (k + 1) c) ≤ variance (balancedRun k (rightStep c)) := by
  have hmono := balancedRun_monotone k (rightStep c)
  have hsum := balancedRun_sum k (rightStep c)
  apply balancedRun_variance_le_allocation (k + 1) c _ _ h1 h2
  · exact hmono.1
  · exact (Nat.le_succ c.n2).trans hmono.2
  · change (balancedRun k (rightStep c)).n1 + (balancedRun k (rightStep c)).n2 =
      c.n1 + (c.n2 + 1) + k at hsum
    omega

theorem left_balancedCumulative_le (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    updateVariance c.n1 + balancedCumulative k (leftStep c) ≤
      balancedCumulative (k + 1) c := by
  have hl := balancedCumulative_decomposition k (leftStep c)
    (by simp only [leftStep]; omega) (by simpa only [leftStep] using h2)
  have hd : (updateVariance c.n1 + balancedCumulative k (leftStep c)) +
      variance (balancedRun k (leftStep c)) = variance c := by
    rw [add_assoc, hl, ← leftStep_variance c h1]
  have hb := balancedCumulative_decomposition (k + 1) c h1 h2
  have hf := left_balanced_variance_le k c h1 h2
  apply (add_le_add_iff_right (variance (balancedRun (k + 1) c))).mp
  calc
    _ ≤ (updateVariance c.n1 + balancedCumulative k (leftStep c)) +
        variance (balancedRun k (leftStep c)) := add_le_add_left hf _
    _ = _ := hd.trans hb.symm

theorem right_balancedCumulative_le (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    updateVariance c.n2 + balancedCumulative k (rightStep c) ≤
      balancedCumulative (k + 1) c := by
  have hr := balancedCumulative_decomposition k (rightStep c)
    (by simpa only [rightStep] using h1) (by simp only [rightStep]; omega)
  have hd : (updateVariance c.n2 + balancedCumulative k (rightStep c)) +
      variance (balancedRun k (rightStep c)) = variance c := by
    rw [add_assoc, hr, ← rightStep_variance c h2]
  have hb := balancedCumulative_decomposition (k + 1) c h1 h2
  have hf := right_balanced_variance_le k c h1 h2
  apply (add_le_add_iff_right (variance (balancedRun (k + 1) c))).mp
  calc
    _ ≤ (updateVariance c.n2 + balancedCumulative k (rightStep c)) +
        variance (balancedRun k (rightStep c)) := add_le_add_left hf _
    _ = _ := hd.trans hb.symm

/-- The variance maximization arising from the actual Bellman recursion is
achieved by the deterministic smaller-count rule. -/
theorem cumulativeVariance_eq_balancedCumulative (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    cumulativeVariance k c = balancedCumulative k c := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih =>
    rw [cumulativeVariance,
      ih (leftStep c) (by simp only [leftStep]; omega) (by simpa only [leftStep] using h2),
      ih (rightStep c) (by simpa only [rightStep] using h1) (by simp only [rightStep]; omega)]
    apply le_antisymm
    · exact max_le (left_balancedCumulative_le k c h1 h2) (right_balancedCumulative_le k c h1 h2)
    · by_cases h : c.n1 ≤ c.n2
      · simp only [balancedCumulative, balancedInnovation, balancedStep, if_pos h]
        exact le_max_left _ _
      · simp only [balancedCumulative, balancedInnovation, balancedStep, if_neg h]
        exact le_max_right _ _

def balancedPayoff : ℕ → ℝ → ℝ → Counts → ℝ
  | 0, m1, m2, _ => max m1 m2
  | k + 1, m1, m2, c =>
    if c.n1 ≤ c.n2 then
      ∫ z : ℝ, balancedPayoff k (m1 + z) m2 (leftStep c) ∂gaussianReal 0 (updateVariance c.n1)
    else
      ∫ z : ℝ, balancedPayoff k m1 (m2 + z) (rightStep c) ∂gaussianReal 0 (updateVariance c.n2)

theorem balancedPayoff_eq (k : ℕ) (m1 m2 : ℝ) (c : Counts) :
    balancedPayoff k m1 m2 c = m2 + hingeLoss (m2 - m1) (balancedCumulative k c) := by
  induction k generalizing m1 m2 c with
  | zero =>
    have h := payoff_eq 0 m1 m2 c
    simpa only [payoff, cumulativeVariance, balancedPayoff, balancedCumulative] using h
  | succ k ih =>
    simp only [balancedPayoff, ih, left_formula_integral, right_formula_integral]
    by_cases h : c.n1 ≤ c.n2
    · simp only [if_pos h, balancedCumulative, balancedInnovation, balancedStep]
      rfl
    · simp only [if_neg h, balancedCumulative, balancedInnovation, balancedStep]
      rfl

/-- Full finite-horizon two-arm optimality (corrected Lemma 16), proved
against the actual recursive Bellman maximum rather than assumed. -/
theorem balanced_policy_optimal (k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    balancedPayoff k m1 m2 c = payoff k m1 m2 c := by
  rw [balancedPayoff_eq, payoff_eq, cumulativeVariance_eq_balancedCumulative k c h1 h2]

theorem payoff_eq_balanced_formula (k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    payoff k m1 m2 c = m2 + hingeLoss (m2 - m1) (balancedCumulative k c) := by
  rw [payoff_eq, cumulativeVariance_eq_balancedCumulative k c h1 h2]

/-- The exact multiround Gaussian variance used by Lemma 17 is certified
for the optimal two-arm policy. -/
theorem cumulativeVariance_eq_transitionVariance (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    (cumulativeVariance k c : ℝ) = transitionVariance k c := by
  have hd := balancedCumulative_decomposition k c h1 h2
  have hdR := congrArg (fun x : ℝ≥0 => (x : ℝ)) hd
  rw [cumulativeVariance_eq_balancedCumulative k c h1 h2]
  simp only [NNReal.coe_add, variance, posteriorVariance, NNReal.coe_inv, NNReal.coe_natCast] at hdR
  unfold transitionVariance
  simp only [one_div]
  linarith

def leftAction (k : ℕ) (m1 m2 : ℝ) (c : Counts) : ℝ :=
  ∫ z : ℝ, payoff k (m1 + z) m2 (leftStep c) ∂gaussianReal 0 (updateVariance c.n1)

def rightAction (k : ℕ) (m1 m2 : ℝ) (c : Counts) : ℝ :=
  ∫ z : ℝ, payoff k m1 (m2 + z) (rightStep c) ∂gaussianReal 0 (updateVariance c.n2)

theorem leftAction_eq (k : ℕ) (m1 m2 : ℝ) (c : Counts) :
    leftAction k m1 m2 c = m2 + hingeLoss (m2 - m1)
      (updateVariance c.n1 + cumulativeVariance k (leftStep c)) := by
  unfold leftAction
  simp only [payoff_eq]
  exact left_formula_integral _ _ _ _

theorem rightAction_eq (k : ℕ) (m1 m2 : ℝ) (c : Counts) :
    rightAction k m1 m2 c = m2 + hingeLoss (m2 - m1)
      (updateVariance c.n2 + cumulativeVariance k (rightStep c)) := by
  unfold rightAction
  simp only [payoff_eq]
  exact right_formula_integral _ _ _ _

theorem smaller_count_left_optimal (k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) (h : c.n1 ≤ c.n2) :
    payoff (k + 1) m1 m2 c = leftAction k m1 m2 c := by
  rw [payoff_eq_balanced_formula _ _ _ _ h1 h2, leftAction_eq,
    cumulativeVariance_eq_balancedCumulative k (leftStep c)
      (by simp only [leftStep]; omega) (by simpa only [leftStep] using h2)]
  simp only [balancedCumulative, balancedInnovation, balancedStep, if_pos h]
  rfl

theorem smaller_count_right_optimal (k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) (h : c.n2 < c.n1) :
    payoff (k + 1) m1 m2 c = rightAction k m1 m2 c := by
  have hn : ¬ c.n1 ≤ c.n2 := by omega
  rw [payoff_eq_balanced_formula _ _ _ _ h1 h2, rightAction_eq,
    cumulativeVariance_eq_balancedCumulative k (rightStep c)
      (by simpa only [rightStep] using h1) (by simp only [rightStep]; omega)]
  simp only [balancedCumulative, balancedInnovation, balancedStep, if_neg hn]
  rfl

end

end Draft0.TwoArmBellman
