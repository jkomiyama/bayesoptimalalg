import Junpei.UniformSamplingProbability
import Junpei.UniformSamplingTail

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.UniformSampling

open Junpei.FiniteGaussian Junpei.TailRatio

noncomputable section

theorem tendsto_atTop_of_sq_div_pos (x : ℕ → ℝ) (q : ℝ) (hq : 0 < q)
    (hx : ∀ᶠ T in atTop, 0 ≤ x T)
    (hlim : Tendsto (fun T => (x T)^2 / (T : ℝ)) atTop (𝓝 q)) :
    Tendsto x atTop atTop := by
  apply tendsto_atTop.mpr
  intro b
  filter_upwards [hx, (tendsto_order.mp hlim).1 (q/2) (by linarith),
    (tendsto_natCast_atTop_atTop : Tendsto (fun T : ℕ => (T : ℝ)) atTop atTop).eventually
      (eventually_gt_atTop (max 1 ((max b 0)^2 * (2/q))))] with T hxp hrat hT
  have hTp : 0 < (T : ℝ) := lt_trans (by positivity : (0 : ℝ) < 1)
    ((le_max_left _ _).trans_lt hT)
  have hsq := (lt_div_iff₀ hTp).mp hrat
  have hlarge : (max b 0)^2 < q/2 * (T : ℝ) := by
    have hm := mul_lt_mul_of_pos_left ((le_max_right _ _).trans_lt hT)
      (show 0 < q/2 by positivity)
    have he : q/2 * ((max b 0)^2 * (2/q)) = (max b 0)^2 := by field_simp
    rwa [he] at hm
  have hb : b ≤ max b 0 := le_max_left _ _
  have hb0 : 0 ≤ max b 0 := le_max_right _ _
  nlinarith

theorem threshold_rate_limit {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (μ : Means A) (n : ℕ → A → ℕ) (K : ℝ) (hK : 0 < K)
    (hn : ∀ a, Tendsto (fun T : ℕ => (n T a : ℝ) / (T : ℝ)) atTop (𝓝 (1/K)))
    (best a : A) (hgap : 0 < μ best - μ a) :
    Tendsto (fun T => -Real.log (normalTail (threshold μ (n T) best a)) / (T : ℝ))
      atTop (𝓝 ((μ best - μ a)^2 / (4*K))) := by
  have hi (b : A) : Tendsto (fun T : ℕ => (T : ℝ) / (n T b : ℝ)) atTop (𝓝 K) := by
    have h := (hn b).inv₀ (ne_of_gt (by positivity : 0 < 1/K))
    simpa only [one_div, inv_div, inv_inv] using h
  have hv : Tendsto (fun T : ℕ => (T : ℝ) * (pairVariance (n T) a best : ℝ))
      atTop (𝓝 (2*K)) := by
    convert (hi a).add (hi best) using 1
    · funext T
      simp only [pairVariance, NNReal.coe_add, NNReal.coe_inv, NNReal.coe_natCast,
        mul_add, div_eq_mul_inv]
    · congr 1
      ring
  have hnpos (b : A) : ∀ᶠ T in atTop, 0 < n T b := by
    filter_upwards [(tendsto_order.mp (hn b)).1 0 (by positivity)] with T hT
    have hh := pos_of_mul_pos_left (show 0 < (n T b : ℝ) * (T : ℝ)⁻¹ by
      simpa only [div_eq_mul_inv] using hT) (by positivity : 0 ≤ (T : ℝ)⁻¹)
    exact_mod_cast hh
  have hsquared : Tendsto (fun T => (threshold μ (n T) best a)^2 / (T : ℝ))
      atTop (𝓝 ((μ best - μ a)^2 / (2*K))) := by
    have h : Tendsto (fun T : ℕ => (μ best - μ a)^2 /
        ((T : ℝ) * (pairVariance (n T) a best : ℝ))) atTop
        (𝓝 ((μ best - μ a)^2 / (2*K))) :=
      tendsto_const_nhds.div hv (ne_of_gt (by positivity : 0 < 2*K))
    apply h.congr'
    filter_upwards [hnpos a, hnpos best] with T ha hb
    have hvp : 0 < (pairVariance (n T) a best : ℝ) := by
      simp only [pairVariance, NNReal.coe_add, NNReal.coe_inv, NNReal.coe_natCast]
      positivity
    unfold threshold
    rw [div_pow, Real.sq_sqrt hvp.le]
    simp only [div_eq_mul_inv, mul_inv_rev]
    ring
  have hpositive : ∀ᶠ T in atTop, 0 ≤ threshold μ (n T) best a := by
    exact Filter.Eventually.of_forall (fun T => div_nonneg hgap.le (Real.sqrt_nonneg _))
  have ht := tendsto_atTop_of_sq_div_pos _ _ (by positivity) hpositive hsquared
  have h := gaussian_tail_rate _ _ ht hsquared
  convert h using 1
  congr 1
  ring

theorem finite_union_rate_limit {B : Type*} [Fintype B] [Nonempty B]
    (p : ℕ → B → ℝ) (r : ℕ → ℝ) (q : B → ℝ)
    (hp : ∀ᶠ T in atTop, ∀ b, 0 < p T b)
    (hl : ∀ᶠ T in atTop, ∀ b, p T b ≤ r T)
    (hu : ∀ᶠ T in atTop, r T ≤ ∑ b, p T b)
    (hlim : ∀ b, Tendsto (fun T => -Real.log (p T b) / (T : ℝ)) atTop (𝓝 (q b))) :
    Tendsto (fun T => -Real.log (r T) / (T : ℝ)) atTop
      (𝓝 (-maxAll (fun b => -q b))) := by
  let M : ℝ := Fintype.card B
  have hM : 0 < M := by
    dsimp [M]
    exact_mod_cast (Fintype.card_pos : 0 < Fintype.card B)
  have hpi : Tendsto (fun T => fun b => -(-Real.log (p T b) / (T : ℝ)))
      atTop (𝓝 (fun b => -q b)) := tendsto_pi_nhds.mpr (fun b => (hlim b).neg)
  have hmin : Tendsto (fun T => -maxAll (fun b => -(-Real.log (p T b) / (T : ℝ))))
      atTop (𝓝 (-maxAll (fun b => -q b))) :=
    ((maxAll_lipschitz (A := B)).continuous.tendsto _ |>.comp hpi).neg
  have hlowlim := hmin.sub (tendsto_const_div_atTop_nhds_zero_nat (Real.log M))
  obtain ⟨b0, hb0⟩ := maxAll_attained (fun b => -q b)
  have huplim : Tendsto (fun T => -Real.log (p T b0) / (T : ℝ)) atTop
      (𝓝 (-maxAll (fun b => -q b))) := by
    have he : q b0 = -maxAll (fun b => -q b) := by linarith
    simpa only [he] using hlim b0
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le'
    (by simpa only [sub_zero] using hlowlim) huplim
  · filter_upwards [hp, hl, hu, eventually_ge_atTop (1 : ℕ)] with T hpp hll huu hT
    obtain ⟨b, hb⟩ := maxAll_attained (p T)
    have hsum : (∑ i, p T i) ≤ M * p T b := by
      calc
        _ ≤ ∑ _ : B, p T b := Finset.sum_le_sum (fun i _ =>
          (coordinate_le_maxAll (p T) i).trans_eq hb.symm)
        _ = _ := by simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, M]
    have hrp : 0 < r T := (hpp b).trans_le (hll b)
    have hlog := Real.log_le_log hrp (huu.trans hsum)
    rw [Real.log_mul hM.ne' (hpp b).ne'] at hlog
    have he (i : B) : -(-Real.log (p T i) / (T : ℝ)) ≤
        -(-Real.log (p T b) / (T : ℝ)) := by
      have hh := Real.log_le_log (hpp i)
        ((coordinate_le_maxAll (p T) i).trans_eq hb.symm)
      simpa only [neg_div, neg_neg] using div_le_div_of_nonneg_right hh
        (by positivity : 0 ≤ (T : ℝ))
    have hmax : maxAll (fun i => -(-Real.log (p T i) / (T : ℝ))) =
        -(-Real.log (p T b) / (T : ℝ)) :=
      le_antisymm (maxAll_le _ _ he)
        (coordinate_le_maxAll (fun i => -(-Real.log (p T i) / (T : ℝ))) b)
    rw [hmax]
    have hh := div_le_div_of_nonneg_right (neg_le_neg hlog)
      (by positivity : 0 ≤ (T : ℝ))
    simpa only [neg_add, add_div, neg_neg, neg_div, sub_eq_add_neg, add_comm] using hh
  · filter_upwards [hp, hl, eventually_ge_atTop (1 : ℕ)] with T hpp hll hT
    have hh := Real.log_le_log (hpp b0) (hll b0)
    exact div_le_div_of_nonneg_right (neg_le_neg hh) (by positivity)

end
end Junpei.UniformSampling
