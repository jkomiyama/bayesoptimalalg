import Junpei.UniformSamplingModel

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.UniformSampling

open Junpei.FiniteGaussian Junpei.TailRatio

noncomputable section

theorem gaussian_upper_tail_standardize (δ : ℝ) (v : ℝ≥0) (hv : 0 < v) :
    (gaussianReal (-δ) v).real (Ioi 0) = normalTail (δ / Real.sqrt (v : ℝ)) ∧
      (gaussianReal (-δ) v).real (Ici 0) = normalTail (δ / Real.sqrt (v : ℝ)) := by
  have hvR : 0 < (v : ℝ) := by exact_mod_cast hv
  have hs : 0 < Real.sqrt (v : ℝ) := Real.sqrt_pos.mpr hvR
  have hvar : (⟨Real.sqrt (v : ℝ)^2, sq_nonneg _⟩ : ℝ≥0) * 1 = v := by
    apply NNReal.coe_injective
    change Real.sqrt (v : ℝ)^2 * 1 = (v : ℝ)
    rw [mul_one, Real.sq_sqrt hvR.le]
  have hmap : (gaussianReal 0 1).map (fun z : ℝ => -δ + Real.sqrt (v : ℝ) * z) =
      gaussianReal (-δ) v := by
    have he : (fun z : ℝ => -δ + Real.sqrt (v : ℝ) * z) =
        (fun z : ℝ => -δ + z) ∘ (fun z : ℝ => Real.sqrt (v : ℝ) * z) := rfl
    rw [he, ← Measure.map_map (by fun_prop) (by fun_prop), gaussianReal_map_const_mul,
      mul_zero, hvar, gaussianReal_map_const_add, zero_add]
  have hp : (fun z : ℝ => -δ + Real.sqrt (v : ℝ) * z) ⁻¹' Ioi 0 =
      Ioi (δ / Real.sqrt (v : ℝ)) := by
    ext z
    change 0 < -δ + Real.sqrt (v : ℝ) * z ↔ δ / Real.sqrt (v : ℝ) < z
    rw [div_lt_iff₀ hs]
    constructor <;> intro h <;> nlinarith
  have hstrict : (gaussianReal (-δ) v).real (Ioi 0) =
      normalTail (δ / Real.sqrt (v : ℝ)) := by
    rw [measureReal_def, ← hmap, Measure.map_apply (by fun_prop) measurableSet_Ioi, hp]
    rfl
  constructor
  · exact hstrict
  · haveI : NoAtoms (gaussianReal (-δ) v) := noAtoms_gaussianReal hv.ne'
    exact (congrArg ENNReal.toReal (measure_congr
      (Ioi_ae_eq_Ici (μ := gaussianReal (-δ) v) (a := 0)))).symm.trans hstrict

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

def threshold (μ : Means A) (n : A → ℕ) (best a : A) : ℝ :=
  (μ best - μ a) / Real.sqrt (pairVariance n a best : ℝ)

theorem pairVariance_pos (n : A → ℕ) (a b : A) (hn : ∀ a, 0 < n a) :
    0 < pairVariance n a b := by
  unfold pairVariance
  have ha : (0 : ℝ≥0) < (n a : ℝ≥0) := by exact_mod_cast hn a
  exact add_pos_of_pos_of_nonneg (inv_pos.mpr ha) (by positivity)

theorem pair_crossing_prob (T : ℕ) (μ : Means A) (n : A → ℕ) (best a : A)
    (hab : a ≠ best) (hn : ∀ a, 0 < n a) (hnT : ∀ a, n a ≤ T) :
    (law T).real {x | empirical μ n x best < empirical μ n x a} =
      normalTail (threshold μ n best a) ∧
    (law T).real {x | empirical μ n x best ≤ empirical μ n x a} =
      normalTail (threshold μ n best a) := by
  have h := empirical_gap_law T μ n a best hab hn hnT
  have hlaw : (law T).map (fun x : Arrays A T => empirical μ n x a - empirical μ n x best) =
      gaussianReal (-(μ best - μ a)) (pairVariance n a best) := by
    convert h.map_eq using 1
    congr 1
    ring
  have hs := gaussian_upper_tail_standardize (μ best - μ a) (pairVariance n a best)
    (pairVariance_pos n a best hn)
  have hm : Measurable (fun x : Arrays A T => empirical μ n x a - empirical μ n x best) :=
    ((measurable_pi_apply a).comp (empirical_measurable T μ n)).sub
      ((measurable_pi_apply best).comp (empirical_measurable T μ n))
  simp only [threshold]
  constructor
  · rw [← hs.1, measureReal_def, measureReal_def, ← hlaw,
      Measure.map_apply hm measurableSet_Ioi]
    congr 1
    congr 1
    ext x
    change empirical μ n x best < empirical μ n x a ↔
      0 < empirical μ n x a - empirical μ n x best
    exact sub_pos.symm
  · rw [← hs.2, measureReal_def, measureReal_def, ← hlaw,
      Measure.map_apply hm measurableSet_Ici]
    congr 1
    congr 1
    ext x
    change empirical μ n x best ≤ empirical μ n x a ↔
      0 ≤ empirical μ n x a - empirical μ n x best
    exact sub_nonneg.symm

theorem error_tail_sandwich (T : ℕ) (μ : Means A) (n : A → ℕ) (best : A)
    (hn : ∀ a, 0 < n a) (hnT : ∀ a, n a ≤ T) :
    (∀ a : {a : A // a ≠ best}, normalTail (threshold μ n best a) ≤ error T μ best n) ∧
    error T μ best n ≤ ∑ a : {a : A // a ≠ best}, normalTail (threshold μ n best a) := by
  constructor
  · intro a
    rw [← (pair_crossing_prob T μ n best a a.2 hn hnT).1]
    exact error_lower_pair T μ n best a
  · have h := error_upper_pairs T μ n best
    simpa only [(fun a : {a : A // a ≠ best} =>
      (pair_crossing_prob T μ n best a a.2 hn hnT).2)] using h

end
end Junpei.UniformSampling
