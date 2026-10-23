import Junpei.MockStream
import Junpei.FiniteModelMeasurable

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.MockSample

open Junpei.FiniteGaussian Junpei.FinitePosterior Junpei.FiniteModelExclusion
open Junpei.MockStream

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A] {H : ℕ}

noncomputable section

omit [Nonempty A] in
theorem theta_coordinate_integrable (H : ℕ) (c : Precisions A) (m : Means A)
    (i a : A) : Integrable (fun w => theta H m i w a) (law H c i) := by
  have h := posterior_coordinate_integrable c m a
  rw [← theta_law H c m i] at h
  exact h.comp_measurable (theta_measurable H m i)

theorem theta_max_integrable (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    Integrable (fun w => maxAll (theta H m i w)) (law H c i) := by
  have h := posterior_integrable maxAll_lipschitz c m
  rw [← theta_law H c m i] at h
  exact h.comp_measurable (theta_measurable H m i)

section Measurable
variable [MeasurableSpace A] [MeasurableSingletonClass A]

omit [Nonempty A] in
theorem selected_coordinate_measurable (H : ℕ) (m : Means A) (i : A)
    (S : World A H i → RawStream A H) (J : RawStream A H → A)
    (hS : Measurable S) (hJ : Measurable J) :
    Measurable (fun w => theta H m i w (J (S w))) := by
  have h : Measurable (fun p : World A H i × A => theta H m i p.1 p.2) :=
    measurable_from_prod_countable_left (fun a =>
      (measurable_pi_apply a).comp (theta_measurable H m i))
  exact h.comp (measurable_id.prodMk (hJ.comp hS))

omit [Nonempty A] in
theorem selected_coordinate_integrable (H : ℕ) (c : Precisions A) (m : Means A) (i : A)
    (S : World A H i → RawStream A H) (J : RawStream A H → A)
    (hS : Measurable S) (hJ : Measurable J) :
    Integrable (fun w => theta H m i w (J (S w))) (law H c i) := by
  have hi : Integrable (fun w => ∑ a, |theta H m i w a|) (law H c i) := by
    apply integrable_finset_sum
    intro a _
    simpa only [Real.norm_eq_abs] using (theta_coordinate_integrable H c m i a).norm
  apply hi.mono' (selected_coordinate_measurable H m i S J hS hJ).aestronglyMeasurable
  apply Filter.Eventually.of_forall
  intro w
  rw [Real.norm_eq_abs]
  exact Finset.single_le_sum (fun a _ => abs_nonneg (theta H m i w a))
    (Finset.mem_univ (J (S w)))

def risk (H : ℕ) (c : Precisions A) (m : Means A) (i : A)
    (S : World A H i → RawStream A H) (J : RawStream A H → A) : ℝ :=
  ∫ w, maxAll (theta H m i w) - theta H m i w (J (S w)) ∂law H c i

theorem risk_integrable (H : ℕ) (c : Precisions A) (m : Means A) (i : A)
    (S : World A H i → RawStream A H) (J : RawStream A H → A)
    (hS : Measurable S) (hJ : Measurable J) :
    Integrable (fun w => maxAll (theta H m i w) - theta H m i w (J (S w)))
      (law H c i) :=
  (theta_max_integrable H c m i).sub
    (selected_coordinate_integrable H c m i S J hS hJ)

omit [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
/-- Exchange of the real latent/noise block and its independent identical
copy preserves the full world law, including all other arms' parameters. -/
theorem integral_swap (H : ℕ) (c : Precisions A) (i : A)
    (f : World A H i → ℝ) (hf : Measurable f) :
    (∫ w, f (swap H i w) ∂law H c i) = ∫ w, f w ∂law H c i := by
  rw [← integral_map (swap_preserving H c i).measurable.aemeasurable
    hf.aestronglyMeasurable, (swap_preserving H c i).map_eq]

/-- Swapping both the latent arm-i parameter and its reward stream preserves
the risk of every measurable policy, even one that recommends i. -/
theorem full_risk_coupling (H : ℕ) (c : Precisions A) (m : Means A) (i : A)
    (J : RawStream A H → A) (hJ : Measurable J) :
    (∫ w, maxAll (theta H m i (swap H i w)) -
        theta H m i (swap H i w) (J (mockStream H m i w)) ∂law H c i) =
      risk H c m i (realStream H m i) J := by
  exact integral_swap H c i _
    (((maxAll_lipschitz (A := A)).continuous.measurable.comp
      (theta_measurable H m i)).sub
      (selected_coordinate_measurable H m i _ J (realStream_measurable H m i) hJ))

/-- Literal risk equality in Definition 5: a continuation that never
recommends i has the same actual Bayesian regret under the real stream and
under an independent posterior-predictive copy of just the arm-i stream. -/
theorem risk_coupling (H : ℕ) (c : Precisions A) (m : Means A) (i : A)
    (J : RawStream A H → A) (hJ : Measurable J) (hne : ∀ s, J s ≠ i) :
    risk H c m i (mockStream H m i) J = risk H c m i (realStream H m i) J := by
  have hswap := integral_swap H c i
    (fun w => theta H m i w (J (realStream H m i w)))
    (selected_coordinate_measurable H m i _ J (realStream_measurable H m i) hJ)
  have he : (fun w => theta H m i (swap H i w)
      (J (realStream H m i (swap H i w)))) =
      (fun w => theta H m i w (J (mockStream H m i w))) := by
    funext w
    rw [stream_swap]
    exact theta_other_unchanged H m i _ (hne _) w
  rw [he] at hswap
  unfold risk
  rw [integral_sub (theta_max_integrable H c m i)
    (selected_coordinate_integrable H c m i _ J (mockStream_measurable H m i) hJ),
    integral_sub (theta_max_integrable H c m i)
      (selected_coordinate_integrable H c m i _ J (realStream_measurable H m i) hJ), hswap]

def removeRecommendation (i : A) (J R : RawStream A H → A) (s : RawStream A H) : A :=
  if J s = i then R s else J s

omit [Fintype A] [Nonempty A] in
theorem removeRecommendation_measurable (i : A) (J R : RawStream A H → A)
    (hJ : Measurable J) (hR : Measurable R) : Measurable (removeRecommendation i J R) :=
  hR.ite (hJ (measurableSet_singleton i)) hJ

omit [Fintype A] [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem removeRecommendation_excludes (i : A) (J R : RawStream A H → A)
    (hR : ∀ s, R s ≠ i) (s : RawStream A H) : removeRecommendation i J R s ≠ i := by
  unfold removeRecommendation
  split_ifs with h
  · exact hR s
  · exact h

omit [Fintype A] [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem removeRecommendation_agrees (i : A) (J R : RawStream A H → A)
    (s : RawStream A H) (hJ : J s ≠ i) : removeRecommendation i J R s = J s := by
  simp [removeRecommendation, hJ]

section Removed
variable (i : A) [Nonempty {a : A // a ≠ i}]

omit [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem maxAll_eq_removed (u : Means A) :
    maxAll u = max (u i) (otherMaximum i u) := by
  apply le_antisymm
  · apply maxAll_le
    intro a
    by_cases h : a = i
    · subst a; exact le_max_left _ _
    · exact (coordinate_le_maxAll (fun b : {a : A // a ≠ i} => u b) ⟨a, h⟩).trans
        (le_max_right _ _)
  · apply max_le (coordinate_le_maxAll u i)
    apply maxAll_le
    intro a
    exact coordinate_le_maxAll u a

omit [MeasurableSpace A] [MeasurableSingletonClass A] in
/-- The exact latent loss decomposition printed in Definition 5. -/
theorem loss_decomposition (u : Means A) (j : A) (hji : j ≠ i) :
    maxAll u - u j = otherMaximum i u - u j + max (u i - otherMaximum i u) 0 := by
  have _ := hji
  rw [maxAll_eq_removed i]
  by_cases h : otherMaximum i u ≤ u i
  · rw [max_eq_left h, max_eq_left (sub_nonneg.mpr h)]
    ring
  · have hh : u i ≤ otherMaximum i u := le_of_not_ge h
    rw [max_eq_right hh, max_eq_right (sub_nonpos.mpr hh), add_zero]

omit [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem otherMaximum_swap (H : ℕ) (m : Means A) (w : World A H i) :
    otherMaximum i (theta H m i (swap H i w)) = otherMaximum i (theta H m i w) := by
  unfold otherMaximum
  congr 1
  funext a
  exact theta_other_unchanged H m i a a.property w

omit [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem otherMinimum_le (u : Means A) (j : A) (hji : j ≠ i) :
    otherMinimum i u ≤ u j := by
  have h := coordinate_le_maxAll (fun a : {a : A // a ≠ i} => -u a) ⟨j, hji⟩
  unfold otherMinimum
  linarith

omit [MeasurableSpace A] [MeasurableSingletonClass A] in
/-- The pointwise increase from excluding a terminal recommendation i is
bounded by its advantage over the minimum retained latent mean. -/
theorem recommendation_loss_increase (u : Means A) (J R : RawStream A H → A)
    (hR : ∀ s, R s ≠ i) (s : RawStream A H) :
    (maxAll u - u (removeRecommendation i J R s)) - (maxAll u - u (J s)) ≤
      max (u i - otherMinimum i u) 0 := by
  unfold removeRecommendation
  split_ifs with h
  · rw [h]
    have hm := otherMinimum_le i u (R s) (hR s)
    have hp := le_max_left (u i - otherMinimum i u) (0 : ℝ)
    linarith
  · simp only [sub_self]
    exact le_max_right _ _

omit [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem exclusion_world_integral (H : ℕ) (c : Precisions A) (m : Means A) :
    (∫ w, max (theta H m i w i - otherMinimum i (theta H m i w)) 0 ∂law H c i) =
      exclusion i c m := by
  unfold exclusion
  rw [← theta_law H c m i, integral_map (theta_measurable H m i).aemeasurable]
  exact ((measurable_pi_apply i).sub (otherMinimum_lipschitz i).continuous.measurable).max
    measurable_const |>.aestronglyMeasurable

omit [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem exclusion_world_integrable (H : ℕ) (c : Precisions A) (m : Means A) :
    Integrable (fun w => max (theta H m i w i - otherMinimum i (theta H m i w)) 0)
      (law H c i) := by
  have h := exclusion_integrable i c m
  rw [← theta_law H c m i] at h
  exact h.comp_measurable (theta_measurable H m i)

/-- The second displayed inequality's risk-comparison part, with the actual
Gaussian posterior expectation and no assumed equality or exclusion bound. -/
theorem modified_real_risk_le (H : ℕ) (c : Precisions A) (m : Means A)
    (J R : RawStream A H → A) (hJ : Measurable J) (hR : Measurable R)
    (hne : ∀ s, R s ≠ i) :
    risk H c m i (realStream H m i) (removeRecommendation i J R) ≤
      risk H c m i (realStream H m i) J + exclusion i c m := by
  have hmod := removeRecommendation_measurable i J R hJ hR
  have h := integral_mono
    (risk_integrable H c m i _ _ (realStream_measurable H m i) hmod)
    ((risk_integrable H c m i _ J (realStream_measurable H m i) hJ).add
      (exclusion_world_integrable i H c m)) (fun w => by
        have hp := recommendation_loss_increase i (theta H m i w) J R hne
          (realStream H m i w)
        simp only [Pi.add_apply] at ⊢
        linarith)
  dsimp only [Pi.add_apply] at h
  rw [integral_add
    (risk_integrable H c m i _ J (realStream_measurable H m i) hJ)
    (exclusion_world_integrable i H c m), exclusion_world_integral i H c m] at h
  exact h

/-- Independent-copy mock loss equals the terminal-modified real loss and is
at most the original real loss plus the actual posterior exclusion slack. -/
theorem mock_risk_eq_and_le (H : ℕ) (c : Precisions A) (m : Means A)
    (J R : RawStream A H → A) (hJ : Measurable J) (hR : Measurable R)
    (hne : ∀ s, R s ≠ i) :
    risk H c m i (mockStream H m i) (removeRecommendation i J R) =
      risk H c m i (realStream H m i) (removeRecommendation i J R) ∧
    risk H c m i (mockStream H m i) (removeRecommendation i J R) ≤
      risk H c m i (realStream H m i) J + exclusion i c m := by
  have he := risk_coupling H c m i (removeRecommendation i J R)
    (removeRecommendation_measurable i J R hJ hR)
    (removeRecommendation_excludes i J R hne)
  exact ⟨he, he ▸ modified_real_risk_le i H c m J R hJ hR hne⟩

end Removed
end Measurable
end
end Junpei.MockSample
