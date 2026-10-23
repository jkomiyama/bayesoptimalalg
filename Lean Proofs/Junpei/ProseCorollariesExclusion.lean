import Junpei.FiniteModelExclusion
import Junpei.EmbeddingRandomized

set_option autoImplicit false
set_option warningAsError true

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.ProseCorollaries

open Junpei.FiniteGaussian Junpei.FinitePosterior Junpei.FiniteModelExclusion

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]
variable {S : Type*} [MeasurableSpace S]

noncomputable section

omit [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem otherMinimum_le_retained (i : A) [Nonempty {a : A // a ≠ i}]
    (u : Means A) (j : A) (hji : j ≠ i) : otherMinimum i u ≤ u j := by
  have h := coordinate_le_maxAll (fun a : {a : A // a ≠ i} => -u a) ⟨j, hji⟩
  change -maxAll (fun a : {a : A // a ≠ i} => -u a) ≤ u j
  linarith

omit [MeasurableSpace A] [MeasurableSingletonClass A] in
/-- The exclusion-loss comparison is pathwise, independently of the policy.
It remains valid at ties and for any recommended arm. -/
theorem regret_on_best_arm_le_exclusion (i : A) [Nonempty {a : A // a ≠ i}]
    (u : Means A) (j : A) (hi : maxAll u = u i) :
    maxAll u - u j ≤ max (u i - otherMinimum i u) 0 := by
  rw [hi]
  by_cases hji : j = i
  · subst j
    exact (sub_self _).trans_le (le_max_right _ _)
  · exact (sub_le_sub_left (otherMinimum_le_retained i u j hji) _).trans
      (le_max_left _ _)

theorem posterior_randomized_regret_integrable (c : Precisions A) (m : Means A)
    (ν : Measure S) [IsProbabilityMeasure ν] (J : Means A × S → A) (hJ : Measurable J) :
    Integrable (fun w => maxAll w.1 - w.1 (J w)) ((posteriorLaw c m).prod ν) := by
  have heval : Measurable (fun p : (Means A × S) × A => p.1.1 p.2) :=
    measurable_from_prod_countable_left (fun a => by
      change Measurable (fun p : Means A × S => p.1 a)
      fun_prop)
  have hm : Measurable (fun w : Means A × S => maxAll w.1 - w.1 (J w)) :=
    (maxAll_lipschitz.continuous.measurable.comp measurable_fst).sub
      (heval.comp (measurable_id.prodMk hJ))
  have hb : Integrable (fun w : Means A × S => 2 * ∑ a, |w.1 a|)
      ((posteriorLaw c m).prod ν) := by
    apply Integrable.const_mul
    apply integrable_finset_sum
    intro a _
    exact ((posterior_coordinate_integrable c m a).abs.comp_fst ν)
  apply hb.mono' hm.aestronglyMeasurable
  exact Filter.Eventually.of_forall (fun w =>
    Junpei.EmbeddingRandomized.regret_norm_le_sum w.1 (J w))

/-- Section 5.2's unnumbered assertion: the posterior risk contribution on
any event where arm i is best is bounded by its actual exclusion expectation.
The arbitrary independent S includes all Gaussian observation arrays and
external policy randomizers. The recommendation may be any Borel function. -/
theorem posterior_best_arm_risk_le_exclusion (i : A) [Nonempty {a : A // a ≠ i}]
    (c : Precisions A) (m : Means A) (ν : Measure S) [IsProbabilityMeasure ν]
    (J : Means A × S → A) (hJ : Measurable J)
    (E : Set (Means A × S)) (hE : MeasurableSet E)
    (hbest : ∀ w ∈ E, maxAll w.1 = w.1 i) :
    (∫ w, E.indicator (fun w => maxAll w.1 - w.1 (J w)) w
      ∂(posteriorLaw c m).prod ν) ≤ exclusion i c m := by
  have hi := posterior_randomized_regret_integrable c m ν J hJ
  have hex := (exclusion_integrable i c m).comp_fst ν
  have hp := integral_mono (hi.indicator hE) hex (fun w => by
    by_cases hw : w ∈ E
    · rw [indicator_of_mem hw]
      exact regret_on_best_arm_le_exclusion i w.1 (J w) (hbest w hw)
    · rw [indicator_of_notMem hw]
      exact le_max_right _ _)
  rw [integral_prod _ hex] at hp
  simpa [exclusion] using hp

end
end Junpei.ProseCorollaries
