import Junpei.InitialOptimalHistory
import Junpei.TailRatio
import Junpei.ProseCorollariesErrorRate

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.UniformSampling

open Junpei.FiniteGaussian Junpei.FiniteModelMeasurable

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

noncomputable section

abbrev Arrays (A : Type*) (T : ℕ) := A → Fin T → ℝ

def law (T : ℕ) : Measure (Arrays A T) :=
  Measure.pi (fun _ : A => Draft0.BadEvent.arrayLaw T)

instance law_probability (T : ℕ) : IsProbabilityMeasure (law (A := A) T) := by
  unfold law
  infer_instance

/-- Empirical averages of actual independent unit-variance Gaussian rewards:
the array stores the centered noise, and the true mean is added to each arm.
The allocation is deterministic and may differ by rounding between arms. -/
def empirical {T : ℕ} (μ : Means A) (n : A → ℕ) (x : Arrays A T) : Means A :=
  fun a => μ a + Draft0.BadEvent.sampleMean (n a) (x a)

theorem empirical_measurable (T : ℕ) (μ : Means A) (n : A → ℕ) :
    Measurable (empirical (T := T) μ n) := by
  apply measurable_pi_lambda
  intro a
  exact (((Draft0.FrequentistModel.sampleMean_continuous T (n a)).measurable.comp
    (measurable_pi_apply a)).const_add (μ a))

theorem empirical_coordinate_law (T : ℕ) (μ : Means A) (n : A → ℕ) (a : A)
    (hn : 0 < n a) (hnT : n a ≤ T) :
    HasLaw (fun x : Arrays A T => empirical μ n x a)
      (gaussianReal (μ a) (n a : ℝ≥0)⁻¹) (law T) := by
  have he := (measurePreserving_eval (fun _ : A => Draft0.BadEvent.arrayLaw T) a).hasLaw
  have hm := (Draft0.BadEvent.sampleMean_law (n a) T hn hnT).fun_comp he
  have hadd : HasLaw (fun z : ℝ => μ a + z)
      (gaussianReal (μ a) (n a : ℝ≥0)⁻¹) (gaussianReal 0 (n a : ℝ≥0)⁻¹) := by
    constructor
    · exact (by fun_prop : Measurable (fun z : ℝ => μ a + z)).aemeasurable
    · simpa using gaussianReal_map_const_add (μ := (0 : ℝ)) (v := (n a : ℝ≥0)⁻¹) (μ a)
  exact hadd.fun_comp hm

theorem empirical_independent (T : ℕ) (μ : Means A) (n : A → ℕ) :
    iIndepFun (fun a (x : Arrays A T) => empirical μ n x a) (law T) := by
  exact iIndepFun_pi (fun a =>
    ((Draft0.FrequentistModel.sampleMean_continuous T (n a)).measurable.const_add
      (μ a)).aemeasurable)

def pairVariance (n : A → ℕ) (a b : A) : ℝ≥0 :=
  (n a : ℝ≥0)⁻¹ + (n b : ℝ≥0)⁻¹

theorem empirical_gap_law (T : ℕ) (μ : Means A) (n : A → ℕ) (a b : A)
    (hab : a ≠ b) (hn : ∀ a, 0 < n a) (hnT : ∀ a, n a ≤ T) :
    HasLaw (fun x : Arrays A T => empirical μ n x a - empirical μ n x b)
      (gaussianReal (μ a - μ b) (pairVariance n a b)) (law T) := by
  have ha := empirical_coordinate_law T μ n a (hn a) (hnT a)
  have hb := empirical_coordinate_law T μ n b (hn b) (hnT b)
  have hneg : HasLaw (fun z : ℝ => -z)
      (gaussianReal (-μ b) (n b : ℝ≥0)⁻¹) (gaussianReal (μ b) (n b : ℝ≥0)⁻¹) := by
    constructor
    · exact (by fun_prop : Measurable (fun z : ℝ => -z)).aemeasurable
    · simpa only [zero_sub] using
        gaussianReal_map_const_sub (μ := μ b) (v := (n b : ℝ≥0)⁻¹) 0
  have hind := (empirical_independent T μ n).indepFun hab
  have hind' : IndepFun (fun x : Arrays A T => empirical μ n x a)
      (fun x : Arrays A T => -empirical μ n x b) (law T) :=
    hind.comp measurable_id (by fun_prop)
  have hsum := hind'.hasLaw_fun_add ha (hneg.fun_comp hb)
  rw [gaussianReal_conv_gaussianReal] at hsum
  simpa only [sub_eq_add_neg, pairVariance] using hsum

def recommendation {T : ℕ} (μ : Means A) (n : A → ℕ) (x : Arrays A T) : A :=
  firstMax (empirical μ n x)

theorem recommendation_measurable (T : ℕ) (μ : Means A) (n : A → ℕ) :
    Measurable (recommendation (T := T) μ n) :=
  firstMax_measurable.comp (empirical_measurable T μ n)

theorem recommendation_maximal {T : ℕ} (μ : Means A) (n : A → ℕ) (x : Arrays A T)
    (a : A) : empirical μ n x a ≤ empirical μ n x (recommendation μ n x) := by
  change empirical μ n x a ≤ empirical μ n x (firstMax (empirical μ n x))
  rw [firstMax_attains (empirical μ n x)]
  exact coordinate_le_maxAll _ _

def error (T : ℕ) (μ : Means A) (best : A) (n : A → ℕ) : ℝ :=
  (law T).real {x | recommendation μ n x ≠ best}

def regret (T : ℕ) (μ : Means A) (best : A) (n : A → ℕ) : ℝ :=
  ∫ x, μ best - μ (recommendation μ n x) ∂law T

theorem strict_pair_implies_error {T : ℕ} (μ : Means A) (n : A → ℕ) (best a : A)
    (x : Arrays A T) (h : empirical μ n x best < empirical μ n x a) :
    recommendation μ n x ≠ best := by
  intro he
  have hm := recommendation_maximal μ n x a
  rw [he] at hm
  exact (not_lt_of_ge hm) h

theorem error_implies_weak_pair {T : ℕ} (μ : Means A) (n : A → ℕ) (best : A)
    (x : Arrays A T) (h : recommendation μ n x ≠ best) :
    ∃ a : {a : A // a ≠ best}, empirical μ n x best ≤ empirical μ n x a := by
  exact ⟨⟨recommendation μ n x, h⟩, recommendation_maximal μ n x best⟩

theorem error_lower_pair (T : ℕ) (μ : Means A) (n : A → ℕ) (best a : A) :
    (law T).real {x | empirical μ n x best < empirical μ n x a} ≤ error T μ best n := by
  exact measureReal_mono (fun _ h => strict_pair_implies_error μ n best a _ h)

theorem error_upper_pairs (T : ℕ) (μ : Means A) (n : A → ℕ) (best : A) :
    error T μ best n ≤ ∑ a : {a : A // a ≠ best},
      (law T).real {x | empirical μ n x best ≤ empirical μ n x a} := by
  have hset : {x : Arrays A T | recommendation μ n x ≠ best} ⊆
      ⋃ a : {a : A // a ≠ best}, {x | empirical μ n x best ≤ empirical μ n x a} := by
    intro x hx
    obtain ⟨a, ha⟩ := error_implies_weak_pair μ n best x hx
    exact mem_iUnion.mpr ⟨a, ha⟩
  exact (measureReal_mono hset).trans (measureReal_iUnion_fintype_le _)

end
end Junpei.UniformSampling
