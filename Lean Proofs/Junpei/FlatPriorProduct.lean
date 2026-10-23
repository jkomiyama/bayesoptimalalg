import Junpei.FlatPriorInitialization
import Mathlib.MeasureTheory.Integral.Pi

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FlatPriorProduct

noncomputable section
set_option linter.unusedSectionVars false

open Junpei.FlatPriorInitialization Junpei.FiniteGaussian Junpei.FinitePosterior

def observationLikelihood {n : ℕ} (y : Fin n → ℝ) (θ : ℝ) : ℝ :=
  ∏ j, gaussianPDFReal θ 1 (y j)

theorem observationLikelihood_integral {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) :
    (∫ θ, observationLikelihood y θ) =
      (Real.sqrt (2 * Real.pi))⁻¹ ^ n * flatNormalizer y := by
  unfold observationLikelihood
  simp_rw [unitGaussian_likelihood]
  rw [integral_const_mul, flatLikelihood_integral y hn]

theorem normalized_observationLikelihood {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) (θ : ℝ) :
    observationLikelihood y θ / (∫ x, observationLikelihood y x) =
      gaussianPDFReal (empiricalMean y) (n : ℝ≥0)⁻¹ θ := by
  rw [observationLikelihood_integral y hn]
  unfold observationLikelihood
  rw [unitGaussian_likelihood, flatLikelihood_factorization y hn]
  have hs : (Real.sqrt (2 * Real.pi))⁻¹ ^ n ≠ 0 := by positivity
  field_simp [hs, ne_of_gt (flatNormalizer_pos y hn)]

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

def productLikelihood (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) (θ : Means A) : ℝ :=
  ∏ a, flatLikelihood (y a) (θ a)

theorem productLikelihood_integrable (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ)
    (hn : ∀ a, 0 < n a) : Integrable (productLikelihood n y) volume :=
  Integrable.fintype_prod (fun a => flatLikelihood_integrable (y a) (hn a))

theorem productLikelihood_integral (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ)
    (hn : ∀ a, 0 < n a) :
    (∫ θ, productLikelihood n y θ) = ∏ a, flatNormalizer (y a) := by
  unfold productLikelihood
  rw [integral_fintype_prod_volume_eq_prod]
  exact Finset.prod_congr rfl (fun a _ => flatLikelihood_integral (y a) (hn a))

theorem productLikelihood_integral_pos (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ)
    (hn : ∀ a, 0 < n a) : 0 < ∫ θ, productLikelihood n y θ := by
  rw [productLikelihood_integral n y hn]
  exact Finset.prod_pos (fun a _ => flatNormalizer_pos (y a) (hn a))

theorem normalized_productLikelihood (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ)
    (hn : ∀ a, 0 < n a) (θ : Means A) :
    productLikelihood n y θ / (∫ x, productLikelihood n y x) =
      ∏ a, gaussianPDFReal (empiricalMean (y a)) (n a : ℝ≥0)⁻¹ (θ a) := by
  rw [productLikelihood_integral n y hn]
  unfold productLikelihood
  simp_rw [flatLikelihood_factorization _ (hn _)]
  rw [Finset.prod_mul_distrib]
  field_simp [ne_of_gt (Finset.prod_pos (fun a _ => flatNormalizer_pos (y a) (hn a)))]

/-- The Gaussian product posterior really has its product likelihood density
with respect to the flat product Lebesgue prior. -/
theorem posterior_density (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a) :
    posteriorLaw c m = volume.withDensity
      (fun θ => ENNReal.ofReal (∏ a, gaussianPDFReal (m a) (c a)⁻¹ (θ a))) := by
  rw [Junpei.FiniteModelExclusion.posteriorLaw_eq_pi]
  apply Measure.pi_eq
  intro s hs
  have hprod : Integrable
      (fun θ : Means A => ∏ a, gaussianPDFReal (m a) (c a)⁻¹ (θ a)) volume :=
    Integrable.fintype_prod (fun a => integrable_gaussianPDFReal _ _)
  have hnonneg (θ : Means A) : 0 ≤ ∏ a, gaussianPDFReal (m a) (c a)⁻¹ (θ a) :=
    Finset.prod_nonneg (fun _ _ => gaussianPDFReal_nonneg _ _ _)
  rw [withDensity_apply _ (MeasurableSet.univ_pi hs),
    ← ofReal_integral_eq_lintegral_ofReal hprod.restrict
      (Filter.Eventually.of_forall hnonneg)]
  change ENNReal.ofReal (∫ θ, (∏ a, gaussianPDFReal (m a) (c a)⁻¹ (θ a))
    ∂((Measure.pi (fun _ : A => (volume : Measure ℝ))).restrict (Set.univ.pi s))) = _
  rw [Measure.restrict_pi_pi, integral_fintype_prod_eq_prod,
    ENNReal.ofReal_prod_of_nonneg]
  · apply Finset.prod_congr rfl
    intro a _
    exact (gaussianReal_apply_eq_integral (m a)
      (inv_ne_zero (ne_of_gt (hc a))) (s a)).symm
  · intro a _
    exact integral_nonneg (gaussianPDFReal_nonneg _ _)

/-- End-to-end initialization from the improper flat prior: any strictly
positive count vector, and every observed data set, gives the exact proper
independent Gaussian posterior used by the Bellman model. -/
theorem normalized_product_measure (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ)
    (hn : ∀ a, 0 < n a) :
    volume.withDensity (fun θ => ENNReal.ofReal
      (productLikelihood n y θ / (∫ x, productLikelihood n y x))) =
        posteriorLaw (fun a => (n a : ℝ≥0)) (fun a => empiricalMean (y a)) := by
  simp_rw [normalized_productLikelihood n y hn]
  exact (posterior_density _ _ (fun a => by exact_mod_cast hn a)).symm

theorem normalized_product_sufficient (n : A → ℕ)
    (y z : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a)
    (hmean : ∀ a, empiricalMean (y a) = empiricalMean (z a)) :
    (fun θ => productLikelihood n y θ / (∫ x, productLikelihood n y x)) =
      (fun θ => productLikelihood n z θ / (∫ x, productLikelihood n z x)) := by
  funext θ
  simp_rw [normalized_productLikelihood n _ hn, hmean]

end
end Junpei.FlatPriorProduct
