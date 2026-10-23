import Junpei.FlatPriorProduct

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.AdaptivePosteriorSufficiency

noncomputable section
set_option linter.unusedSectionVars false

open Junpei.FlatPriorInitialization Junpei.FlatPriorProduct
open Junpei.FiniteGaussian Junpei.FinitePosterior

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

abbrev Fiber {N : ℕ} (arms : Fin N → A) (a : A) := {j : Fin N // arms j = a}

def sampleCount {N : ℕ} (arms : Fin N → A) (a : A) : ℕ := Fintype.card (Fiber arms a)

def sampleData {N : ℕ} (arms : Fin N → A) (y : Fin N → ℝ) (a : A) :
    Fin (sampleCount arms a) → ℝ :=
  fun j => y (((Fintype.equivFin (Fiber arms a)).symm j).val)

def likelihood {N : ℕ} (arms : Fin N → A) (y : Fin N → ℝ) (θ : Means A) : ℝ :=
  ∏ j, gaussianPDFReal (θ (arms j)) 1 (y j)

def likelihoodConstant {N : ℕ} (arms : Fin N → A) : ℝ :=
  ∏ a, (Real.sqrt (2 * Real.pi))⁻¹ ^ sampleCount arms a

theorem sampleCount_pos {N : ℕ} (arms : Fin N → A)
    (hinit : ∀ a, ∃ j, arms j = a) (a : A) : 0 < sampleCount arms a := by
  obtain ⟨j, hj⟩ := hinit a
  exact Fintype.card_pos_iff.mpr ⟨⟨j, hj⟩⟩

theorem likelihood_by_arm {N : ℕ} (arms : Fin N → A) (y : Fin N → ℝ) (θ : Means A) :
    likelihood arms y θ =
      ∏ a, observationLikelihood (sampleData arms y a) (θ a) := by
  unfold likelihood
  rw [← Fintype.prod_fiberwise arms (fun j => gaussianPDFReal (θ (arms j)) 1 (y j))]
  apply Finset.prod_congr rfl
  intro a _
  have he := (Fintype.equivFin (Fiber arms a)).symm.prod_comp
    (fun j : Fiber arms a => gaussianPDFReal (θ a) 1 (y j.val))
  change (∏ j : Fiber arms a, gaussianPDFReal (θ (arms j.val)) 1 (y j.val)) = _
  calc
    _ = ∏ j : Fiber arms a, gaussianPDFReal (θ a) 1 (y j.val) := by
      apply Finset.prod_congr rfl
      intro j _
      rw [j.property]
    _ = _ := he.symm

/-- Regrouping the observation likelihood is valid for every realized
sequence of chosen arms, including sequences chosen from earlier rewards. -/
theorem likelihood_factorization {N : ℕ} (arms : Fin N → A) (y : Fin N → ℝ) (θ : Means A) :
    likelihood arms y θ = likelihoodConstant arms *
      productLikelihood (sampleCount arms) (sampleData arms y) θ := by
  rw [likelihood_by_arm]
  unfold observationLikelihood
  simp_rw [unitGaussian_likelihood]
  exact Finset.prod_mul_distrib

theorem likelihoodConstant_pos {N : ℕ} (arms : Fin N → A) :
    0 < likelihoodConstant arms := by
  unfold likelihoodConstant
  apply Finset.prod_pos
  intro a _
  positivity

theorem normalized_likelihood {N : ℕ} (arms : Fin N → A) (y : Fin N → ℝ)
    (hinit : ∀ a, ∃ j, arms j = a) (θ : Means A) :
    likelihood arms y θ / (∫ x, likelihood arms y x) =
      ∏ a, gaussianPDFReal (empiricalMean (sampleData arms y a))
        (sampleCount arms a : ℝ≥0)⁻¹ (θ a) := by
  have hn := sampleCount_pos arms hinit
  simp_rw [likelihood_factorization]
  rw [integral_const_mul]
  have hc := ne_of_gt (likelihoodConstant_pos arms)
  rw [mul_div_mul_left _ _ hc]
  exact normalized_productLikelihood _ _ hn θ

/-- Normalizing the Gaussian likelihood of the labelled reward vector gives
the proper product Gaussian measure. This statement concerns the explicitly
defined likelihood; it does not assert a separate conditional-distribution
theorem for an arbitrary adaptive-history law. Actual adaptive Gaussian
observation disintegration is established in ObservationBridge. -/
theorem normalized_history_posterior {N : ℕ} (arms : Fin N → A) (y : Fin N → ℝ)
    (hinit : ∀ a, ∃ j, arms j = a) :
    volume.withDensity (fun θ => ENNReal.ofReal
      (likelihood arms y θ / (∫ x, likelihood arms y x))) =
      posteriorLaw (fun a => (sampleCount arms a : ℝ≥0))
        (fun a => empiricalMean (sampleData arms y a)) := by
  simp_rw [normalized_likelihood arms y hinit]
  exact (posterior_density _ _ (fun a => by
    exact_mod_cast sampleCount_pos arms hinit a)).symm

/-- A positive, parameter-independent policy-density factor cancels out of
the posterior. This is the likelihood cancellation used for adaptive rules. -/
theorem parameter_independent_factor_cancels {N : ℕ} (arms : Fin N → A) (y : Fin N → ℝ)
    (hinit : ∀ a, ∃ j, arms j = a) (p : ℝ) (hp : 0 < p) (θ : Means A) :
    (p * likelihood arms y θ) / (∫ x, p * likelihood arms y x) =
      ∏ a, gaussianPDFReal (empiricalMean (sampleData arms y a))
        (sampleCount arms a : ℝ≥0)⁻¹ (θ a) := by
  rw [integral_const_mul, mul_div_mul_left _ _ (ne_of_gt hp)]
  exact normalized_likelihood arms y hinit θ

end
end Junpei.AdaptivePosteriorSufficiency
