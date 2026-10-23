import Junpei.JointGaussianMeasurable

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FlatPriorInitialization

noncomputable section

def empiricalMean {n : ℕ} (y : Fin n → ℝ) : ℝ := (∑ j, y j) / n

def residual {n : ℕ} (y : Fin n → ℝ) : ℝ := ∑ j, (y j - empiricalMean y)^2

/-- The likelihood with its parameter-independent unit-Gaussian normalizing
constant omitted. The prior density with respect to Lebesgue measure is one. -/
def flatLikelihood {n : ℕ} (y : Fin n → ℝ) (θ : ℝ) : ℝ :=
  Real.exp (-(∑ j, (y j - θ)^2) / 2)

def flatNormalizer {n : ℕ} (y : Fin n → ℝ) : ℝ :=
  Real.exp (-residual y / 2) * Real.sqrt (2 * Real.pi * ((n : ℝ)⁻¹))

theorem empiricalMean_sum {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) :
    (n : ℝ) * empiricalMean y = ∑ j, y j := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt hn)
  unfold empiricalMean
  field_simp

theorem sum_squares_decomposition {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) (θ : ℝ) :
    (∑ j, (y j - θ)^2) = residual y + n * (θ - empiricalMean y)^2 := by
  have hm := empiricalMean_sum y hn
  unfold residual
  simp only [sub_sq, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  simp only [← Finset.sum_mul, ← Finset.mul_sum]
  rw [← hm]
  ring

theorem flatNormalizer_pos {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) :
    0 < flatNormalizer y := by
  unfold flatNormalizer
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  positivity

/-- Completing the square identifies the whole normalized flat-prior
likelihood, not just its first two moments. -/
theorem flatLikelihood_factorization {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) (θ : ℝ) :
    flatLikelihood y θ = flatNormalizer y *
      gaussianPDFReal (empiricalMean y) (n : ℝ≥0)⁻¹ θ := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hs : Real.sqrt (2 * Real.pi * (n : ℝ)⁻¹) ≠ 0 := by positivity
  unfold flatLikelihood flatNormalizer gaussianPDFReal
  rw [sum_squares_decomposition y hn θ]
  simp only [NNReal.coe_inv, NNReal.coe_natCast]
  have he : -(residual y + (n : ℝ) * (θ - empiricalMean y)^2) / 2 =
      -residual y / 2 + -(θ - empiricalMean y)^2 / (2 * (n : ℝ)⁻¹) := by
    field_simp
    ring
  rw [he, Real.exp_add]
  field_simp

theorem flatLikelihood_integrable {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) :
    Integrable (flatLikelihood y) volume := by
  have he : flatLikelihood y = fun θ => flatNormalizer y *
      gaussianPDFReal (empiricalMean y) (n : ℝ≥0)⁻¹ θ :=
    funext (flatLikelihood_factorization y hn)
  rw [he]
  exact (integrable_gaussianPDFReal _ _).const_mul _

theorem flatLikelihood_integral {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) :
    (∫ θ, flatLikelihood y θ) = flatNormalizer y := by
  simp_rw [flatLikelihood_factorization y hn]
  rw [integral_const_mul, integral_gaussianPDFReal_eq_one]
  · ring
  · exact inv_ne_zero (by exact_mod_cast (ne_of_gt hn))

/-- Every nonempty observed sample gives exactly the proper Gaussian
posterior N(sample mean, reciprocal sample count). -/
theorem normalized_flatLikelihood {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) (θ : ℝ) :
    flatLikelihood y θ / (∫ x, flatLikelihood y x) =
      gaussianPDFReal (empiricalMean y) (n : ℝ≥0)⁻¹ θ := by
  rw [flatLikelihood_integral y hn, flatLikelihood_factorization y hn]
  field_simp [ne_of_gt (flatNormalizer_pos y hn)]

theorem normalized_flat_measure {n : ℕ} (y : Fin n → ℝ) (hn : 0 < n) :
    volume.withDensity (fun θ => ENNReal.ofReal
      (flatLikelihood y θ / (∫ x, flatLikelihood y x))) =
        gaussianReal (empiricalMean y) (n : ℝ≥0)⁻¹ := by
  simp_rw [normalized_flatLikelihood y hn]
  exact (gaussianReal_of_var_ne_zero _ (inv_ne_zero (by
    exact_mod_cast (ne_of_gt hn)))).symm

/-- Restoring all unit-Gaussian observation constants changes the likelihood
by a strictly positive scalar independent of the unknown mean. -/
theorem unitGaussian_likelihood {n : ℕ} (y : Fin n → ℝ) (θ : ℝ) :
    (∏ j, gaussianPDFReal θ 1 (y j)) =
      (Real.sqrt (2 * Real.pi))⁻¹ ^ n * flatLikelihood y θ := by
  simp only [gaussianPDFReal, NNReal.coe_one, mul_one]
  rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin,
    ← Real.exp_sum]
  congr 2
  rw [← Finset.sum_div, ← Finset.sum_neg_distrib]

end
end Junpei.FlatPriorInitialization
