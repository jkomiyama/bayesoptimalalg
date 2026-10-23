import Draft0.MainTheorem

set_option autoImplicit false
set_option warningAsError true

/-!
Limited comparison checks for junpei_version.pdf. These prove the revised
rate algebra conditional on its stated component bounds. They do not
formalize or certify the revised probabilistic or policy lemmas.
-/

open Filter Asymptotics
open scoped NNReal

namespace NewVersionComparison.RateChecks

open Draft0.Rates

theorem revised_product176 (T : ℕ) :
    invPower 6 T * invPower 170 T = invPower 176 T := by
  exact invPower_product 6 170 T

theorem revised_product178 (T : ℕ) :
    invPower 8 T * invPower 170 T = invPower 178 T := by
  exact invPower_product 8 170 T

theorem cutoff33_satisfies_revised_threshold :
    (180 : ℝ) < ((33 : ℝ) - 6) ^ 2 / 4 := by norm_num

theorem all_cutoffs_ge33_satisfy_revised_threshold (CU : ℝ) (hCU : 33 ≤ CU) :
    180 < (CU - 6) ^ 2 / 4 := by nlinarith

theorem revised_interval_probability_conditional (T : ℕ) (delta : ℝ) (variance : ℝ≥0)
    (hT : 5 ≤ T) (hv : 0 < variance) (hvu : (variance : ℝ) ≤ 2)
    (hvl : invPower 2 T ≤ (variance : ℝ))
    (hdelta : delta ^ 2 ≤ 160 * Real.log (T : ℝ) * (variance : ℝ)) :
    Draft0.PredictiveBound.lowerConstant * invPower 170 T ≤
      (ProbabilityTheory.gaussianReal delta variance
        (Set.Icc (invPower 2 T) (2 * invPower 2 T))).toReal := by
  exact (mul_le_mul_of_nonneg_left
    (invPower_antitone (by norm_num : 162 ≤ 170) (by omega : 1 ≤ T))
    Draft0.PredictiveBound.lowerConstant_pos.le).trans
      (Draft0.PredictiveBound.gaussian_interval_probability_lower_nat 160 T
        hT hv hvu hvl hdelta)

theorem revised_first_margin {c1 c3 : ℝ} {g0 g1 g3 : ℕ → ℝ}
    (hc1 : 0 < c1) (hc3 : 0 < c3)
    (h0 : g0 =o[atTop] invPower 180)
    (h1 : ∀ᶠ T in atTop, c1 * invPower 6 T ≤ g1 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 170 T ≤ g3 T) :
    ∀ᶠ T in atTop, ((c1 * c3) / 2) * invPower 176 T ≤
      g1 T * g3 T - 2 * g0 T := by
  exact eventual_subtraction_lower_bound (mul_pos hc1 hc3) (by norm_num)
    (by norm_num : 176 < 180) h0.isBigO
    (eventual_product_lower_bound hc1.le hc3.le h1 h3)

theorem revised_second_margin {c2 c3 : ℝ} {g0 g2 g3 : ℕ → ℝ}
    (hc2 : 0 < c2) (hc3 : 0 < c3)
    (h0 : g0 =o[atTop] invPower 180)
    (h2 : ∀ᶠ T in atTop, c2 * invPower 8 T ≤ g2 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 170 T ≤ g3 T) :
    ∀ᶠ T in atTop, ((c2 * c3) / 2) * invPower 178 T ≤
      g2 T * g3 T - 4 * g0 T := by
  exact eventual_subtraction_lower_bound (mul_pos hc2 hc3) (by norm_num)
    (by norm_num : 178 < 180) h0.isBigO
    (eventual_product_lower_bound hc2.le hc3.le h2 h3)

theorem revised_margins_positive {c1 c2 c3 : ℝ} {g0 g1 g2 g3 : ℕ → ℝ}
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hc3 : 0 < c3)
    (h0 : g0 =o[atTop] invPower 180)
    (h1 : ∀ᶠ T in atTop, c1 * invPower 6 T ≤ g1 T)
    (h2 : ∀ᶠ T in atTop, c2 * invPower 8 T ≤ g2 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 170 T ≤ g3 T) :
    ∀ᶠ T in atTop, 0 < g1 T * g3 T - 2 * g0 T ∧
      0 < g2 T * g3 T - 4 * g0 T := by
  filter_upwards [eventual_subtraction_positive (mul_pos hc1 hc3)
      (by norm_num : (0 : ℝ) < 2) (by norm_num : 176 < 180) h0.isBigO
      (eventual_product_lower_bound hc1.le hc3.le h1 h3),
    eventual_subtraction_positive (mul_pos hc2 hc3)
      (by norm_num : (0 : ℝ) < 4) (by norm_num : 178 < 180) h0.isBigO
      (eventual_product_lower_bound hc2.le hc3.le h2 h3)] with T hfirst hsecond
  exact ⟨hfirst, hsecond⟩

theorem revised176_can_be_weakened_to178 {T : ℕ} (hT : 1 ≤ T) :
    invPower 178 T ≤ invPower 176 T := by
  exact invPower_antitone (by norm_num) hT

theorem revised166_can_be_weakened_to170 {T : ℕ} (hT : 1 ≤ T) :
    invPower 170 T ≤ invPower 166 T := by
  exact invPower_antitone (by norm_num) hT

/-- Existing draft0 exponent 206 alone does not supply revised exponent 170. -/
theorem rate206_does_not_supply170 : ¬ (invPower 170 =O[atTop] invPower 206) := by
  exact inverse_power_not_bigOmega_nat (by norm_num)

theorem rate212_does_not_supply176 : ¬ (invPower 176 =O[atTop] invPower 212) := by
  exact inverse_power_not_bigOmega_nat (by norm_num)

theorem rate214_does_not_supply178 : ¬ (invPower 178 =O[atTop] invPower 214) := by
  exact inverse_power_not_bigOmega_nat (by norm_num)

theorem first_arm_positive_advantage_le_exclusion (mu1 mu2 mu3 : ℝ) :
    max (mu3 - mu1) 0 ≤ max (mu3 - min mu1 mu2) 0 := by
  exact max_le_max (sub_le_sub_left (min_le_left mu1 mu2) mu3) le_rfl

#print axioms revised_margins_positive
#print axioms revised_interval_probability_conditional
#print axioms Draft0.MainTheorem.end_to_end_certificate

end NewVersionComparison.RateChecks
