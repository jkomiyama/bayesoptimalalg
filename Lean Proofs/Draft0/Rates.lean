import Mathlib.Analysis.Asymptotics.SpecificAsymptotics
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.Tactic

/-!
Corrected integer-budget asymptotic algebra for Appendix E.3 and the regret
step of Theorem 2. The hypotheses giving bounds on the actual EBI functions,
event probabilities, and simple regret are explicitly conditional. This file
does not claim those stochastic hypotheses have been established.
-/

open Filter Asymptotics

namespace Draft0.Rates

/-- Polynomial rate on the paper's natural-number budget domain. -/
noncomputable def invPower (k : ℕ) (T : ℕ) : ℝ := ((T : ℝ) ^ k)⁻¹

theorem invPower_nonneg (k T : ℕ) : 0 ≤ invPower k T := by
  exact inv_nonneg.mpr (pow_nonneg (Nat.cast_nonneg T) k)

theorem invPower_pos (k T : ℕ) (hT : 0 < T) : 0 < invPower k T := by
  exact inv_pos.mpr (pow_pos (Nat.cast_pos.mpr hT) k)

theorem invPower_antitone {small large T : ℕ} (h : small ≤ large) (hT : 1 ≤ T) :
    invPower large T ≤ invPower small T := by
  have hreal : (1 : ℝ) ≤ T := by exact_mod_cast hT
  have hpos : (0 : ℝ) < T := lt_of_lt_of_le zero_lt_one hreal
  exact (inv_le_inv₀ (pow_pos hpos large) (pow_pos hpos small)).mpr
    (pow_le_pow_right₀ hreal h)

theorem invPower_product (p q T : ℕ) :
    invPower p T * invPower q T = invPower (p + q) T := by
  simp only [invPower]
  rw [← mul_inv, ← pow_add]

/-- Exponents add: 6+110=116, not 110. -/
theorem corrected_g1_g3_exponent (T : ℕ) :
    invPower 6 T * invPower 110 T = invPower 116 T := by
  exact invPower_product 6 110 T

/-- Exponents add: 8+110=118, not 110. -/
theorem corrected_g2_g3_exponent (T : ℕ) :
    invPower 8 T * invPower 110 T = invPower 118 T := by
  exact invPower_product 8 110 T

theorem product_lower_bound {p q T : ℕ} {c d x y : ℝ}
    (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hx : c * invPower p T ≤ x) (hy : d * invPower q T ≤ y) :
    (c * d) * invPower (p + q) T ≤ x * y := by
  have hcP : 0 ≤ c * invPower p T := mul_nonneg hc (invPower_nonneg p T)
  have hdP : 0 ≤ d * invPower q T := mul_nonneg hd (invPower_nonneg q T)
  have h := mul_le_mul hx hy hdP (hcP.trans hx)
  calc
    (c * d) * invPower (p + q) T =
        (c * invPower p T) * (d * invPower q T) := by
      rw [← invPower_product]
      ring
    _ ≤ x * y := h

theorem eventual_product_lower_bound {p q : ℕ} {c d : ℝ} {f g : ℕ → ℝ}
    (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hf : ∀ᶠ T in atTop, c * invPower p T ≤ f T)
    (hg : ∀ᶠ T in atTop, d * invPower q T ≤ g T) :
    ∀ᶠ T in atTop, (c * d) * invPower (p + q) T ≤ f T * g T := by
  filter_upwards [hf, hg] with T hT kT
  exact product_lower_bound hc hd hT kT

theorem inverse_power_littleO_real {small large : ℕ} (h : small < large) :
    (fun t : ℝ => (t ^ large)⁻¹) =o[atTop] (fun t : ℝ => (t ^ small)⁻¹) := by
  apply (Asymptotics.isLittleO_pow_pow_atTop_of_lt (𝕜 := ℝ) h).inv_rev
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with t ht hz
  exact (pow_ne_zero small ht.ne' hz).elim

/-- Faster inverse powers are little-o of slower ones on integer budgets. -/
theorem inverse_power_littleO_nat {small large : ℕ} (h : small < large) :
    invPower large =o[atTop] invPower small := by
  exact (inverse_power_littleO_real h).comp_tendsto
    (tendsto_natCast_atTop_atTop : Tendsto (fun T : ℕ => (T : ℝ)) atTop atTop)

theorem inverse_power_not_bigOmega_nat {small large : ℕ} (h : small < large) :
    ¬ (invPower small =O[atTop] invPower large) := by
  apply (inverse_power_littleO_nat h).not_isBigO
  apply Filter.Eventually.frequently
  filter_upwards [eventually_gt_atTop (0 : ℕ)] with T hT
  exact (invPower_pos large T hT).ne'

/-- Integer-budget countermodels also reject the original exponent 110 inference. -/
theorem rate116_not_rate110 : ¬ (invPower 110 =O[atTop] invPower 116) := by
  exact inverse_power_not_bigOmega_nat (by norm_num)

theorem rate118_not_rate110 : ¬ (invPower 110 =O[atTop] invPower 118) := by
  exact inverse_power_not_bigOmega_nat (by norm_num)

/-- A faster upper error rate is negligible relative to a slower benefit rate. -/
theorem error_littleO {a k : ℕ} {g0 : ℕ → ℝ}
    (hka : k < a) (h0 : g0 =O[atTop] invPower a) :
    g0 =o[atTop] invPower k := by
  exact h0.trans_isLittleO (inverse_power_littleO_nat hka)

/--
Subtracting m*g0 preserves half the lower-bound constant, provided the error
has strictly faster polynomial decay. This repairs both -2*g0 and -4*g0.
-/
theorem eventual_subtraction_lower_bound {a k : ℕ} {c m : ℝ}
    {g0 benefit : ℕ → ℝ} (hc : 0 < c) (hm : 0 < m)
    (hka : k < a) (h0 : g0 =O[atTop] invPower a)
    (hbenefit : ∀ᶠ T in atTop, c * invPower k T ≤ benefit T) :
    ∀ᶠ T in atTop, (c / 2) * invPower k T ≤ benefit T - m * g0 T := by
  have hsmall : ∀ᶠ T in atTop,
      ‖g0 T‖ ≤ (c / (2 * m)) * ‖invPower k T‖ :=
    (error_littleO hka h0).def
      (div_pos hc (mul_pos (by norm_num : (0 : ℝ) < 2) hm))
  filter_upwards [hbenefit, hsmall] with T hT hsmallT
  have hnorm : ‖invPower k T‖ = invPower k T :=
    Real.norm_of_nonneg (invPower_nonneg k T)
  rw [hnorm] at hsmallT
  have hratio : m * (c / (2 * m)) = c / 2 := by
    field_simp
  have herror : m * g0 T ≤ (c / 2) * invPower k T := by
    calc
      m * g0 T ≤ m * ‖g0 T‖ :=
        mul_le_mul_of_nonneg_left (le_abs_self (g0 T)) hm.le
      _ ≤ m * (c / (2 * m) * invPower k T) :=
        mul_le_mul_of_nonneg_left hsmallT hm.le
      _ = (c / 2) * invPower k T := by rw [← mul_assoc, hratio]
  linarith

theorem eventual_subtraction_positive {a k : ℕ} {c m : ℝ}
    {g0 benefit : ℕ → ℝ} (hc : 0 < c) (hm : 0 < m)
    (hka : k < a) (h0 : g0 =O[atTop] invPower a)
    (hbenefit : ∀ᶠ T in atTop, c * invPower k T ≤ benefit T) :
    ∀ᶠ T in atTop, 0 < benefit T - m * g0 T := by
  filter_upwards [eventual_subtraction_lower_bound hc hm hka h0 hbenefit,
    eventually_gt_atTop (0 : ℕ)] with T hT hpos
  exact (mul_pos (half_pos hc) (invPower_pos k T hpos)).trans_le hT

/-- A concrete positive eventual lower bound implies the usual Omega relation. -/
theorem omega_of_eventual_lower_bound {k : ℕ} {c : ℝ} {f : ℕ → ℝ}
    (hc : 0 < c) (hf : ∀ᶠ T in atTop, c * invPower k T ≤ f T) :
    invPower k =O[atTop] f := by
  apply IsBigOWith.isBigO (c := c⁻¹)
  apply IsBigOWith.of_bound
  filter_upwards [hf] with T hT
  rw [Real.norm_of_nonneg (invPower_nonneg k T)]
  have hfpos : 0 ≤ f T := (mul_nonneg hc.le (invPower_nonneg k T)).trans hT
  rw [Real.norm_of_nonneg hfpos]
  calc
    invPower k T = c⁻¹ * (c * invPower k T) := by
      rw [← mul_assoc, inv_mul_cancel₀ hc.ne', one_mul]
    _ ≤ c⁻¹ * f T := mul_le_mul_of_nonneg_left hT (inv_nonneg.mpr hc.le)

theorem subtraction_preserves_omega {a k : ℕ} {c m : ℝ}
    {g0 benefit : ℕ → ℝ} (hc : 0 < c) (hm : 0 < m)
    (hka : k < a) (h0 : g0 =O[atTop] invPower a)
    (hbenefit : ∀ᶠ T in atTop, c * invPower k T ≤ benefit T) :
    invPower k =O[atTop] (fun T => benefit T - m * g0 T) := by
  exact omega_of_eventual_lower_bound (half_pos hc)
    (eventual_subtraction_lower_bound hc hm hka h0 hbenefit)

/-- Explicit corrected first EBI comparison, conditional on its component bounds. -/
theorem corrected_g1_margin {a : ℕ} {c1 c3 : ℝ} {g0 g1 g3 : ℕ → ℝ}
    (hc1 : 0 < c1) (hc3 : 0 < c3) (ha : 116 < a)
    (h0 : g0 =O[atTop] invPower a)
    (h1 : ∀ᶠ T in atTop, c1 * invPower 6 T ≤ g1 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 110 T ≤ g3 T) :
    ∀ᶠ T in atTop,
      ((c1 * c3) / 2) * invPower 116 T ≤ g1 T * g3 T - 2 * g0 T := by
  exact eventual_subtraction_lower_bound (mul_pos hc1 hc3) (by norm_num) ha h0
    (eventual_product_lower_bound hc1.le hc3.le h1 h3)

/-- Explicit corrected second EBI comparison, conditional on its component bounds. -/
theorem corrected_g2_margin {a : ℕ} {c2 c3 : ℝ} {g0 g2 g3 : ℕ → ℝ}
    (hc2 : 0 < c2) (hc3 : 0 < c3) (ha : 118 < a)
    (h0 : g0 =O[atTop] invPower a)
    (h2 : ∀ᶠ T in atTop, c2 * invPower 8 T ≤ g2 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 110 T ≤ g3 T) :
    ∀ᶠ T in atTop,
      ((c2 * c3) / 2) * invPower 118 T ≤ g2 T * g3 T - 4 * g0 T := by
  exact eventual_subtraction_lower_bound (mul_pos hc2 hc3) (by norm_num) ha h0
    (eventual_product_lower_bound hc2.le hc3.le h2 h3)

/-- The first margin also admits the conservative exponent 118. -/
theorem corrected_g1_margin_conservative {a : ℕ} {c1 c3 : ℝ} {g0 g1 g3 : ℕ → ℝ}
    (hc1 : 0 < c1) (hc3 : 0 < c3) (ha : 116 < a)
    (h0 : g0 =O[atTop] invPower a)
    (h1 : ∀ᶠ T in atTop, c1 * invPower 6 T ≤ g1 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 110 T ≤ g3 T) :
    ∀ᶠ T in atTop,
      ((c1 * c3) / 2) * invPower 118 T ≤ g1 T * g3 T - 2 * g0 T := by
  filter_upwards [corrected_g1_margin hc1 hc3 ha h0 h1 h3,
    eventually_ge_atTop (1 : ℕ)] with T hT hpos
  exact (mul_le_mul_of_nonneg_left (invPower_antitone (by norm_num) hpos)
    (half_pos (mul_pos hc1 hc3)).le).trans hT

/-- Both corrected differences are eventually strictly positive. -/
theorem corrected_margins_positive {a : ℕ} {c1 c2 c3 : ℝ}
    {g0 g1 g2 g3 : ℕ → ℝ}
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hc3 : 0 < c3) (ha : 118 < a)
    (h0 : g0 =O[atTop] invPower a)
    (h1 : ∀ᶠ T in atTop, c1 * invPower 6 T ≤ g1 T)
    (h2 : ∀ᶠ T in atTop, c2 * invPower 8 T ≤ g2 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 110 T ≤ g3 T) :
    ∀ᶠ T in atTop, 0 < g1 T * g3 T - 2 * g0 T ∧
      0 < g2 T * g3 T - 4 * g0 T := by
  have h116 : 116 < a := by omega
  filter_upwards [eventual_subtraction_positive (mul_pos hc1 hc3)
      (by norm_num : (0 : ℝ) < 2) h116 h0
      (eventual_product_lower_bound hc1.le hc3.le h1 h3),
    eventual_subtraction_positive (mul_pos hc2 hc3)
      (by norm_num : (0 : ℝ) < 4) ha h0
      (eventual_product_lower_bound hc2.le hc3.le h2 h3)] with T hleft hright
  exact ⟨hleft, hright⟩

/-- Conditional transfer from the margins to the actual EBI comparisons. -/
theorem corrected_EBI_comparisons {a : ℕ} {c1 c2 c3 : ℝ}
    {g0 g1 g2 g3 leftComparison rightComparison : ℕ → ℝ}
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hc3 : 0 < c3) (ha : 118 < a)
    (h0 : g0 =O[atTop] invPower a)
    (h1 : ∀ᶠ T in atTop, c1 * invPower 6 T ≤ g1 T)
    (h2 : ∀ᶠ T in atTop, c2 * invPower 8 T ≤ g2 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 110 T ≤ g3 T)
    (hleft : ∀ᶠ T in atTop, g1 T * g3 T - 2 * g0 T ≤ leftComparison T)
    (hright : ∀ᶠ T in atTop, g2 T * g3 T - 4 * g0 T ≤ rightComparison T) :
    ∀ᶠ T in atTop, 0 < leftComparison T ∧ 0 < rightComparison T := by
  filter_upwards [corrected_margins_positive hc1 hc2 hc3 ha h0 h1 h2 h3,
    hleft, hright] with T hT hl hr
  exact ⟨hT.1.trans_le hl, hT.2.trans_le hr⟩

/-- Choosing CU=30 leaves the original exponent expression above 118. -/
theorem concrete_CU_rate : (118 : ℝ) < ((30 : ℝ) - 6) ^ 2 / 4 := by
  norm_num

theorem concrete_CU_exponent : ((30 : ℝ) - 6) ^ 2 / 4 = (144 : ℝ) := by
  norm_num

theorem concrete_CU_nat_rate : (118 : ℕ) < 144 := by
  norm_num

/-- The replacement nonterminal probability exponent 206 gives margin 212. -/
theorem repaired_g1_g3_exponent (T : ℕ) :
    invPower 6 T * invPower 206 T = invPower 212 T := by
  exact invPower_product 6 206 T

theorem repaired_g2_g3_exponent (T : ℕ) :
    invPower 8 T * invPower 206 T = invPower 214 T := by
  exact invPower_product 8 206 T

/-- CU=40 accommodates both margins for the replacement probability bound. -/
theorem repaired_CU_exponent : ((40 : ℝ) - 6) ^ 2 / 4 = (289 : ℝ) := by
  norm_num

theorem repaired_CU_nat_rate : (214 : ℕ) < 289 := by
  norm_num

theorem repaired_g1_margin {a : ℕ} {c1 c3 : ℝ} {g0 g1 g3 : ℕ → ℝ}
    (hc1 : 0 < c1) (hc3 : 0 < c3) (ha : 212 < a)
    (h0 : g0 =O[atTop] invPower a)
    (h1 : ∀ᶠ T in atTop, c1 * invPower 6 T ≤ g1 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 206 T ≤ g3 T) :
    ∀ᶠ T in atTop,
      ((c1 * c3) / 2) * invPower 212 T ≤ g1 T * g3 T - 2 * g0 T := by
  exact eventual_subtraction_lower_bound (mul_pos hc1 hc3) (by norm_num) ha h0
    (eventual_product_lower_bound hc1.le hc3.le h1 h3)

theorem repaired_g2_margin {a : ℕ} {c2 c3 : ℝ} {g0 g2 g3 : ℕ → ℝ}
    (hc2 : 0 < c2) (hc3 : 0 < c3) (ha : 214 < a)
    (h0 : g0 =O[atTop] invPower a)
    (h2 : ∀ᶠ T in atTop, c2 * invPower 8 T ≤ g2 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 206 T ≤ g3 T) :
    ∀ᶠ T in atTop,
      ((c2 * c3) / 2) * invPower 214 T ≤ g2 T * g3 T - 4 * g0 T := by
  exact eventual_subtraction_lower_bound (mul_pos hc2 hc3) (by norm_num) ha h0
    (eventual_product_lower_bound hc2.le hc3.le h2 h3)

theorem repaired_margins_positive {a : ℕ} {c1 c2 c3 : ℝ}
    {g0 g1 g2 g3 : ℕ → ℝ}
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hc3 : 0 < c3) (ha : 214 < a)
    (h0 : g0 =O[atTop] invPower a)
    (h1 : ∀ᶠ T in atTop, c1 * invPower 6 T ≤ g1 T)
    (h2 : ∀ᶠ T in atTop, c2 * invPower 8 T ≤ g2 T)
    (h3 : ∀ᶠ T in atTop, c3 * invPower 206 T ≤ g3 T) :
    ∀ᶠ T in atTop, 0 < g1 T * g3 T - 2 * g0 T ∧
      0 < g2 T * g3 T - 4 * g0 T := by
  have h212 : 212 < a := by omega
  filter_upwards [eventual_subtraction_positive (mul_pos hc1 hc3)
      (by norm_num : (0 : ℝ) < 2) h212 h0
      (eventual_product_lower_bound hc1.le hc3.le h1 h3),
    eventual_subtraction_positive (mul_pos hc2 hc3)
      (by norm_num : (0 : ℝ) < 4) ha h0
      (eventual_product_lower_bound hc2.le hc3.le h2 h3)] with T hleft hright
  exact ⟨hleft, hright⟩

/-- Simple regret retains the positive reward gap multiplying error probability. -/
theorem corrected_regret_lower_bound {gap probability eventProb regret : ℝ}
    (hgap : 0 ≤ gap) (hregret : gap * probability ≤ regret)
    (hevent : eventProb ≤ probability) : gap * eventProb ≤ regret := by
  exact (mul_le_mul_of_nonneg_left hevent hgap).trans hregret

theorem eventual_regret_rate {k : ℕ} {gap c : ℝ}
    {probability eventProb regret : ℕ → ℝ}
    (hgap : 0 < gap)
    (hregret : ∀ᶠ T in atTop, gap * probability T ≤ regret T)
    (hevent : ∀ᶠ T in atTop, eventProb T ≤ probability T)
    (hrate : ∀ᶠ T in atTop, c * invPower k T ≤ eventProb T) :
    ∀ᶠ T in atTop, (gap * c) * invPower k T ≤ regret T := by
  filter_upwards [hregret, hevent, hrate] with T hr he hp
  calc
    (gap * c) * invPower k T = gap * (c * invPower k T) := mul_assoc _ _ _
    _ ≤ gap * eventProb T := mul_le_mul_of_nonneg_left hp hgap.le
    _ ≤ regret T := corrected_regret_lower_bound hgap.le hr he

theorem regret_omega {k : ℕ} {gap c : ℝ}
    {probability eventProb regret : ℕ → ℝ}
    (hgap : 0 < gap) (hc : 0 < c)
    (hregret : ∀ᶠ T in atTop, gap * probability T ≤ regret T)
    (hevent : ∀ᶠ T in atTop, eventProb T ≤ probability T)
    (hrate : ∀ᶠ T in atTop, c * invPower k T ≤ eventProb T) :
    invPower k =O[atTop] regret := by
  exact omega_of_eventual_lower_bound (mul_pos hgap hc)
    (eventual_regret_rate hgap hregret hevent hrate)

section ExpectedRegret

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]

/-- Expected loss when every incorrect recommendation incurs the same gap. -/
noncomputable def fixedGapRegret (μ : Measure Ω) (wrong : Set Ω) (gap : ℝ) : ℝ :=
  ∫ outcome, wrong.indicator (fun _ => gap) outcome ∂μ

/-- The expectation itself has the factor gap, using the actual Bochner integral. -/
theorem fixedGapRegret_eq (μ : Measure Ω) {wrong : Set Ω}
    (hwrong : MeasurableSet wrong) (gap : ℝ) :
    fixedGapRegret μ wrong gap = gap * μ.real wrong := by
  rw [fixedGapRegret, integral_indicator_const gap hwrong, smul_eq_mul, mul_comm]

/-- An error-forcing event gives a gap-scaled regret lower bound. -/
theorem fixedGapRegret_event_lower_bound (μ : Measure Ω) [IsProbabilityMeasure μ]
    {wrong event : Set Ω} (hwrong : MeasurableSet wrong)
    (hevent : event ⊆ wrong) {gap : ℝ} (hgap : 0 ≤ gap) :
    gap * μ.real event ≤ fixedGapRegret μ wrong gap := by
  rw [fixedGapRegret_eq μ hwrong gap]
  exact mul_le_mul_of_nonneg_left (measureReal_mono hevent) hgap

end ExpectedRegret

end Draft0.Rates
