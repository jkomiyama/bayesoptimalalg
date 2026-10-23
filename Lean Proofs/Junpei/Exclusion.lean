import Draft0.PosteriorLaw
import Draft0.ThreeArmComparison
import Draft0.GaussianTail
import Junpei.CountComparison

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.Exclusion

open Draft0.BayesianModel Draft0.PosteriorLaw Draft0.PosteriorRegret
open Draft0.Gaussian Draft0.ThreeArmComparison

noncomputable section

def otherMinimum (i : Arm) (u : Means) : ℝ :=
  if i = 0 then min (u 1) (u 2)
  else if i = 1 then min (u 0) (u 2) else min (u 0) (u 1)

/-- The revised paper's actual posterior exclusion loss. -/
def exclusion (c : Counts) (m : Means) (i : Arm) : ℝ :=
  ∫ u : Means, max (u i - otherMinimum i u) 0 ∂posteriorLaw c m

def pairIntegral (c : Counts) (m : Means) (i j : Arm) : ℝ :=
  ∫ u : Means, max (u i - u j) 0 ∂posteriorLaw c m

def pairSlack (c : Counts) (m : Means) (i j : Arm) : ℝ :=
  hingeLoss (m j - m i) ((c i : ℝ≥0)⁻¹ + (c j : ℝ≥0)⁻¹)

theorem otherMinimum_integrable (c : Counts) (m : Means) (i : Arm) :
    Integrable (otherMinimum i) (posteriorLaw c m) := by
  have hmin (a b : Arm) : Integrable (fun u : Means => min (u a) (u b))
      (posteriorLaw c m) := by
    have h := ((posterior_coordinate_integrable c m a).sub
      ((posterior_coordinate_integrable c m a).sub
        (posterior_coordinate_integrable c m b)).pos_part)
    convert h using 1
    funext u
    dsimp
    by_cases hab : u a ≤ u b
    · rw [min_eq_left hab, max_eq_right (by linarith : u a - u b ≤ 0), sub_zero]
    · rw [min_eq_right (by linarith : u b ≤ u a),
        max_eq_left (by linarith : 0 ≤ u a - u b)]
      ring
  fin_cases i <;> simpa [otherMinimum] using hmin _ _

theorem exclusion_integrable (c : Counts) (m : Means) (i : Arm) :
    Integrable (fun u : Means => max (u i - otherMinimum i u) 0) (posteriorLaw c m) :=
  ((posterior_coordinate_integrable c m i).sub (otherMinimum_integrable c m i)).pos_part

theorem pair_integrable (c : Counts) (m : Means) (i j : Arm) :
    Integrable (fun u : Means => max (u i - u j) 0) (posteriorLaw c m) :=
  ((posterior_coordinate_integrable c m i).sub (posterior_coordinate_integrable c m j)).pos_part

theorem otherMinimum_le (i j : Arm) (u : Means) (hij : j ≠ i) :
    otherMinimum i u ≤ u j := by
  fin_cases i <;> fin_cases j <;> simp_all [otherMinimum]

theorem pairIntegral_le_exclusion (c : Counts) (m : Means) (i j : Arm) (hij : j ≠ i) :
    pairIntegral c m i j ≤ exclusion c m i := by
  exact integral_mono (pair_integrable c m i j) (exclusion_integrable c m i)
    (fun u => max_le_max (sub_le_sub_left (otherMinimum_le i j u hij) _) le_rfl)

theorem exclusion_nonnegative (c : Counts) (m : Means) (i : Arm) :
    0 ≤ exclusion c m i := integral_nonneg (fun _ => le_max_right _ _)

theorem positive_part_const_sub_integral (a : ℝ) (v : ℝ≥0) :
    (∫ x : ℝ, max (a - x) 0 ∂gaussianReal 0 v) = hingeLoss (-a) v := by
  have hm : (gaussianReal 0 v).map (fun x : ℝ => -x) = gaussianReal 0 v := by
    simpa using gaussianReal_map_neg (μ := (0 : ℝ)) (v := v)
  have he : (∫ x : ℝ, max (x - (-a)) 0 ∂(gaussianReal 0 v).map (fun x : ℝ => -x)) =
      ∫ x : ℝ, max (a - x) 0 ∂gaussianReal 0 v := by
    rw [integral_map (by fun_prop) (by fun_prop)]
    congr 1
    funext x
    congr 1
    ring
  rw [hm] at he
  exact he.symm

theorem positive_pair_double_integral (a b : ℝ) (v w : ℝ≥0) :
    (∫ x : ℝ, ∫ y : ℝ, max ((a + x) - (b + y)) 0
      ∂gaussianReal 0 w ∂gaussianReal 0 v) = hingeLoss (b - a) (v + w) := by
  have he (x : ℝ) : (∫ y : ℝ, max ((a + x) - (b + y)) 0 ∂gaussianReal 0 w) =
      hingeLoss (b - a - x) w := by
    have h := positive_part_const_sub_integral (a + x - b) w
    convert h using 1
    · congr 1
      funext y
      congr 1
      ring
    · congr 1
      ring
  simp_rw [he]
  exact hingeLoss_shift_tower (b - a) v w

theorem pair_integral_triple (c : Counts) (m : Means) (i j : Arm) :
    pairIntegral c m i j = ∫ x : Triple,
      max (tripleShift m x i - tripleShift m x j) 0 ∂centeredTripleLaw c := by
  unfold pairIntegral posteriorLaw
  rw [integral_map (tripleShift_measurable m).aemeasurable (by fun_prop)]

theorem pair_integrable_triple (c : Counts) (m : Means) (i j : Arm) :
    Integrable (fun x : Triple => max (tripleShift m x i - tripleShift m x j) 0)
      (centeredTripleLaw c) := by
  have h := pair_integrable c m i j
  unfold posteriorLaw at h
  exact h.comp_measurable (tripleShift_measurable m)

theorem pairIntegral_01 (c : Counts) (m : Means) :
    pairIntegral c m 0 1 = pairSlack c m 0 1 := by
  rw [pair_integral_triple]
  have hi := pair_integrable_triple c m 0 1
  simp [tripleShift, shift, Fin.ext_iff] at hi ⊢
  unfold centeredTripleLaw at hi ⊢
  rw [integral_prod _ hi]
  dsimp only [Prod.fst, Prod.snd]
  have hf (x : ℝ) : (∫ y : ℝ × ℝ, max ((m 0 + x) - (m 1 + y.1)) 0
      ∂(gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹)) =
      ∫ y : ℝ, max ((m 0 + x) - (m 1 + y)) 0 ∂gaussianReal 0 (c 1 : ℝ≥0)⁻¹ := by
    simpa using integral_fun_fst (μ := gaussianReal 0 (c 1 : ℝ≥0)⁻¹)
      (ν := gaussianReal 0 (c 2 : ℝ≥0)⁻¹) (fun y : ℝ => max ((m 0 + x) - (m 1 + y)) 0)
  simp_rw [hf]
  exact positive_pair_double_integral _ _ _ _

theorem pairIntegral_02 (c : Counts) (m : Means) :
    pairIntegral c m 0 2 = pairSlack c m 0 2 := by
  rw [pair_integral_triple]
  have hi := pair_integrable_triple c m 0 2
  simp [tripleShift, shift, Fin.ext_iff] at hi ⊢
  unfold centeredTripleLaw at hi ⊢
  rw [integral_prod _ hi]
  dsimp only [Prod.fst, Prod.snd]
  have hf (x : ℝ) : (∫ y : ℝ × ℝ, max ((m 0 + x) - (m 2 + y.2)) 0
      ∂(gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹)) =
      ∫ y : ℝ, max ((m 0 + x) - (m 2 + y)) 0 ∂gaussianReal 0 (c 2 : ℝ≥0)⁻¹ := by
    simpa using integral_fun_snd (μ := gaussianReal 0 (c 1 : ℝ≥0)⁻¹)
      (ν := gaussianReal 0 (c 2 : ℝ≥0)⁻¹) (fun y : ℝ => max ((m 0 + x) - (m 2 + y)) 0)
  simp_rw [hf]
  exact positive_pair_double_integral _ _ _ _

theorem pairIntegral_10 (c : Counts) (m : Means) :
    pairIntegral c m 1 0 = pairSlack c m 1 0 := by
  rw [pair_integral_triple]
  have hi := pair_integrable_triple c m 1 0
  simp [tripleShift, shift, Fin.ext_iff] at hi ⊢
  unfold centeredTripleLaw at hi ⊢
  rw [integral_prod _ hi]
  dsimp only [Prod.fst, Prod.snd]
  have hf (x : ℝ) : (∫ y : ℝ × ℝ, max ((m 1 + y.1) - (m 0 + x)) 0
      ∂(gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹)) =
      ∫ y : ℝ, max ((m 1 + y) - (m 0 + x)) 0 ∂gaussianReal 0 (c 1 : ℝ≥0)⁻¹ := by
    simpa using integral_fun_fst (μ := gaussianReal 0 (c 1 : ℝ≥0)⁻¹)
      (ν := gaussianReal 0 (c 2 : ℝ≥0)⁻¹) (fun y : ℝ => max ((m 1 + y) - (m 0 + x)) 0)
  simp_rw [hf]
  have he (x : ℝ) : (∫ y : ℝ, max ((m 1 + y) - (m 0 + x)) 0
      ∂gaussianReal 0 (c 1 : ℝ≥0)⁻¹) = hingeLoss (m 0 - m 1 + x) (c 1 : ℝ≥0)⁻¹ := by
    unfold hingeLoss
    congr 1
    funext y
    congr 1
    ring
  simp_rw [he]
  rw [hingeLoss_add_tower]
  unfold pairSlack
  rw [add_comm]

theorem pairIntegral_20 (c : Counts) (m : Means) :
    pairIntegral c m 2 0 = pairSlack c m 2 0 := by
  rw [pair_integral_triple]
  have hi := pair_integrable_triple c m 2 0
  simp [tripleShift, shift, Fin.ext_iff] at hi ⊢
  unfold centeredTripleLaw at hi ⊢
  rw [integral_prod _ hi]
  dsimp only [Prod.fst, Prod.snd]
  have hf (x : ℝ) : (∫ y : ℝ × ℝ, max ((m 2 + y.2) - (m 0 + x)) 0
      ∂(gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹)) =
      ∫ y : ℝ, max ((m 2 + y) - (m 0 + x)) 0 ∂gaussianReal 0 (c 2 : ℝ≥0)⁻¹ := by
    simpa using integral_fun_snd (μ := gaussianReal 0 (c 1 : ℝ≥0)⁻¹)
      (ν := gaussianReal 0 (c 2 : ℝ≥0)⁻¹) (fun y : ℝ => max ((m 2 + y) - (m 0 + x)) 0)
  simp_rw [hf]
  have he (x : ℝ) : (∫ y : ℝ, max ((m 2 + y) - (m 0 + x)) 0
      ∂gaussianReal 0 (c 2 : ℝ≥0)⁻¹) = hingeLoss (m 0 - m 2 + x) (c 2 : ℝ≥0)⁻¹ := by
    unfold hingeLoss
    congr 1
    funext y
    congr 1
    ring
  simp_rw [he]
  rw [hingeLoss_add_tower]
  unfold pairSlack
  rw [add_comm]

theorem pairIntegral_12 (c : Counts) (m : Means) :
    pairIntegral c m 1 2 = pairSlack c m 1 2 := by
  rw [pair_integral_triple]
  have hi := pair_integrable_triple c m 1 2
  simp [tripleShift, shift, Fin.ext_iff] at hi ⊢
  unfold centeredTripleLaw at hi ⊢
  have hp : Integrable (fun x : ℝ × ℝ => max ((m 1 + x.1) - (m 2 + x.2)) 0)
      ((gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹)) :=
    hi.of_comp_snd (by exact MeasureTheory.IsProbabilityMeasure.ne_zero _)
  have he := integral_fun_snd (μ := gaussianReal 0 (c 0 : ℝ≥0)⁻¹)
    (ν := (gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹))
    (fun x : ℝ × ℝ => max ((m 1 + x.1) - (m 2 + x.2)) 0)
  simp only [measureReal_univ_eq_one, one_smul] at he
  rw [he, integral_prod _ hp]
  exact positive_pair_double_integral _ _ _ _

theorem pairIntegral_21 (c : Counts) (m : Means) :
    pairIntegral c m 2 1 = pairSlack c m 2 1 := by
  rw [pair_integral_triple]
  have hi := pair_integrable_triple c m 2 1
  simp [tripleShift, shift, Fin.ext_iff] at hi ⊢
  unfold centeredTripleLaw at hi ⊢
  have hp : Integrable (fun x : ℝ × ℝ => max ((m 2 + x.2) - (m 1 + x.1)) 0)
      ((gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹)) :=
    hi.of_comp_snd (by exact MeasureTheory.IsProbabilityMeasure.ne_zero _)
  have he := integral_fun_snd (μ := gaussianReal 0 (c 0 : ℝ≥0)⁻¹)
    (ν := (gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹))
    (fun x : ℝ × ℝ => max ((m 2 + x.2) - (m 1 + x.1)) 0)
  simp only [measureReal_univ_eq_one, one_smul] at he
  rw [he, integral_prod_symm _ hp]
  exact positive_pair_double_integral _ _ _ _

theorem pairIntegral_eq_pairSlack (c : Counts) (m : Means) (i j : Arm) (hij : i ≠ j) :
    pairIntegral c m i j = pairSlack c m i j := by
  fin_cases i <;> fin_cases j
  all_goals first | exact False.elim (hij rfl) | exact pairIntegral_01 c m |
    exact pairIntegral_02 c m | exact pairIntegral_10 c m | exact pairIntegral_12 c m |
    exact pairIntegral_20 c m | exact pairIntegral_21 c m

theorem pairSlack_le_exclusion (c : Counts) (m : Means) (i j : Arm) (hij : i ≠ j) :
    pairSlack c m i j ≤ exclusion c m i := by
  rw [← pairIntegral_eq_pairSlack c m i j hij]
  exact pairIntegral_le_exclusion c m i j hij.symm

theorem stateSlack_le_exclusion (c : Counts) (m : Means) :
    stateSlack c m ≤ exclusion c m 2 := by
  have h := pairSlack_le_exclusion c m 2 0 (by decide)
  simpa [stateSlack, slack, pairSlack, posteriorVariance, add_comm] using h

theorem exclusion_third_le_pairSlacks (c : Counts) (m : Means) :
    exclusion c m 2 ≤ pairSlack c m 2 0 + pairSlack c m 2 1 := by
  have hi := (pair_integrable c m 2 0).add (pair_integrable c m 2 1)
  have h := integral_mono (exclusion_integrable c m 2) hi (fun u => by
    change max (u 2 - min (u 0) (u 1)) 0 ≤ max (u 2 - u 0) 0 + max (u 2 - u 1) 0
    by_cases hu : u 0 ≤ u 1
    · rw [min_eq_left hu]
      linarith [le_max_right (u 2 - u 1) 0]
    · rw [min_eq_right (le_of_not_ge hu)]
      linarith [le_max_right (u 2 - u 0) 0])
  simp only [Pi.add_apply] at h
  rw [integral_add (pair_integrable c m 2 0) (pair_integrable c m 2 1)] at h
  change exclusion c m 2 ≤ pairIntegral c m 2 0 + pairIntegral c m 2 1 at h
  rw [pairIntegral_20, pairIntegral_21] at h
  exact h

end

end Junpei.Exclusion
