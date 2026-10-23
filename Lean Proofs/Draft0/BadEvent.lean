import Draft0.Drift
import Draft0.SmallBall
import Draft0.PredictiveBound
import Mathlib.Probability.HasLaw
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
Frequentist sample arrays for means (0,0,1), and probabilities of the bad
empirical-mean event. The raw observations are independent unit-variance
Gaussian coordinates, rather than assumed posterior innovations.
-/

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal BigOperators

namespace Draft0.BadEvent

noncomputable section

def arrayLaw (q : ℕ) : Measure (Fin q → ℝ) :=
  Measure.pi (fun _ : Fin q => gaussianReal 0 1)

instance arrayLaw_probability (q : ℕ) : IsProbabilityMeasure (arrayLaw q) := by
  unfold arrayLaw
  infer_instance

def weightVariance (a : ℝ) : ℝ≥0 := ⟨a ^ 2, sq_nonneg a⟩

def weightedSum {q : ℕ} (a : Fin q → ℝ) (ω : Fin q → ℝ) : ℝ := ∑ i, a i * ω i

theorem weighted_coordinates_independent {q : ℕ} (a : Fin q → ℝ) :
    iIndepFun (fun i (ω : Fin q → ℝ) => a i * ω i) (arrayLaw q) := by
  exact iIndepFun_pi (fun _ => by fun_prop)

theorem weighted_coordinate_law {q : ℕ} (a : Fin q → ℝ) (i : Fin q) :
    HasLaw (fun ω : Fin q → ℝ => a i * ω i) (gaussianReal 0 (weightVariance (a i))) (arrayLaw q) := by
  have heval : HasLaw (Function.eval i) (gaussianReal 0 1) (arrayLaw q) :=
    (measurePreserving_eval (fun _ : Fin q => gaussianReal 0 1) i).hasLaw
  have hscale : HasLaw (fun x : ℝ => a i * x) (gaussianReal 0 (weightVariance (a i)))
      (gaussianReal 0 1) := by
    constructor
    · exact (by fun_prop : AEMeasurable (fun x : ℝ => a i * x) (gaussianReal 0 1))
    · rw [gaussianReal_map_const_mul, mul_zero, mul_one]
      rfl
  exact hscale.fun_comp heval

theorem weighted_finset_sum_law {q : ℕ} (a : Fin q → ℝ) (s : Finset (Fin q)) :
    HasLaw (fun ω : Fin q → ℝ => ∑ i ∈ s, a i * ω i)
      (gaussianReal 0 (∑ i ∈ s, weightVariance (a i))) (arrayLaw q) := by
  classical
  have hind := weighted_coordinates_independent a
  induction s using Finset.induction_on with
  | empty =>
      constructor
      · simp only [Finset.sum_empty]
        exact aemeasurable_const
      · simp [gaussianReal_zero_var]
  | insert i s hi ih =>
      have hInd := hind.indepFun_finset_sum_of_notMem (fun _ => by fun_prop) hi
      have hInd' : IndepFun (fun ω : Fin q → ℝ => a i * ω i)
          (fun ω : Fin q → ℝ => ∑ j ∈ s, a j * ω j) (arrayLaw q) := by
        simpa only [Finset.sum_fn] using hInd.symm
      have hsum := hInd'.hasLaw_fun_add (weighted_coordinate_law a i) ih
      rw [gaussianReal_conv_gaussianReal, zero_add] at hsum
      simpa only [Finset.sum_insert hi] using hsum

theorem weightedSum_law {q : ℕ} (a : Fin q → ℝ) :
    HasLaw (weightedSum a) (gaussianReal 0 (∑ i, weightVariance (a i))) (arrayLaw q) := by
  exact weighted_finset_sum_law a Finset.univ

def meanCoefficient {q : ℕ} (n : ℕ) (i : Fin q) : ℝ :=
  if (i : ℕ) < n then (n : ℝ)⁻¹ else 0

def sampleMean {q : ℕ} (n : ℕ) (ω : Fin q → ℝ) : ℝ := weightedSum (meanCoefficient n) ω

theorem fin_sum_if_lt (n q : ℕ) (hnq : n ≤ q) (a b : ℝ) :
    (∑ i : Fin q, if (i : ℕ) < n then a else b) = (n : ℝ) * a + ((q - n : ℕ) : ℝ) * b := by
  classical
  have hc : (Finset.univ.filter (fun i : Fin q => (i : ℕ) < n)).card = n := by
    simpa only [Fintype.card_subtype] using Fintype.card_fin_lt_of_le hnq
  have hcomp : (Finset.univ.filter (fun i : Fin q => ¬ ((i : ℕ) < n))).card = q - n := by
    have h := Finset.filter_card_add_filter_neg_card_eq_card
      (s := (Finset.univ : Finset (Fin q))) (fun i => (i : ℕ) < n)
    rw [hc, Finset.card_univ, Fintype.card_fin] at h
    omega
  rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const, hc, hcomp]
  simp only [nsmul_eq_mul]

theorem meanVariance_eq (n q : ℕ) (hn : 0 < n) (hnq : n ≤ q) :
    (∑ i : Fin q, weightVariance (meanCoefficient n i)) = (n : ℝ≥0)⁻¹ := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  ext
  simp only [NNReal.coe_sum, weightVariance, NNReal.coe_mk, NNReal.coe_inv, NNReal.coe_natCast]
  have hpoint : ∀ i : Fin q, (meanCoefficient n i) ^ 2 =
      if (i : ℕ) < n then ((n : ℝ)⁻¹) ^ 2 else 0 := by
    intro i
    unfold meanCoefficient
    split_ifs <;> simp
  simp_rw [hpoint]
  rw [fin_sum_if_lt n q hnq]
  field_simp
  ring

theorem sampleMean_law (n q : ℕ) (hn : 0 < n) (hnq : n ≤ q) :
    HasLaw (sampleMean (q := q) n) (gaussianReal 0 (n : ℝ≥0)⁻¹) (arrayLaw q) := by
  have h := weightedSum_law (meanCoefficient (q := q) n)
  rw [meanVariance_eq n q hn hnq] at h
  exact h

def driftCoefficient {q : ℕ} (n : ℕ) (i : Fin q) : ℝ := meanCoefficient n i - (q : ℝ)⁻¹

def driftVariance (n q : ℕ) : ℝ≥0 := ∑ i : Fin q, weightVariance (driftCoefficient n i)

theorem driftVariance_coe (n q : ℕ) (hn : 0 < n) (hnq : n ≤ q) :
    (driftVariance n q : ℝ) = (n : ℝ)⁻¹ - (q : ℝ)⁻¹ := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  have hq0 : (q : ℝ) ≠ 0 := by exact_mod_cast (show q ≠ 0 by omega)
  unfold driftVariance
  simp only [NNReal.coe_sum, weightVariance, NNReal.coe_mk]
  have hpoint : ∀ i : Fin q, (driftCoefficient n i) ^ 2 =
      if (i : ℕ) < n then ((n : ℝ)⁻¹ - (q : ℝ)⁻¹) ^ 2 else ((q : ℝ)⁻¹) ^ 2 := by
    intro i
    unfold driftCoefficient meanCoefficient
    split_ifs <;> simp
  simp_rw [hpoint]
  rw [fin_sum_if_lt n q hnq, Nat.cast_sub hnq]
  field_simp
  ring

theorem sampleMean_difference_eq (n q : ℕ) (ω : Fin q → ℝ) :
    sampleMean n ω - sampleMean q ω = weightedSum (driftCoefficient n) ω := by
  unfold sampleMean weightedSum
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _hi
  unfold driftCoefficient
  simp only [meanCoefficient, if_pos i.isLt]
  ring

theorem sampleMean_difference_law (n q : ℕ) :
    HasLaw (fun ω : Fin q → ℝ => sampleMean n ω - sampleMean q ω)
      (gaussianReal 0 (driftVariance n q)) (arrayLaw q) := by
  simpa only [sampleMean_difference_eq] using weightedSum_law (driftCoefficient (q := q) n)

theorem reciprocal_square_ge_decrement (n : ℕ) (hn : 0 < n) :
    (n : ℝ)⁻¹ - ((n : ℝ) + 1)⁻¹ ≤ 1 / (n : ℝ) ^ 2 := by
  have hnr : 0 < (n : ℝ) := by exact_mod_cast hn
  rw [← one_div (n : ℝ), ← one_div ((n : ℝ) + 1),
    Counts.varianceDecrement_identity n hn]
  unfold Counts.varianceDecrement
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith only [hnr]

theorem reciprocalSqSum_ge_variance (n q : ℕ) (hn : 0 < n) (hnq : n ≤ q) :
    (n : ℝ)⁻¹ - (q : ℝ)⁻¹ ≤ Drift.reciprocalSqSum n q := by
  induction q with
  | zero => omega
  | succ q ih =>
      by_cases hnq' : n ≤ q
      · unfold Drift.reciprocalSqSum
        rw [Finset.sum_Ico_succ_top hnq']
        have hprev := ih hnq'
        have hstep := reciprocal_square_ge_decrement q (by omega)
        unfold Drift.reciprocalSqSum at hprev
        push_cast
        linarith only [hprev, hstep]
      · have hEq : n = q + 1 := by omega
        subst n
        simp [Drift.reciprocalSqSum]

theorem gaussian_law_subgaussian {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} {X : Ω → ℝ} {v : ℝ≥0} (hLaw : HasLaw X (gaussianReal 0 v) μ) :
    HasSubgaussianMGF X v μ := by
  have hid : HasSubgaussianMGF (fun x : ℝ => x) v (gaussianReal 0 v) := by
    constructor
    · exact fun t => integrable_exp_mul_gaussianReal t
    · intro t
      simp [mgf_fun_id_gaussianReal]
  have hmap : HasSubgaussianMGF (fun x : ℝ => x) v (μ.map X) := by
    rw [hLaw.map_eq]
    exact hid
  have h := HasSubgaussianMGF.of_map (X := fun x : ℝ => x) (Y := X) hLaw.aemeasurable hmap
  simpa only [Function.comp_apply] using h

def driftThreshold (T n q : ℕ) : ℝ :=
  4 * Real.sqrt (Real.log (T : ℝ)) * Real.sqrt (Drift.reciprocalSqSum n q)

def driftFailure (T n q : ℕ) : Set (Fin q → ℝ) :=
  {ω | driftThreshold T n q < |sampleMean n ω - sampleMean q ω|}

theorem driftFailure_probability_le (T n q : ℕ) (hT : 5 ≤ T) (hn : 0 < n) (hnq : n ≤ q) :
    (arrayLaw q).real (driftFailure T n q) ≤ 2 * Rates.invPower 8 T := by
  by_cases hnqeq : n = q
  · subst n
    have hset : driftFailure T q q = ∅ := by
      ext ω
      simp [driftFailure, driftThreshold, Drift.reciprocalSqSum]
    rw [hset]
    simp only [measureReal_empty]
    exact mul_nonneg (by norm_num) (Rates.invPower_nonneg 8 T)
  have hnqlt : n < q := by omega
  have hnr : 0 < (n : ℝ) := by exact_mod_cast hn
  have hqr : 0 < (q : ℝ) := by exact_mod_cast (show 0 < q by omega)
  have hTr : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hlog := Drift.log_horizon_ge_one T hT
  have hlog0 : 0 ≤ Real.log (T : ℝ) := by linarith only [hlog]
  have hvarEq := driftVariance_coe n q hn hnq
  have hv : 0 < driftVariance n q := by
    change 0 < (driftVariance n q : ℝ)
    rw [hvarEq]
    exact sub_pos.mpr ((inv_lt_inv₀ hqr hnr).mpr (by exact_mod_cast hnqlt))
  have hVarBound : (driftVariance n q : ℝ) ≤ Drift.reciprocalSqSum n q := by
    rw [hvarEq]
    exact reciprocalSqSum_ge_variance n q hn hnq
  have hthreshold0 : 0 ≤ driftThreshold T n q := by unfold driftThreshold; positivity
  have hthresholdsq : (driftThreshold T n q) ^ 2 =
      16 * Real.log (T : ℝ) * Drift.reciprocalSqSum n q := by
    unfold driftThreshold
    rw [mul_pow, mul_pow, Real.sq_sqrt hlog0,
      Real.sq_sqrt (Drift.reciprocalSqSum_nonnegative n q)]
    norm_num
  have hscaled := mul_le_mul_of_nonneg_left hVarBound hlog0
  have hquot : 8 * Real.log (T : ℝ) ≤ (driftThreshold T n q) ^ 2 / (2 * (driftVariance n q : ℝ)) := by
    apply (le_div_iff₀ (by positivity)).mpr
    rw [hthresholdsq]
    nlinarith only [hscaled]
  have hexp : Real.exp (-(driftThreshold T n q) ^ 2 / (2 * (driftVariance n q : ℝ))) ≤
      Rates.invPower 8 T := by
    unfold Rates.invPower
    rw [← PredictiveBound.exp_neg_nat_log 8 hTr]
    apply Real.exp_le_exp.mpr
    rw [neg_div]
    norm_num
    linarith only [hquot]
  have hsub := gaussian_law_subgaussian (sampleMean_difference_law n q)
  have hright := hsub.measure_ge_le hthreshold0
  have hleft := hsub.neg.measure_ge_le hthreshold0
  have hsubset : driftFailure T n q ⊆
      {ω | driftThreshold T n q ≤ sampleMean n ω - sampleMean q ω} ∪
        {ω | driftThreshold T n q ≤ -(sampleMean n ω - sampleMean q ω)} := by
    intro ω hω
    change driftThreshold T n q < |sampleMean n ω - sampleMean q ω| at hω
    rw [lt_abs] at hω
    rcases hω with hω | hω
    · exact Or.inl (le_of_lt hω)
    · exact Or.inr (le_of_lt hω)
  have hmono := measureReal_mono (μ := arrayLaw q) hsubset
  have hunion := measureReal_union_le (μ := arrayLaw q)
    {ω | driftThreshold T n q ≤ sampleMean n ω - sampleMean q ω}
    {ω | driftThreshold T n q ≤ -(sampleMean n ω - sampleMean q ω)}
  simp only [Pi.neg_apply] at hleft
  linarith only [hmono, hunion, hright, hleft, hexp]

def goodDriftSet (T q : ℕ) : Set (Fin q → ℝ) :=
  {ω | ∀ n : ℕ, 1 ≤ n → n ≤ q →
    |sampleMean n ω - sampleMean q ω| ≤ driftThreshold T n q}

theorem goodDrift_compl_probability_le (T q : ℕ) (hT : 5 ≤ T) (hqT : q ≤ T) :
    (arrayLaw q).real (goodDriftSet T q)ᶜ ≤ 2 * Rates.invPower 7 T := by
  have hsubset : (goodDriftSet T q)ᶜ ⊆ ⋃ i : Fin q, driftFailure T ((i : ℕ) + 1) q := by
    intro ω hω
    simp only [goodDriftSet, Set.mem_compl_iff, Set.mem_setOf_eq] at hω
    push_neg at hω
    rcases hω with ⟨n, hn, hnq, hf⟩
    let i : Fin q := ⟨n - 1, by omega⟩
    apply Set.mem_iUnion.mpr
    refine ⟨i, ?_⟩
    have hi : (i : ℕ) + 1 = n := by dsimp [i]; omega
    rw [hi]
    exact hf
  have hmono := measureReal_mono (μ := arrayLaw q) hsubset
  have hunion := measureReal_iUnion_fintype_le (μ := arrayLaw q)
    (fun i : Fin q => driftFailure T ((i : ℕ) + 1) q)
  have hsum : (∑ i : Fin q, (arrayLaw q).real (driftFailure T ((i : ℕ) + 1) q)) ≤
      (q : ℝ) * (2 * Rates.invPower 8 T) := by
    calc
      _ ≤ ∑ _i : Fin q, 2 * Rates.invPower 8 T := by
        apply Finset.sum_le_sum
        intro i _hi
        exact driftFailure_probability_le T _ q hT (by omega) (by have hi := i.isLt; omega)
      _ = _ := by simp
  have hqTr : (q : ℝ) ≤ (T : ℝ) := by exact_mod_cast hqT
  have hscale := mul_le_mul_of_nonneg_right hqTr
    (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) (Rates.invPower_nonneg 8 T))
  have hrate : (T : ℝ) * (2 * Rates.invPower 8 T) = 2 * Rates.invPower 7 T := by
    have hT0 : (T : ℝ) ≠ 0 := by exact_mod_cast (show T ≠ 0 by omega)
    unfold Rates.invPower
    field_simp
  rw [hrate] at hscale
  linarith only [hmono, hunion, hsum, hscale]

def goodMeanSet (T q : ℕ) : Set (Fin q → ℝ) :=
  {ω | |sampleMean q ω| ≤ Rates.invPower 2 T}

theorem goodMeanSet_measurable (T q : ℕ) : MeasurableSet (goodMeanSet T q) := by
  unfold goodMeanSet
  apply measurableSet_le
  · unfold sampleMean weightedSum
    fun_prop
  · exact measurable_const

theorem goodDriftSet_measurable (T q : ℕ) : MeasurableSet (goodDriftSet T q) := by
  unfold goodDriftSet
  simp only [Set.setOf_forall]
  apply MeasurableSet.iInter
  intro n
  apply MeasurableSet.iInter
  intro _hn
  apply MeasurableSet.iInter
  intro _hnq
  apply measurableSet_le
  · unfold sampleMean weightedSum
    fun_prop
  · exact measurable_const

theorem f2_polynomial_lower (T : ℕ) (hT : 1 ≤ T) :
    (1 / 4 : ℝ) * Rates.invPower 2 T ≤ SmallBall.f2 (T : ℝ) := by
  have hTr : 1 ≤ (T : ℝ) := by exact_mod_cast hT
  have htpos : 0 < (T : ℝ) := by linarith only [hTr]
  have hpiBound : Real.pi ≤ 4 * (T : ℝ) := by linarith only [Real.pi_le_four, hTr]
  have hmul := mul_le_mul_of_nonneg_right hpiBound (by positivity : 0 ≤ (T : ℝ) ^ 3)
  have hsq : ((1 / 2 : ℝ) * Rates.invPower 2 T) ^ 2 ≤ 1 / (Real.pi * (T : ℝ) ^ 3) := by
    calc
      _ = 1 / (4 * (T : ℝ) ^ 4) := by
        unfold Rates.invPower
        field_simp
        ring
      _ ≤ _ := by
        apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
        nlinarith only [hmul]
  have hsqrt := Real.le_sqrt_of_sq_le hsq
  unfold SmallBall.f2
  nlinarith only [hsqrt]

theorem goodMean_probability_lower (T q : ℕ) (hT : T = 2 * q + 1) (hq : 0 < q) :
    (1 / 4 : ℝ) * Rates.invPower 2 T ≤ (arrayLaw q).real (goodMeanSet T q) := by
  have hLaw := sampleMean_law q q hq le_rfl
  have hmass : (arrayLaw q).real (goodMeanSet T q) =
      (gaussianReal 0 (q : ℝ≥0)⁻¹ (Icc (-Rates.invPower 2 T) (Rates.invPower 2 T))).toReal := by
    rw [← hLaw.map_eq, Measure.map_apply_of_aemeasurable hLaw.aemeasurable measurableSet_Icc]
    congr 1
    ext ω
    simp only [goodMeanSet, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc, abs_le]
  have h := SmallBall.appendixB_probability_lower q hq
  have hTr : (T : ℝ) = 2 * (q : ℝ) + 1 := by exact_mod_cast hT
  rw [← hTr] at h
  rw [hmass]
  exact (f2_polynomial_lower T (by omega)).trans h

theorem goodMean_probability_lower_general (T q : ℕ)
    (hT : 3 ≤ T) (hq : 0 < q) (hqT : q ≤ T) (hTq : T ≤ 3 * q) :
    (1 / 4 : ℝ) * Rates.invPower 2 T ≤ (arrayLaw q).real (goodMeanSet T q) := by
  have hLaw := sampleMean_law q q hq le_rfl
  have hmass : (arrayLaw q).real (goodMeanSet T q) =
      (gaussianReal 0 (q : ℝ≥0)⁻¹ (Icc (-Rates.invPower 2 T) (Rates.invPower 2 T))).toReal := by
    rw [← hLaw.map_eq, Measure.map_apply_of_aemeasurable hLaw.aemeasurable measurableSet_Icc]
    congr 1
    ext ω
    simp only [goodMeanSet, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc, abs_le]
  have hqr : 0 < (q : ℝ) := by exact_mod_cast hq
  have hTr : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hqTr : (q : ℝ) ≤ (T : ℝ) := by exact_mod_cast hqT
  have hTqr : (T : ℝ) ≤ 3 * (q : ℝ) := by exact_mod_cast hTq
  have hvlower : (T : ℝ)⁻¹ ≤ (((q : ℝ≥0)⁻¹) : ℝ) := by
    simp only [NNReal.coe_natCast]
    exact (inv_le_inv₀ hTr hqr).mpr hqTr
  have hvupper : (((q : ℝ≥0)⁻¹) : ℝ) ≤ 3 / (T : ℝ) := by
    simp only [NNReal.coe_natCast]
    rw [inv_eq_one_div]
    apply (div_le_div_iff₀ hqr hTr).mpr
    simpa only [one_mul] using hTqr
  have h := SmallBall.gaussian_smallBall_probability_lower (by exact_mod_cast hT)
    (by positivity : 0 < (q : ℝ≥0)⁻¹) hvlower hvupper
  rw [hmass]
  exact (f2_polynomial_lower T (by omega)).trans h

theorem drift_error_smaller_than_mean_lower (T : ℕ) (hT : 5 ≤ T) :
    2 * Rates.invPower 7 T ≤ (1 / 8 : ℝ) * Rates.invPower 2 T := by
  have hTr : (5 : ℝ) ≤ T := by exact_mod_cast hT
  have htpos : 0 < (T : ℝ) := by linarith only [hTr]
  have hpow : (5 : ℝ) ^ 5 ≤ (T : ℝ) ^ 5 := pow_le_pow_left₀ (by norm_num) hTr 5
  have hInv : 2 * Rates.invPower 5 T ≤ 1 / 8 := by
    unfold Rates.invPower
    rw [← div_eq_mul_inv]
    apply (div_le_iff₀ (by positivity)).mpr
    norm_num at hpow
    linarith only [hpow]
  calc
    2 * Rates.invPower 7 T = (2 * Rates.invPower 5 T) * Rates.invPower 2 T := by
      rw [← Rates.invPower_product 5 2 T]
      ring
    _ ≤ (1 / 8 : ℝ) * Rates.invPower 2 T :=
      mul_le_mul_of_nonneg_right hInv (Rates.invPower_nonneg 2 T)

def goodArmSet (T q : ℕ) : Set (Fin q → ℝ) := goodMeanSet T q ∩ goodDriftSet T q

/-- Actual frequentist Y∩Z probability for one iid Gaussian arm. The Gaussian
drift estimate is proved above, rather than supplied as a comparison premise. -/
theorem goodArm_probability_lower (T q : ℕ) (hT : T = 2 * q + 1) (hq : 2 ≤ q) :
    (1 / 8 : ℝ) * Rates.invPower 2 T ≤ (arrayLaw q).real (goodArmSet T q) := by
  have hmean := goodMean_probability_lower T q hT (by omega)
  have hdrift := goodDrift_compl_probability_le T q (by omega) (by omega)
  have hsmall := drift_error_smaller_than_mean_lower T (by omega)
  have hdecomp := measureReal_inter_add_diff (μ := arrayLaw q)
    (s := goodMeanSet T q) (goodDriftSet_measurable T q)
  have hsubset : goodMeanSet T q \ goodDriftSet T q ⊆ (goodDriftSet T q)ᶜ := by
    exact fun _ h => h.2
  have hdiff := measureReal_mono (μ := arrayLaw q) hsubset
  unfold goodArmSet
  linarith only [hmean, hdrift, hsmall, hdecomp, hdiff]

theorem goodArm_probability_lower_general (T q : ℕ)
    (hT : 5 ≤ T) (hq : 0 < q) (hqT : q ≤ T) (hTq : T ≤ 3 * q) :
    (1 / 8 : ℝ) * Rates.invPower 2 T ≤ (arrayLaw q).real (goodArmSet T q) := by
  have hmean := goodMean_probability_lower_general T q (by omega) hq hqT hTq
  have hdrift := goodDrift_compl_probability_le T q hT hqT
  have hsmall := drift_error_smaller_than_mean_lower T hT
  have hdecomp := measureReal_inter_add_diff (μ := arrayLaw q)
    (s := goodMeanSet T q) (goodDriftSet_measurable T q)
  have hsubset : goodMeanSet T q \ goodDriftSet T q ⊆ (goodDriftSet T q)ᶜ := by
    exact fun _ h => h.2
  have hdiff := measureReal_mono (μ := arrayLaw q) hsubset
  unfold goodArmSet
  linarith only [hmean, hdrift, hsmall, hdecomp, hdiff]

def underestimationInterval (T : ℕ) : Set ℝ :=
  Icc (-40 * Real.sqrt (Real.log (T : ℝ)) - 1) (-40 * Real.sqrt (Real.log (T : ℝ)))

def underestimationConstant : ℝ := (Real.sqrt (2 * Real.pi))⁻¹

theorem underestimationConstant_positive : 0 < underestimationConstant := by
  unfold underestimationConstant
  positivity

theorem underestimationPDF_lower (T : ℕ) (hT : 5 ≤ T) (x : ℝ)
    (hx : x ∈ underestimationInterval T) :
    underestimationConstant * Rates.invPower 882 T ≤ gaussianPDFReal 1 1 x := by
  have htpos : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hlog := Drift.log_horizon_ge_one T hT
  have hs0 := Real.sqrt_nonneg (Real.log (T : ℝ))
  have hs1 : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.one_le_sqrt.mpr hlog
  have hsq : (Real.sqrt (Real.log (T : ℝ))) ^ 2 = Real.log (T : ℝ) :=
    Real.sq_sqrt (by linarith only [hlog])
  have hdiff0 : x - 1 ≤ 0 := by linarith only [hx.2, hs0]
  have hdiff : |x - 1| ≤ 42 * Real.sqrt (Real.log (T : ℝ)) := by
    rw [abs_of_nonpos hdiff0]
    linarith only [hx.1, hs1]
  have hquad : (x - 1) ^ 2 ≤ 1764 * Real.log (T : ℝ) := by
    have h := (sq_le_sq₀ (abs_nonneg (x - 1)) (by positivity)).mpr hdiff
    rw [sq_abs] at h
    nlinarith only [h, hsq]
  have hexp : Rates.invPower 882 T ≤ Real.exp (-(x - 1) ^ 2 / 2) := by
    unfold Rates.invPower
    rw [← PredictiveBound.exp_neg_nat_log 882 htpos]
    apply Real.exp_le_exp.mpr
    norm_num
    nlinarith only [hquad]
  have hmul := mul_le_mul_of_nonneg_left hexp underestimationConstant_positive.le
  simpa only [gaussianPDFReal, NNReal.coe_one, mul_one, underestimationConstant] using hmul

theorem underestimation_probability_lower (T : ℕ) (hT : 5 ≤ T) :
    underestimationConstant * Rates.invPower 882 T ≤
      (gaussianReal 1 1).real (underestimationInterval T) := by
  have hmass : (gaussianReal 1 1).real (underestimationInterval T) =
      ∫ x in underestimationInterval T, gaussianPDFReal 1 1 x := by
    rw [measureReal_def, gaussianReal_apply_eq_integral 1 (by norm_num : (1 : ℝ≥0) ≠ 0),
      ENNReal.toReal_ofReal]
    exact integral_nonneg (gaussianPDFReal_nonneg 1 1)
  have hmono : (∫ _ in underestimationInterval T, underestimationConstant * Rates.invPower 882 T) ≤
      ∫ x in underestimationInterval T, gaussianPDFReal 1 1 x := by
    apply setIntegral_mono_on
      (integrableOn_const (by exact isCompact_Icc.measure_ne_top))
      (integrable_gaussianPDFReal 1 1).integrableOn measurableSet_Icc
    intro x hx
    exact underestimationPDF_lower T hT x hx
  have hconst : (∫ _ in underestimationInterval T, underestimationConstant * Rates.invPower 882 T) =
      underestimationConstant * Rates.invPower 882 T := by
    unfold underestimationInterval
    rw [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Icc,
      ENNReal.toReal_ofReal (by linarith)]
    ring
  rw [← hconst, hmass]
  exact hmono

abbrev FullSample (q : ℕ) := ℝ × ((Fin q → ℝ) × (Fin q → ℝ))

/-- Actual frequentist model with q iid N(0,1) samples for each inferior arm
and the first N(1,1) sample of the superior arm, independent across arms. -/
def frequentistLaw (q : ℕ) : Measure (FullSample q) :=
  (gaussianReal 1 1).prod ((arrayLaw q).prod (arrayLaw q))

instance frequentistLaw_probability (q : ℕ) : IsProbabilityMeasure (frequentistLaw q) := by
  unfold frequentistLaw
  infer_instance

def badEvent (T q : ℕ) : Set (FullSample q) :=
  underestimationInterval T ×ˢ (goodArmSet T q ×ˢ goodArmSet T q)

theorem badEvent_measurable (T q : ℕ) : MeasurableSet (badEvent T q) := by
  exact measurableSet_Icc.prod
    (((goodMeanSet_measurable T q).inter (goodDriftSet_measurable T q)).prod
      ((goodMeanSet_measurable T q).inter (goodDriftSet_measurable T q)))

theorem badEvent_probability_eq (T q : ℕ) :
    (frequentistLaw q).real (badEvent T q) =
      (gaussianReal 1 1).real (underestimationInterval T) *
        (arrayLaw q).real (goodArmSet T q) * (arrayLaw q).real (goodArmSet T q) := by
  simp only [frequentistLaw, badEvent, Measure.prod_prod, measureReal_def, ENNReal.toReal_mul]
  ring

/-- A concrete polynomial probability bound for the actual bad event,
including both arms' full drift constraints and the superior arm's draw. -/
theorem badEvent_probability_lower (T q : ℕ) (hT : T = 2 * q + 1) (hq : 2 ≤ q) :
    (underestimationConstant / 64) * Rates.invPower 886 T ≤
      (frequentistLaw q).real (badEvent T q) := by
  have hx := underestimation_probability_lower T (by omega)
  have hy := goodArm_probability_lower T q hT hq
  have hprod1 := Rates.product_lower_bound underestimationConstant_positive.le
    (by norm_num : (0 : ℝ) ≤ 1 / 8) hx hy
  have hprod2 := Rates.product_lower_bound
    (mul_nonneg underestimationConstant_positive.le (by norm_num : (0 : ℝ) ≤ 1 / 8))
    (by norm_num : (0 : ℝ) ≤ 1 / 8) hprod1 hy
  rw [badEvent_probability_eq]
  convert hprod2 using 1
  ring

theorem badEvent_probability_lower_general (T q : ℕ)
    (hT : 5 ≤ T) (hq : 0 < q) (hqT : q ≤ T) (hTq : T ≤ 3 * q) :
    (underestimationConstant / 64) * Rates.invPower 886 T ≤
      (frequentistLaw q).real (badEvent T q) := by
  have hx := underestimation_probability_lower T hT
  have hy := goodArm_probability_lower_general T q hT hq hqT hTq
  have hprod1 := Rates.product_lower_bound underestimationConstant_positive.le
    (by norm_num : (0 : ℝ) ≤ 1 / 8) hx hy
  have hprod2 := Rates.product_lower_bound
    (mul_nonneg underestimationConstant_positive.le (by norm_num : (0 : ℝ) ≤ 1 / 8))
    (by norm_num : (0 : ℝ) ≤ 1 / 8) hprod1 hy
  rw [badEvent_probability_eq]
  convert hprod2 using 1
  ring

def rawMeanPath {q : ℕ} (ω : FullSample q) : Counts.MeanPath :=
  fun i n => if i = 1 then sampleMean n ω.2.1 else if i = 2 then sampleMean n ω.2.2 else ω.1

/-- The polynomial-probability bad event entails the exact printed W event
for every count state satisfying its caps. -/
theorem badEvent_implies_W (T q : ℕ) (ω : FullSample q) (c : Counts.Counts)
    (hω : ω ∈ badEvent T q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) (h3 : c.n3 ≤ 1) :
    Drift.paperW T q 40 (rawMeanPath ω) c := by
  rcases hω with ⟨hx, ⟨hy1, hz1⟩, ⟨hy2, hz2⟩⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, h1, h2, h3⟩
  · simpa only [Drift.eventX, rawMeanPath, show (3 : ℕ) ≠ 1 by decide,
      show (3 : ℕ) ≠ 2 by decide, if_false] using hx.2
  · unfold Drift.eventY rawMeanPath
    norm_num
    simpa only [goodMeanSet, Set.mem_setOf_eq, Rates.invPower, one_div] using hy1
  · unfold Drift.eventY rawMeanPath
    norm_num
    simpa only [goodMeanSet, Set.mem_setOf_eq, Rates.invPower, one_div] using hy2
  · intro n hn hnq
    simpa only [rawMeanPath, if_pos rfl, driftThreshold] using hz1 n hn hnq
  · intro n hn hnq
    simpa only [rawMeanPath, show (2 : ℕ) ≠ 1 by decide, if_false, if_pos rfl,
      driftThreshold] using hz2 n hn hnq

def arrayPrefix {q N : ℕ} (h : q ≤ N) (ω : Fin N → ℝ) : Fin q → ℝ :=
  fun i => ω (Fin.castLE h i)

theorem arrayPrefix_preserving {q N : ℕ} (h : q ≤ N) :
    MeasurePreserving (arrayPrefix h) (arrayLaw N) (arrayLaw q) := by
  have hind0 : iIndepFun (fun i (ω : Fin N → ℝ) => ω i) (arrayLaw N) :=
    iIndepFun_pi (fun _ => aemeasurable_id)
  have hind : iIndepFun (fun i (ω : Fin N → ℝ) => ω (Fin.castLE h i)) (arrayLaw N) :=
    hind0.precomp (Fin.castLE_injective h)
  have hmap := (iIndepFun_iff_map_fun_eq_pi_map
    (fun i => (measurable_pi_apply (Fin.castLE h i)).aemeasurable)).mp hind
  have heval (i : Fin q) : (arrayLaw N).map (fun ω => ω (Fin.castLE h i)) = gaussianReal 0 1 :=
    (measurePreserving_eval (fun _ : Fin N => gaussianReal 0 1) (Fin.castLE h i)).map_eq
  constructor
  · unfold arrayPrefix
    fun_prop
  · simpa only [arrayPrefix, heval] using hmap

theorem sampleMean_arrayPrefix_eq {q N : ℕ} (h : q ≤ N) (n : ℕ) (hnq : n ≤ q)
    (ω : Fin N → ℝ) : sampleMean n (arrayPrefix h ω) = sampleMean n ω := by
  classical
  let e : Fin q ↪ Fin N := ⟨Fin.castLE h, Fin.castLE_injective h⟩
  unfold sampleMean weightedSum arrayPrefix
  calc
    (∑ i : Fin q, meanCoefficient n i * ω (Fin.castLE h i)) =
        ∑ j ∈ Finset.univ.map e, meanCoefficient n j * ω j := by
      rw [Finset.sum_map]
      rfl
    _ = ∑ j : Fin N, meanCoefficient n j * ω j := by
      apply Finset.sum_subset
      · exact fun _ _ => Finset.mem_univ _
      · intro j _hju hjnot
        have hqj : q ≤ (j : ℕ) := by
          by_contra! hjq
          let i : Fin q := ⟨(j : ℕ), hjq⟩
          apply hjnot
          apply Finset.mem_map.mpr
          refine ⟨i, Finset.mem_univ _, ?_⟩
          apply Fin.ext
          rfl
        simp only [meanCoefficient, if_neg (by omega : ¬ ((j : ℕ) < n)), zero_mul]

theorem sampleMean_one {T : ℕ} (hT : 0 < T) (ω : Fin T → ℝ) :
    sampleMean 1 ω = ω ⟨0, hT⟩ := by
  rw [← sampleMean_arrayPrefix_eq (show 1 ≤ T by omega) 1 le_rfl]
  simp [sampleMean, weightedSum, meanCoefficient, arrayPrefix]
  congr 1

abbrev RawArrays (T : ℕ) := Fin 3 → Fin T → ℝ

def rawArraysLaw (T : ℕ) : Measure (RawArrays T) :=
  Measure.pi (fun _ : Fin 3 => arrayLaw T)

instance rawArraysLaw_probability (T : ℕ) : IsProbabilityMeasure (rawArraysLaw T) := by
  unfold rawArraysLaw
  infer_instance

/-- Projection of complete independent raw sample streams to the samples
used in the bad event. Arm indices here are 0,1,2; the superior arm has mean 1. -/
def projectRawArrays (T q : ℕ) (hqT : q ≤ T) (hT : 0 < T) (ω : RawArrays T) : FullSample q :=
  (ω 2 ⟨0, hT⟩ + 1, (arrayPrefix hqT (ω 0), arrayPrefix hqT (ω 1)))

theorem projectRawArrays_preserving (T q : ℕ) (hqT : q ≤ T) (hT : 0 < T) :
    MeasurePreserving (projectRawArrays T q hqT hT) (rawArraysLaw T) (frequentistLaw q) := by
  have hprefix := arrayPrefix_preserving hqT
  have hfirst : MeasurePreserving (fun ω : Fin T → ℝ => ω ⟨0, hT⟩ + 1)
      (arrayLaw T) (gaussianReal 1 1) := by
    have heval := measurePreserving_eval (fun _ : Fin T => gaussianReal 0 1) (⟨0, hT⟩ : Fin T)
    have hshift : MeasurePreserving (fun x : ℝ => x + 1) (gaussianReal 0 1) (gaussianReal 1 1) := by
      constructor
      · fun_prop
      · simpa using (gaussianReal_map_add_const (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) 1)
    exact hshift.comp heval
  have htwo := measurePreserving_piFinTwo (fun _ : Fin 2 => arrayLaw T)
  have hpair := (hprefix.prod hprefix).comp htwo
  have hthree := measurePreserving_piFinSuccAbove (fun _ : Fin 3 => arrayLaw T) (2 : Fin 3)
  have hfull := (hfirst.prod hpair).comp hthree
  convert hfull using 1

def rawBadEvent (T q : ℕ) (hqT : q ≤ T) (hT : 0 < T) : Set (RawArrays T) :=
  (projectRawArrays T q hqT hT) ⁻¹' badEvent T q

theorem rawBadEvent_probability_lower (T q : ℕ) (horizon : T = 2 * q + 1) (hq : 2 ≤ q) :
    (underestimationConstant / 64) * Rates.invPower 886 T ≤
      (rawArraysLaw T).real (rawBadEvent T q (by omega) (by omega)) := by
  have hp := projectRawArrays_preserving T q (by omega) (by omega)
  have heq : (rawArraysLaw T).real (rawBadEvent T q (by omega) (by omega)) =
      (frequentistLaw q).real (badEvent T q) := by
    simp only [measureReal_def]
    rw [← hp.map_eq, Measure.map_apply hp.measurable (badEvent_measurable T q)]
    rfl
  rw [heq]
  exact badEvent_probability_lower T q horizon hq

theorem rawBadEvent_probability_lower_general (T q : ℕ)
    (hT : 5 ≤ T) (hq : 0 < q) (hqT : q ≤ T) (hTq : T ≤ 3 * q) :
    (underestimationConstant / 64) * Rates.invPower 886 T ≤
      (rawArraysLaw T).real (rawBadEvent T q hqT (by omega)) := by
  have hp := projectRawArrays_preserving T q hqT (by omega)
  have heq : (rawArraysLaw T).real (rawBadEvent T q hqT (by omega)) =
      (frequentistLaw q).real (badEvent T q) := by
    simp only [measureReal_def]
    rw [← hp.map_eq, Measure.map_apply hp.measurable (badEvent_measurable T q)]
    rfl
  rw [heq]
  exact badEvent_probability_lower_general T q hT hq hqT hTq

theorem rawBadEvent_even_probability_lower (T q : ℕ) (horizon : T = 2 * q) (hq : 3 ≤ q) :
    (underestimationConstant / 64) * Rates.invPower 886 T ≤
      (rawArraysLaw T).real (rawBadEvent T q (by omega) (by omega)) := by
  exact rawBadEvent_probability_lower_general T q (by omega) (by omega) (by omega) (by omega)

/-- Empirical means computed from the complete raw arrays, with true means
(0,0,1). This uses the paper's one-based arm labels. -/
def fullMeanPath {T : ℕ} (ω : RawArrays T) : Counts.MeanPath :=
  fun i n => if i = 1 then sampleMean n (ω 0)
    else if i = 2 then sampleMean n (ω 1) else sampleMean n (ω 2) + 1

theorem rawBadEvent_implies_full_W (T q : ℕ) (hqT : q ≤ T) (hT : 0 < T)
    (ω : RawArrays T) (c : Counts.Counts)
    (hω : ω ∈ rawBadEvent T q hqT hT)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) (h3 : c.n3 ≤ 1) :
    Drift.paperW T q 40 (fullMeanPath ω) c := by
  have hw := badEvent_implies_W T q (projectRawArrays T q hqT hT ω) c hω h1 h2 h3
  rcases hw with ⟨hX, hY1, hY2, hZ1, hZ2, _⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, h1, h2, h3⟩
  · simpa [Drift.eventX, rawMeanPath, fullMeanPath, projectRawArrays, sampleMean_one hT] using hX
  · simpa [Drift.eventY, rawMeanPath, fullMeanPath, projectRawArrays,
      sampleMean_arrayPrefix_eq hqT q le_rfl] using hY1
  · simpa [Drift.eventY, rawMeanPath, fullMeanPath, projectRawArrays,
      sampleMean_arrayPrefix_eq hqT q le_rfl] using hY2
  · intro n hn hnq
    have h := hZ1 n hn hnq
    simpa [rawMeanPath, fullMeanPath, projectRawArrays,
      sampleMean_arrayPrefix_eq hqT n hnq, sampleMean_arrayPrefix_eq hqT q le_rfl] using h
  · intro n hn hnq
    have h := hZ2 n hn hnq
    simpa [rawMeanPath, fullMeanPath, projectRawArrays,
      sampleMean_arrayPrefix_eq hqT n hnq, sampleMean_arrayPrefix_eq hqT q le_rfl] using h

end

end Draft0.BadEvent
