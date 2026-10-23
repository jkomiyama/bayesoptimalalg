import Draft0.BadEvent

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.Probability

open Draft0 BadEvent

noncomputable section

def f3 (T : ℕ) : ℝ := 2 * Real.exp (-(7 / 2 : ℝ) * Real.log (T : ℝ))

theorem f3_eq_rpow (T : ℕ) (hT : 0 < T) :
    f3 T = 2 * (T : ℝ) ^ (-(7 / 2 : ℝ)) := by
  rw [f3, Real.rpow_def_of_pos (by exact_mod_cast hT)]
  congr 1
  congr 1
  ring

def driftThreshold3 (T n q : ℕ) : ℝ :=
  3 * Real.sqrt (Real.log (T : ℝ)) * Real.sqrt (Drift.reciprocalSqSum n q)

def driftFailure3 (T n q : ℕ) : Set (Fin q → ℝ) :=
  {ω | driftThreshold3 T n q < |sampleMean n ω - sampleMean q ω|}

def goodDriftSet3 (T q : ℕ) : Set (Fin q → ℝ) :=
  {ω | ∀ n : ℕ, 1 ≤ n → n ≤ q →
    |sampleMean n ω - sampleMean q ω| ≤ driftThreshold3 T n q}

def goodArmSet3 (T q : ℕ) : Set (Fin q → ℝ) :=
  goodMeanSet T q ∩ goodDriftSet3 T q

theorem goodDriftSet3_measurable (T q : ℕ) : MeasurableSet (goodDriftSet3 T q) := by
  unfold goodDriftSet3
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

theorem goodArmSet3_measurable (T q : ℕ) : MeasurableSet (goodArmSet3 T q) :=
  (goodMeanSet_measurable T q).inter (goodDriftSet3_measurable T q)

theorem driftFailure3_probability_le (T n q : ℕ) (hT : 5 ≤ T) (hn : 0 < n) (hnq : n ≤ q) :
    (arrayLaw q).real (driftFailure3 T n q) ≤ 2 * Real.exp (-(9 / 2 : ℝ) * Real.log (T : ℝ)) := by
  by_cases hnqeq : n = q
  · subst n
    have hset : driftFailure3 T q q = ∅ := by
      ext ω
      simp [driftFailure3, driftThreshold3, Drift.reciprocalSqSum]
    rw [hset]
    simp only [measureReal_empty]
    exact mul_nonneg (by norm_num) (Real.exp_pos _).le
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
  have hthreshold0 : 0 ≤ driftThreshold3 T n q := by unfold driftThreshold3; positivity
  have hthresholdsq : (driftThreshold3 T n q) ^ 2 =
      9 * Real.log (T : ℝ) * Drift.reciprocalSqSum n q := by
    unfold driftThreshold3
    rw [mul_pow, mul_pow, Real.sq_sqrt hlog0,
      Real.sq_sqrt (Drift.reciprocalSqSum_nonnegative n q)]
    norm_num
  have hscaled := mul_le_mul_of_nonneg_left hVarBound hlog0
  have hquot : (9 / 2 : ℝ) * Real.log (T : ℝ) ≤ (driftThreshold3 T n q) ^ 2 / (2 * (driftVariance n q : ℝ)) := by
    apply (le_div_iff₀ (by positivity)).mpr
    rw [hthresholdsq]
    nlinarith only [hscaled]
  have hexp : Real.exp (-(driftThreshold3 T n q) ^ 2 / (2 * (driftVariance n q : ℝ))) ≤
      Real.exp (-(9 / 2 : ℝ) * Real.log (T : ℝ)) := by
    apply Real.exp_le_exp.mpr
    rw [neg_div]
    norm_num
    linarith only [hquot]
  have hsub := gaussian_law_subgaussian (sampleMean_difference_law n q)
  have hright := hsub.measure_ge_le hthreshold0
  have hleft := hsub.neg.measure_ge_le hthreshold0
  have hsubset : driftFailure3 T n q ⊆
      {ω | driftThreshold3 T n q ≤ sampleMean n ω - sampleMean q ω} ∪
        {ω | driftThreshold3 T n q ≤ -(sampleMean n ω - sampleMean q ω)} := by
    intro ω hω
    change driftThreshold3 T n q < |sampleMean n ω - sampleMean q ω| at hω
    rw [lt_abs] at hω
    rcases hω with hω | hω
    · exact Or.inl (le_of_lt hω)
    · exact Or.inr (le_of_lt hω)
  have hmono := measureReal_mono (μ := arrayLaw q) hsubset
  have hunion := measureReal_union_le (μ := arrayLaw q)
    {ω | driftThreshold3 T n q ≤ sampleMean n ω - sampleMean q ω}
    {ω | driftThreshold3 T n q ≤ -(sampleMean n ω - sampleMean q ω)}
  simp only [Pi.neg_apply] at hleft
  linarith only [hmono, hunion, hright, hleft, hexp]

theorem goodDriftSet3_compl_probability_le (T q : ℕ) (hT : 5 ≤ T) (hqT : q ≤ T) :
    (arrayLaw q).real (goodDriftSet3 T q)ᶜ ≤ f3 T := by
  have hsubset : (goodDriftSet3 T q)ᶜ ⊆ ⋃ i : Fin q, driftFailure3 T ((i : ℕ) + 1) q := by
    intro ω hω
    simp only [goodDriftSet3, Set.mem_compl_iff, Set.mem_setOf_eq] at hω
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
    (fun i : Fin q => driftFailure3 T ((i : ℕ) + 1) q)
  have hsum : (∑ i : Fin q, (arrayLaw q).real (driftFailure3 T ((i : ℕ) + 1) q)) ≤
      (q : ℝ) * (2 * Real.exp (-(9 / 2 : ℝ) * Real.log (T : ℝ))) := by
    calc
      _ ≤ ∑ _i : Fin q, 2 * Real.exp (-(9 / 2 : ℝ) * Real.log (T : ℝ)) := by
        apply Finset.sum_le_sum
        intro i _hi
        exact driftFailure3_probability_le T _ q hT (by omega) (by have hi := i.isLt; omega)
      _ = _ := by simp
  have hqTr : (q : ℝ) ≤ (T : ℝ) := by exact_mod_cast hqT
  have hscale := mul_le_mul_of_nonneg_right hqTr
    (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2)
      (Real.exp_pos (-(9 / 2 : ℝ) * Real.log (T : ℝ))).le)
  have hrate : (T : Real) * (2 * Real.exp (-(9 / 2 : Real) * Real.log (T : Real))) = f3 T := by
    unfold f3
    calc
      _ = 2 * (Real.exp (Real.log (T : Real)) * Real.exp (-(9 / 2 : Real) * Real.log (T : Real))) := by
        rw [Real.exp_log (by exact_mod_cast (show 0 < T by omega))]
        ring
      _ = 2 * Real.exp (Real.log (T : Real) + -(9 / 2 : Real) * Real.log (T : Real)) := by rw [Real.exp_add]
      _ = _ := by congr 2; ring
  rw [hrate] at hscale
  linarith only [hmono, hunion, hsum, hscale]

theorem lemma5 (T q : ℕ) (hT : 5 ≤ T) (hqT : q ≤ T) :
    (arrayLaw q).real (goodMeanSet T q ∩ (goodDriftSet3 T q)ᶜ) ≤
      2 * (T : ℝ) ^ (-(7 / 2 : ℝ)) := by
  rw [← f3_eq_rpow T (by omega)]
  exact (measureReal_mono (inter_subset_right)).trans
    (goodDriftSet3_compl_probability_le T q hT hqT)

theorem lemma4 (T q : ℕ) (hT : T = 2 * q + 1) (hq : 0 < q) :
    SmallBall.f2 (T : ℝ) ≤ (arrayLaw q).real (goodMeanSet T q) := by
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
  exact h

theorem goodArmSet3_probability_lower (T q : ℕ)
    (horizon : T = 2 * q + 1) (hq : 2 ≤ q) :
    SmallBall.f2 (T : ℝ) - f3 T ≤ (arrayLaw q).real (goodArmSet3 T q) := by
  have hmean := lemma4 T q horizon (by omega)
  have hdrift := goodDriftSet3_compl_probability_le T q (by omega) (by omega)
  have hdecomp := measureReal_inter_add_diff (μ := arrayLaw q)
    (s := goodMeanSet T q) (goodDriftSet3_measurable T q)
  have hsubset : goodMeanSet T q \ goodDriftSet3 T q ⊆ (goodDriftSet3 T q)ᶜ := by
    exact fun _ h => h.2
  have hdiff := measureReal_mono (μ := arrayLaw q) hsubset
  unfold goodArmSet3
  linarith only [hmean, hdrift, hdecomp, hdiff]

theorem f2_minus_f3_nonnegative (T : ℕ) (hT : 8 ≤ T) :
    0 ≤ SmallBall.f2 (T : ℝ) - f3 T := by
  have hTr : (8 : ℝ) ≤ T := by exact_mod_cast hT
  have hTp : 0 < (T : ℝ) := by linarith
  have hlog : 0 ≤ Real.log (T : ℝ) := Real.log_nonneg (by linarith)
  have hsmall : f3 T ≤ 2 * Rates.invPower 3 T := by
    unfold f3 Rates.invPower
    rw [← PredictiveBound.exp_neg_nat_log 3 hTp]
    apply mul_le_mul_of_nonneg_left _ (by norm_num)
    apply Real.exp_le_exp.mpr
    norm_num
    nlinarith
  have hrate : 2 * Rates.invPower 3 T ≤ (1 / 4 : ℝ) * Rates.invPower 2 T := by
    unfold Rates.invPower
    field_simp [ne_of_gt hTp]
    nlinarith only [hTr]
  have hf2 := f2_polynomial_lower T (by omega)
  linarith

end

end Junpei.Probability
