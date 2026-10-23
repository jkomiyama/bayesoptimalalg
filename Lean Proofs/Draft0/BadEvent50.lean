import Draft0.BadEvent

/-!
The complete frequentist bad event with CU=50. This stronger underestimation
event accommodates the corrected even-budget drift exponent. All raw sample
streams and their Gaussian product laws are reused from Draft0.BadEvent.
-/

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.BadEvent50

open Draft0.BadEvent

noncomputable section

def underestimationInterval (T : ℕ) : Set ℝ :=
  Icc (-50 * Real.sqrt (Real.log (T : ℝ)) - 1) (-50 * Real.sqrt (Real.log (T : ℝ)))

theorem underestimationPDF_lower (T : ℕ) (hT : 5 ≤ T) (x : ℝ)
    (hx : x ∈ underestimationInterval T) :
    underestimationConstant * Rates.invPower 1352 T ≤ gaussianPDFReal 1 1 x := by
  have htpos : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hlog := Drift.log_horizon_ge_one T hT
  have hs0 := Real.sqrt_nonneg (Real.log (T : ℝ))
  have hs1 : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.one_le_sqrt.mpr hlog
  have hsq : (Real.sqrt (Real.log (T : ℝ))) ^ 2 = Real.log (T : ℝ) :=
    Real.sq_sqrt (by linarith only [hlog])
  have hdiff0 : x - 1 ≤ 0 := by linarith only [hx.2, hs0]
  have hdiff : |x - 1| ≤ 52 * Real.sqrt (Real.log (T : ℝ)) := by
    rw [abs_of_nonpos hdiff0]
    linarith only [hx.1, hs1]
  have hquad : (x - 1) ^ 2 ≤ 2704 * Real.log (T : ℝ) := by
    have h := (sq_le_sq₀ (abs_nonneg (x - 1)) (by positivity)).mpr hdiff
    rw [sq_abs] at h
    nlinarith only [h, hsq]
  have hexp : Rates.invPower 1352 T ≤ Real.exp (-(x - 1) ^ 2 / 2) := by
    unfold Rates.invPower
    rw [← PredictiveBound.exp_neg_nat_log 1352 htpos]
    apply Real.exp_le_exp.mpr
    norm_num
    nlinarith only [hquad]
  have hmul := mul_le_mul_of_nonneg_left hexp underestimationConstant_positive.le
  simpa only [gaussianPDFReal, NNReal.coe_one, mul_one, underestimationConstant] using hmul

theorem underestimation_probability_lower (T : ℕ) (hT : 5 ≤ T) :
    underestimationConstant * Rates.invPower 1352 T ≤
      (gaussianReal 1 1).real (underestimationInterval T) := by
  have hmass : (gaussianReal 1 1).real (underestimationInterval T) =
      ∫ x in underestimationInterval T, gaussianPDFReal 1 1 x := by
    rw [measureReal_def, gaussianReal_apply_eq_integral 1 (by norm_num : (1 : ℝ≥0) ≠ 0),
      ENNReal.toReal_ofReal]
    exact integral_nonneg (gaussianPDFReal_nonneg 1 1)
  have hmono : (∫ _ in underestimationInterval T, underestimationConstant * Rates.invPower 1352 T) ≤
      ∫ x in underestimationInterval T, gaussianPDFReal 1 1 x := by
    apply setIntegral_mono_on
      (integrableOn_const (by exact isCompact_Icc.measure_ne_top))
      (integrable_gaussianPDFReal 1 1).integrableOn measurableSet_Icc
    intro x hx
    exact underestimationPDF_lower T hT x hx
  have hconst : (∫ _ in underestimationInterval T, underestimationConstant * Rates.invPower 1352 T) =
      underestimationConstant * Rates.invPower 1352 T := by
    unfold underestimationInterval
    rw [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Icc,
      ENNReal.toReal_ofReal (by linarith)]
    ring
  rw [← hconst, hmass]
  exact hmono

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

theorem badEvent_probability_lower_general (T q : ℕ)
    (hT : 5 ≤ T) (hq : 0 < q) (hqT : q ≤ T) (hTq : T ≤ 3 * q) :
    (underestimationConstant / 64) * Rates.invPower 1356 T ≤
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

theorem badEvent_implies_W (T q : ℕ) (ω : FullSample q) (c : Counts.Counts)
    (hω : ω ∈ badEvent T q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) (h3 : c.n3 ≤ 1) :
    Drift.paperW T q 50 (rawMeanPath ω) c := by
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

def rawBadEvent (T q : ℕ) (hqT : q ≤ T) (hT : 0 < T) : Set (RawArrays T) :=
  (projectRawArrays T q hqT hT) ⁻¹' badEvent T q

theorem rawBadEvent_measurable (T q : ℕ) (hqT : q ≤ T) (hT : 0 < T) :
    MeasurableSet (rawBadEvent T q hqT hT) :=
  (badEvent_measurable T q).preimage (projectRawArrays_preserving T q hqT hT).measurable

theorem rawBadEvent_probability_lower_general (T q : ℕ)
    (hT : 5 ≤ T) (hq : 0 < q) (hqT : q ≤ T) (hTq : T ≤ 3 * q) :
    (underestimationConstant / 64) * Rates.invPower 1356 T ≤
      (rawArraysLaw T).real (rawBadEvent T q hqT (by omega)) := by
  have hp := projectRawArrays_preserving T q hqT (by omega)
  have heq : (rawArraysLaw T).real (rawBadEvent T q hqT (by omega)) =
      (frequentistLaw q).real (badEvent T q) := by
    simp only [measureReal_def]
    rw [← hp.map_eq, Measure.map_apply hp.measurable (badEvent_measurable T q)]
    rfl
  rw [heq]
  exact badEvent_probability_lower_general T q hT hq hqT hTq

theorem rawBadEvent_odd_probability_lower (T q : ℕ) (horizon : T = 2 * q + 1) (hq : 2 ≤ q) :
    (underestimationConstant / 64) * Rates.invPower 1356 T ≤
      (rawArraysLaw T).real (rawBadEvent T q (by omega) (by omega)) := by
  exact rawBadEvent_probability_lower_general T q (by omega) (by omega) (by omega) (by omega)

theorem rawBadEvent_even_probability_lower (T q : ℕ) (horizon : T = 2 * q) (hq : 3 ≤ q) :
    (underestimationConstant / 64) * Rates.invPower 1356 T ≤
      (rawArraysLaw T).real (rawBadEvent T q (by omega) (by omega)) := by
  exact rawBadEvent_probability_lower_general T q (by omega) (by omega) (by omega) (by omega)

theorem rawBadEvent_implies_full_W (T q : ℕ) (hqT : q ≤ T) (hT : 0 < T)
    (ω : RawArrays T) (c : Counts.Counts)
    (hω : ω ∈ rawBadEvent T q hqT hT)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) (h3 : c.n3 ≤ 1) :
    Drift.paperW T q 50 (fullMeanPath ω) c := by
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

end Draft0.BadEvent50
