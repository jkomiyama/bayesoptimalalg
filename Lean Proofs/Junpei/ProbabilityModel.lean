import Junpei.Probability
import Junpei.CountComparison

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.Probability

open Draft0 BadEvent

noncomputable section

def frequentistLawDelta (q : ℕ) (Δ : ℝ) : Measure (FullSample q) :=
  (gaussianReal Δ 1).prod ((arrayLaw q).prod (arrayLaw q))

instance frequentistLawDelta_probability (q : ℕ) (Δ : ℝ) :
    IsProbabilityMeasure (frequentistLawDelta q Δ) := by
  unfold frequentistLawDelta
  infer_instance

def underestimationSet (T : ℕ) (CU : ℝ) : Set ℝ :=
  Iic (-CU * Real.sqrt (Real.log (T : ℝ)))

def badEvent (T q : ℕ) (CU : ℝ) : Set (FullSample q) :=
  underestimationSet T CU ×ˢ (goodArmSet3 T q ×ˢ goodArmSet3 T q)

theorem badEvent_measurable (T q : ℕ) (CU : ℝ) : MeasurableSet (badEvent T q CU) :=
  measurableSet_Iic.prod ((goodArmSet3_measurable T q).prod (goodArmSet3_measurable T q))

def projectRawArraysDelta (T q : ℕ) (Δ : ℝ) (hqT : q ≤ T) (hT : 0 < T)
    (ω : RawArrays T) : FullSample q :=
  (ω 2 ⟨0, hT⟩ + Δ, (arrayPrefix hqT (ω 0), arrayPrefix hqT (ω 1)))

theorem projectRawArraysDelta_preserving (T q : ℕ) (Δ : ℝ)
    (hqT : q ≤ T) (hT : 0 < T) :
    MeasurePreserving (projectRawArraysDelta T q Δ hqT hT)
      (rawArraysLaw T) (frequentistLawDelta q Δ) := by
  have hprefix := arrayPrefix_preserving hqT
  have hfirst : MeasurePreserving (fun ω : Fin T → ℝ => ω ⟨0, hT⟩ + Δ)
      (arrayLaw T) (gaussianReal Δ 1) := by
    have heval := measurePreserving_eval (fun _ : Fin T => gaussianReal 0 1) (⟨0, hT⟩ : Fin T)
    have hshift : MeasurePreserving (fun x : ℝ => x + Δ) (gaussianReal 0 1) (gaussianReal Δ 1) := by
      constructor
      · fun_prop
      · simpa using (gaussianReal_map_add_const (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) Δ)
    exact hshift.comp heval
  have htwo := measurePreserving_piFinTwo (fun _ : Fin 2 => arrayLaw T)
  have hpair := (hprefix.prod hprefix).comp htwo
  have hthree := measurePreserving_piFinSuccAbove (fun _ : Fin 3 => arrayLaw T) (2 : Fin 3)
  have hfull := (hfirst.prod hpair).comp hthree
  convert hfull using 1

def rawBadEvent (T q : ℕ) (Δ CU : ℝ) (hqT : q ≤ T) (hT : 0 < T) : Set (RawArrays T) :=
  projectRawArraysDelta T q Δ hqT hT ⁻¹' badEvent T q CU

theorem rawBadEvent_measurable (T q : ℕ) (Δ CU : ℝ) (hqT : q ≤ T) (hT : 0 < T) :
    MeasurableSet (rawBadEvent T q Δ CU hqT hT) :=
  (projectRawArraysDelta_preserving T q Δ hqT hT).measurable
    (badEvent_measurable T q CU)

def rawMeanPath {T : ℕ} (Δ : ℝ) (ω : RawArrays T) : Counts.MeanPath :=
  fun i n => if i = 1 then sampleMean n (ω 0)
    else if i = 2 then sampleMean n (ω 1) else sampleMean n (ω 2) + Δ

theorem rawBadEvent_implies_newW (T q : ℕ) (Δ CU : ℝ) (hqT : q ≤ T) (hT : 0 < T)
    (ω : RawArrays T) (hω : ω ∈ rawBadEvent T q Δ CU hqT hT)
    (c : Counts.Counts) (hc1 : c.n1 ≤ q) (hc2 : c.n2 ≤ q) (hc3 : c.n3 ≤ 1) :
    JunpeiComparison.newW T q CU (rawMeanPath Δ ω) c := by
  change (projectRawArraysDelta T q Δ hqT hT ω).1 ∈ underestimationSet T CU ∧
    (projectRawArraysDelta T q Δ hqT hT ω).2.1 ∈ goodArmSet3 T q ∧
    (projectRawArraysDelta T q Δ hqT hT ω).2.2 ∈ goodArmSet3 T q at hω
  rcases hω with ⟨hX, ⟨hY1, hZ1⟩, ⟨hY2, hZ2⟩⟩
  unfold JunpeiComparison.newW
  refine ⟨?_, ?_, ?_, ?_, ?_, hc1, hc2, hc3⟩
  · unfold Drift.eventX rawMeanPath
    norm_num
    rw [sampleMean_one hT]
    simpa [projectRawArraysDelta, underestimationSet] using hX
  · unfold Drift.eventY rawMeanPath
    norm_num
    have he := sampleMean_arrayPrefix_eq hqT q le_rfl (ω 0)
    simpa [goodMeanSet, projectRawArraysDelta, he, Rates.invPower, one_div] using hY1
  · unfold Drift.eventY rawMeanPath
    norm_num
    have he := sampleMean_arrayPrefix_eq hqT q le_rfl (ω 1)
    simpa [goodMeanSet, projectRawArraysDelta, he, Rates.invPower, one_div] using hY2
  · intro n hn hnq
    have h := hZ1 n hn hnq
    have hen := sampleMean_arrayPrefix_eq hqT n hnq (ω 0)
    have heq := sampleMean_arrayPrefix_eq hqT q le_rfl (ω 0)
    simpa [rawMeanPath, projectRawArraysDelta, driftThreshold3, hen, heq] using h
  · intro n hn hnq
    have h := hZ2 n hn hnq
    have hen := sampleMean_arrayPrefix_eq hqT n hnq (ω 1)
    have heq := sampleMean_arrayPrefix_eq hqT q le_rfl (ω 1)
    simpa [rawMeanPath, projectRawArraysDelta, driftThreshold3, hen, heq] using h

theorem badEvent_probability_eq (T q : ℕ) (Δ CU : ℝ) :
    (frequentistLawDelta q Δ).real (badEvent T q CU) =
      (gaussianReal Δ 1).real (underestimationSet T CU) *
        ((arrayLaw q).real (goodArmSet3 T q)) ^ 2 := by
  unfold frequentistLawDelta badEvent Measure.real
  rw [Measure.prod_prod, Measure.prod_prod, ENNReal.toReal_mul, ENNReal.toReal_mul]
  ring

theorem rawBadEvent_probability_eq (T q : ℕ) (Δ CU : ℝ) (hqT : q ≤ T) (hT : 0 < T) :
    (rawArraysLaw T).real (rawBadEvent T q Δ CU hqT hT) =
      (gaussianReal Δ 1).real (underestimationSet T CU) *
        ((arrayLaw q).real (goodArmSet3 T q)) ^ 2 := by
  have hp := projectRawArraysDelta_preserving T q Δ hqT hT
  have h := hp.measure_preimage (badEvent_measurable T q CU).nullMeasurableSet
  change rawArraysLaw T (rawBadEvent T q Δ CU hqT hT) =
    frequentistLawDelta q Δ (badEvent T q CU) at h
  unfold Measure.real
  rw [h]
  exact badEvent_probability_eq T q Δ CU

end

end Junpei.Probability
