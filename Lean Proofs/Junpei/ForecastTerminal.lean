import Junpei.Forecast

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal
open Draft0.Counts Draft0.Drift

namespace Junpei.Forecast

noncomputable section

theorem terminal_gap_bound (T q : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (hlarge : 9 ≤ T)
    (hW : JunpeiComparison.newW T q CU p c)
    (hf : (c.n1 = q ∧ c.n2 = q - 1) ∨ (c.n1 = q - 1 ∧ c.n2 = q)) :
    |terminalGap p c| ≤ 10 * Real.sqrt (Real.log (T : ℝ)) / (T : ℝ) := by
  rcases hW with ⟨_, hY1, hY2, hZ1, hZ2, h1, h2, _⟩
  have hq : 2 ≤ q := by omega
  have hTr : (9 : ℝ) ≤ T := by exact_mod_cast hlarge
  have hTp : 0 < (T : ℝ) := by linarith
  have hqr : (2 : ℝ) ≤ q := by exact_mod_cast hq
  have hqsub : 0 < ((q - 1 : ℕ) : ℝ) := by exact_mod_cast (show 0 < q - 1 by omega)
  have hn1 : 1 ≤ c.n1 := by rcases hf with h | h <;> omega
  have hn2 : 1 ≤ c.n2 := by rcases hf with h | h <;> omega
  have hsumq : reciprocalSqSum q q = 0 := by simp [reciprocalSqSum]
  have hsumprev : reciprocalSqSum (q - 1) q = 1 / ((q - 1 : ℕ) : ℝ) ^ 2 := by
    unfold reciprocalSqSum
    have hIco : Finset.Ico (q - 1) q = {q - 1} := Nat.Ico_pred_singleton (by omega)
    rw [hIco, Finset.sum_singleton]
  have hsqrtprev : Real.sqrt (reciprocalSqSum (q - 1) q) = 1 / ((q - 1 : ℕ) : ℝ) := by
    rw [hsumprev]
    have he : 1 / ((q - 1 : ℕ) : ℝ) ^ 2 = (1 / ((q - 1 : ℕ) : ℝ)) ^ 2 := by ring
    rw [he, Real.sqrt_sq (by positivity)]
  have hroots : Real.sqrt (reciprocalSqSum c.n1 q) +
      Real.sqrt (reciprocalSqSum c.n2 q) = 1 / ((q - 1 : ℕ) : ℝ) := by
    rcases hf with h | h
    · rw [h.1, h.2, hsumq, hsqrtprev]
      simp
    · rw [h.1, h.2, hsumq, hsqrtprev]
      simp
  have hZ1' := hZ1 c.n1 hn1 h1
  have hZ2' := hZ2 c.n2 hn2 h2
  have htri1 := abs_sub_le (p 1 c.n1) (p 1 q) (p 2 c.n2)
  have htri2 := abs_sub_le (p 1 q) (p 2 q) (p 2 c.n2)
  rw [abs_sub_comm (p 2 q) (p 2 c.n2)] at htri2
  have hanchor := abs_sub (p 1 q) (p 2 q)
  unfold eventY at hY1 hY2
  have hanchors : |p 1 q - p 2 q| ≤ 2 / (T : ℝ) ^ 2 := by
    calc
      _ ≤ |p 1 q| + |p 2 q| := hanchor
      _ ≤ 1 / (T : ℝ) ^ 2 + 1 / (T : ℝ) ^ 2 := add_le_add hY1 hY2
      _ = _ := by ring
  have habs : |terminalGap p c| ≤
      3 * Real.sqrt (Real.log (T : ℝ)) *
        (Real.sqrt (reciprocalSqSum c.n1 q) + Real.sqrt (reciprocalSqSum c.n2 q)) +
      2 / (T : ℝ) ^ 2 := by
    unfold terminalGap
    nlinarith only [htri1, htri2, hanchors, hZ1', hZ2']
  rw [hroots] at habs
  have hrecip : 1 / ((q - 1 : ℕ) : ℝ) ≤ 3 / (T : ℝ) := by
    apply (div_le_div_iff₀ hqsub hTp).mpr
    have hcast : ((q - 1 : ℕ) : ℝ) = (q : ℝ) - 1 := by
      rw [Nat.cast_sub (show 1 ≤ q by omega), Nat.cast_one]
    have hTcast : (T : ℝ) = 2 * (q : ℝ) + 1 := by exact_mod_cast hT
    rw [hcast]
    linarith
  have hlog := log_horizon_ge_one T (by omega)
  have hsqrt : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.one_le_sqrt.mpr hlog
  have htiny : 2 / (T : ℝ) ^ 2 ≤ Real.sqrt (Real.log (T : ℝ)) / (T : ℝ) := by
    have ha : 2 / (T : ℝ) ^ 2 ≤ 1 / (T : ℝ) := by
      apply (div_le_div_iff₀ (by positivity) hTp).mpr
      nlinarith only [hTr]
    exact ha.trans (div_le_div_of_nonneg_right hsqrt hTp.le)
  have hscale := mul_le_mul_of_nonneg_left hrecip
    (by positivity : 0 ≤ 3 * Real.sqrt (Real.log (T : ℝ)))
  calc
    |terminalGap p c| ≤ _ := habs
    _ ≤ 3 * Real.sqrt (Real.log (T : ℝ)) * (3 / (T : ℝ)) +
        Real.sqrt (Real.log (T : ℝ)) / (T : ℝ) := add_le_add hscale htiny
    _ = _ := by ring

end

end Junpei.Forecast
