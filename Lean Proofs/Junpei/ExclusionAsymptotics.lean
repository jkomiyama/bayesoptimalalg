import Junpei.ExclusionBounds
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics

open Filter Asymptotics

namespace Junpei.ExclusionAsymptotics

open Draft0.Rates Junpei.ExclusionTail

noncomputable section

/-- The revised paper's strict cutoff inequality suffices; no integer rounding
of the real exponent or extra cutoff hypothesis is used. -/
theorem paperG0_littleO180_equation11 (CU : ℝ)
    (hcondition : 180 < (CU - 6) ^ 2 / 4) :
    (fun T : ℕ => paperG0 T CU) =o[atTop] invPower 180 := by
  apply isLittleO_of_tendsto'
  · filter_upwards [eventually_gt_atTop (0 : ℕ)] with T hT
    intro hz
    exact ((invPower_pos 180 T hT).ne' hz).elim
  have hdecay := (tendsto_rpow_neg_atTop
    (show 0 < (CU - 6) ^ 2 / 4 - 180 by linarith)).comp
    (tendsto_natCast_atTop_atTop : Tendsto (fun T : ℕ => (T : ℝ)) atTop atTop)
  have hscaled := hdecay.const_mul (2 / Real.sqrt Real.pi)
  simp only [mul_zero] at hscaled
  apply hscaled.congr'
  filter_upwards [eventually_gt_atTop (0 : ℕ)] with T hT
  have hTpos : (0 : ℝ) < T := by exact_mod_cast hT
  dsimp only [Function.comp_apply]
  unfold paperG0 invPower
  rw [div_inv_eq_mul, ← Real.rpow_natCast (T : ℝ) 180]
  rw [mul_assoc, ← Real.rpow_add hTpos]
  congr 1
  congr 1
  norm_num
  ring

theorem paperG0_bigO180_equation11 (CU : ℝ)
    (hcondition : 180 < (CU - 6) ^ 2 / 4) :
    (fun T : ℕ => paperG0 T CU) =O[atTop] invPower 180 :=
  (paperG0_littleO180_equation11 CU hcondition).isBigO

end
end Junpei.ExclusionAsymptotics
