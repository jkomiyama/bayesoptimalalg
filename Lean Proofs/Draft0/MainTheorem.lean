import Draft0.DynamicPolicy
import Draft0.EvenDynamicPolicy
import Draft0.PathInduction
import Draft0.FrequentistModel
import Draft0.BadEvent50
import Draft0.Optimality
import Draft0.PosteriorLaw

/-!
End-to-end regret bounds for the explicitly defined posterior Bayes optimal
Gaussian algorithm on true means (0,0,1). The event probabilities, Bellman
comparisons, all action choices and terminal recommendation are proved.
-/

open MeasureTheory ProbabilityTheory Filter Asymptotics

namespace Draft0.MainTheorem

open BayesianModel PolicyPaths PathInduction FrequentistModel

noncomputable section

theorem modelCounts_project (c : BayesianModel.Counts) :
    DynamicPolicy.modelCounts (projectCounts c) = c := by
  ext i
  fin_cases i <;> simp [DynamicPolicy.modelCounts, projectCounts]

theorem modelMeans_project (p : Path) (c : BayesianModel.Counts) :
    DynamicPolicy.modelMeans (paperPath p) (projectCounts c) = meansAt p c := by
  ext i
  fin_cases i <;> simp [DynamicPolicy.modelMeans, paperPath, projectCounts, meansAt]

theorem decision_selected (T s : ℕ) (p : Path) :
    DynamicPolicy.decision T (s + 1) (paperPath p) (projectCounts (counts T p s)) =
      selected T p (s + 1) := by
  simp only [DynamicPolicy.decision, modelCounts_project, modelMeans_project,
    selected, Nat.add_sub_cancel]

theorem invariant_horizon (T q s : ℕ) (p : Path)
    (hc : MainBridge.invariant q s (counts T p s)) :
    Counts.horizonConsistent (s + 1) (projectCounts (counts T p s)) := by
  have h₀ := counts_positive T p s 0
  have h₁ := counts_positive T p s 1
  simp only [Counts.horizonConsistent, projectCounts]
  exact ⟨h₀, h₁, hc.2.2.1, by simpa only [Nat.add_sub_cancel] using hc.2.2.2⟩

theorem fullMeanPath_eq {T : ℕ} (ω : BadEvent.RawArrays T) :
    paperPath (empiricalPath ω) = BadEvent.fullMeanPath ω := by
  funext i n
  simp [paperPath, empiricalPath, trueMean, BadEvent.fullMeanPath]

theorem W50_implies_W40 (T q : ℕ) (p : Counts.MeanPath) (c : Counts.Counts)
    (hW : Drift.paperW T q 50 p c) : Drift.paperW T q 40 p c := by
  refine ⟨?_, hW.2⟩
  have hx := hW.1
  unfold Drift.eventX at *
  nlinarith [Real.sqrt_nonneg (Real.log (T : ℝ))]

theorem rawBadEvent50_goodPath (T q : ℕ) (hq : 2 ≤ q) (hqT : q ≤ T) (hT : 0 < T)
    (ω : BadEvent.RawArrays T) (hω : ω ∈ BadEvent50.rawBadEvent T q hqT hT) :
    goodPath T q (empiricalPath ω) := by
  have hw := BadEvent50.rawBadEvent_implies_full_W T q hqT hT ω
    (⟨1, 1, 1⟩ : Counts.Counts) hω (by change 1 ≤ q; omega)
      (by change 1 ≤ q; omega) (by norm_num)
  rw [← fullMeanPath_eq ω] at hw
  have h40 := W50_implies_W40 T q _ _ hw
  exact ⟨h40.1, h40.2.1, h40.2.2.1, h40.2.2.2.1, h40.2.2.2.2.1⟩

theorem odd_goodPath_wrong (N : ℕ)
    (hN : ∀ (T : ℕ), N ≤ T → ∀ (q t : ℕ) (p : Counts.MeanPath) (c : Counts.Counts),
      T = 2 * q + 1 → 4 ≤ t → t ≤ T → Counts.horizonConsistent t c →
      Drift.paperW T q 40 p c → DynamicPolicy.admissible q c (DynamicPolicy.decision T t p c))
    (T q : ℕ) (hlarge : N ≤ T) (hT : T = 2 * q + 1) (hq : 2 ≤ q)
    (p : Path) (hp : goodPath T q p) : recommendation T p ≠ 2 := by
  have hpolicy (s : ℕ) (hs : 3 ≤ s) (hsT : s < T)
      (hc : MainBridge.invariant q s (counts T p s)) :
      MainBridge.admissible q (counts T p s) (selected T p (s + 1)) := by
    have ha := hN T hlarge q (s + 1) (paperPath p) (projectCounts (counts T p s))
      hT (by omega) (by omega) (invariant_horizon T q s p hc)
      (goodPath_W T q s p (counts T p s) hp hc)
    rw [decision_selected] at ha
    exact ha
  have hi := counts_invariant T q hq (by omega) p hpolicy T (by omega) le_rfl
  exact goodPath_recommendation_wrong T q hq (by omega) (Or.inr hT) p hp hi

/-- No unproved event or policy hypothesis occurs in this odd-budget main bound. -/
theorem odd_main_polynomial_lower_bound :
    ∃ N : ℕ, ∀ (T q : ℕ), N ≤ T → T = 2 * q + 1 →
      (BadEvent.underestimationConstant / 64) * Rates.invPower 1356 T ≤ regret T := by
  obtain ⟨N, hN⟩ := DynamicPolicy.eventually_all_W_admissible
  refine ⟨max N 5, ?_⟩
  intro T q hlarge hT
  have h5 : 5 ≤ T := (le_max_right N 5).trans hlarge
  have hq : 2 ≤ q := by omega
  rw [regret_eq_wrong_probability]
  apply (BadEvent50.rawBadEvent_odd_probability_lower T q hT hq).trans
  refine measureReal_mono ?_ (measure_ne_top _ _)
  intro ω hω
  exact odd_goodPath_wrong N hN T q ((le_max_left N 5).trans hlarge) hT hq
    (empiricalPath ω) (rawBadEvent50_goodPath T q hq (by omega) (by omega) ω hω)

theorem even_rawBadEvent_wrong (N : ℕ)
    (hN : ∀ (T : ℕ), N ≤ T → ∀ (q t : ℕ) (p : Counts.MeanPath) (c : Counts.Counts),
      T = 2 * q → 4 ≤ t → t ≤ T → Counts.horizonConsistent t c →
      Drift.paperW T q 50 p c → DynamicPolicy.admissible q c (DynamicPolicy.decision T t p c))
    (T q : ℕ) (hlarge : N ≤ T) (hT : T = 2 * q) (hq : 3 ≤ q)
    (ω : BadEvent.RawArrays T) (hω : ω ∈ BadEvent50.rawBadEvent T q (by omega) (by omega)) :
    actualRecommendation T ω ≠ 2 := by
  let p := empiricalPath ω
  have hp := rawBadEvent50_goodPath T q (by omega) (by omega) (by omega) ω hω
  have hpolicy (s : ℕ) (hs : 3 ≤ s) (hsT : s < T)
      (hc : MainBridge.invariant q s (counts T p s)) :
      MainBridge.admissible q (counts T p s) (selected T p (s + 1)) := by
    have hw := BadEvent50.rawBadEvent_implies_full_W T q (by omega) (by omega) ω
      (projectCounts (counts T p s)) hω hc.1 hc.2.1 hc.2.2.1.le
    rw [← fullMeanPath_eq ω] at hw
    have ha := hN T hlarge q (s + 1) (paperPath p) (projectCounts (counts T p s))
      hT (by omega) (by omega) (invariant_horizon T q s p hc) hw
    rw [decision_selected] at ha
    exact ha
  have hi := counts_invariant T q (by omega) (by omega) p hpolicy T (by omega) le_rfl
  exact goodPath_recommendation_wrong T q (by omega) (by omega) (Or.inl hT) p hp hi

theorem even_main_polynomial_lower_bound :
    ∃ N : ℕ, ∀ (T q : ℕ), N ≤ T → T = 2 * q →
      (BadEvent.underestimationConstant / 64) * Rates.invPower 1356 T ≤ regret T := by
  obtain ⟨N, hN⟩ := EvenDynamicPolicy.eventually_all_W_admissible
  refine ⟨max N 12, ?_⟩
  intro T q hlarge hT
  have h12 : 12 ≤ T := (le_max_right N 12).trans hlarge
  have hq : 3 ≤ q := by omega
  rw [regret_eq_wrong_probability]
  apply (BadEvent50.rawBadEvent_even_probability_lower T q hT hq).trans
  refine measureReal_mono ?_ (measure_ne_top _ _)
  intro ω hω
  exact even_rawBadEvent_wrong N hN T q ((le_max_left N 12).trans hlarge) hT hq ω hω

/-- The complete main lower bound, valid for every sufficiently large natural
budget, not only an odd subsequence. There are no stochastic or policy premises. -/
theorem main_polynomial_lower_bound :
    ∃ N : ℕ, ∀ T : ℕ, N ≤ T →
      (BadEvent.underestimationConstant / 64) * Rates.invPower 1356 T ≤ regret T := by
  obtain ⟨No, ho⟩ := odd_main_polynomial_lower_bound
  obtain ⟨Ne, he⟩ := even_main_polynomial_lower_bound
  refine ⟨max No Ne, ?_⟩
  intro T hT
  have hmod := Nat.mod_lt T (by norm_num : 0 < 2)
  have hdiv := Nat.mod_add_div T 2
  have hparity : T = 2 * (T / 2) ∨ T = 2 * (T / 2) + 1 := by omega
  rcases hparity with hEven | hOdd
  · exact he T (T / 2) ((le_max_right No Ne).trans hT) hEven
  · exact ho T (T / 2) ((le_max_left No Ne).trans hT) hOdd

/-- A concrete witness for the paper's main polynomial-regret assertion:
three arms, true means (0,0,1), and positive exponent C=1356. -/
theorem main_theorem :
    ∃ C : ℕ, 0 < C ∧ Rates.invPower C =O[atTop] regret := by
  refine ⟨1356, by norm_num, ?_⟩
  obtain ⟨N, hN⟩ := main_polynomial_lower_bound
  apply Rates.omega_of_eventual_lower_bound
    (div_pos BadEvent.underestimationConstant_positive (by norm_num : (0 : ℝ) < 64))
  exact Filter.eventually_atTop.mpr ⟨N, hN⟩

/-- The same defined Bellman algorithm is posterior Bayes optimal and has the
proved frequentist polynomial lower bound. The only strategy qualification is
the integrability required for its expected regret to be defined. -/
theorem end_to_end_certificate :
    (∀ (k : ℕ) (c : BayesianModel.Counts) (m : Means), (∀ i, 0 < c i) →
      ∀ s : Optimality.Strategy k, Optimality.Admissible s c m →
        Optimality.expectedRegret (Optimality.optimal k c m) c m ≤
          Optimality.expectedRegret s c m) ∧
    (∃ C : ℕ, 0 < C ∧ Rates.invPower C =O[atTop] regret) := by
  exact ⟨fun _ _ _ hc s hs => Optimality.bayes_optimal s _ _ hc hs, main_theorem⟩

end

end Draft0.MainTheorem
