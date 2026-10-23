import Draft0.PosteriorRegret

open MeasureTheory ProbabilityTheory

namespace Draft0.Optimality

open BayesianModel PosteriorRegret

noncomputable section

/-- Unrolling a history-dependent strategy at a fixed observed state gives
an arm at each node and a continuation for every observed innovation. -/
inductive Strategy : ℕ → Type
  | stop (i : Arm) : Strategy 0
  | sample {k : ℕ} (i : Arm) (next : ℝ → Strategy k) : Strategy (k + 1)

def utility : {k : ℕ} → Strategy k → Counts → Means → ℝ
  | 0, .stop i, _, m => m i
  | _ + 1, .sample i next, c, m =>
      ∫ z : ℝ, utility (next z) (advance c i) (shift m i z)
        ∂gaussianReal 0 (innovation c i)

/-- Admissibility states that each continuation's actual expected payoff is
integrable. It places no optimality or paper-estimate condition on a strategy. -/
def Admissible : {k : ℕ} → Strategy k → Counts → Means → Prop
  | 0, .stop _, _, _ => True
  | _ + 1, .sample i next, c, m =>
      (∀ z, Admissible (next z) (advance c i) (shift m i z)) ∧
      Integrable (fun z : ℝ => utility (next z) (advance c i) (shift m i z))
        (gaussianReal 0 (innovation c i))

def optimal : (k : ℕ) → Counts → Means → Strategy k
  | 0, _, m => .stop (recommend m)
  | k + 1, c, m =>
    let i := choose k c m
    .sample i (fun z => optimal k (advance c i) (shift m i z))

/-- The selected Bellman policy attains the recursive value. -/
theorem optimal_utility (k : ℕ) (c : Counts) (m : Means) :
    utility (optimal k c m) c m = value k c m := by
  induction k generalizing c m with
  | zero => exact recommend_max m
  | succ k ih =>
    simp only [optimal, utility, ih]
    exact choose_optimal k c m

theorem optimal_admissible (k : ℕ) (c : Counts) (m : Means) :
    Admissible (optimal k c m) c m := by
  induction k generalizing c m with
  | zero => trivial
  | succ k ih =>
    constructor
    · intro z; exact ih _ _
    · simp only [optimal_utility]
      exact action_integrable k c m (choose k c m)

/-- Bellman's value dominates every admissible history-dependent strategy,
including strategies with observations changing all later actions. -/
theorem utility_le_value {k : ℕ} (s : Strategy k) (c : Counts) (m : Means)
    (hs : Admissible s c m) : utility s c m ≤ value k c m := by
  induction k generalizing c m with
  | zero =>
    cases s with
    | stop i =>
      change m i ≤ terminal m
      rw [← recommend_max]
      exact recommend_greedy m i
  | succ k ih =>
    cases s with
    | sample i next =>
      have h := integral_mono hs.2 (action_integrable k c m i)
        (fun z => ih (next z) _ _ (hs.1 z))
      exact h.trans (action_le_value k c m i)

def expectedRegret : {k : ℕ} → Strategy k → Counts → Means → ℝ
  | 0, .stop i, c, m => potential c m - m i
  | _ + 1, .sample i next, c, m =>
    ∫ z : ℝ, expectedRegret (next z) (advance c i) (shift m i z)
      ∂gaussianReal 0 (innovation c i)

theorem expectedRegret_eq {k : ℕ} (s : Strategy k) (c : Counts) (m : Means)
    (hc : ∀ i, 0 < c i) (hs : Admissible s c m) :
    expectedRegret s c m = potential c m - utility s c m := by
  induction k generalizing c m with
  | zero => cases s; rfl
  | succ k ih =>
    cases s with
    | sample i next =>
      have hp : ∀ j, 0 < advance c i j := by
        intro j
        exact lt_of_lt_of_le (hc j) (Nat.le_add_right _ _)
      simp only [expectedRegret, ih (next _) _ _ hp (hs.1 _), utility]
      rw [integral_sub (shifted_integrable (potential_lipschitz _) _ _ _) hs.2]
      have ht := congrFun (potential_tower c i (hc i)) m
      exact congrArg (fun x : ℝ => x - ∫ z : ℝ, utility (next z) (advance c i) (shift m i z)
        ∂gaussianReal 0 (innovation c i)) ht

/-- Global posterior Bayes optimality after the required initial draws.
No premise assumes that the selected policy is optimal. -/
theorem bayes_optimal {k : ℕ} (s : Strategy k) (c : Counts) (m : Means)
    (hc : ∀ i, 0 < c i) (hs : Admissible s c m) :
    expectedRegret (optimal k c m) c m ≤ expectedRegret s c m := by
  rw [expectedRegret_eq _ _ _ hc (optimal_admissible k c m),
    expectedRegret_eq s c m hc hs, optimal_utility]
  exact sub_le_sub_left (utility_le_value s c m hs) _

end

end Draft0.Optimality
