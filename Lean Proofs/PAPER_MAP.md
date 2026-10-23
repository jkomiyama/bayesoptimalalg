# Paper-to-Lean map

This guide follows the numbering in `junpei_final.pdf` (48 pages), including the corrected Footnote 14 and Equation (26). It describes correspondence; Lean checks the mathematical statements in [LeanProofs.lean](LeanProofs.lean). Run the commands in [README.md](README.md).

Links point to declaration lines. Every name has prefix `BayesOptimal.` (omitted from link labels). **Proved** refers to the formal result with its stated assumptions. Definitions introduce objects; open or unformalized entries are not claimed as proofs. Page references use PDF pages alongside the stable theorem/equation numbers.

## Contents

- [Numbered theorems and lemmas](#numbered-theorems-and-lemmas)
- [Definitions](#definitions)
- [Equations](#equations)
- [Remarks and footnotes](#remarks-and-footnotes)
- [Supporting arguments and constructions](#supporting-arguments-and-constructions)
- [Coverage limits and manuscript consistency](#coverage-limits-and-manuscript-consistency)

## Numbered theorems and lemmas

<a id="theorem-1"></a>

### Theorem 1: Bayes optimal actions, recommendations, and posterior risk

**Status:** Proved. **PDF pages:** 10-11.

FiniteHorizon.theorem1 is the principal endpoint for any finite collection of N Borel sampling rules and one terminal rule, defined only on the paper's proper Gaussian states, for arbitrary finite K and positive real precisions. Its premise is actual posterior-risk minimization against concrete Borel full-history policies at each state and remaining budget H <= N. It derives the Bellman risk identity, almost-sure EBI-maximizing sampling, and maximal-mean recommendation. Padding the finite rule collection and extending outside positive precisions preserve direct execution and risk. The supporting generic theorems also derive these conclusions from single-state minimization. No Bellman-action premise, rule beyond the finite budget, or optimality at a larger horizon is assumed.

- [`GeneralOptimality.FiniteHorizon.theorem1`](LeanProofs.lean#L22955)
- [`GeneralOptimality.FiniteHorizon.rawTree_extend`](LeanProofs.lean#L22905)
- [`GeneralOptimality.FiniteHorizon.actualRisk_extend`](LeanProofs.lean#L22922)
- [`GeneralOptimality.FiniteHorizon.exists_sequentiallyOptimal`](LeanProofs.lean#L23012)
- [`GeneralOptimality.ProperDomain.theorem1`](LeanProofs.lean#L22814)
- [`GeneralOptimality.theorem1`](LeanProofs.lean#L22644)
- [`GeneralOptimality.minimizing_actualRisk_eq_loss`](LeanProofs.lean#L22550)
- [`GeneralOptimality.minimizesHistoryRisk_iff_actualRisk_eq_loss`](LeanProofs.lean#L22557)
- [`GeneralOptimality.minimizing_draw_EBI_ae`](LeanProofs.lean#L22609)
- [`GeneralOptimality.minimizing_recommend_ae`](LeanProofs.lean#L22619)
- [`GeneralOptimality.toRaw_compile`](LeanProofs.lean#L22465)
- [`GeneralOptimality.actualRisk_compile`](LeanProofs.lean#L22494)
- [`GeneralOptimality.ProperDomain.rawTree_extend`](LeanProofs.lean#L22766)
- [`GeneralOptimality.ProperDomain.actualRisk_extend`](LeanProofs.lean#L22781)
- [`DecisionRules.optimal_action_iff_EBI_max`](LeanProofs.lean#L12401)
- [`DecisionRules.terminal_recommendation_iff`](LeanProofs.lean#L12416)

<a id="theorem-2"></a>

### Theorem 2: Polynomial frequentist regret and zero exponential-rate liminf

**Status:** Proved. **PDF pages:** 11; proof 14-15.

Three arms with means (0,0,Delta), every Delta>0; one policy-independent threshold and coefficient; polynomial bound at sufficiently large odd horizons. The liminf statement is over all horizons. RealPrecisionMain uses only initialization risk minimization against concrete Borel history competitors.

- [`RealPrecisionMain.theorem2`](LeanProofs.lean#L24729)
- [`RealPrecisionMain.theorem2_paper_exponent`](LeanProofs.lean#L24740)
- [`RealPrecisionMain.theorem2_exponential_rate`](LeanProofs.lean#L24751)
- [`InitialHistoryMain.theorem2`](LeanProofs.lean#L23199)
- [`Main.theorem2`](LeanProofs.lean#L19445)
- [`Main.theorem2_exponential_rate`](LeanProofs.lean#L19470)

<a id="lemma-3"></a>

### Lemma 3: Probability of severe underestimation of arm 3

**Status:** Proved. **PDF pages:** 13.

Exact printed f1; arbitrary Delta>0 and CU>0, eventually every natural horizon. The finite bound records its explicit threshold hypotheses.

- [`Probability.lemma3`](LeanProofs.lean#L19148)
- [`Probability.lemma3_eventually`](LeanProofs.lean#L19202)
- [`TailRatio.shifted_left_tail`](LeanProofs.lean#L18994)

<a id="lemma-4"></a>

### Lemma 4: Small-ball probability for each zero-mean arm

**Status:** Proved. **PDF pages:** 13; proof 24.

Odd T=2q+1, q>0; actual sample mean variance 1/q; exact printed f2.

- [`Probability.lemma4`](LeanProofs.lean#L17215)
- [`Foundations.SmallBall.f2`](LeanProofs.lean#L1279)

<a id="lemma-5"></a>

### Lemma 5: Drift-event failure bound

**Status:** Proved. **PDF pages:** 13; proof 24.

T>=5, q<=T; revised coefficient 3 in event Z; actual Gaussian arrays, finite union bound, and zero-variance endpoint.

- [`Probability.driftFailure3_probability_le`](LeanProofs.lean#L17108)
- [`Probability.goodDriftSet3_compl_probability_le`](LeanProofs.lean#L17169)
- [`Probability.lemma5`](LeanProofs.lean#L17208)

<a id="lemma-6"></a>

### Lemma 6: Arm 3 EBI upper bound

**Status:** Proved. **PDF pages:** 14; proof 24-27.

Consistent revised W states and cutoff 180<(CU-6)^2/4. Explicit paperG0 envelope and little-o(T^-180), uniform in round, state, and policy.

- [`PolicyChoice.lemma6_upper`](LeanProofs.lean#L14268)
- [`PolicyChoice.lemma6_absolute_upper`](LeanProofs.lean#L23514)
- [`ExclusionAsymptotics.paperG0_littleO180_equation11`](LeanProofs.lean#L17769)

<a id="lemma-7"></a>

### Lemma 7: EBI lower bounds and exclusion of saturated actions

**Status:** Proved. **PDF pages:** 14; proof 37.

Sufficiently large odd T, 4<=t<T, consistent W states. Both nonbest arms have uniform positive polynomial EBI; a saturated action cannot maximize EBI.

- [`PolicyChoice.lemmas7_8_uniform`](LeanProofs.lean#L23529)
- [`PolicyChoice.lemma7_individual_lower`](LeanProofs.lean#L14295)
- [`PolicyChoice.lemma7_saturated_left`](LeanProofs.lean#L14311)
- [`PolicyChoice.lemma7_saturated_right`](LeanProofs.lean#L14325)

<a id="lemma-8"></a>

### Lemma 8: Last-round exclusion of arm 3

**Status:** Proved. **PDF pages:** 14; proof 37.

Separate final-round proof including tied means. Terminal count multiset {q,q-1}, mean-gap bound, and both remaining-arm improvements of order T^-22.

- [`PolicyChoice.lemmas7_8_uniform`](LeanProofs.lean#L23529)
- [`PolicyChoice.terminal_pair`](LeanProofs.lean#L14339)
- [`PolicyChoice.lemma8_individual_lower`](LeanProofs.lean#L14345)
- [`Forecast.terminal_gap_bound`](LeanProofs.lean#L14171)

<a id="conjecture-9"></a>

### Conjecture 9: Almost-every-instance exponential convergence

**Status:** Open. **PDF pages:** 20.

Explicit manuscript conjecture. No proved theorem or custom axiom encodes it.

<a id="lemma-10"></a>

### Lemma 10: Mills inequalities

**Status:** Proved. **PDF pages:** 23.

Standard normal density and actual tail for every x>0; exact printed upper and lower factors.

- [`TailRatio.mills_lower`](LeanProofs.lean#L19112)
- [`TailRatio.mills_upper`](LeanProofs.lean#L19087)
- [`TailRatio.tail_remainder_identity`](LeanProofs.lean#L19074)

<a id="lemma-11"></a>

### Lemma 11: Exclusion-loss Gaussian bound

**Status:** Proved. **PDF pages:** 24-25.

Minimum-based exclusion loss; proper consistent W states. Also includes the displayed independent-Gaussian comparison on p.25.

- [`ExclusionBounds.lemma11`](LeanProofs.lean#L13318)
- [`ExclusionBounds.lemma11_from_equation11`](LeanProofs.lean#L13344)
- [`ExclusionVarianceComparison.lemma11_independent_comparison`](LeanProofs.lean#L17993)

<a id="lemma-12"></a>

### Lemma 12: EBI bounded by exclusion loss via mock sampling

**Status:** Proved. **PDF pages:** 25-27.

Any finite arm set with at least two arms, any finite horizon, positive real precisions. Literal minimum-index retained recommendation, independent mock stream, budget and actual-risk sandwich are constructed.

- [`FiniteModelExclusion.lemma12_any_finite_arm`](LeanProofs.lean#L8864)
- [`ExclusionPerm.lemma12_k3`](LeanProofs.lean#L13469)
- [`MockRisk.lemma12_by_mock_samples`](LeanProofs.lean#L11348)
- [`MockTie.literal_mock_loss_chain`](LeanProofs.lean#L11827)

<a id="lemma-13"></a>

### Lemma 13: Full/reduced posterior loss comparison

**Status:** Proved. **PDF pages:** 27-28.

Any finite arm set with a nonempty retained set. Both policy embeddings and forced retained-arm comparison are proved with actual Gaussian risks; integer three-arm endpoints are also retained.

- [`PolicyEmbedding.lemma13_loss_by_policy_embeddings`](LeanProofs.lean#L16930)
- [`PolicyEmbedding.lemma13_actionLoss_by_policy_embeddings`](LeanProofs.lean#L17024)
- [`ExclusionLoss.lemma13_loss`](LeanProofs.lean#L17915)
- [`ExclusionLoss.lemma13_actionLoss`](LeanProofs.lean#L17925)

<a id="lemma-14"></a>

### Lemma 14: Normalized two-arm terminal-regret integral

**Status:** Proved. **PDF pages:** 28-29.

Arbitrary real means, nonnegative variances, positive total variance; exact Gaussian normalization factor.

- [`Marginal.lemma14_normalized`](LeanProofs.lean#L13657)
- [`Foundations.Gaussian.twoArmPosteriorLoss_eq`](LeanProofs.lean#L1238)
- [`Foundations.Gaussian.twoArmPosteriorLoss_normalized`](LeanProofs.lean#L1250)

<a id="lemma-15"></a>

### Lemma 15: One-step improvement and g1 bounds

**Status:** Proved. **PDF pages:** 29-31.

Exact revised probability product; g1 is zero at gap zero, while actual improvement is positive. Count-uniform exponents 4 and 6, plus terminal exponent 22 including zero gap.

- [`Marginal.lemma15_g1_lower`](LeanProofs.lean#L13701)
- [`Marginal.lemma15_g1_uniform4`](LeanProofs.lean#L23822)
- [`Marginal.lemma15_g1_uniform6`](LeanProofs.lean#L23836)
- [`Marginal.lemma15_terminal_uniform22`](LeanProofs.lean#L13915)
- [`Foundations.Gaussian.oneStepImprovement_positive`](LeanProofs.lean#L1026)

<a id="lemma-16"></a>

### Lemma 16: Strict advantage of smaller-count terminal action and g2 bounds

**Status:** Proved. **PDF pages:** 31-33.

Distinct positive integer counts, arbitrary real gap; strict comparison includes zero gap. Small-gap uniform exponents 6 and 8.

- [`Marginal.lemma16_g2_lower`](LeanProofs.lean#L13749)
- [`Marginal.lemma16_g2_count_strict`](LeanProofs.lean#L23994)
- [`Marginal.lemma16_g2_uniform6`](LeanProofs.lean#L23881)
- [`Marginal.lemma16_g2_uniform8`](LeanProofs.lean#L23900)

<a id="lemma-17"></a>

### Lemma 17: Two-arm smaller-count policy is Bayes optimal

**Status:** Proved. **PDF pages:** 33-34.

All positive integer precisions and real means, any finite remaining horizon. Covers both tied-count choices and adaptive competitors; deterministic terminal-variance comparison and Gaussian forecast are supporting results.

- [`TwoArmOptimal.lemma17_every_posterior`](LeanProofs.lean#L14621)
- [`TwoArmOptimal.lemma17_balanced_policy`](LeanProofs.lean#L14639)
- [`TwoArmOptimal.lemma17_minimal_terminal_variance`](LeanProofs.lean#L14648)
- [`TwoArmOptimal.lemma17_variance_identity`](LeanProofs.lean#L14656)
- [`TwoArmOptimal.lemma17_gaussian_forecast`](LeanProofs.lean#L14661)

<a id="lemma-18"></a>

### Lemma 18: Posterior forecast reaches the small-gap interval

**Status:** Proved. **PDF pages:** 34-35.

Any deterministic count-only smaller-count tie rule. Requires t<T so at least one forecast draw occurs. Uniform lower bound with exponent 170 under revised W; predictive law remains distinct from frequentist law.

- [`Forecast.lemma18`](LeanProofs.lean#L14958)
- [`Forecast.ruleForecast_gaussian`](LeanProofs.lean#L14921)
- [`Forecast.ruleAccumulatedVariance_coe`](LeanProofs.lean#L14929)
- [`Forecast.general_forecast_probability170`](LeanProofs.lean#L14122)

<a id="lemma-19"></a>

### Lemma 19: Reduced two-arm EBI gains by shared-sample coupling

**Status:** Proved. **PDF pages:** 35-37.

Uniform exponents 176 and 178. For the forced-left comparison use rightTie; for forced-right use leftTie. Each comparison uses one fixed rule on both sides, as permitted by the paper.

- [`Coupling19.lemma19_individual_gains_via_coupling`](LeanProofs.lean#L15760)
- [`Coupling19.lemma19_saturated_left_gap_via_coupling`](LeanProofs.lean#L15899)
- [`Coupling19.lemma19_saturated_right_gap_via_coupling`](LeanProofs.lean#L15924)
- [`Dynamic.lemma19_individual_gains`](LeanProofs.lean#L12873)

## Definitions

<a id="definition-1"></a>

### Definition 1: Frequentist exponential rate

**Status:** Definition and consequences. **PDF pages:** 4.

Formal analysis uses extended-real normalized negative log, assigning +infinity to zero regret. The positive-liminf-to-exponential-envelope implication used in Remark 2 is proved; no literal syntax for every o(T) expression is asserted.

- [`RegretExponent.rate`](LeanProofs.lean#L19257)
- [`RegretRatio.exponential_envelope_of_positive_rate`](LeanProofs.lean#L19754)

<a id="definition-2"></a>

### Definition 2: Algorithms and posterior-state policies

**Status:** Definition and constructions. **PDF pages:** 6.

FiniteHorizon.Policy A N contains exactly N Borel sampling rules, indexed by Fin N, and one Borel terminal rule on strictly positive real-precision Gaussian states. Its direct execution uses only the current state, fresh scalar randomizer, and selected reward. Padding unused indices and extending the proper-state domain preserve the raw strategy and actual risk. Concrete Borel history policies include past arms, rewards, and seeds; GeneralOptimality.compile explicitly represents the state policy as such a history algorithm.

- [`GeneralOptimality.FiniteHorizon.Policy`](LeanProofs.lean#L22864)
- [`GeneralOptimality.FiniteHorizon.extend`](LeanProofs.lean#L22874)
- [`GeneralOptimality.FiniteHorizon.rawTree`](LeanProofs.lean#L22896)
- [`GeneralOptimality.FiniteHorizon.rawTree_extend`](LeanProofs.lean#L22905)
- [`GeneralOptimality.FiniteHorizon.actualRisk_extend`](LeanProofs.lean#L22922)
- [`HistoryPolicy.Policy`](LeanProofs.lean#L21984)
- [`HistoryPolicy.toRaw`](LeanProofs.lean#L22008)
- [`HistoryPolicy.toRaw_budget`](LeanProofs.lean#L22070)
- [`GeneralOptimality.ProperDomain.State`](LeanProofs.lean#L22699)
- [`GeneralOptimality.ProperDomain.Policy`](LeanProofs.lean#L22706)
- [`GeneralOptimality.ProperDomain.extend`](LeanProofs.lean#L22720)
- [`GeneralOptimality.ProperDomain.rawTree_extend`](LeanProofs.lean#L22766)
- [`GeneralOptimality.ProperDomain.actualRisk_extend`](LeanProofs.lean#L22781)
- [`GeneralOptimality.compile`](LeanProofs.lean#L22444)
- [`GeneralOptimality.toRaw_compile`](LeanProofs.lean#L22465)
- [`GeneralOptimality.actualRisk_compile`](LeanProofs.lean#L22494)

<a id="definition-3"></a>

### Definition 3: Flat-prior generalized Bayes optimal algorithm

**Status:** Definition and constructions. **PDF pages:** 6-7.

Flat likelihood is normalized after initialization; no risk is integrated against an improper probability prior. FiniteHorizon.SequentiallyOptimal states actual-risk optimality of the finite rule collection at every proper Gaussian state and remaining budget H <= N, against concrete Borel history competitors. This exact finite policy class has an explicit Borel canonical witness, including N = 0. Proper-domain and generic representations supply the supporting proof chain. Initialization-only endpoints remain sufficient for the main theorem.

- [`GeneralOptimality.FiniteHorizon.SequentiallyOptimal`](LeanProofs.lean#L22940)
- [`GeneralOptimality.FiniteHorizon.MinimizesHistoryRisk`](LeanProofs.lean#L22928)
- [`GeneralOptimality.FiniteHorizon.exists_sequentiallyOptimal`](LeanProofs.lean#L23012)
- [`FlatPriorProduct.normalized_product_measure`](LeanProofs.lean#L12228)
- [`GeneralOptimality.ProperDomain.MinimizesHistoryRisk`](LeanProofs.lean#L22787)
- [`GeneralOptimality.ProperDomain.SequentiallyOptimalUpTo`](LeanProofs.lean#L22798)
- [`GeneralOptimality.SequentiallyOptimalUpTo`](LeanProofs.lean#L22637)
- [`GeneralOptimality.exists_sequentiallyOptimalUpTo`](LeanProofs.lean#L22667)
- [`RealPrecisionMain.InitialHistoryRiskOptimal`](LeanProofs.lean#L24680)
- [`RealPrecisionMain.sequentiallyOptimal_implies_initial`](LeanProofs.lean#L24697)

<a id="definition-4"></a>

### Definition 4: Expected Bellman improvement

**Status:** Definition and consequences. **PDF pages:** 10.

EBI is skipped-round loss minus forced-action loss. Its nonnegativity and maximizing-action equivalence are proved.

- [`DecisionRules.EBI`](LeanProofs.lean#L12371)
- [`DecisionRules.EBI_eq`](LeanProofs.lean#L12374)
- [`Information.expected_bellman_improvement_nonnegative`](LeanProofs.lean#L14811)
- [`DecisionRules.argmax_EBI_iff_argmin_actionLoss`](LeanProofs.lean#L12380)

<a id="definition-5"></a>

### Definition 5: Mock-sample loss

**Status:** Definition and constructions. **PDF pages:** 25-26.

Independent posterior-copy latent parameter and conditional mock rewards; real sampling budget preserved. Literal minimum-index retained recommendation and its risk agree with the earlier selector's loss.

- [`MockTie.literalMockPolicy`](LeanProofs.lean#L11656)
- [`MockTie.literalMockLoss`](LeanProofs.lean#L11794)
- [`MockTie.literalMockPolicy_budget`](LeanProofs.lean#L11662)
- [`MockTie.literalMockPolicy_recommends_minimum`](LeanProofs.lean#L11724)
- [`MockTie.literalMockLoss_eq_original`](LeanProofs.lean#L11815)
- [`MockTie.literal_mock_loss_chain`](LeanProofs.lean#L11827)

## Equations

<a id="equation-1"></a>

### Equation (1): Frequentist simple regret

**Status:** Definition and identities. **PDF pages:** 4.

Main.regret implements the actual reward-stack expectation at means (0,0,Delta); regret_eq identifies it with Delta times the wrong-recommendation probability. A generic finite-instance regret integral also supports error-rate equivalence.

- [`Main.regret`](LeanProofs.lean#L19387)
- [`Main.regret_eq`](LeanProofs.lean#L19390)
- [`ProseCorollaries.frequentistRegret`](LeanProofs.lean#L24105)
- [`UniformSampling.observedRegret`](LeanProofs.lean#L25539)

<a id="equation-2"></a>

### Equation (2): Sequential posterior risk minimization

**Status:** Definition and existence. **PDF pages:** 7.

FiniteHorizon.SequentiallyOptimal is fixed-budget sequential posterior risk minimization for the literal finite collection of rules, arbitrary finite K, and strictly positive real precisions. Only remaining horizons H <= N are quantified. FiniteHorizon.theorem1 derives the risk identity and local action rules from this premise. The explicit canonical policy belongs to this finite class and attains the Bellman loss. Exact padding and proper-domain extensions transport actual risk without adding policy or optimality hypotheses.

- [`GeneralOptimality.FiniteHorizon.SequentiallyOptimal`](LeanProofs.lean#L22940)
- [`GeneralOptimality.FiniteHorizon.MinimizesHistoryRisk`](LeanProofs.lean#L22928)
- [`GeneralOptimality.FiniteHorizon.minimizing_extend_iff`](LeanProofs.lean#L22933)
- [`GeneralOptimality.FiniteHorizon.canonical_actualRisk`](LeanProofs.lean#L22997)
- [`GeneralOptimality.FiniteHorizon.canonical_sequentiallyOptimal`](LeanProofs.lean#L23003)
- [`GeneralOptimality.FiniteHorizon.theorem1`](LeanProofs.lean#L22955)
- [`GeneralOptimality.ProperDomain.SequentiallyOptimalUpTo`](LeanProofs.lean#L22798)
- [`GeneralOptimality.ProperDomain.MinimizesHistoryRisk`](LeanProofs.lean#L22787)
- [`GeneralOptimality.ProperDomain.minimizing_extend_iff`](LeanProofs.lean#L22791)
- [`GeneralOptimality.MinimizesHistoryRisk`](LeanProofs.lean#L22537)
- [`GeneralOptimality.minimizesHistoryRisk_iff_actualRisk_eq_loss`](LeanProofs.lean#L22557)
- [`GeneralOptimality.exists_sequentiallyOptimalUpTo`](LeanProofs.lean#L22667)
- [`GeneralOptimality.ProperDomain.theorem1`](LeanProofs.lean#L22814)

<a id="equation-3"></a>

### Equation (3): Risk minimization at initialization

**Status:** Definition and reduction. **PDF pages:** 7.

Concrete full-history competition at counts one; actual-risk minimization is not replaced by a Bellman-equality assumption. Equivalence of representations and transfer into the main theorem are proved. The complete repair and regret-preservation result is for three arms with integer-count states. The arbitrary-K wording of the unnumbered page-7 paragraph is broader than this certificate; Theorem 1 is a distinct general finite-arm result.

- [`PosteriorStateHistory.InitialHistoryRiskOptimal`](LeanProofs.lean#L23164)
- [`PosteriorStateHistory.initialHistoryRiskOptimal_iff`](LeanProofs.lean#L23167)
- [`RealPrecisionMain.InitialHistoryRiskOptimal`](LeanProofs.lean#L24680)
- [`RealPrecisionMain.initialHistoryRiskOptimal_iff_restriction`](LeanProofs.lean#L24686)
- [`InitialHistoryMain.regret_repair_eq`](LeanProofs.lean#L23193)

<a id="equation-4"></a>

### Equation (4): Posterior-predictive Gaussian reward law

**Status:** Proved law. **PDF pages:** 9.

Scalar predictive Gaussian law together with the full latent-parameter/observation joint-law bridge. Positive real precisions are allowed.

- [`Foundations.Gaussian.section51_predictive_law`](LeanProofs.lean#L794)
- [`ObservationBridge.scalar_observation_joint_law`](LeanProofs.lean#L10158)
- [`ObservationBridge.observation_joint_law`](LeanProofs.lean#L10246)

<a id="equation-5"></a>

### Equation (5): Minimum-index posterior-mean maximizer

**Status:** Definition and selector proofs. **PDF pages:** 9.

Exact minimum maximizing arm for finite linearly ordered arm types; attainment, least index, and Borel measurability are proved.

- [`MockTie.minimumMaximizer`](LeanProofs.lean#L11385)
- [`MockTie.minimumMaximizer_attains`](LeanProofs.lean#L11387)
- [`MockTie.minimumMaximizer_le`](LeanProofs.lean#L11390)
- [`MockTie.minimumMaximizer_measurable`](LeanProofs.lean#L11416)

<a id="equation-6"></a>

### Equation (6): Terminal posterior expected regret

**Status:** Definition and identities. **PDF pages:** 9.

Terminal regret is an actual posterior integral; terminalRegret_eq gives expected maximum minus selected mean. Optimal terminal loss is connected to this integral.

- [`FinitePosterior.terminalRegret`](LeanProofs.lean#L8487)
- [`FinitePosterior.terminalRegret_eq`](LeanProofs.lean#L8491)
- [`DecisionRules.terminal_recommendation_iff`](LeanProofs.lean#L12416)
- [`DecisionRules.optimal_posterior_loss`](LeanProofs.lean#L12426)

<a id="equation-7"></a>

### Equation (7): Action loss as expected next-state loss

**Status:** Proved recursion. **PDF pages:** 9.

actionLoss_eq identifies the Gaussian innovation recursion with the posterior-loss integral; rawRisk_succ_data proves the corresponding actual observation recursion.

- [`FinitePosterior.actionLoss`](LeanProofs.lean#L8500)
- [`FinitePosterior.actionLoss_eq`](LeanProofs.lean#L8503)
- [`EmbeddingExecution.rawRisk_succ_data`](LeanProofs.lean#L10526)
- [`ObservationBridge.observation_integral`](LeanProofs.lean#L10293)

<a id="equation-8"></a>

### Equation (8): Bellman loss is the minimum action loss

**Status:** Definition and minimum proofs. **PDF pages:** 9.

FiniteGaussian.value is a reward-value recursion; FinitePosterior.loss subtracts it from posterior expected maximum. The exact minimum and attaining selector are proved.

- [`FinitePosterior.loss`](LeanProofs.lean#L8498)
- [`DecisionRules.actionLoss_minimal`](LeanProofs.lean#L12391)
- [`DecisionRules.chosen_actionLoss_eq`](LeanProofs.lean#L12397)
- [`JointGaussianMeasurable.minimumOptimal_actionLoss`](LeanProofs.lean#L12000)

<a id="equation-9"></a>

### Equation (9): EBI equals skipped-round loss minus action loss

**Status:** Definition and identity. **PDF pages:** 10.

Remaining-horizon indexing differs from paper round indexing; EBI_eq is the loss-form identity.

- [`DecisionRules.EBI`](LeanProofs.lean#L12371)
- [`DecisionRules.EBI_eq`](LeanProofs.lean#L12374)

<a id="equation-10"></a>

### Equation (10): Main polynomial regret lower bound

**Status:** Proved. **PDF pages:** 11.

Actual frequentist law, arbitrary Delta>0, all sufficiently large odd T, policy-independent constants. Paper exponent endpoint uses (33+Delta)^2/2+4.

- [`RealPrecisionMain.theorem2`](LeanProofs.lean#L24729)
- [`RealPrecisionMain.theorem2_paper_exponent`](LeanProofs.lean#L24740)
- [`InitialHistoryMain.theorem2`](LeanProofs.lean#L23199)
- [`Main.theorem2_paper_exponent`](LeanProofs.lean#L19647)

<a id="equation-11"></a>

### Equation (11): Cutoff condition 180<(CU-6)^2/4

**Status:** Hypothesis and consequences. **PDF pages:** 12.

A parameter choice in the manuscript, not a standalone universal assertion. Its consequences for little-o(T^-180), CU>=10, and the main CU=33 specialization are proved.

- [`ExclusionAsymptotics.paperG0_littleO180_equation11`](LeanProofs.lean#L17769)
- [`ExclusionBounds.cutoff_condition_implies_ten`](LeanProofs.lean#L13314)
- [`Main.theorem2_paper_exponent`](LeanProofs.lean#L19647)

<a id="equation-12"></a>

### Equation (12): OCBA oracle static-allocation objective

**Status:** Contextual definition; unformalized. **PDF pages:** 19.

Displayed in the literature review. No standalone formal definition or optimal-allocation theorem for this OCBA objective is claimed in this bundle; cited large-deviation results are outside scope.

<a id="equation-13"></a>

### Equation (13): Absolute full-versus-reduced value-loss difference

**Status:** Proved. **PDF pages:** 27.

Both full-to-reduced and reduced-to-full policy embeddings, imaginary stream, discarded dummy rewards, integrability, measurability, and the final exclusion bound are proved.

- [`PolicyEmbedding.lemma13_loss_by_policy_embeddings`](LeanProofs.lean#L16930)
- [`ExclusionLoss.lemma13_loss`](LeanProofs.lean#L17915)

<a id="equation-14"></a>

### Equation (14): Absolute forced-action loss difference

**Status:** Proved. **PDF pages:** 27.

Same constructed policy embeddings with the first action fixed to a retained arm.

- [`PolicyEmbedding.lemma13_actionLoss_by_policy_embeddings`](LeanProofs.lean#L17024)
- [`ExclusionLoss.lemma13_actionLoss`](LeanProofs.lean#L17925)

<a id="equation-15"></a>

### Equation (15): Centered latent two-arm difference is Gaussian

**Status:** Proved law. **PDF pages:** 28.

Difference of independent posterior Gaussian coordinates. The normalized terminal loss formula uses this law.

- [`Foundations.Gaussian.posterior_difference_law`](LeanProofs.lean#L1094)
- [`CommonSample.centered_pair_difference_law`](LeanProofs.lean#L15973)
- [`Marginal.lemma14_normalized`](LeanProofs.lean#L13657)

<a id="equation-16"></a>

### Equation (16): Posterior-mean difference innovation is Gaussian

**Status:** Proved law. **PDF pages:** 29.

Scalar conjugate-update law supplies variance 1/[n(n+1)]. Reflection handles either arm; multi-step distribution and regret bridges use this update.

- [`Foundations.Gaussian.section51_increment_law`](LeanProofs.lean#L801)
- [`Foundations.Gaussian.section51_increment_law_nat`](LeanProofs.lean#L823)
- [`CommonSample.observation_conjugate_update`](LeanProofs.lean#L15202)

<a id="equation-17"></a>

### Equation (17): One-step improvement probability-product bound

**Status:** Proved bound. **PDF pages:** 30.

Exact printed g1 lower bound; pointwise event argument and actual Gaussian integral identities are formalized. The absolute gap handles both signs.

- [`Marginal.lemma15_g1_lower`](LeanProofs.lean#L13701)
- [`Comparison.Gaussian.lemma15_repaired_event_bound`](LeanProofs.lean#L13589)
- [`Foundations.Gaussian.oneStepImprovement_eq_hinge`](LeanProofs.lean#L977)

<a id="equation-18"></a>

### Equation (18): Difference of two action losses and conditional Gaussian integration

**Status:** Alternative proof; displayed equality not separately proved. **PDF pages:** 32.

The exact displayed triple-integral equality is not separately certified. The linked Gaussian hinge-integral argument proves the required Lemma 16 conclusion. This entry does not certify the literal printed integral chain.

- [`Marginal.hinge_increment_integral_shift`](LeanProofs.lean#L13733)
- [`Marginal.absolute_shift_hinge_integrable`](LeanProofs.lean#L13722)
- [`Marginal.lemma16_g2_lower`](LeanProofs.lean#L13749)
- [`Marginal.lemma16_g2_count_lower`](LeanProofs.lean#L23863)

<a id="equation-19"></a>

### Equation (19): Small-gap interval probability is Omega(T^-2)

**Status:** Proved bound via interval lemma. **PDF pages:** 33.

gaussian_small_interval_lower gives the finite uniform interval estimate; g2_uniform6_variance instantiates it inside the printed product bound. No separate asymptotic equality to a polynomial is claimed.

- [`Marginal.gaussian_small_interval_lower`](LeanProofs.lean#L23671)
- [`Marginal.g2_uniform6_variance`](LeanProofs.lean#L23730)
- [`Marginal.lemma16_g2_uniform6`](LeanProofs.lean#L23881)

<a id="equation-20"></a>

### Equation (20): Deterministic allocation forecast variance and order independence

**Status:** Proved law and count identity. **PDF pages:** 33.

Shared-stream terminal-gap law and telescoping allocation variance cover arbitrary deterministic schedules. Risk depends only on terminal counts; the balanced-policy specialization is also named.

- [`CommonSample.terminal_gap_law`](LeanProofs.lean#L16264)
- [`CommonSample.allocationVariance_decomposition`](LeanProofs.lean#L15052)
- [`CommonSample.risk_depends_only_terminal_counts`](LeanProofs.lean#L15128)
- [`TwoArmOptimal.lemma17_gaussian_forecast`](LeanProofs.lean#L14661)

<a id="equation-21"></a>

### Equation (21): Smaller-count forecast Gaussian law

**Status:** Proved law and variance identity. **PDF pages:** 34.

Any fixed count-only smaller-count tie rule. Exact accumulated variance is initial minus final reciprocal counts.

- [`Forecast.ruleForecast_gaussian`](LeanProofs.lean#L14921)
- [`Forecast.ruleAccumulatedVariance_coe`](LeanProofs.lean#L14929)

<a id="equation-22"></a>

### Equation (22): Anchor reciprocal-variance sum is at most twice forecast variance

**Status:** Proved bound. **PDF pages:** 35.

Both final count orientations {q,q-1}; positive remaining forecast variance required.

- [`Forecast.finalVariance_residual_lower`](LeanProofs.lean#L13981)
- [`Forecast.finalVariance_anchor`](LeanProofs.lean#L14007)

<a id="equation-23"></a>

### Equation (23): Squared current mean gap is at most 160 log(T) times forecast variance

**Status:** Proved bound. **PDF pages:** 35.

Derived from revised W with drift coefficient 3 and count constraints. Both general-rule and earlier fixed-orientation endpoints exist.

- [`Forecast.general_drift160`](LeanProofs.lean#L14034)
- [`Comparison.new_equation23_drift160`](LeanProofs.lean#L12472)

<a id="equation-24"></a>

### Equation (24): Each reduced two-arm EBI is at least c1*T^-176

**Status:** Proved bound. **PDF pages:** 35.

Actual common-sample coupling and an alternative Gaussian Bellman proof are both included.

- [`Coupling19.lemma19_individual_gains_via_coupling`](LeanProofs.lean#L15760)
- [`Dynamic.lemma19_individual_gains`](LeanProofs.lean#L12873)

<a id="equation-25"></a>

### Equation (25): Strict advantage over a saturated action is at least c2*T^-178

**Status:** Proved bound. **PDF pages:** 35.

Both saturated orientations, with the count-only tie convention chosen for the forced-arm comparison.

- [`Coupling19.lemma19_saturated_left_gap_via_coupling`](LeanProofs.lean#L15899)
- [`Coupling19.lemma19_saturated_right_gap_via_coupling`](LeanProofs.lean#L15924)
- [`Dynamic.lemma19_saturated_left_gap`](LeanProofs.lean#L12894)
- [`Dynamic.lemma19_saturated_right_gap`](LeanProofs.lean#L12911)

<a id="equation-26"></a>

### Equation (26): Simple regret equals gap times error probability

**Status:** Proved identity. **PDF pages:** 41.

For the three-arm instance (0,0,Delta), the actual frequentist simple regret is Delta times the probability of not recommending the best arm. Lean uses zero-based indices: 2 : Fin 3 is arm 3 in the paper.

- [`Main.regret_eq`](LeanProofs.lean#L19390)

## Remarks and footnotes

<a id="remark-1"></a>

### Remark 1: Gaussian extensions of cited frequentist and Bayesian guarantees

**Status:** Contextual extensions; unformalized. **PDF pages:** 5.

SR, change-of-measure lower bounds, and general-prior Bayesian O(1/T) statements are not independently reproved. Full external algorithms and regularity hypotheses are not specified in the manuscript.

<a id="remark-2"></a>

### Remark 2: Regret ratio has infinite extended-real limit superior

**Status:** Proved. **PDF pages:** 12.

Any nonnegative comparison-regret sequence with positive lower exponential rate; denominator zero means ratio +infinity. Initialization-only actual-risk and real-precision endpoints included.

- [`RegretRatio.remark2`](LeanProofs.lean#L19783)
- [`InitialHistoryMain.remark2`](LeanProofs.lean#L23227)
- [`RealPrecisionMain.remark2`](LeanProofs.lean#L24760)

<a id="remark-3"></a>

### Remark 3: Two-arm action-loss ties, strict saturated comparisons, balanced counts

**Status:** Proved. **PDF pages:** 18.

Literal equal-action-loss and both strict saturated-arm cases, plus balanced terminal count statement.

- [`ProseCorollaries.remark3_equal_action_losses`](LeanProofs.lean#L24373)
- [`ProseCorollaries.remark3_saturated_left_strict`](LeanProofs.lean#L24424)
- [`ProseCorollaries.remark3_saturated_right_strict`](LeanProofs.lean#L24456)
- [`ProseCorollaries.remark3_balanced_terminal_counts`](LeanProofs.lean#L24404)

<a id="footnote-6"></a>

### Footnote 6: CU>=33 suffices for the main cutoff

**Status:** Parameter specialization. **PDF pages:** 12.

The actual main proof specializes CU=33 and obtains the explicit exponent. The footnote's parameter choice is used within that proof rather than exported as a separate footnote theorem.

- [`Main.theorem2_paper_exponent`](LeanProofs.lean#L19647)
- [`ExclusionBounds.paperG0_littleO180`](LeanProofs.lean#L13382)

<a id="footnote-13"></a>

### Footnote 13: Negative mean-gap case in Lemma 15

**Status:** Sign symmetry in formal statements. **PDF pages:** 30.

Formal bounds are stated using abs(gap), so both signs are included without an additional sign assumption.

- [`Marginal.lemma15_g1_lower`](LeanProofs.lean#L13701)
- [`Marginal.lemma15_g1_uniform6`](LeanProofs.lean#L23836)

<a id="historical-footnote-14-smaller-scale-claim"></a>

### Historical Footnote 14 smaller-scale claim: Counterexample to necessity of the smaller O(T^(-3/2)) scale

**Status:** Refuted. **PDF pages:** 39-40 (historical discussion of the earlier claim).

This refutes the earlier smaller-scale necessity claim, not the corrected footnote in junpei_final.pdf. Lean proves a positive polynomial lower bound for the logarithmically enlarged gap and proves that gap is not O(T^(-3/2)). The explicit lower bound here has order T^(-4).

- [`ProseCorollaries.Footnote14.counterexample_to_necessary_scale`](LeanProofs.lean#L21508)
- [`Footnote14Scale.logarithmicGap_not_bigO_paper_scale`](LeanProofs.lean#L21543)
- [`Footnote14Scale.polynomial_tail_does_not_require_paper_scale`](LeanProofs.lean#L21554)

<a id="corrected-footnote-14"></a>

### Corrected Footnote 14: Polynomial Gaussian-tail lower bounds exactly characterize the logarithmic gap scale

**Status:** Proved. **PDF pages:** 38; Appendix G.4, 47.

For every deterministic nonnegative d : Nat -> Real, an eventual lower bound c*T^(-p) for some fixed real c>0 and p>0 on the actual N(0,T^(-3)) tail at d(T) is equivalent to d(T)=O(T^(-3/2)*sqrt(log T)) in Lean IsBigO notation. Both directions are proved without monotonicity or additional regularity. This matches the corrected footnote in junpei_final.pdf and Appendix G.4. The historical smaller-scale claim is a separate entry.

- [`Footnote14Characterization.corrected_footnote14_iff`](LeanProofs.lean#L21945)
- [`Footnote14Characterization.tail_standardize`](LeanProofs.lean#L21712)
- [`Footnote14Characterization.eventuallyPolynomialTail_of_isBigO`](LeanProofs.lean#L21751)
- [`Footnote14Characterization.isBigO_of_eventuallyPolynomialTail`](LeanProofs.lean#L21797)
- [`Footnote14Characterization.realExponentPolynomialTail_iff_eventually`](LeanProofs.lean#L21914)
- [`Footnote14Characterization.eventuallyPolynomialTail_iff_isBigO`](LeanProofs.lean#L21824)

## Supporting arguments and constructions

<a id="flat-posterior-normalization"></a>

### Flat posterior normalization: Flat likelihood, sufficient statistics, and initialized posterior

**Status:** Proved. **PDF pages:** 6-8.

Normalization is against Lebesgue measure; the resulting product Gaussian is a probability law. Explicit labelled adaptive likelihood normalization is proved, not an independent arbitrary-history regular-conditional-distribution theorem.

- [`FlatPriorInitialization.sum_squares_decomposition`](LeanProofs.lean#L12054)
- [`FlatPriorInitialization.normalized_flat_measure`](LeanProofs.lean#L12111)
- [`FlatPriorProduct.normalized_product_measure`](LeanProofs.lean#L12228)
- [`FlatPriorProduct.normalized_product_sufficient`](LeanProofs.lean#L12236)
- [`AdaptivePosteriorSufficiency.normalized_history_posterior`](LeanProofs.lean#L12335)
- [`AdaptivePosteriorSufficiency.parameter_independent_factor_cancels`](LeanProofs.lean#L12347)

<a id="flat-posterior-actual-risk"></a>

### Flat posterior actual risk: Normalized flat posterior agrees with actual continuation law

**Status:** Proved. **PDF pages:** 6-9.

Links normalized flat likelihood to the proper Gaussian posterior, observation joint law, latent/noise world, and original continuation-risk integral.

- [`FixedGaussian.FlatInitialization.normalizedFlatLaw_eq_posterior`](LeanProofs.lean#L21318)
- [`FixedGaussian.FlatInitialization.normalizedFlat_observation_joint_law`](LeanProofs.lean#L21332)
- [`FixedGaussian.FlatInitialization.flatWorldLaw_eq_worldLaw`](LeanProofs.lean#L21348)
- [`FixedGaussian.FlatInitialization.flat_continuationRisk_eq`](LeanProofs.lean#L21416)

<a id="borel-bellman-functions"></a>

### Borel Bellman functions: Finite losses and measurable optimal selectors

**Status:** Proved. **PDF pages:** 9-10.

Jointly in means and positive real precisions; exact minimum-index action. Concrete actual regret is integrable for regular Borel policy trees.

- [`JointGaussianMeasurable.value_joint_measurable`](LeanProofs.lean#L11907)
- [`JointGaussianMeasurable.potential_joint_measurable`](LeanProofs.lean#L11947)
- [`JointGaussianMeasurable.loss_joint_measurable`](LeanProofs.lean#L11969)
- [`JointGaussianMeasurable.actionLoss_joint_measurable`](LeanProofs.lean#L11973)
- [`JointGaussianMeasurable.minimumOptimal_joint_measurable`](LeanProofs.lean#L11987)
- [`RealPrecisionPolicy.exists_globally_borel_sequentiallyOptimal`](LeanProofs.lean#L22365)

<a id="history-competitors"></a>

### History competitors: Borel full-history algorithms and posterior-state execution

**Status:** Proved constructions. **PDF pages:** 3-7.

Past arms, rewards, and randomizers represented explicitly. Compiler preserves recommendation, sampled-arm trace, real budget, measurability, and actual risk.

- [`HistoryPolicy.toRaw_run_eq`](LeanProofs.lean#L22092)
- [`HistoryPolicy.toRaw_trace_eq`](LeanProofs.lean#L22100)
- [`HistoryPolicy.toRaw_run_measurable`](LeanProofs.lean#L22019)
- [`HistoryPolicy.toRaw_budget`](LeanProofs.lean#L22070)
- [`HistoryPolicy.familyRisk_eq_actualRisk`](LeanProofs.lean#L22144)
- [`PosteriorStateHistory.actualRisk_compile`](LeanProofs.lean#L23129)
- [`PosteriorStateHistory.minimizesHistoryRisk_iff`](LeanProofs.lean#L23148)

<a id="initial-optimality-repair"></a>

### Initial optimality repair: Initial-only risk minimization implies almost-sure optimal repair

**Status:** Proved. **PDF pages:** 7.

From actual initial-risk optimality, not from an assumed sequential/local Bellman premise; recommendation and sampled-arm trace agreement are proved. The complete repair and regret-preservation result is for three arms with integer-count states. The arbitrary-K wording of the unnumbered page-7 paragraph is broader than this certificate; Theorem 1 is a distinct general finite-arm result.

- [`InitialOptimal.InitialRiskOptimal`](LeanProofs.lean#L20358)
- [`InitialOptimal.minimizing_rawTree_repair_ae`](LeanProofs.lean#L20343)
- [`InitialOptimal.minimizing_trace_repair_ae`](LeanProofs.lean#L23475)
- [`InitialOptimal.initialRiskOptimal_repair_ae`](LeanProofs.lean#L20366)

<a id="absolute-continuity-transfer"></a>

### Absolute-continuity transfer: Predictive null sets are null at every fixed Gaussian parameter

**Status:** Proved. **PDF pages:** 7.

Mutual absolute continuity for finite predictive and fixed-parameter Gaussian arrays, including independent policy seeds and measurable maps; transfers optimal-repair equality.

- [`FixedGaussian.fixed_predictive_equivalent`](LeanProofs.lean#L18150)
- [`FixedGaussian.initialRiskOptimal_repair_fixed_ae`](LeanProofs.lean#L20412)
- [`FixedGaussian.initialRiskOptimal_fixed_payoff_repair_eq`](LeanProofs.lean#L20420)

<a id="actual-stack-execution"></a>

### Actual stack execution: Calendar arrays, per-arm reward stacks, initialization, and Main.regret

**Status:** Proved. **PDF pages:** 3-4; 7; 14-15.

The execution-law bridge links the posterior raw-tree argument to the original frequentist per-arm-stack regret definition. It is not an assumed risk-preservation identity.

- [`FixedGaussian.rawTree_stack_law_eq`](LeanProofs.lean#L20724)
- [`FixedGaussian.initialRiskOptimal_regret_repair_eq`](LeanProofs.lean#L21189)
- [`InitialHistoryMain.regret_repair_eq`](LeanProofs.lean#L23193)
- [`RealPrecisionMain.regret_repair_eq`](LeanProofs.lean#L24719)

<a id="real-precision-restriction"></a>

### Real-precision restriction: Full real-precision state policy restricted to reachable integer states

**Status:** Proved. **PDF pages:** 6-10.

Exact raw-tree and risk identities. Main endpoints require initialization-only optimality at the given horizon; no larger-horizon optimality assumption.

- [`RealPrecisionMain.rawTree_restriction`](LeanProofs.lean#L24643)
- [`RealPrecisionMain.continuationRisk_restriction`](LeanProofs.lean#L24665)
- [`RealPrecisionMain.initialHistoryRiskOptimal_iff_restriction`](LeanProofs.lean#L24686)
- [`RealPrecisionMain.exists_initialHistoryRiskOptimal`](LeanProofs.lean#L24770)

<a id="ebi-nonnegativity"></a>

### EBI nonnegativity: Ignoring a sample is feasible and cannot improve optimum

**Status:** Proved. **PDF pages:** 10.

Arbitrary finite arm set and positive real precisions; no EBI sign hypothesis.

- [`Information.expected_bellman_improvement_nonnegative`](LeanProofs.lean#L14811)

<a id="probability-and-path-bridge"></a>

### Probability and path bridge: Actual bad-event probability and policy-independent path induction

**Status:** Proved. **PDF pages:** 13-15.

Revised coefficient-3 event, actual independent arrays, measurability, probability factorization, and bad-event implication. Main proof turns these into wrong recommendation and regret.

- [`Probability.rawBadEvent_implies_newW`](LeanProofs.lean#L17332)
- [`Main.jointBadEvent_probability`](LeanProofs.lean#L19413)
- [`Main.eventually_pointwise_event_lower`](LeanProofs.lean#L19419)
- [`Main.theorem2`](LeanProofs.lean#L19445)

<a id="exclusion-regret-contribution"></a>

### Exclusion regret contribution: Best-arm-event posterior regret is bounded by exclusion loss

**Status:** Proved. **PDF pages:** 16.

Arbitrary measurable terminal decision with independent auxiliary noise; min-based exclusion expectation.

- [`ProseCorollaries.posterior_best_arm_risk_le_exclusion`](LeanProofs.lean#L24332)

<a id="error-regret-rate-equivalence"></a>

### Error/regret rate equivalence: Error probability and simple regret have equal exponential rates

**Status:** Proved. **PDF pages:** 18.

Finite unique-best positive-gap instances; min/max-gap sandwich, equality of lower and upper extended-real rates, including zero-error horizons.

- [`ProseCorollaries.error_regret_sandwich`](LeanProofs.lean#L24110)
- [`ProseCorollaries.error_regret_same_exponential_rate`](LeanProofs.lean#L24228)
- [`ProseCorollaries.error_regret_same_upper_exponential_rate`](LeanProofs.lean#L24247)

<a id="uniform-sampling"></a>

### Uniform sampling: Uniform-sampling frequentist exponent minGap^2/(4K)

**Status:** Proved. **PDF pages:** 5.

Explicit exact-budget floor/remainder allocation, all sufficiently large horizons; full regret-rate limit and error exponent. Observed recommendation uses rewards, without a latent-parameter argument.

- [`UniformSampling.uniform_frequentist_rate_limit`](LeanProofs.lean#L25430)
- [`UniformSampling.uniform_error_log_limit`](LeanProofs.lean#L25446)
- [`UniformSampling.observedRecommendation`](LeanProofs.lean#L25487)
- [`UniformSampling.observed_uniform_frequentist_rate_limit`](LeanProofs.lean#L25556)

<a id="mock-policy-construction"></a>

### Mock-policy construction: Independent posterior copy and literal minimum-index mock policy

**Status:** Proved. **PDF pages:** 25-27.

Shared independent Gaussian blocks, actual risk, measurable execution, budget preservation, and exact terminal rule. No separate compression of arbitrary auxiliary blocks into one scalar seed is claimed.

- [`MockTie.literalMockPolicy_run_measurable`](LeanProofs.lean#L11679)
- [`MockTie.literalMockPolicy_regular`](LeanProofs.lean#L11790)
- [`MockTie.literalMockPolicy_terminal_spec`](LeanProofs.lean#L11733)
- [`MockTie.literalMockLoss_eq_original`](LeanProofs.lean#L11815)
- [`MockTie.literal_mock_loss_chain`](LeanProofs.lean#L11827)

<a id="two-policy-embeddings"></a>

### Two policy embeddings: Both directions of full/reduced policy comparison

**Status:** Proved. **PDF pages:** 27-28.

Reverse simulation uses an independent imaginary stream and discards real dummy observations. The final endpoints discharge simulator measurability, feasibility, risk identities, and coupling bounds.

- [`EmbeddingDirect.lemma13_first_embedding`](LeanProofs.lean#L9633)
- [`EmbeddingDirect.lemma13_first_embedding_forced`](LeanProofs.lean#L9644)
- [`EmbeddingCoupling.lemma13_second_embedding`](LeanProofs.lean#L16708)
- [`EmbeddingCoupling.lemma13_second_embedding_forced`](LeanProofs.lean#L16719)
- [`PolicyEmbedding.lemma13_loss_by_policy_embeddings`](LeanProofs.lean#L16930)

<a id="common-sample-coupling"></a>

### Common-sample coupling: Allocation order invariance and coupled risk gains

**Status:** Proved. **PDF pages:** 33; 35-37.

Per-arm Gaussian innovation streams realize conjugate observation laws. Product-stream risk equals recursive posterior risk; coupled differences give the Lemma 19 bounds.

- [`CommonSample.terminalMeans_permutation`](LeanProofs.lean#L15164)
- [`CommonSample.stream_integral_eq_recursive_risk`](LeanProofs.lean#L16327)
- [`CommonSample.stream_integral_permutation`](LeanProofs.lean#L16365)
- [`Coupling19.left_common_product_gain`](LeanProofs.lean#L16478)
- [`Coupling19.right_common_product_gain`](LeanProofs.lean#L16513)

<a id="lemma-19-stronger-tie-interpretation"></a>

### Lemma 19 stronger tie interpretation: One globally fixed left-tie rule cannot serve both forced-arm comparisons

**Status:** Refuted stronger interpretation. **PDF pages:** 36.

Negative certificate is within the lemma's count/state domain, but the stronger interpretation is not the paper's allowed orientation choice. This is not a refutation of Lemma 19.

- [`Coupling19.fixed_tie_same_forced_arm_assertion_false`](LeanProofs.lean#L15263)
- [`Coupling19.counterexample_within_lemma19_domain`](LeanProofs.lean#L15285)

<a id="general-proper-prior-bellman-assertion"></a>

### General proper-prior Bellman assertion: Non-Gaussian proper priors with finite first moment

**Status:** Unformalized. **PDF pages:** 7.

Current general finite-arm optimality theorems concern proper product-Gaussian posteriors. No separate theorem for every non-Gaussian finite-first-moment prior is supplied.

<a id="kg-ei-definitions-and-literature"></a>

### KG/EI definitions and literature: One-step heuristics and literature comparisons

**Status:** Outside formal targets. **PDF pages:** 11; 18-20.

No independent verification of all cited algorithms, literature rates, historical claims, or numerical illustrations is claimed.

<a id="appendix-f-proposal"></a>

### Appendix F proposal: Enlarged-event improvement

**Status:** Explicitly unproved. **PDF pages:** 38.

The manuscript's proposed enlarged-event argument remains unproved. The footnote counterexample is not a proof of this proposal; the main theorem keeps the original event.

## Coverage limits and manuscript consistency

- **Equation (18):** its literal triple-integral equality is not separately proved. Lemma 16 is established through the linked Gaussian hinge-integral proof.
- **Initialization-only repair:** the complete repair and preservation of frequentist regret are proved for the three-arm count model used in Theorem 2. The general-K unnumbered wording on page 7 extends beyond this certificate. Theorem 1 separately covers arbitrary finite arm sets for sequentially posterior-optimal policies.
- **Final PDF pages 39-40:** the new explicit bounds with factors `exp(-3/2)/(sqrt(2*pi)*T)` and `exp(-3/2)/(C*sqrt(2*pi)) * T^(-(C^2+1)/2)` do not have matching certificates at those precise constants here. The corrected equivalence and historical counterexample are proved using other polynomial bounds. This does not claim the sharper bounds are false.
- **Historical footnote discussion:** pages 39-40 still describe the old `O(T^(-3/2))` necessity claim as though it were the current Footnote 14. The corrected footnote is the logarithmically enlarged equivalence. Its proof and the counterexample to the old claim are distinct entries above.
- **Open and contextual claims:** Conjecture 9, the Appendix F enlarged-event proposal, general non-Gaussian prior extensions, cited-literature results, and numerical illustrations are not independently proved here. Compression of an arbitrary auxiliary Gaussian block into one scalar randomizer has no separate certificate.
- **Flat posterior:** the normalized labelled-likelihood and execution-law results are the specific statements above; they are not a general arbitrary-history regular-conditional-distribution theorem.

Appendix G's former filenames identify sections of the single source. Earlier `Junpei.*` declarations are now `BayesOptimal.*`; earlier `Draft0.*` foundations are now `BayesOptimal.Foundations.*`; the former `Junpei.BayesOptimal.*` rules are now `BayesOptimal.DecisionRules.*`. The source retains all 1,517 theorem/lemma declarations. No manuscript text has been changed.
