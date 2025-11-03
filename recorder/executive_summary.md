# Executive Summary: Text-Based Distillation for Multi-Turn Tool-Calling Agents

**Related Documentation:**
- [Technical Methods](./technical_methods.md) - Detailed methodology and reproducibility information
- [Recording Infrastructure README](./README.md) - Tools and workflows for data collection

---

## Problem Statement

Can simple distillation from visible text tokens, without access to model log-probabilities, effectively improve a small student model's performance on complex multi-turn tool-calling tasks? This question is critical because text-only distillation is compatible with closed-source frontier models and is straightforward to productize.

## Conclusion

For practitioners seeking to improve small models on multi-turn tool-calling tasks:
- **Text-only distillation from a single frontier model is insufficient**
- **Ensemble distillation with rejection sampling may work but is fragile and domain-dependent**

The fundamental challenge is that visible text tokens alone do not capture the reasoning processes and decision-making strategies that make frontier models effective at multi-turn tool use. Access to model log-probabilities is essential in traditional distillation, but frontier models from closed labs do not make this information available.

Ensembling and rejection sampling are well-known techniques that could play a role in future work if we rely on open-source teachers with verifiable rewards, where we have more control over the distillation process and can access model internals.




## Study Overview

We evaluated distillation approaches using tau2-bench, a benchmark designed to evaluate autonomous agents in realistic multi-turn scenarios. Tau2-bench tasks require agents to:
- Conduct extended conversations spanning 10-20+ turns with a user simulator
- Execute strategic database operations through function calling (queries, updates, deletions)
- Extract information from users through natural conversation
- Maintain consistency between conversation state and database state
- Achieve specific outcomes measured by exact database state matching

These tasks are challenging because they demand planning across long horizons, robust error recovery, and coordinated reasoning about both dialogue flow and database consequences. Unlike single-turn benchmarks, errors compound across turns, and agents must adapt their strategy based on user responses and intermediate outcomes.

We evaluated the base student model Qwen3-30B-A3B trained on demonstrations from frontier teachers: GPT-5, Claude Sonnet 4.5, and Gemini 2.5 Pro, across airline (10 test tasks) and retail (22 test tasks) customer service scenarios. Agent evaluation used temperature 1.0 with 3 trials per task and a Gemini 2.5 Flash user simulator at temperature 0.0 for consistency.

## Key Findings

### Primary Result: Simple Distillation Does Not Work

**Single-model distillation (GPT-5 only) shows no improvement:**
- Airline: Base 50.0% → SFT 50.0% (no change)
- Retail: Base 42.4% → SFT 42.4% (no change)

**Single-model distillation with rejection sampling shows no improvement:**
- Airline: Base 50.0% → SFT+rejection 50.0% (no change)
- Retail: Base 42.4% → SFT+rejection 28.8% (degradation)

**Multi-model ensemble distillation (3 teachers) shows no improvement:**
- Airline: Base 46.7% → SFT 50.0% (+3.3pp, marginal)
- Retail: Base 36.4% → SFT 36.4% (no change)

### Positive Control: Ensemble + Rejection Sampling

**Multi-model ensemble with rejection sampling shows selective improvement:**
- Airline: Base 46.7% → SFT+rejection 50.0% (+3.3pp, marginal)
- Retail: Base 36.4% → SFT+rejection 51.5% (+15.1pp, substantial)

The retail domain improvement serves as a **positive control**, which was a critical validation goal of this study. The positive control demonstrates that:
1. Our fine-tuning code and pipeline logic can in fact produce improvements when given the right data
2. The tau2-bench evaluation, as we have set it up with all its components (stochastic sampling, user simulator, database verification, multi-turn rollouts), can detect these improvements

Given the complexity of the experimental pipeline, involving data recording infrastructure, multi-turn stochastic evaluation, temperature 1.0 sampling, and numerous configuration parameters, it was essential to establish this validation. Without a positive control, negative results would be ambiguous: they could reflect either genuine ineffectiveness of simple distillation or subtle bugs in the experimental setup.

The +15.1pp improvement on retail demonstrates our pipeline is sensitive enough to capture performance gains. This increases confidence that the null results for simple distillation methods reflect their genuine ineffectiveness rather than experimental artifacts. The inconsistency across domains (retail success vs airline marginal effect) and the requirement for both ensemble teachers and rejection sampling indicates the effect is fragile and highly context-dependent.

**Additional support for the positive control [evidence not shown]:** In separate experiments using 8 teachers (4 open-source, 4 closed-source) with deterministic agent sampling (temperature 0), we observed positive control improvements on airline tasks as well. This suggests the positive control mechanism is robust across different experimental conditions, further validating the reliability of our measurement approach.

### Teacher Performance Context

Teacher baselines provide important context:
- GPT-5: 70.0% (airline), 68.2% (retail)
- Claude Sonnet 4.5: 70.0% (airline), 81.8% (retail)
- Gemini 2.5 Pro: 50.0% (airline), 59.1% (retail)

The student base model performs substantially below all teachers, but simple distillation methods fail to close this gap meaningfully.

## Business Implications

**The simple, productizable approach (single-model text distillation) is not viable for improving multi-turn tool-calling performance.** This holds even when augmented with rejection sampling to filter successful examples.

The only configuration showing improvement requires:
1. Multiple frontier model teachers (increased cost and complexity)
2. Rejection sampling (reduced data efficiency)
3. Domain-specific effectiveness (retail only in our experiments)

These requirements significantly limit productization potential and raise questions about generalization.

## Technical Notes

**Stochasticity and noise:** Multi-turn agentic tasks with temperature 1.0 sampling introduce substantial variance. Our evaluation uses 3 trials per task with a stochastic user simulator, which adds measurement noise. The presence of a positive control increases confidence that null results reflect genuine lack of improvement rather than experimental issues.

