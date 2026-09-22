#!/usr/bin/env python3
"""generate_hard_decisions_synthetic.py
======================================
Synthetic Data Generation Engine for Hard Decision Tasks (P1–P6 Curriculum).

Addresses the empirical failure modes diagnosed on jevbench hard problems:
1. temporal_numeric: Parameterized business hours SLA, timezone arithmetic, calendar leap years, tiered pricing.
2. probability: Bayes screening, expected monetary value, lot sampling without replacement, system reliability.
3. tradeoff: Explicit rule precedence hierarchies, exception clauses, emergency overrides.
4. long_policy: Multi-section policy documents, contract sublimits, rider exceptions.
5. trap: Surface claim lexical overlap with subtle negation / constraint violation.
6. multi_hop: Multi-step inference chains across disjoint evidence paragraphs.
7. adversarial: Misleading keyword distractors with high surface lexical similarity.

Architecture:
- Deterministic Math & Temporal Generators: 100% exact ground truth, zero hallucination risk.
- Flagship LLM Generator: Qwen 3.8 125B Q4 via local llama-server with strict JSON schema and concise reasoning.
- Train-Serving Parity (P2): Outputs grouped NLI candidate pairs with matching hypothesis format:
    "The correct answer is: {key}: {criteria}"
- Memory Safe: Automatically unloads Qwen 3.8 125B from VRAM upon completion via POST /models/unload.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
TEACHER_MODEL_ID = os.environ.get("NLI_HARD_TEACHER", "qwen-3.8-125b-q4")

DOMAINS = [
    "Clinical Healthcare & Triage",
    "Corporate Financial Compliance & AML",
    "Cloud Infrastructure & Incident Response",
    "Maritime Shipping & Customs Logistics",
    "Cybersecurity Access Control & Zero Trust",
    "Commercial Real Estate & Zoning Regulations",
    "Aerospace Maintenance & Safety Auditing",
    "Government Procurement & Defense Contracting",
    "Intellectual Property & Patent Licensing",
    "Telecommunications Spectrum & Interconnect",
]


# =============================================================================
# 1. Deterministic Generators: Temporal & Numeric
# =============================================================================

def gen_business_hours_sla(seed: int) -> Dict[str, Any]:
    """Generates an SLA deadline problem across business hours and weekends."""
    rng = random.Random(seed)
    
    # Start on a weekday
    year = 2026
    month = rng.choice([3, 4, 5, 6, 9, 10])
    # Pick a start weekday (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)
    weekday_offset = rng.randint(0, 4)
    # Start day of month ensuring valid weekday
    base_date = datetime.date(year, month, 1)
    while base_date.weekday() != weekday_offset:
        base_date += datetime.timedelta(days=1)
    
    start_hour = rng.choice([9, 10, 11, 13, 14, 15])
    start_minute = rng.choice([0, 15, 30, 45])
    start_dt = datetime.datetime(year, base_date.month, base_date.day, start_hour, start_minute)
    
    # Business hours: 09:00 to 17:00 (8 hours/day)
    bh_open = 9
    bh_close = 17
    bh_per_day = bh_close - bh_open  # 8 hours
    
    # SLA duration: 12, 16, 20, 24, 28, or 32 business hours
    sla_hours = rng.choice([12, 16, 20, 24, 28])
    
    # Compute ground truth deadline
    curr_dt = start_dt
    rem_minutes = sla_hours * 60
    
    while rem_minutes > 0:
        # Minutes left in current business day
        close_dt = datetime.datetime(curr_dt.year, curr_dt.month, curr_dt.day, bh_close, 0)
        mins_today = int((close_dt - curr_dt).total_seconds() // 60)
        
        if rem_minutes <= mins_today:
            curr_dt = curr_dt + datetime.timedelta(minutes=rem_minutes)
            rem_minutes = 0
        else:
            rem_minutes -= mins_today
            # Advance to next business day at 09:00
            curr_dt = datetime.datetime(curr_dt.year, curr_dt.month, curr_dt.day, bh_open, 0) + datetime.timedelta(days=1)
            # Skip Saturday (5) and Sunday (6)
            while curr_dt.weekday() >= 5:
                curr_dt += datetime.timedelta(days=1)
                
    gt_str = curr_dt.strftime("%A, %B %d at %H:%M UTC")
    
    # Distractors:
    # Distractor 1: Wall-clock hours without business hours exclusion
    d1_dt = start_dt + datetime.timedelta(hours=sla_hours)
    d1_str = d1_dt.strftime("%A, %B %d at %H:%M UTC")
    
    # Distractor 2: Off-by-one business day (e.g. counting start day as a full day)
    d2_dt = curr_dt + datetime.timedelta(days=1 if curr_dt.weekday() < 4 else 3)
    d2_str = d2_dt.strftime("%A, %B %d at %H:%M UTC")
    
    # Distractor 3: Counting weekends as business days
    curr_no_wknd = start_dt
    rem_m = sla_hours * 60
    while rem_m > 0:
        c_dt = datetime.datetime(curr_no_wknd.year, curr_no_wknd.month, curr_no_wknd.day, bh_close, 0)
        m_today = int((c_dt - curr_no_wknd).total_seconds() // 60)
        if rem_m <= m_today:
            curr_no_wknd += datetime.timedelta(minutes=rem_m)
            rem_m = 0
        else:
            rem_m -= m_today
            curr_no_wknd = datetime.datetime(curr_no_wknd.year, curr_no_wknd.month, curr_no_wknd.day, bh_open, 0) + datetime.timedelta(days=1)
    d3_str = curr_no_wknd.strftime("%A, %B %d at %H:%M UTC")
    
    # Ensure options are unique
    options_set = {gt_str}
    distractors = []
    for d in [d1_str, d2_str, d3_str]:
        if d not in options_set:
            distractors.append(d)
            options_set.add(d)
    
    # Fallback if any duplicate
    hour_shifts = [2, -2, 4]
    h_idx = 0
    while len(distractors) < 3:
        fallback_dt = curr_dt + datetime.timedelta(hours=hour_shifts[h_idx % len(hour_shifts)])
        fb_str = fallback_dt.strftime("%A, %B %d at %H:%M UTC")
        if fb_str not in options_set:
            distractors.append(fb_str)
            options_set.add(fb_str)
        h_idx += 1
        
    all_options = [gt_str] + distractors[:3]
    rng.shuffle(all_options)
    gold_key = chr(ord('A') + all_options.index(gt_str))
    opt_dict = {chr(ord('A') + i): opt for i, opt in enumerate(all_options)}
    
    instruction = (
        f"You are a technical support operations manager. Based on the support agreement terms, "
        f"determine the exact SLA resolution deadline."
    )
    state = (
        f"Support Agreement Section 3.2 (Service Level Terms):\n"
        f"- Standard Business Hours are defined as Monday through Friday, 09:00 to 17:00 UTC.\n"
        f"- Weekends (Saturday and Sunday) and statutory holidays are strictly excluded from SLA time calculations.\n"
        f"- Time elapsed outside of Standard Business Hours pauses the SLA clock.\n\n"
        f"Incident Record:\n"
        f"- Ticket INC-{seed:05d} was created on {start_dt.strftime('%A, %B %d, %Y at %H:%M UTC')}.\n"
        f"- Priority: Level 2 (Target Resolution SLA: {sla_hours} business hours)."
    )
    
    return {
        "id": f"synth_temporal_sla_{seed:05d}",
        "family": "temporal_numeric",
        "domain": "Cloud Infrastructure & Incident Response",
        "instruction": instruction,
        "state": state,
        "options": opt_dict,
        "gold": gold_key,
        "rationale": (
            f"The ticket was logged at {start_dt.strftime('%H:%M UTC on %A')}. Counting only business hours "
            f"(09:00–17:00, 8h/day, Mon–Fri), {sla_hours} business hours elapse at {gt_str}."
        ),
    }


def gen_multi_timezone_event_sequence(seed: int) -> Dict[str, Any]:
    """Generates cross-timezone sequencing / duration problems."""
    rng = random.Random(seed)
    
    # Define city offsets relative to UTC
    cities = [
        {"city": "San Francisco", "tz": "PST", "offset": -8},
        {"city": "New York", "tz": "EST", "offset": -5},
        {"city": "London", "tz": "GMT", "offset": 0},
        {"city": "Tokyo", "tz": "JST", "offset": +9},
        {"city": "Sydney", "tz": "AEDT", "offset": +11},
    ]
    sampled_cities = rng.sample(cities, 4)
    
    # Create base UTC timestamp
    base_utc = datetime.datetime(2026, 7, 10, rng.randint(4, 18), rng.choice([0, 15, 30, 45]))
    
    # Generate 4 events in UTC order with deliberate time deltas
    deltas = [0, rng.randint(45, 120), rng.randint(150, 300), rng.randint(350, 500)]
    events = []
    
    for i, c in enumerate(sampled_cities):
        evt_utc = base_utc + datetime.timedelta(minutes=deltas[i])
        # Convert to local time
        evt_local = evt_utc + datetime.timedelta(hours=c["offset"])
        events.append({
            "service": f"Cluster-{chr(ord('A') + i)}",
            "city": c["city"],
            "tz": c["tz"],
            "local_time_str": evt_local.strftime("%B %d, %H:%M") + f" {c['tz']}",
            "utc_dt": evt_utc,
            "utc_time_str": evt_utc.strftime("%H:%M UTC"),
        })
        
    # Question: Identify the exact chronological duration between the earliest event and the latest event
    total_minutes = deltas[-1]
    hours = total_minutes // 60
    mins = total_minutes % 60
    gt_duration = f"{hours} hours and {mins} minutes"
    
    # Distractor 1: Naive local clock difference ignoring timezone offsets
    first_local = events[0]["utc_dt"] + datetime.timedelta(hours=sampled_cities[0]["offset"])
    last_local = events[-1]["utc_dt"] + datetime.timedelta(hours=sampled_cities[-1]["offset"])
    naive_diff_mins = abs(int((last_local - first_local).total_seconds() // 60))
    d1_duration = f"{naive_diff_mins // 60} hours and {naive_diff_mins % 60} minutes"
    
    # Distractor 2: Off by timezone difference
    offset_gap = abs(sampled_cities[-1]["offset"] - sampled_cities[0]["offset"]) * 60
    d2_mins = abs(total_minutes - offset_gap)
    d2_duration = f"{d2_mins // 60} hours and {d2_mins % 60} minutes"
    
    # Distractor 3: Simple arithmetic shift
    d3_mins = total_minutes + 90
    d3_duration = f"{d3_mins // 60} hours and {d3_mins % 60} minutes"
    
    options_set = {gt_duration}
    distractors = []
    for d in [d1_duration, d2_duration, d3_duration]:
        if d not in options_set:
            distractors.append(d)
            options_set.add(d)
    while len(distractors) < 3:
        d_extra = f"{(total_minutes + 60) // 60} hours and {(total_minutes + 30) % 60} minutes"
        distractors.append(d_extra)
        
    all_options = [gt_duration] + distractors[:3]
    rng.shuffle(all_options)
    gold_key = chr(ord('A') + all_options.index(gt_duration))
    opt_dict = {chr(ord('A') + i): opt for i, opt in enumerate(all_options)}
    
    log_lines = "\n".join([
        f"- [{e['service']}] Incident alarm logged at {e['local_time_str']} ({e['city']})"
        for e in events
    ])
    
    instruction = (
        "You are an incident response lead investigating a multi-region outage. "
        "Calculate the exact elapsed time between the initial cluster alarm and the final cluster alarm."
    )
    state = (
        f"Global Telemetry Ingestion Log:\n{log_lines}\n\n"
        f"Standard Reference Timezone Offsets:\n"
        f"- San Francisco (PST): UTC-8\n"
        f"- New York (EST): UTC-5\n"
        f"- London (GMT): UTC+0\n"
        f"- Tokyo (JST): UTC+9\n"
        f"- Sydney (AEDT): UTC+11"
    )
    
    return {
        "id": f"synth_temporal_tz_{seed:05d}",
        "family": "temporal_numeric",
        "domain": "Cybersecurity Access Control & Zero Trust",
        "instruction": instruction,
        "state": state,
        "options": opt_dict,
        "gold": gold_key,
        "rationale": (
            f"Converting all timestamps to UTC: the earliest event is at {events[0]['utc_time_str']} "
            f"and the latest event is at {events[-1]['utc_time_str']}. "
            f"The true elapsed time is {gt_duration}."
        ),
    }


def gen_tiered_pricing(seed: int) -> Dict[str, Any]:
    """Generates tiered utility / cloud compute billing calculations."""
    rng = random.Random(seed)
    
    tier1_limit = rng.choice([1000, 2000, 5000])
    tier2_limit = tier1_limit + rng.choice([3000, 5000, 8000])
    
    rate1 = rng.choice([0.10, 0.12, 0.15])
    rate2 = rng.choice([0.08, 0.09, 0.10])
    rate3 = rng.choice([0.05, 0.06, 0.07])
    
    # Consumption falls in Tier 3
    excess = rng.randint(500, 2500)
    total_units = tier2_limit + excess
    
    # True cost calculation:
    cost1 = tier1_limit * rate1
    cost2 = (tier2_limit - tier1_limit) * rate2
    cost3 = excess * rate3
    gt_total = cost1 + cost2 + cost3
    gt_str = f"${gt_total:,.2f}"
    
    # Distractor 1: Flat rate at Tier 3 rate (common user error)
    d1 = total_units * rate3
    d1_str = f"${d1:,.2f}"
    
    # Distractor 2: Applying Tier 1 rate to everything
    d2 = total_units * rate1
    d2_str = f"${d2:,.2f}"
    
    # Distractor 3: Off-by-one tier boundary (e.g. tier2 from 0 to tier2_limit)
    d3 = tier2_limit * rate2 + excess * rate3
    d3_str = f"${d3:,.2f}"
    
    options_set = {gt_str}
    distractors = []
    for d in [d1_str, d2_str, d3_str]:
        if d not in options_set:
            distractors.append(d)
            options_set.add(d)
            
    while len(distractors) < 3:
        d_extra = f"${(gt_total + 150.0):,.2f}"
        distractors.append(d_extra)
        
    all_options = [gt_str] + distractors[:3]
    rng.shuffle(all_options)
    gold_key = chr(ord('A') + all_options.index(gt_str))
    opt_dict = {chr(ord('A') + i): opt for i, opt in enumerate(all_options)}
    
    instruction = (
        "You are an enterprise procurement auditor. Based on the contract's tiered billing schedule, "
        "calculate the exact total invoiced cost for this billing cycle."
    )
    state = (
        f"Contract Pricing Schedule (Compute / Storage Egress):\n"
        f"- Tier 1 (First {tier1_limit:,} units): ${rate1:.2f} per unit\n"
        f"- Tier 2 (Next {tier2_limit - tier1_limit:,} units, up to {tier2_limit:,} units): ${rate2:.2f} per unit\n"
        f"- Tier 3 (Units in excess of {tier2_limit:,}): ${rate3:.2f} per unit\n\n"
        f"Consumption Record:\n"
        f"Account ACC-{seed:05d} consumed {total_units:,} units during the statement period."
    )
    
    return {
        "id": f"synth_numeric_tiered_{seed:05d}",
        "family": "temporal_numeric",
        "domain": "Cloud Infrastructure & Incident Response",
        "instruction": instruction,
        "state": state,
        "options": opt_dict,
        "gold": gold_key,
        "rationale": (
            f"Tier 1: {tier1_limit:,} * ${rate1:.2f} = ${cost1:,.2f}. "
            f"Tier 2: {tier2_limit - tier1_limit:,} * ${rate2:.2f} = ${cost2:,.2f}. "
            f"Tier 3: {excess:,} * ${rate3:.2f} = ${cost3:,.2f}. "
            f"Total = ${gt_total:,.2f}."
        ),
    }


# =============================================================================
# 2. Deterministic Generators: Probability & Statistics
# =============================================================================

def gen_bayes_screening(seed: int) -> Dict[str, Any]:
    """Generates Bayes rule / diagnostic screening probability questions."""
    rng = random.Random(seed)
    
    # Prevalence: 1%, 2%, or 5%
    prev = rng.choice([0.01, 0.02, 0.05])
    # Sensitivity (true positive rate): 90%, 95%, or 98%
    sens = rng.choice([0.90, 0.95, 0.98])
    # False positive rate: 4%, 5%, or 8%
    fpr = rng.choice([0.04, 0.05, 0.08])
    
    p_d = prev
    p_not_d = 1.0 - prev
    p_pos_given_d = sens
    p_pos_given_not_d = fpr
    
    # P(Pos) = P(Pos|D)P(D) + P(Pos|~D)P(~D)
    p_pos = (p_pos_given_d * p_d) + (p_pos_given_not_d * p_not_d)
    # Posterior P(D|Pos) = P(Pos|D)P(D) / P(Pos)
    posterior = (p_pos_given_d * p_d) / p_pos
    gt_pct = posterior * 100.0
    gt_str = f"{gt_pct:.1f}%"
    
    # Distractor 1: Base-rate neglect (confusing sensitivity with posterior)
    d1 = f"{sens * 100.0:.1f}%"
    
    # Distractor 2: Sensitivity minus FPR
    d2 = f"{(sens - fpr) * 100.0:.1f}%"
    
    # Distractor 3: False positive rate
    d3 = f"{fpr * 100.0:.1f}%"
    
    options_set = {gt_str}
    distractors = []
    for d in [d1, d2, d3]:
        if d not in options_set:
            distractors.append(d)
            options_set.add(d)
            
    while len(distractors) < 3:
        distractors.append(f"{gt_pct + 12.5:.1f}%")
        
    all_options = [gt_str] + distractors[:3]
    rng.shuffle(all_options)
    gold_key = chr(ord('A') + all_options.index(gt_str))
    opt_dict = {chr(ord('A') + i): opt for i, opt in enumerate(all_options)}
    
    instruction = (
        "You are a medical biostatistician reviewing diagnostic test performance. "
        "Given a patient with a positive test result, what is the calibrated posterior probability that they truly have the condition?"
    )
    state = (
        f"Epidemiological & Assay Parameters:\n"
        f"- Disease prevalence in target population: {prev*100:.1f}%\n"
        f"- Test Sensitivity (True Positive Rate): {sens*100:.1f}%\n"
        f"- Test False Positive Rate (probability healthy individual tests positive): {fpr*100:.1f}%\n\n"
        f"Clinical Observation:\n"
        f"A randomly selected patient from this population undergoes testing and receives a positive result."
    )
    
    return {
        "id": f"synth_prob_bayes_{seed:05d}",
        "family": "probability",
        "domain": "Clinical Healthcare & Triage",
        "instruction": instruction,
        "state": state,
        "options": opt_dict,
        "gold": gold_key,
        "rationale": (
            f"Using Bayes' Rule: P(D|+) = [P(+|D) * P(D)] / [P(+|D)*P(D) + P(+|~D)*P(~D)] = "
            f"[{sens:.2f} * {prev:.2f}] / [({sens:.2f} * {prev:.2f}) + ({fpr:.2f} * {1-prev:.2f})] = {gt_str}."
        ),
    }


def gen_expected_monetary_value(seed: int) -> Dict[str, Any]:
    """Generates expected monetary value decision problems."""
    rng = random.Random(seed)
    
    cost_a = rng.choice([50000, 100000, 150000])
    cost_b = rng.choice([30000, 60000, 80000])
    
    # State probabilities (Bull, Base, Bear)
    p_bull = rng.choice([0.20, 0.25, 0.30])
    p_base = rng.choice([0.40, 0.50])
    p_bear = round(1.0 - p_bull - p_base, 2)
    
    # Payoffs for Option A
    payoff_a_bull = rng.choice([300000, 400000, 500000])
    payoff_a_base = rng.choice([120000, 160000, 200000])
    payoff_a_bear = rng.choice([-20000, 0, 30000])
    
    emv_a = (p_bull * payoff_a_bull) + (p_base * payoff_a_base) + (p_bear * payoff_a_bear) - cost_a
    
    # Payoffs for Option B
    payoff_b_bull = rng.choice([180000, 220000, 250000])
    payoff_b_base = rng.choice([100000, 120000, 140000])
    payoff_b_bear = rng.choice([40000, 50000, 60000])
    
    emv_b = (p_bull * payoff_b_bull) + (p_base * payoff_b_base) + (p_bear * payoff_b_bear) - cost_b
    
    if emv_a > emv_b:
        best_opt = "Option A"
        best_emv = emv_a
        diff = emv_a - emv_b
    else:
        best_opt = "Option B"
        best_emv = emv_b
        diff = emv_b - emv_a
        
    gt_str = f"Select {best_opt} (Net Expected Value: ${best_emv:,.0f})"
    
    # Distractors:
    d1_str = f"Select {'Option B' if best_opt == 'Option A' else 'Option A'} (Net Expected Value: ${min(emv_a, emv_b):,.0f})"
    # Distractor 2: Gross expected value without subtracting upfront cost
    gross_best = best_emv + (cost_a if best_opt == "Option A" else cost_b)
    d2_str = f"Select {best_opt} (Gross Expected Value: ${gross_best:,.0f})"
    # Distractor 3: Equal allocation
    d3_str = "Both options have identical risk-adjusted returns of $0.00"
    
    options_set = {gt_str}
    distractors = []
    for d in [d1_str, d2_str, d3_str]:
        if d not in options_set:
            distractors.append(d)
            options_set.add(d)
            
    while len(distractors) < 3:
        distractors.append(f"Select Option A with manual hedging of ${best_emv*0.5:,.0f}")
        
    all_options = [gt_str] + distractors[:3]
    rng.shuffle(all_options)
    gold_key = chr(ord('A') + all_options.index(gt_str))
    opt_dict = {chr(ord('A') + i): opt for i, opt in enumerate(all_options)}
    
    instruction = (
        "You are an enterprise risk officer. Based on the decision tree probabilities and upfront capital requirements, "
        "determine the optimal strategic option that maximizes net expected monetary value (EMV)."
    )
    state = (
        f"Capital Expenditure Options & Scenario Probabilities:\n"
        f"- Market Scenarios: High Growth ({p_bull*100:.0f}%), Moderate Growth ({p_base*100:.0f}%), Contraction ({p_bear*100:.0f}%)\n\n"
        f"Option A (High-Capacity Expansion):\n"
        f"- Upfront Investment Cost: ${cost_a:,}\n"
        f"- Gross Payoffs: High Growth = ${payoff_a_bull:,} | Moderate = ${payoff_a_base:,} | Contraction = ${payoff_a_bear:,}\n\n"
        f"Option B (Lean Modular Expansion):\n"
        f"- Upfront Investment Cost: ${cost_b:,}\n"
        f"- Gross Payoffs: High Growth = ${payoff_b_bull:,} | Moderate = ${payoff_b_base:,} | Contraction = ${payoff_b_bear:,}"
    )
    
    return {
        "id": f"synth_prob_emv_{seed:05d}",
        "family": "probability",
        "domain": "Corporate Financial Compliance & AML",
        "instruction": instruction,
        "state": state,
        "options": opt_dict,
        "gold": gold_key,
        "rationale": (
            f"EMV(A) = ({p_bull}*${payoff_a_bull:,} + {p_base}*${payoff_a_base:,} + {p_bear}*${payoff_a_bear:,}) - ${cost_a:,} = ${emv_a:,.0f}. "
            f"EMV(B) = ({p_bull}*${payoff_b_bull:,} + {p_base}*${payoff_b_base:,} + {p_bear}*${payoff_b_bear:,}) - ${cost_b:,} = ${emv_b:,.0f}. "
            f"{best_opt} provides the higher net expected value."
        ),
    }


# =============================================================================
# 3. LLM Generator: Tradeoff, Long Policy, Trap, Multi-hop, Adversarial
# =============================================================================

FAMILY_PROMPTS = {
    "tradeoff": "where an explicit exception or emergency override clause resolves conflicting policy rules.",
    "long_policy": "with multi-section policy terms where a specific rider sublimit or exception overrides the general clause.",
    "trap": "with subtle negation or constraint traps where distractors have misleading surface lexical overlap.",
    "multi_hop": "requiring a 3-step reasoning chain connecting disjoint policy statements to reach the conclusion.",
    "adversarial": "where prominent distractor keywords tempt surface matching but a precise factual detail disproves them.",
}


def query_qwen_for_scenarios(
    family: str,
    batch_idx: int,
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    """Queries qwen-3.8-125b-q4 for 1 structured decision scenario."""
    prompt_text = FAMILY_PROMPTS.get(family, "with explicit policy constraints and hard distractors.")
    domain = random.choice(DOMAINS)
    
    schema_example = json.dumps({
        "instruction": "Based on the policy hierarchy, which rule applies?",
        "state": "Section 1: General rule... Section 4: Emergency override...",
        "options": {
            "A": "Apply Section 1...",
            "B": "Apply Section 4...",
            "C": "Neither...",
            "D": "Both..."
        },
        "gold": "B",
        "rationale": "Section 4 explicitly overrides Section 1."
    })
    
    system_prompt = (
        f"You are a benchmark generator. Keep internal reasoning strictly under 30 words. "
        f"Output ONLY raw JSON strictly following this schema and key names: {schema_example}. Do NOT wrap in markdown."
    )
    
    user_prompt = (
        f"Generate exactly 1 hard decision scenario for the family '{family}' {prompt_text} "
        f"Domain focus: {domain}."
    )
    
    payload = {
        "model": TEACHER_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1800,
        "temperature": 0.7,
    }
    
    req = urllib.request.Request(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    
    content_str = ""
    try:
        raw_resp = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")
        parsed = json.loads(raw_resp)
        content_str = parsed["choices"][0]["message"].get("content", "").strip()
        if not content_str:
            print(f"  [Warning] Received empty content for family '{family}' call {batch_idx}", file=sys.stderr)
            return []
        if content_str.startswith("```json"):
            content_str = content_str[7:]
        if content_str.startswith("```"):
            content_str = content_str[3:]
        if content_str.endswith("```"):
            content_str = content_str[:-3]
        content_str = content_str.strip()
        data = json.loads(content_str)
        
        if isinstance(data, dict) and "scenarios" in data and isinstance(data["scenarios"], list):
            s_list = data["scenarios"]
        elif isinstance(data, list):
            s_list = data
        else:
            s_list = [data]
            
        valid_scenarios = []
        for s in s_list:
            opts = s.get("options", {})
            gold_raw = str(s.get("gold", "")).strip()
            # Normalize gold e.g. "Option B" -> "B"
            gold = ""
            for ch in gold_raw:
                if ch.upper() in opts:
                    gold = ch.upper()
                    break
            if not gold and gold_raw in opts:
                gold = gold_raw
                
            if (
                s.get("instruction")
                and s.get("state")
                and isinstance(opts, dict)
                and len(opts) >= 3
                and gold in opts
            ):
                s["gold"] = gold
                s["family"] = family
                s["id"] = f"synth_{family}_{batch_idx:04d}_{len(valid_scenarios)}"
                valid_scenarios.append(s)
            else:
                print(f"  [Debug] Validation failed on scenario: keys={list(s.keys())}, gold={gold!r}, opts_keys={list(opts.keys())}", file=sys.stderr)
        return valid_scenarios
    except Exception as e:
        print(f"  [Warning] Generation failed for family '{family}' call {batch_idx}: {e}\nRaw prefix: {content_str[:250]!r}", file=sys.stderr)
        return []


def unload_teacher_model() -> None:
    """Unloads the 125B model from GPU memory to free the RTX 5090 for PyTorch."""
    print(f"\nUnloading teacher model '{TEACHER_MODEL_ID}' from GPU memory...")
    try:
        req = urllib.request.Request(
            f"{LLAMA_SERVER_URL}/models/unload",
            data=json.dumps({"model": TEACHER_MODEL_ID}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=15)
        print("-> Teacher model unloaded successfully. GPU VRAM freed.")
    except Exception as e:
        print(f"  Note: Unload request returned: {e}")


# =============================================================================
# 4. Grouped NLI Conversion (P1 & P2 Parity)
# =============================================================================

def convert_scenario_to_grouped_nli(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converts a decision scenario into K candidate NLI pairs sharing group_id."""
    instruction = scenario["instruction"].strip()
    state = scenario["state"].strip()
    premise = f"{instruction}\n\n{state}" if state else instruction
    
    options = scenario["options"]
    gold_key = scenario["gold"]
    k_options = len(options)
    
    pairs = []
    for opt_key, opt_text in options.items():
        is_gold = (opt_key == gold_key)
        
        # Serving-template parity (P2)
        hypothesis = f"The correct answer is: {opt_key}: {opt_text}"
        
        # Calibrated soft targets
        if is_gold:
            p_soft = 0.88
            soft_labels = [0.05, 0.88, 0.07]
        else:
            p_soft = round(0.12 / max(1, k_options - 1), 4)
            soft_labels = [round(0.88 - p_soft, 4), p_soft, 0.07]
            
        pairs.append({
            "id": f"{scenario['id']}_opt_{opt_key}",
            "premise": premise,
            "hypothesis": hypothesis,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": scenario["id"],
            "is_gold": is_gold,
            "soft_target": p_soft,
            "soft_labels": soft_labels,
            "source": f"synth_hard_{scenario.get('family', 'decision')}",
            "language": "en",
            "image": "",
            "metadata": {
                "scenario_id": scenario["id"],
                "family": scenario.get("family"),
                "domain": scenario.get("domain"),
                "option_key": opt_key,
                "is_gold": is_gold,
                "rationale": scenario.get("rationale", ""),
            },
        })
    return pairs


# =============================================================================
# 5. Main Execution Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Engine for Hard Decision Tasks")
    parser.add_argument("--out-dir", default="./data", help="Output directory for datasets")
    parser.add_argument("--n-deterministic-each", type=int, default=200, help="Number of samples per deterministic generator")
    parser.add_argument("--n-llm-each", type=int, default=10, help="Number of LLM scenarios to generate per family")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM generation, run only deterministic generators")
    parser.add_argument("--resume", action="store_true", help="Resume from existing generated raw scenarios")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    raw_scenarios_path = os.path.join(args.out_dir, "synth_hard_scenarios_raw.jsonl")
    nli_output_path = os.path.join(args.out_dir, "synth_hard_decisions_grouped.jsonl")
    
    scenarios: List[Dict[str, Any]] = []
    seen_ids = set()
    
    if args.resume and os.path.exists(raw_scenarios_path):
        print(f"Resuming from existing scenarios at {raw_scenarios_path}...")
        with open(raw_scenarios_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    scenarios.append(item)
                    seen_ids.add(item["id"])
        print(f"Loaded {len(scenarios)} existing scenarios.")
        
    # -------------------------------------------------------------------------
    # 1. Deterministic Generation
    # -------------------------------------------------------------------------
    print(f"\n--- Generating Deterministic Datasets ({args.n_deterministic_each} per archetype) ---")
    t0 = time.time()
    
    # SLA business hours
    for i in range(args.n_deterministic_each):
        s = gen_business_hours_sla(10000 + i)
        if s["id"] not in seen_ids:
            scenarios.append(s)
            seen_ids.add(s["id"])
            
    # Multi-timezone
    for i in range(args.n_deterministic_each):
        s = gen_multi_timezone_event_sequence(20000 + i)
        if s["id"] not in seen_ids:
            scenarios.append(s)
            seen_ids.add(s["id"])
            
    # Tiered pricing
    for i in range(args.n_deterministic_each):
        s = gen_tiered_pricing(30000 + i)
        if s["id"] not in seen_ids:
            scenarios.append(s)
            seen_ids.add(s["id"])
            
    # Bayes probability
    for i in range(args.n_deterministic_each):
        s = gen_bayes_screening(40000 + i)
        if s["id"] not in seen_ids:
            scenarios.append(s)
            seen_ids.add(s["id"])
            
    # Expected monetary value
    for i in range(args.n_deterministic_each):
        s = gen_expected_monetary_value(50000 + i)
        if s["id"] not in seen_ids:
            scenarios.append(s)
            seen_ids.add(s["id"])
            
    dt_det = time.time() - t0
    print(f"Deterministic generation complete: {len(scenarios)} total scenarios in {dt_det:.2f}s.")
    
    # Write checkpoint
    with open(raw_scenarios_path, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s) + "\n")
            
    # -------------------------------------------------------------------------
    # 2. LLM Generation (Qwen 3.8 125B Q4)
    # -------------------------------------------------------------------------
    if not args.skip_llm:
        print(f"\n--- Generating LLM Scenarios via {TEACHER_MODEL_ID} ({args.n_llm_each} scenarios per family) ---")
        llm_families = ["tradeoff", "long_policy", "trap", "multi_hop", "adversarial"]
        
        for fam in llm_families:
            print(f"\nGenerating family: '{fam}'...")
            fam_start = time.time()
            fam_count = 0
            
            for b in range(args.n_llm_each):
                batch_scenarios = query_qwen_for_scenarios(fam, b)
                for s in batch_scenarios:
                    if s["id"] not in seen_ids:
                        scenarios.append(s)
                        seen_ids.add(s["id"])
                        fam_count += 1
                        # Append immediately to checkpoint file
                        with open(raw_scenarios_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(s) + "\n")
                print(f"  Item {b+1}/{args.n_llm_each} -> +{len(batch_scenarios)} scenario (total in family: {fam_count})")
            
            print(f"Finished '{fam}': {fam_count} scenarios in {time.time() - fam_start:.1f}s.")
            
        # Unload teacher model when LLM generation finishes
        unload_teacher_model()
        
    # -------------------------------------------------------------------------
    # 3. Convert all Scenarios to Grouped NLI Pairs
    # -------------------------------------------------------------------------
    print(f"\n--- Converting {len(scenarios)} Scenarios to Grouped NLI Dataset ---")
    all_nli_pairs: List[Dict[str, Any]] = []
    
    family_counts = {}
    for s in scenarios:
        pairs = convert_scenario_to_grouped_nli(s)
        all_nli_pairs.extend(pairs)
        fam = s.get("family", "unknown")
        family_counts[fam] = family_counts.get(fam, 0) + 1
        
    print(f"Writing {len(all_nli_pairs):,} grouped NLI items to {nli_output_path}...")
    with open(nli_output_path, "w", encoding="utf-8") as f:
        for p in all_nli_pairs:
            f.write(json.dumps(p) + "\n")
            
    print("\nDataset Summary by Failure Archetype:")
    for fam, count in sorted(family_counts.items()):
        n_pairs = count * 4  # approximately 4 options per scenario
        print(f"  - {fam:20s}: {count:,} scenarios (~{n_pairs:,} candidate NLI pairs)")
    print(f"Total Grouped NLI Candidate Pairs: {len(all_nli_pairs):,}")
    print("Done!")


if __name__ == "__main__":
    main()
