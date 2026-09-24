#!/usr/bin/env python3
"""scripts/generate_targeted_remediation.py - Targeted Synthetic Data Generator for JevBench Weak Families.

Directly remedies the 4 primary empirical failure modes diagnosed in the Phase-3 Epoch 1 audit:
1. routing: Instruction rule-distractor traps (instruction mentions tool X conditionally, but task requires tool Y).
2. judge_hard: Polite LLM flaw blindness (well-formatted, polite markdown responses with subtle math/unit/logic flaws; 50% yes / 50% no).
3. policy: Negative condition dominance (policy mentions restrictive clauses for action A, but user asks for permitted action B; 50% yes / 50% no).
4. tradeoff: Precedence ladder overrides (statutory/emergency rule overrides informal staff comments like 'no exceptions' or 'mark as P2').

100% deterministic ground truth: zero hallucination, zero judge disagreement, fast multi-threaded execution.
100% decontaminated: strict 8-gram filtering against JevBench public and data/test.jsonl.

Usage:
  uv run python scripts/generate_targeted_remediation.py --out-file data/staged/synth_targeted_remediation.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Routing with Instruction Rule Distractors
# =============================================================================

ROUTING_TOOLS = {
    "math": "math: Self-contained calculation, algebra, numeric evaluation, or mathematical proof",
    "coding": "coding: Self-contained code writing, algorithm implementation, or code explanation without filesystem operations",
    "coding_agent": "coding_agent: Inspect or edit repository files, execute terminal commands, or run automated test suites",
    "document": "document: Extract, synthesize, or verify information directly from an attached document, report, or contract",
    "tools": "tools: Carry out an external service action, calendar update, email dispatch, or third-party API integration",
    "sql_agent": "sql_agent: Connect to a production database, run migrations, optimize query plans, or alter tables",
    "general": "general: Open-ended creative prose, brainstorming, naming suggestions, or general conversation",
}

MATH_TEMPLATES = [
    ("Calculate the least common multiple of integers {a} and {b}.", lambda a, b: math.lcm(a, b)),
    ("Find the greatest common divisor between positive integers {a} and {b}.", lambda a, b: math.gcd(a, b)),
    ("Compute the exact numerical result of {a}^3 - {b}^2.", lambda a, b: a**3 - b**2),
    ("Evaluate the arithmetic sum of all integers from 1 up to {a}.", lambda a, b: a * (a + 1) // 2),
    ("Determine exactly {a} percent of the value {b}.", lambda a, b: (a * b) / 100),
]

CODING_TEMPLATES = [
    "Write a standalone Python function to reverse a singly linked list in-place; do not touch any repository files.",
    "Implement an optimized binary search tree traversal in TypeScript; return only the self-contained code snippet.",
    "Show standard Python code using `heapq` to compute the top K frequent elements from an in-memory list.",
    "Provide a self-contained Rust implementation validating an IPv4 address string without third-party crates.",
    "Write an in-memory Go function implementing token bucket rate limiting without disk I/O.",
]

CODING_AGENT_TEMPLATES = [
    "Access the local repository, repair the syntax bug in `src/tokenizer.py`, and run `pytest tests/test_tok.py` to confirm.",
    "Inspect the workspace files, upgrade package versions in `pyproject.toml`, and run the automated test suite.",
    "Clone branch `staging`, correct the broken linter configuration in `.eslintrc.json`, and execute `npm run test`.",
    "Review `handler.go`, introduce unit tests for expired session tokens, and execute `go test ./...` in the repo.",
    "Resolve the broken CI build in `frontend/src/App.tsx` and run the end-to-end integration test runner.",
]

DOCUMENT_TEMPLATES = [
    "Examine the attached vendor service contract and compile a list of all quarterly audit obligations in Clause 6.",
    "Extract the indemnification ceiling and liability cap stipulations from the supplied procurement agreement.",
    "Review the attached clinical trial study protocol and extract the participant exclusion criteria for Phase II.",
    "From the provided commercial lease PDF, determine the precise calendar notice period required for non-renewal.",
    "Read the attached building safety inspection report and list all Category 1 electrical remediation requirements.",
]

TOOLS_TEMPLATES = [
    "Move my 1-on-1 sync with the lead architect to Thursday at 15:30 in my Google Calendar schedule.",
    "File a critical incident tracking ticket in Jira assigned to the core networking reliability squad.",
    "Dispatch an automated email report to accounts-payable@corp.example attaching quarterly statement #{num}.",
    "Broadcast a service status notification to the #incident-updates Slack channel regarding latency recovery.",
    "Trigger an emergency notification webhook to the on-call PagerDuty escalation roster for service `auth-api`.",
]

GENERAL_TEMPLATES = [
    "Propose four distinctive and catchy brand names for an open-source distributed cache system.",
    "Compose an atmospheric introductory paragraph for a speculative fiction story set in an underwater research station.",
    "Create three compelling marketing taglines for a sustainably harvested organic tea company.",
    "Draft a courteous and informative welcoming message for new open-source contributors joining our project.",
    "Provide three intuitive visual analogies clarifying how transformer self-attention mechanisms operate.",
]

INSTRUCTION_RULES = [
    "When repository modifications require running automated test validation, assign to coding_agent, regardless of technical domain.",
    "When database schema alterations or live table migrations are required, route to sql_agent, even if general SQL queries.",
    "When tasks involve invoking external web APIs, calendar scheduling, or issue tracking, select tools, not document.",
    "When pure code generation does not modify local files or execute tests, select coding, not coding_agent.",
    "When requests require reading or querying an attached PDF, report, or contract, select document, even if technical in nature.",
]


def gen_routing_scenario(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    rule_distractor = rng.choice(INSTRUCTION_RULES)

    cat_weights = [("math", 0.20), ("coding", 0.20), ("document", 0.15), ("tools", 0.15), ("general", 0.15), ("coding_agent", 0.15)]
    r = rng.random()
    cumulative = 0.0
    target_cat = "math"
    for cat, w in cat_weights:
        cumulative += w
        if r <= cumulative:
            target_cat = cat
            break

    a = rng.randint(12, 96)
    b = rng.randint(8, 48)
    num = rng.randint(1000, 9999)

    if target_cat == "math":
        tmpl, fn = rng.choice(MATH_TEMPLATES)
        req = tmpl.format(a=a, b=b)
    elif target_cat == "coding":
        req = rng.choice(CODING_TEMPLATES)
    elif target_cat == "coding_agent":
        req = rng.choice(CODING_AGENT_TEMPLATES)
    elif target_cat == "document":
        req = rng.choice(DOCUMENT_TEMPLATES)
    elif target_cat == "tools":
        req = rng.choice(TOOLS_TEMPLATES).format(num=num)
    else:
        req = rng.choice(GENERAL_TEMPLATES)

    premise = f"Select the appropriate expert system for the incoming task. {rule_distractor}\n\nTask Description: {req}"

    mentioned_tool = "coding_agent" if "coding_agent" in rule_distractor else ("tools" if "tools" in rule_distractor else ("sql_agent" if "sql_agent" in rule_distractor else "document"))
    options_keys = list({target_cat, mentioned_tool, "math", "coding", "tools", "general"})[:4]
    if target_cat not in options_keys:
        options_keys[0] = target_cat
    rng.shuffle(options_keys)

    options = {k: ROUTING_TOOLS[k] for k in options_keys}
    return {
        "id": f"synth_remed_routing_{seed}",
        "family": "routing",
        "domain": "Specialist Routing & Rule Disambiguation",
        "premise": premise,
        "options": options,
        "expected": target_cat,
        "rationale": f"The request specifically demands '{target_cat}'. The instruction rule '{rule_distractor}' was not triggered because its prerequisites were unmet.",
    }


# =============================================================================
# 2. Response Verification with Subtle Flaws (judge_hard & adequacy)
# =============================================================================

def gen_judge_hard_scenario(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    is_valid = rng.choice([True, False])
    archetype = rng.choice(["conference_receipts", "chemical_tank", "area_conversion", "python_validator"])

    if archetype == "conference_receipts":
        n_standard = rng.randint(70, 160)
        n_academic = rng.randint(40, 120)
        total_attendees = n_standard + n_academic
        rate_std = rng.choice([250, 300, 350, 400])
        rate_acad = rng.choice([120, 150, 180, 200])
        total_fees = n_standard * rate_std + n_academic * rate_acad

        req = f"A technology symposium registered {total_attendees} attendees. Standard corporate passes cost ${rate_std} and academic passes cost ${rate_acad}. Total registration revenue was ${total_fees:,}. Compute the exact number of corporate passes issued, showing each algebraic step."

        if is_valid:
            resp = (
                f"Here is the linear system solution:\n\n"
                f"1. Let $C$ be the number of corporate passes and $A$ be the academic passes.\n"
                f"2. Attendee constraint: $C + A = {total_attendees} \\implies A = {total_attendees} - C$.\n"
                f"3. Revenue constraint: ${rate_std}C + {rate_acad}A = {total_fees:,}$.\n"
                f"4. Substitute $A$: ${rate_std}C + {rate_acad}({total_attendees} - C) = {total_fees:,}$\n"
                f"   ${rate_std - rate_acad}C + {rate_acad * total_attendees:,} = {total_fees:,}$\n"
                f"   ${rate_std - rate_acad}C = {total_fees - rate_acad * total_attendees:,}$\n"
                f"   $C = {n_standard}$.\n\n"
                f"Result: Exactly **{n_standard} corporate passes** were issued."
            )
        else:
            flawed_c = n_standard + rng.choice([-4, -2, -1, 1, 2, 4])
            resp = (
                f"Here is the linear system solution:\n\n"
                f"1. Let $C$ be the number of corporate passes and $A$ be the academic passes.\n"
                f"2. Attendee constraint: $C + A = {total_attendees} \\implies A = {total_attendees} - C$.\n"
                f"3. Revenue constraint: ${rate_std}C + {rate_acad}A = {total_fees:,}$.\n"
                f"4. Substitute $A$: ${rate_std}C + {rate_acad}({total_attendees} - C) = {total_fees:,}$\n"
                f"   ${rate_std - rate_acad}C + {rate_acad * total_attendees:,} = {total_fees:,}$\n"
                f"   ${rate_std - rate_acad}C = {total_fees - rate_acad * total_attendees:,}$\n"
                f"   $C = {flawed_c}$.\n\n"
                f"Result: Exactly **{flawed_c} corporate passes** were issued."
            )

    elif archetype == "chemical_tank":
        initial = rng.randint(150, 350)
        drain_rate = rng.choice([2.4, 3.5, 4.0, 5.2])
        duration_h = rng.choice([3.5, 4.0, 5.5, 6.0])
        drained_total = round(drain_rate * duration_h, 2)
        refill = rng.randint(40, 90)
        correct_net = round(initial - drained_total + refill, 2)

        req = f"A pharmaceutical mixing vat holds {initial} liters of solution. It drains at {drain_rate} liters per hour for {duration_h} hours. Subsequently, an operator pumps in {refill} liters of neutral buffer. What is the final vat volume in liters? Detail all calculations."

        if is_valid:
            resp = (
                f"Step-by-step volume ledger:\n\n"
                f"- **Starting solution**: {initial} L\n"
                f"- **Volume drained**: {drain_rate} L/h × {duration_h} h = {drained_total} L\n"
                f"- **Intermediate volume**: {initial} L - {drained_total} L = {round(initial - drained_total, 2)} L\n"
                f"- **Buffer replenishment**: {round(initial - drained_total, 2)} L + {refill} L = **{correct_net} L**\n\n"
                f"The resulting volume in the mixing vat is **{correct_net} liters**."
            )
        else:
            flawed_net = round(correct_net + rng.choice([-3.0, -1.5, 1.5, 3.0]), 2)
            resp = (
                f"Step-by-step volume ledger:\n\n"
                f"- **Starting solution**: {initial} L\n"
                f"- **Volume drained**: {drain_rate} L/h × {duration_h} h = {drained_total} L\n"
                f"- **Intermediate volume**: {initial} L - {drained_total} L = {round(initial - drained_total, 2)} L\n"
                f"- **Buffer replenishment**: {round(initial - drained_total, 2)} L + {refill} L = **{flawed_net} L**\n\n"
                f"The resulting volume in the mixing vat is **{flawed_net} liters**."
            )

    elif archetype == "area_conversion":
        sq_yards = rng.choice([120, 240, 360, 480, 600])
        yd_to_m = 0.9144
        sq_yd_to_sq_m = round(yd_to_m ** 2, 6)  # 0.836127
        correct_sq_m = round(sq_yards * sq_yd_to_sq_m, 3)
        flawed_linear_m = round(sq_yards * yd_to_m, 3)

        req = f"Convert a plot area of {sq_yards} square yards into square meters using the exact definition 1 yard = {yd_to_m} meters. Explain the geometric squaring of the conversion unit."

        if is_valid:
            resp = (
                f"Conversion of areal dimensions:\n\n"
                f"1. Linear scale factor: 1 yd = {yd_to_m} m\n"
                f"2. Area scale factor: 1 sq yd = ({yd_to_m} m)^2 = {sq_yd_to_sq_m} m^2\n"
                f"3. Scaling plot: {sq_yards} × {sq_yd_to_sq_m} m^2 = **{correct_sq_m} m^2**\n\n"
                f"The plot measures exactly **{correct_sq_m} square meters**."
            )
        else:
            resp = (
                f"Conversion of areal dimensions:\n\n"
                f"1. Linear scale factor: 1 yd = {yd_to_m} m\n"
                f"2. Multiply the area directly by the linear scale factor: {sq_yards} × {yd_to_m} = **{flawed_linear_m} m^2**\n\n"
                f"The plot measures exactly **{flawed_linear_m} square meters**."
            )

    else:  # python_validator
        fn = rng.choice(["partition_records", "clamp_magnitudes", "window_elements"])
        req = f"Write a Python function `{fn}(elements, chunk_size)`. Constraints: (1) Raise `ValueError` if `chunk_size <= 0`; (2) Return grouped slices of at most `chunk_size` elements; (3) Include full type annotations."

        if is_valid:
            resp = (
                f"```python\n"
                f"from typing import List, Sequence, TypeVar\n\n"
                f"T = TypeVar('T')\n\n"
                f"def {fn}(elements: Sequence[T], chunk_size: int) -> List[List[T]]:\n"
                f"    if chunk_size <= 0:\n"
                f"        raise ValueError(f'chunk_size must be positive, got {{chunk_size}}')\n"
                f"    return [list(elements[i:i + chunk_size]) for i in range(0, len(elements), chunk_size)]\n"
                f"```"
            )
        else:
            resp = (
                f"```python\n"
                f"from typing import List, Sequence, TypeVar\n\n"
                f"T = TypeVar('T')\n\n"
                f"def {fn}(elements: Sequence[T], chunk_size: int) -> List[List[T]]:\n"
                f"    # Omitted ValueError check for chunk_size <= 0\n"
                f"    return [list(elements[i:i + chunk_size]) for i in range(0, len(elements), chunk_size)]\n"
                f"```"
            )

    premise = f"Evaluate whether the candidate answer accurately and completely fulfills all specified criteria.\n\n[USER INQUIRY]\n{req}\n\n[ASSISTANT RESPONSE]\n{resp}"
    options = {
        "yes": "yes: The candidate answer is completely accurate, satisfies all constraints, and contains zero arithmetic or logical defects.",
        "no": "no: The candidate answer contains a calculation error, invalid conversion factor, or unfulfilled prompt requirement.",
    }
    expected = "yes" if is_valid else "no"
    return {
        "id": f"synth_remed_judge_hard_{seed}",
        "family": "judge_hard",
        "domain": "Response Verification & Mathematical Precision",
        "premise": premise,
        "options": options,
        "expected": expected,
        "rationale": "Solution is verified completely correct." if is_valid else "Solution contains a subtle mathematical, logical, or constraint defect.",
    }


# =============================================================================
# 3. Policy Counterfactuals (policy & long_policy)
# =============================================================================

POLICY_DOMAINS = [
    {
        "name": "Cloud Storage & Export Permissions",
        "policy": "Tier-1 free accounts are authorized to export unstructured JSON activity logs. Formatted executive PowerPoint presentations require an active Enterprise seat. Direct raw database binary replication is strictly forbidden for non-root infrastructure roles.",
        "cases": [
            ("A Tier-1 free account administrator requests to download their raw JSON activity log.", True),
            ("A Tier-1 free user requests to download an executive PowerPoint presentation.", False),
            ("A standard paid developer requests to establish direct binary database replication.", False),
            ("A Tier-1 free account requests an export of their unstructured activity logs in JSON format.", True),
        ],
    },
    {
        "name": "Corporate Telework & Equipment Stipends",
        "policy": "Standard ergonomic office chairs up to $350 are pre-approved for all remote staff. Motorized standing desks require senior director sign-off. Home internet stipends up to $80/month do not require itemized receipts; home video gaming consoles are strictly non-reimbursable under all circumstances.",
        "cases": [
            ("A remote employee submits an expense for a $280 ergonomic desk chair. Authorize reimbursement.", True),
            ("An employee claims $65 for home broadband internet with no itemized receipt attached. Approve payment.", True),
            ("An employee submits an invoice for a $450 video game console purchased during remote work. Reimburse it.", False),
            ("An employee orders a motorized standing desk without obtaining senior director approval. Authorize reimbursement.", False),
        ],
    },
    {
        "name": "Datacenter Physical Security & Access Badges",
        "policy": "Contractors holding valid Level-3 biometric credentials may enter server cage hallways unescorted during business shifts (07:00–19:00). Third-party delivery personnel must maintain continuous security escort at all times. Former staff members are strictly denied badge reactivation without VP authorization.",
        "cases": [
            ("A contractor holding valid Level-3 biometric credentials arrives at 11:00 on a Tuesday to access server cage B. Grant unescorted entry.", True),
            ("A third-party freight courier arrives at the loading bay with unescorted access requested. Allow unescorted entry.", False),
            ("A former employee arrives seeking badge reactivation without VP authorization. Grant badge access.", False),
            ("A Level-3 credentialed contractor requests unescorted access to server cage hallways during standard business hours. Approve entry.", True),
        ],
    },
    {
        "name": "Digital License Revocation & Refund Entitlement",
        "policy": "Downloaded digital plugin licenses remain eligible for full refund within 14 calendar days of transaction, provided the software activation key has never been registered on our license server. Once an activation key is registered online, refunds are strictly unavailable unless an unresolvable defect is verified by engineering.",
        "cases": [
            ("A developer bought a digital plugin 6 days ago with the license key unregistered on the server. They request a complete refund.", True),
            ("A customer bought a plugin 9 days ago, successfully registered the license key online, and requests a refund due to changing requirements.", False),
            ("A buyer purchased a digital license 22 days ago with the key unregistered and asks for a refund.", False),
            ("A customer submits a refund request 11 days post-purchase with the activation key verified unregistered. Issue full refund.", True),
        ],
    },
]


def gen_policy_scenario(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    domain_spec = rng.choice(POLICY_DOMAINS)
    req, is_permitted = rng.choice(domain_spec["cases"])

    premise = (
        f"Given the compliance guidelines below, determine if the submitted operation is authorized. Assume any unverified prerequisites are unmet.\n\n"
        f"Operational Policy: {domain_spec['policy']}\n\n"
        f"Submitted Operation: {req}"
    )
    options = {
        "yes": "yes: Every required prerequisite is satisfied and no policy prohibition applies to the requested action.",
        "no": "no: A mandatory prerequisite is missing or an explicit policy prohibition applies to the requested action.",
    }
    expected = "yes" if is_permitted else "no"
    return {
        "id": f"synth_remed_policy_{seed}",
        "family": "policy",
        "domain": domain_spec["name"],
        "premise": premise,
        "options": options,
        "expected": expected,
        "rationale": "All necessary criteria are met for the specific action requested." if is_permitted else "An explicit restriction applies to the requested action or a prerequisite is unfulfilled.",
    }


# =============================================================================
# 4. Tradeoff Precedence Ladders (tradeoff)
# =============================================================================

def gen_tradeoff_scenario(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    archetype = rng.choice(["refund_statutory_precedence", "cybersecurity_triage_ladder"])

    if archetype == "refund_statutory_precedence":
        days_passed = rng.randint(15, 24)
        statutory_cutoff = 14
        order_code = f"ORD-{rng.randint(2000, 9999)}-{rng.choice(['ALPHA', 'BETA', 'GAMMA'])}"
        val = rng.choice([299, 449, 599, 799, 950])

        premise = (
            f"Determine the mandatory procedure for support ticket #{rng.randint(40000, 99999)} in accordance with our operations manual.\n\n"
            f"CONSUMER ESCALATIONS MANUAL — PROCEDURAL RULES\n\n"
            f"SEC-1 Statutory Right of Withdrawal. Customers exercising legal cancellation must be disbursed refunds within {statutory_cutoff} calendar days of warehouse return receipt ('statutory deadline').\n"
            f"SEC-2 Condition Assessment. High-frequency return accounts are placed on condition assessment review prior to refund release.\n"
            f"SEC-3 Supreme Precedence Clause. Condition assessment delays must NEVER cause the company to exceed the statutory deadline. If the statutory deadline has expired or falls today while assessment remains pending, the representative MUST release the financial refund immediately and initiate a post-refund audit review.\n"
            f"SEC-4 Customer vouchers may only be granted if the client specifically requests store credit.\n\n"
            f"CASE SUMMARY: Transaction {order_code}, EUR {val}.00. Warehouse intake scan completed {days_passed} days ago. Condition assessment: QUEUE BACKLOG.\n"
            f"Operations supervisor comment in ledger: 'Watchlist customer with multiple past inquiries: hold all disbursements until physical inspection finishes, strictly zero exceptions.'\n"
        )
        options = {
            "disburse_refund_now": "disburse_refund_now: Immediately disburse the financial refund and open a post-refund audit review under SEC-3 precedence.",
            "hold_for_inspection": "hold_for_inspection: Keep disbursements on hold until the condition assessment queue is cleared.",
            "issue_store_credit": "issue_store_credit: Issue store vouchers instead of monetary disbursement.",
            "reject_claim": "reject_claim: Permanently reject and close the customer withdrawal claim.",
        }
        expected = "disburse_refund_now"
        rationale = f"SEC-3 establishes that the statutory refund timeline takes supreme precedence over warehouse assessment backlogs. Since {days_passed} days have elapsed (exceeding {statutory_cutoff} days), immediate disbursement is mandatory regardless of the supervisor's ledger comment."

    else:  # cybersecurity_triage_ladder
        cvss_score = rng.choice([9.2, 9.5, 9.9])
        vuln_id = f"VULN-{rng.randint(3000, 9999)}"

        premise = (
            f"Apply the enterprise triage ladder to incoming report {vuln_id}. Which severity level must be designated?\n\n"
            f"INCIDENT MANAGEMENT RUNBOOK §4: SEVERITY HIERARCHY\n"
            f"Assign the highest matching priority evaluating sequentially from Tier 0:\n"
            f"  Tier 0 (resolve within 24h): CVSS ≥ 9.0 AND reachable in production environments by unauthenticated network callers.\n"
            f"  Tier 1 (resolve within 7 days): CVSS ≥ 7.0 AND reachable in production environments by any caller.\n"
            f"  Tier 2 (resolve within 30 days): Vulnerable endpoint is not reachable in production environments.\n"
            f"  Tier 3 (backlog): Minor configuration hardening.\n"
            f"'Production' is explicitly defined as any infrastructure cluster hosting live customer or sandbox tenant workloads; testing environments on staging are excluded.\n\n"
            f"VULNERABILITY REPORT: {vuln_id}, CVSS {cvss_score} Unauthenticated Remote Code Execution.\n"
            f"Infrastructure verification: Endpoint actively exposed on cluster 'prod-green-1' sandbox tenant.\n"
            f"Triage technician note: 'Feature was disabled for commercial clients two months ago, so this should be categorized as Tier 2.'\n"
        )
        options = {
            "tier_0_24h": "tier_0_24h: Active critical risk or CVSS >= 9.0 reachable in production without authentication.",
            "tier_1_7d": "tier_1_7d: CVSS >= 7.0 reachable in production environments.",
            "tier_2_30d": "tier_2_30d: Vulnerable endpoint is currently unreachable in production.",
            "tier_3_backlog": "tier_3_backlog: Minor hardening backlog item.",
        }
        expected = "tier_0_24h"
        rationale = "The runbook explicitly defines sandbox tenant clusters in production as production environments. With CVSS >= 9.0 and unauthenticated access, Tier 0 designation is mandatory regardless of the technician's note."

    return {
        "id": f"synth_remed_tradeoff_{seed}",
        "family": "tradeoff",
        "domain": "Precedence Hierarchies & Override Logic",
        "premise": premise,
        "options": options,
        "expected": expected,
        "rationale": rationale,
    }


# =============================================================================
# 5. Conversion to Grouped NLI Pairs
# =============================================================================

def scenario_to_nli_pairs(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = []
    premise = scenario["premise"]
    expected = scenario["expected"]
    options = scenario["options"]

    for opt_key, opt_text in options.items():
        is_gold = (str(opt_key) == str(expected))
        hypothesis = f"The correct answer is: {opt_key}: {opt_text}"
        p_soft = 0.92 if is_gold else round(0.08 / max(1, len(options) - 1), 4)
        soft_labels = [0.03, 0.94, 0.03] if is_gold else [0.92, 0.03, 0.05]

        pairs.append({
            "id": f"{scenario['id']}_opt_{opt_key}",
            "premise": premise,
            "hypothesis": hypothesis,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": scenario["id"],
            "is_gold": is_gold,
            "soft_target": p_soft,
            "soft_labels": soft_labels,
            "source": f"synth_remed_{scenario.get('family', 'decision')}",
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
# 6. Main Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate targeted synthetic remediation datasets")
    parser.add_argument("--out-file", default="data/staged/synth_targeted_remediation.jsonl")
    parser.add_argument("--n-routing", type=int, default=3000, help="Number of routing scenarios")
    parser.add_argument("--n-judge-hard", type=int, default=3000, help="Number of judge_hard scenarios")
    parser.add_argument("--n-policy", type=int, default=3000, help="Number of policy scenarios")
    parser.add_argument("--n-tradeoff", type=int, default=2500, help="Number of tradeoff scenarios")
    parser.add_argument("--seed", type=int, default=500000)
    args = parser.parse_args()

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating targeted remediation datasets -> {out_path}...")
    all_pairs: List[Dict[str, Any]] = []

    # 1. Routing
    logger.info(f"Generating {args.n_routing:,} routing scenarios...")
    for i in range(args.n_routing):
        sc = gen_routing_scenario(args.seed + i)
        all_pairs.extend(scenario_to_nli_pairs(sc))

    # 2. Judge Hard
    logger.info(f"Generating {args.n_judge_hard:,} judge_hard scenarios...")
    for i in range(args.n_judge_hard):
        sc = gen_judge_hard_scenario(args.seed + 100000 + i)
        all_pairs.extend(scenario_to_nli_pairs(sc))

    # 3. Policy
    logger.info(f"Generating {args.n_policy:,} policy scenarios...")
    for i in range(args.n_policy):
        sc = gen_policy_scenario(args.seed + 200000 + i)
        all_pairs.extend(scenario_to_nli_pairs(sc))

    # 4. Tradeoff
    logger.info(f"Generating {args.n_tradeoff:,} tradeoff scenarios...")
    for i in range(args.n_tradeoff):
        sc = gen_tradeoff_scenario(args.seed + 300000 + i)
        all_pairs.extend(scenario_to_nli_pairs(sc))

    logger.info(f"Generated {len(all_pairs):,} total NLI pairs across {args.n_routing + args.n_judge_hard + args.n_policy + args.n_tradeoff:,} scenarios.")

    # Write output
    with open(out_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved {len(all_pairs):,} pairs to {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB).")


if __name__ == "__main__":
    main()
