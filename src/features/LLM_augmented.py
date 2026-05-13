import re
import json
import anthropic
from dataclasses import dataclass
from features.system_prompt import SYSTEM_PROMPT, FALLBACK_BATCH_SYSTEM_PROMPT, TEMPLATE_GENERATION_SYSTEM_PROMPT

client = anthropic.Anthropic()


# --- Step 1: Sample a representative subset of your logs ---

def sample_representative_logs(logs: list[dict], n: int = 500) -> list[dict]:
    """
    Sample logs to cover diverse patterns.
    Stratify by parent image to ensure coverage.
    """
    from collections import defaultdict
    import random

    buckets = defaultdict(list)
    for log in logs:
        key = log.get("parent_image_path", "unknown").split("\\")[-1].lower()
        buckets[key].append(log)

    sampled = []
    per_bucket = max(1, n // len(buckets))
    for bucket in buckets.values():
        sampled.extend(random.sample(bucket, min(per_bucket, len(bucket))))

    return sampled[:n]


# --- Step 2: Ask LLM to generate reusable templates from the sample ---

def generate_template_library(sampled_logs: list[dict]) -> list[dict]:
    """
    Ask Claude ONCE to analyze a sample and produce
    a reusable pattern → template mapping.
    """
    formatted = json.dumps([{
        "command_line": l.get("command_line", ""),
        "image_path":   l.get("image_path", ""),
        "parent_image": l.get("parent_image_path", "")
    } for l in sampled_logs], indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=TEMPLATE_GENERATION_SYSTEM_PROMPT,  # reuse from previous step
        messages=[{"role": "user", "content": f"""Analyze these eCar log samples and extract \
a reusable TEMPLATE LIBRARY.

For each distinct behavioral pattern you identify, produce:
- "pattern_id": short unique name (e.g. "lolbin_certutil_decode")
- "match_rules": dict of regex patterns to match against command_line, image_path, parent_image
- "template": enriched description with {{placeholders}} for variable parts
  (use {{command_line}}, {{image_path}}, {{parent_image}}, {{filename}}, {{args}})
- "threat_level": benign | suspicious | malicious

Return ONLY a JSON array, no markdown.

Logs sample:
{formatted}"""}]
    )

    return json.loads(response.content[0].text)


# --- Step 3: Apply templates locally at scale ---

@dataclass
class Template:
    pattern_id:   str
    match_rules:  dict   # field -> regex
    template:     str
    threat_level: str


def compile_templates(raw_templates: list[dict]) -> list[Template]:
    return [Template(**t) for t in raw_templates]


def match_template(log: dict, templates: list[Template]) -> Template | None:
    """Find the first template whose rules match this log."""
    for tmpl in templates:
        if all(
            re.search(pattern, log.get(field, ""), re.IGNORECASE)
            for field, pattern in tmpl.match_rules.items()
        ):
            return tmpl
    return None  # no match → fallback needed


def apply_template(log: dict, template: Template) -> str:
    """Fill template placeholders with actual log values."""
    cmd   = log.get("command_line", "")
    image = log.get("image_path", "")
    parent = log.get("parent_image_path", "")

    return template.template.format(
        command_line = cmd,
        image_path   = image,
        parent_image = parent,
        filename     = image.split("\\")[-1],
        args         = " ".join(cmd.split()[1:])  # everything after the binary
    )


def enrich_batch(logs: list[dict], batch_size: int = 20) -> list[str]:
    """
    Call LLM once per batch of unmatched logs.
    Returns enriched descriptions in the same order.
    """
    results = []

    for i in range(0, len(logs), batch_size):
        batch = logs[i:i + batch_size]
        formatted = json.dumps([{
            "id":           idx,
            "command_line": l.get("command_line", ""),
            "image_path":   l.get("image_path", ""),
            "parent_image": l.get("parent_image_path", "")
        } for idx, l in enumerate(batch)], indent=2)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"""Enrich each eCar log below with a \
3-sentence semantic description (same rules as before).
Return ONLY a JSON array of {{"id": int, "description": str}}, no markdown.

Logs:
{formatted}"""}]
        )

        batch_results = json.loads(response.content[0].text)
        # Sort by id to preserve order
        batch_results.sort(key=lambda x: x["id"])
        results.extend([r["description"] for r in batch_results])

    return results


# --- Step 5: Full pipeline ---

def enrich_all_logs(logs: list[dict]) -> list[str]:
    print(f"Total logs: {len(logs)}")

    # 1. Sample and generate templates (1-2 LLM calls)
    sample   = sample_representative_logs(logs, n=500)
    raw_tmpl = generate_template_library(sample)
    templates = compile_templates(raw_tmpl)
    print(f"Generated {len(templates)} templates")

    # 2. Apply templates locally — free and instant
    enriched      = []
    unmatched_idx = []
    unmatched_logs = []

    for i, log in enumerate(logs):
        tmpl = match_template(log, templates)
        if tmpl:
            enriched.append(apply_template(log, tmpl))
        else:
            enriched.append(None)          # placeholder
            unmatched_idx.append(i)
            unmatched_logs.append(log)

    print(f"Matched:   {len(logs) - len(unmatched_logs)} logs (free)")
    print(f"Unmatched: {len(unmatched_logs)} logs (LLM batches)")

    # 3. Enrich unmatched logs in batches
    if unmatched_logs:
        unmatched_enriched = enrich_batch(unmatched_logs, batch_size=20)
        for idx, desc in zip(unmatched_idx, unmatched_enriched):
            enriched[idx] = desc

    return enriched


def enrich_process_nodes(process_logs: list[dict], batch_size: int = 20) -> list[str]:
    """Create enriched descriptions for a list of process logs.

    Deduplicate identical process entries before calling the LLM, then restore order.
    """
    unique_map = {}
    unique_logs = []
    ordered_keys = []

    for log in process_logs:
        key = (
            log.get("command_line", ""),
            log.get("image_path", ""),
            log.get("parent_image_path", ""),
        )
        ordered_keys.append(key)
        if key not in unique_map:
            unique_map[key] = len(unique_logs)
            unique_logs.append(
                {
                    "command_line": log.get("command_line", ""),
                    "image_path": log.get("image_path", ""),
                    "parent_image_path": log.get("parent_image_path", ""),
                }
            )

    enriched_unique = enrich_all_logs(unique_logs)
    return [enriched_unique[unique_map[key]] for key in ordered_keys]