#!/usr/bin/env python3
"""
Structured LLM Prompt Pipeline & JSON Extractor
================================================
Turns unstructured text (invoices, resumes) into Pydantic-validated JSON using
Google Gemini or OpenAI.

Pipeline:
    1. Parameterized prompt templates  (string.Template, schema injected automatically)
    2. Few-shot examples               (user/assistant turns before the real document)
    3. JSON-mode API call              (Gemini or OpenAI, temperature 0)
    4. Pydantic validation             (types, ranges, cross-field arithmetic checks)
    5. Self-repair loop                (validation errors are fed back to the model)

Install:
    pip install pydantic google-generativeai openai

Usage:
    export GEMINI_API_KEY=...            # or OPENAI_API_KEY=...
    python llm_json_extractor.py --type invoice --sample --provider gemini
    python llm_json_extractor.py --type resume  --file resume.txt --provider openai --out out.json
    python llm_json_extractor.py --type invoice --sample --provider mock   # offline demo, no API key
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from string import Template
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

T = TypeVar("T", bound=BaseModel)


# ----------------------------------------------------------------------------
# 1. Pydantic schemas (the contract the LLM output must satisfy)
# ----------------------------------------------------------------------------
class LineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def _line_math(self) -> "LineItem":
        if abs(self.quantity * self.unit_price - self.total) > 0.02:
            raise ValueError(
                f"line '{self.description}': quantity*unit_price "
                f"({self.quantity * self.unit_price:.2f}) != total ({self.total:.2f})"
            )
        return self


class Invoice(BaseModel):
    invoice_number: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)
    customer_name: Optional[str] = None
    issue_date: date
    due_date: Optional[date] = None
    currency: str = Field(pattern=r"^[A-Z]{3}$", description="ISO 4217 code, e.g. USD, INR")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax: float = Field(ge=0, default=0)
    total_amount: float = Field(ge=0)

    @model_validator(mode="after")
    def _invoice_math(self) -> "Invoice":
        line_sum = sum(li.total for li in self.line_items)
        if abs(line_sum - self.subtotal) > 0.02:
            raise ValueError(f"sum of line totals ({line_sum:.2f}) != subtotal ({self.subtotal:.2f})")
        if abs(self.subtotal + self.tax - self.total_amount) > 0.02:
            raise ValueError(
                f"subtotal + tax ({self.subtotal + self.tax:.2f}) != total_amount ({self.total_amount:.2f})"
            )
        if self.due_date and self.due_date < self.issue_date:
            raise ValueError("due_date is before issue_date")
        return self


class Experience(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = Field(None, description="YYYY-MM or YYYY")
    end_date: Optional[str] = Field(None, description="YYYY-MM, YYYY, or null if current")
    highlights: list[str] = []


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    year: Optional[int] = Field(None, ge=1950, le=2100)


class Resume(BaseModel):
    full_name: str = Field(min_length=1)
    email: Optional[str] = Field(None, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[Education] = []

    @field_validator("skills")
    @classmethod
    def _dedupe_skills(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for s in (x.strip() for x in v):
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out


# ----------------------------------------------------------------------------
# 2. Parameterized prompt templates + few-shot examples
# ----------------------------------------------------------------------------
SYSTEM_TEMPLATE = Template(
    """You are a precise information-extraction engine.
Convert the user's ${doc_type} text into a single JSON object that conforms EXACTLY
to this JSON Schema:

${schema}

Rules:
- Output ONLY the JSON object. No prose, no markdown code fences.
- Use null for missing optional fields. Never invent or guess values.
- Dates must be ISO 8601 (YYYY-MM-DD). Numbers must be plain numerics: no currency
  symbols, no thousands separators.
${extra_rules}"""
)

USER_TEMPLATE = Template("Extract the ${doc_type} below.\n\n<document>\n${document}\n</document>")

REPAIR_TEMPLATE = Template(
    "Your previous output failed validation:\n${errors}\n\n"
    "Return a corrected JSON object only. Re-read the document; fix the listed problems "
    "without inventing data."
)

DOC_CONFIG: dict[str, dict] = {
    "invoice": {
        "model": Invoice,
        "extra_rules": "- Currency must be a 3-letter ISO code (map symbols: $->USD, \u20b9->INR, \u20ac->EUR).\n"
        "- Line totals must equal quantity*unit_price; subtotal + tax must equal total_amount.",
        "few_shot": [
            (
                "INVOICE No. 8841\nFrom: Brightline Studio\nBilled to: Harlow & Co\n"
                "Issued: March 3, 2026   Due: April 2, 2026\n"
                "3 x Logo concept @ $200.00 = $600.00\n1 x Brand guide @ $400.00 = $400.00\n"
                "Subtotal $1,000.00  Tax (10%) $100.00  TOTAL DUE $1,100.00",
                {
                    "invoice_number": "8841",
                    "vendor_name": "Brightline Studio",
                    "customer_name": "Harlow & Co",
                    "issue_date": "2026-03-03",
                    "due_date": "2026-04-02",
                    "currency": "USD",
                    "line_items": [
                        {"description": "Logo concept", "quantity": 3, "unit_price": 200.0, "total": 600.0},
                        {"description": "Brand guide", "quantity": 1, "unit_price": 400.0, "total": 400.0},
                    ],
                    "subtotal": 1000.0,
                    "tax": 100.0,
                    "total_amount": 1100.0,
                },
            )
        ],
    },
    "resume": {
        "model": Resume,
        "extra_rules": "- Keep 'highlights' as short bullet strings taken from the text.\n"
        "- 'skills' is a flat list of individual skills.",
        "few_shot": [
            (
                "Rahul Mehta | rahul.mehta@example.com | +91 98765 43210\n"
                "Backend engineer with 4 years building APIs.\n"
                "Experience:\nAcme Corp, Software Engineer, Jun 2022 - Present\n"
                "- Cut API latency by 35%\n"
                "Education: B.E. Computer Science, VIT, 2022\n"
                "Skills: Python, Django, PostgreSQL, Docker",
                {
                    "full_name": "Rahul Mehta",
                    "email": "rahul.mehta@example.com",
                    "phone": "+91 98765 43210",
                    "summary": "Backend engineer with 4 years building APIs.",
                    "skills": ["Python", "Django", "PostgreSQL", "Docker"],
                    "experience": [
                        {
                            "company": "Acme Corp",
                            "title": "Software Engineer",
                            "start_date": "2022-06",
                            "end_date": None,
                            "highlights": ["Cut API latency by 35%"],
                        }
                    ],
                    "education": [{"institution": "VIT", "degree": "B.E. Computer Science", "year": 2022}],
                },
            )
        ],
    },
}

SAMPLE_DOCS: dict[str, str] = {
    "invoice": (
        "INVOICE #INV-2026-0042\n"
        "Acme Cloud Services Pvt Ltd, Indore\n"
        "Bill to: Nova Retail LLP\n"
        "Date: 14 Oct 2026    Payment due: 13 Nov 2026\n\n"
        "2 x Managed Hosting (monthly) @ \u20b912,500 = \u20b925,000\n"
        "1 x SSL Certificate (annual) @ \u20b93,000 = \u20b93,000\n\n"
        "Subtotal: \u20b928,000\nGST 18%: \u20b95,040\nTOTAL: \u20b933,040"
    ),
    "resume": (
        "PRIYA SHARMA\npriya.sharma@example.com  |  +91 91234 56789\n\n"
        "Data analyst with a focus on dashboards and ML-driven forecasting.\n\n"
        "WORK\nDeloitte (Intern), Analyst Intern, Jan 2026 - Jun 2026\n"
        " * Built Power BI dashboards used by 3 teams\n"
        " * Automated weekly reports with Python\n\n"
        "EDUCATION\nB.Tech, AI & Data Science, RGPV affiliated college, 2027\n\n"
        "SKILLS: Python, Pandas, SQL, Power BI, python, React"
    ),
}


def build_messages(doc_type: str, document: str) -> tuple[str, list[dict]]:
    """Return (system_prompt, messages) with schema + few-shot turns baked in."""
    cfg = DOC_CONFIG[doc_type]
    schema = json.dumps(cfg["model"].model_json_schema(), indent=2)
    system = SYSTEM_TEMPLATE.substitute(doc_type=doc_type, schema=schema, extra_rules=cfg["extra_rules"])

    messages: list[dict] = []
    for example_text, example_json in cfg["few_shot"]:
        messages.append({"role": "user", "content": USER_TEMPLATE.substitute(doc_type=doc_type, document=example_text)})
        messages.append({"role": "assistant", "content": json.dumps(example_json)})
    messages.append({"role": "user", "content": USER_TEMPLATE.substitute(doc_type=doc_type, document=document)})
    return system, messages


# ----------------------------------------------------------------------------
# 3. LLM providers (thin adapters with one method: complete(system, messages) -> str)
# ----------------------------------------------------------------------------
class GeminiProvider:
    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        genai.configure(api_key=key)
        self._genai, self.model = genai, model

    def complete(self, system: str, messages: list[dict]) -> str:
        model = self._genai.GenerativeModel(
            self.model,
            system_instruction=system,
            generation_config=self._genai.GenerationConfig(temperature=0, response_mime_type="application/json"),
        )
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]} for m in messages
        ]
        return model.generate_content(contents).text


class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY.")
        self.client, self.model = OpenAI(), model

    def complete(self, system: str, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, *messages],
        )
        return resp.choices[0].message.content


class MockProvider:
    """Offline demo: first reply has a deliberate arithmetic error to show the repair loop."""

    def __init__(self, doc_type: str):
        self.doc_type, self.calls = doc_type, 0

    def complete(self, system: str, messages: list[dict]) -> str:
        self.calls += 1
        if self.doc_type == "invoice":
            data = {
                "invoice_number": "INV-2026-0042", "vendor_name": "Acme Cloud Services Pvt Ltd",
                "customer_name": "Nova Retail LLP", "issue_date": "2026-10-14", "due_date": "2026-11-13",
                "currency": "INR",
                "line_items": [
                    {"description": "Managed Hosting (monthly)", "quantity": 2, "unit_price": 12500, "total": 25000},
                    {"description": "SSL Certificate (annual)", "quantity": 1, "unit_price": 3000, "total": 3000},
                ],
                "subtotal": 28000, "tax": 5040, "total_amount": 33040,
            }
            if self.calls == 1:
                data["total_amount"] = 33400  # deliberate error
        else:
            data = {
                "full_name": "Priya Sharma", "email": "priya.sharma@example.com", "phone": "+91 91234 56789",
                "summary": "Data analyst with a focus on dashboards and ML-driven forecasting.",
                "skills": ["Python", "Pandas", "SQL", "Power BI", "python", "React"],
                "experience": [{
                    "company": "Deloitte", "title": "Analyst Intern", "start_date": "2026-01",
                    "end_date": "2026-06",
                    "highlights": ["Built Power BI dashboards used by 3 teams", "Automated weekly reports with Python"],
                }],
                "education": [{"institution": "RGPV affiliated college", "degree": "B.Tech, AI & Data Science", "year": 2027}],
            }
        return json.dumps(data)


# ----------------------------------------------------------------------------
# 4 + 5. Extraction with validation and self-repair
# ----------------------------------------------------------------------------
def _clean_json(raw: str) -> str:
    """Strip markdown fences / stray prose around a JSON object."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start : end + 1] if start != -1 and end != -1 else raw


def _format_errors(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "\n".join(f"- {'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in exc.errors())
    return f"- {exc}"


def extract(provider, doc_type: str, document: str, max_retries: int = 3, verbose: bool = True) -> BaseModel:
    """Run the full pipeline. Raises RuntimeError if no valid output after max_retries."""
    model_cls: Type[BaseModel] = DOC_CONFIG[doc_type]["model"]
    system, messages = build_messages(doc_type, document)

    last_error = ""
    for attempt in range(1, max_retries + 1):
        raw = provider.complete(system, messages)
        try:
            return model_cls.model_validate(json.loads(_clean_json(raw)))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = _format_errors(exc)
            if verbose:
                print(f"[attempt {attempt}/{max_retries}] validation failed:\n{last_error}", file=sys.stderr)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_TEMPLATE.substitute(errors=last_error)},
            ]
    raise RuntimeError(f"No valid {doc_type} JSON after {max_retries} attempts. Last errors:\n{last_error}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Extract validated JSON from unstructured text with an LLM.")
    p.add_argument("--type", choices=DOC_CONFIG, required=True, help="document type / schema")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="path to a UTF-8 text file")
    src.add_argument("--sample", action="store_true", help="use the built-in sample document")
    p.add_argument("--provider", choices=["gemini", "openai", "mock"], default="gemini")
    p.add_argument("--model", help="override the default model name")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = p.parse_args()

    document = SAMPLE_DOCS[args.type] if args.sample else args.file.read_text(encoding="utf-8")

    if args.provider == "gemini":
        provider = GeminiProvider(args.model) if args.model else GeminiProvider()
    elif args.provider == "openai":
        provider = OpenAIProvider(args.model) if args.model else OpenAIProvider()
    else:
        provider = MockProvider(args.type)

    result = extract(provider, args.type, document, max_retries=args.retries)
    output = result.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Saved validated {args.type} JSON to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
