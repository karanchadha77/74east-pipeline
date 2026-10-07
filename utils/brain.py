"""
Claude Code CLI Brain — Uses `claude` CLI as the reasoning engine.

Instead of paying for API calls, this shells out to the Claude Code CLI
which is included in your Pro/Max subscription.

Supports pluggable LLM backends via profile.yaml `ai` section.
When a profile with `ai` config is provided, requests are routed through
the configured backend per component. Without a profile, falls back to
direct Claude CLI calls (original behavior).
"""

import os
import subprocess
import json
import re
import hashlib
from pathlib import Path
from typing import Optional
from utils.llm import claude_cmd

def CACHE_DIR_FN():
    from utils.usercontext import cache_dir
    return cache_dir()


class ClaudeBrain:
    """Interface to AI reasoning — routes through pluggable LLM backends."""

    def __init__(self, verbose: bool = True, profile: dict = None):
        self.verbose = verbose
        self.profile = profile
        self._verify_cli()

    def _verify_cli(self):
        """Check that claude CLI is installed and accessible (skipped if not needed)."""
        # Skip CLI check if profile uses a non-CLI backend for all components
        if self.profile:
            ai_config = self.profile.get("ai", {})
            default = ai_config.get("default_backend", "claude_cli")
            components = ai_config.get("components", {})
            all_backends = set(components.values()) | {default}
            if "claude_cli" not in all_backends:
                if self.verbose:
                    print(f"  🧠 Using non-CLI backends: {', '.join(all_backends)}")
                return

        try:
            result = subprocess.run(
                [claude_cmd(), "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Claude CLI not responding. Install with: "
                    "npm install -g @anthropic-ai/claude-code"
                )
            if self.verbose:
                version = result.stdout.strip()
                print(f"  🧠 Claude CLI ready: {version}")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Install with:\n"
                "  npm install -g @anthropic-ai/claude-code\n"
                "Then run: claude auth"
            )

    def ask(self, prompt: str, timeout: int = 120, component: str = "general") -> str:
        """
        Send a prompt to the configured LLM backend.

        When a profile with `ai` config is available, routes through the
        pluggable backend for the given component. Otherwise falls back to
        direct Claude CLI calls (original behavior).
        """
        if self.verbose:
            preview = prompt[:80].replace('\n', ' ')
            print(f"  AI: {preview}...")

        # Route through pluggable backend if profile is available
        if self.profile:
            from utils.llm import get_backend
            backend = get_backend(component, self.profile)
            return backend.ask(prompt, timeout=timeout)

        # Fallback: direct Claude CLI (original behavior)
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        result = subprocess.run(
            [claude_cmd(), "-p", "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown error"
            raise RuntimeError(f"Claude CLI error: {error_msg}")

        # Parse the JSON output
        try:
            data = json.loads(result.stdout)
            # Claude Code JSON output has a "result" field
            return data.get("result", result.stdout)
        except json.JSONDecodeError:
            # Fallback: return raw stdout
            return result.stdout.strip()

    def ask_json(self, prompt: str, timeout: int = 120, component: str = "general") -> dict:
        """Ask the LLM and parse JSON from the response."""
        # Route through pluggable backend if profile is available
        if self.profile:
            from utils.llm import get_backend
            backend = get_backend(component, self.profile)
            return backend.ask_json(prompt, timeout=timeout)

        # Fallback: original behavior
        full_prompt = prompt + (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown fencing, no explanation, no preamble. Just the JSON object."
        )
        raw = self.ask(full_prompt, timeout=timeout)

        # Strip markdown code fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  Warning: JSON parse failed, raw response:\n{raw[:500]}")
            raise ValueError(f"LLM didn't return valid JSON: {e}")

    def ask_cached(self, prompt: str, cache_key: Optional[str] = None) -> str:
        """Ask with disk caching — useful for repeated identical questions."""
        if cache_key is None:
            cache_key = hashlib.md5(prompt.encode()).hexdigest()

        cache_file = CACHE_DIR_FN() / f"{cache_key}.txt"
        if cache_file.exists():
            if self.verbose:
                print(f"  💾 Cache hit: {cache_key}")
            return cache_file.read_text()

        result = self.ask(prompt)
        cache_file.write_text(result)
        return result

    def match_job(self, job_description: str, profile: dict, resume_text: str = "") -> dict:
        """Score a job posting against the user's profile with enhanced context."""
        if profile.get("firm"):
            return self.match_client_lead(job_description, profile, resume_text)
        # Build enhanced profile context
        skills = profile.get("skills", {})
        primary_skills = ", ".join(skills.get("primary", profile["preferences"].get("keywords", [])))
        secondary_skills = ", ".join(skills.get("secondary", []))
        ideal_job = profile.get("ideal_job_description", "")
        favorites = profile.get("favorite_companies", [])

        prompt = f"""You are a job matching engine. Be critical and honest.

APPLICANT PROFILE:
- Roles sought: {', '.join(profile['preferences']['roles'])}
- Primary skills: {primary_skills}
- Secondary skills: {secondary_skills}
- Location preference: {', '.join(profile['preferences']['locations'])}
- Remote only: {profile['preferences']['remote_only']}
- Ideal job: {ideal_job[:500]}
- Favorite companies (bonus +10 if match): {', '.join(favorites)}

APPLICANT RESUME:
{resume_text[:4000] if resume_text else '(No resume provided)'}

JOB POSTING:
{job_description[:8000]}

Return this exact JSON structure:
{{
  "score": <integer 0-100>,
  "apply": <true or false>,
  "reasoning": "<one sentence explaining the score>",
  "title_match": "<how well the title matches>",
  "skill_overlap": ["<matching skills>"],
  "missing_skills": ["<skills they want that applicant lacks>"],
  "red_flags": ["<any concerns>"],
  "cover_letter": "<2-3 paragraph tailored cover letter, or empty string — see rule below>",
  "improve_match": "<one suggestion to improve match for this job>"
}}

Scoring guidelines:
- 90-100: Perfect match (role, skills, location all align)
- 70-89: Strong match (most criteria met)
- 50-69: Partial match (some gaps but worth considering)
- Below 50: Poor match
- If the company is in the favorites list, add 10 bonus points (max 100)
- Set "apply" to true only if score >= {profile['preferences'].get('min_match_score', 65)}
- Write the cover_letter ONLY if score >= {profile['preferences'].get('min_match_score', 65)}; otherwise set cover_letter to "" — do not waste effort on poor matches
"""
        return self.ask_json(prompt, component="scoring")

    def match_client_lead(self, job_description: str, profile: dict, resume_text: str = "") -> dict:
        """Client-hunt mode: score a job posting as a sales lead for the firm in profile['firm'].

        The hiring company is the prospect, not the employer. Returns the same JSON
        keys as the job-seeker prompt so the tracker and dashboard work unchanged:
        "apply" means "pursue this lead" and "cover_letter" holds the outreach pitch.
        """
        firm = profile["firm"]
        min_score = profile["preferences"].get("min_match_score", 65)
        favorites = profile.get("favorite_companies", [])

        prompt = f"""You are a lead qualification engine for {firm.get('name', 'an outsourced services firm')}. Be critical and honest.

A company has posted a job. The hiring company is a PROSPECT. Decide whether the firm below
could do this work for them as an outsourced retainer instead of (or alongside) the hire.
An on-site or full-time posting is NOT disqualifying by itself: an owner hiring an Accounting
Manager or Controller is a buying signal for a fractional/outsourced controller. Judge the
company, its industry and the scope of work, not whether a person could take the job.

FIRM BRIEF:
{firm.get('brief', '')[:6000]}

ADDITIONAL FIRM MATERIAL:
{resume_text[:2000] if resume_text else '(none)'}

Favorite / priority prospects (bonus +10 if match): {', '.join(favorites) or '(none)'}

JOB POSTING:
{job_description[:8000]}

Return this exact JSON structure:
{{
  "score": <integer 0-100>,
  "apply": <true or false — true means pursue this company as a client lead>,
  "reasoning": "<one sentence: why this company is or isn't a good client lead>",
  "title_match": "<engagement type this maps to, e.g. fractional controller retainer, books cleanup, AP back office, or 'none'>",
  "skill_overlap": ["<firm capabilities that match the work described>"],
  "missing_skills": ["<work they need that the firm does not do or is out of scope>"],
  "red_flags": ["<skip criteria hit: US residency/CPA signature/on-site required, tax/payroll-only, low budget, no remote systems access, etc.>"],
  "cover_letter": "<short outreach pitch to the owner/hiring manager, or empty string — see rule below>",
  "improve_match": "<best angle to pitch, or the buying signal to look for before reaching out>"
}}

Scoring guidelines:
- 90-100: US freight brokerage/3PL/trucking company needing controller-level, recurring accounting work; buying signals present (factoring, multiple entities, messy/late books, lender needs financials, bookkeeper left, rapid growth)
- 70-89: US freight/logistics company with recurring accounting needs, or a non-freight US SMB clearly needing controller-level outsourced work (likely USD 3,000+/month)
- 50-69: Plausible but weak — scope unclear, mostly bookkeeping, or industry outside the firm's focus
- Below 50: Skip — tax/payroll-only, hourly data entry, large enterprise with an in-house finance team, non-US company, roles legally requiring a US CPA signature or US residency, or work the firm does not do
- Recruiting agencies posting on behalf of an unnamed client: cap the score at 60
- If the company is in the favorites list, add 10 bonus points (max 100)
- Set "apply" to true only if score >= {min_score}
- Write the cover_letter ONLY if score >= {min_score}; otherwise set it to "".
  Pitch rules: written as {firm.get('lead', 'the founder')}, the named lead, with the team behind him; 120-180 words;
  reference specifics from the posting; offer the firm as an outsourced alternative to the hire;
  cite only capabilities and anonymized proof points from the brief — never invent clients, numbers or credentials;
  end with a low-commitment call to action (short call). No subject line, no placeholders.
"""
        return self.ask_json(prompt, component="scoring")

    def score_profile(self, profile: dict, resume_text: str = "") -> dict:
        """Analyze the user's profile and resume for job market readiness."""
        skills = profile.get("skills", {})
        primary = ", ".join(skills.get("primary", []))
        secondary = ", ".join(skills.get("secondary", []))
        roles = ", ".join(profile["preferences"]["roles"])
        ideal = profile.get("ideal_job_description", "")

        prompt = f"""Analyze this job seeker's profile and resume for job market readiness.

PROFILE:
- Target roles: {roles}
- Primary skills: {primary}
- Secondary skills: {secondary}
- Ideal job: {ideal[:500]}
- Location: {profile['personal'].get('location', 'Not specified')}

RESUME TEXT:
{resume_text[:6000] if resume_text else '(No resume provided)'}

Return a JSON assessment:
{{
  "profile_score": <0-100>,
  "strengths": ["<top 3 strengths>"],
  "gaps": ["<skills or experience gaps>"],
  "resume_suggestions": ["<specific resume improvements>"],
  "keyword_recommendations": ["<keywords to add to profile>"],
  "role_fit": {{
    "<role name>": <0-100 fit score>
  }},
  "summary": "<2-3 sentence overall assessment>"
}}
"""
        return self.ask_json(prompt, timeout=180, component="profile_analysis")

    def answer_question(self, question: str, profile: dict, context: str = "") -> str:
        """Answer a custom application question using AI."""
        return self.ask(f"""You are filling out a job application for someone.
Answer this question concisely and professionally (1-3 sentences max).

Applicant info:
- Name: {profile['personal']['first_name']} {profile['personal']['last_name']}
- Location: {profile['personal']['location']}
- Looking for: {', '.join(profile['preferences']['roles'])}

Additional context: {context}

Question: {question}

Answer (be concise, direct, professional):""", component="form_analysis")

    def analyze_form(self, form_html: str, profile: dict) -> list:
        """Analyze a form's HTML and return fill instructions."""
        return self.ask_json(f"""You are a form-filling automation assistant.

APPLICANT PROFILE:
{json.dumps(profile['personal'], indent=2)}

COMMON ANSWERS:
{json.dumps(profile.get('common_answers', {}), indent=2)}

FORM HTML (may be truncated):
{form_html[:12000]}

Analyze every visible input field. Return a JSON array of actions:
[
  {{"action": "fill", "selector": "<CSS selector>", "value": "<text to type>", "field_name": "<what field this is>"}},
  {{"action": "select", "selector": "<CSS selector>", "value": "<option value>", "field_name": "<what field>"}},
  {{"action": "check", "selector": "<CSS selector>", "field_name": "<what checkbox>"}},
  {{"action": "upload", "selector": "<CSS selector>", "file_key": "resume", "field_name": "resume upload"}},
  {{"action": "click", "selector": "<CSS selector>", "field_name": "submit button"}}
]

Rules:
- Use the most specific CSS selector possible (prefer #id, then [name=...], then [aria-label=...])
- For file uploads, always set file_key to "resume"
- Put the submit/next button click LAST
- Skip hidden fields and CSRF tokens
- For dropdowns, use the actual option value attribute
""", component="form_analysis")
