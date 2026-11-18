"""Generate the sample resumes and edge-case files under ``sample_data/``.

Run from the repo root::

    python scripts/generate_samples.py

Output is deterministic: all content is fixed, PDF/DOCX metadata uses a fixed timestamp and
DOCX zip entries use a fixed mtime, so re-running overwrites every file with identical bytes.
``sample_data/MANIFEST.md`` (the answer key for demos and tests) is generated from the same data.
"""

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_data"
RESUME_DIR = SAMPLE_DIR / "resumes"
EDGE_DIR = SAMPLE_DIR / "edge_cases"

# Fixed values so output is byte-identical across runs (not meaningful dates).
FIXED_TIMESTAMP = datetime(2024, 1, 1, 0, 0, 0)
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class Job:
    title: str
    company: str
    location: str
    dates: str
    bullets: tuple[str, ...]


@dataclass(frozen=True)
class Section:
    heading: str
    items: tuple[str, ...]


@dataclass(frozen=True)
class Resume:
    first: str
    last: str
    fmt: str  # "pdf" | "docx" | "txt"
    title: str
    location: str
    phone: str
    years: int
    summary: str
    skills: tuple[tuple[str, str], ...]  # (category, comma-separated items)
    jobs: tuple[Job, ...]
    education: tuple[str, ...]
    extras: tuple[Section, ...] = ()
    python_placement: str = "-"  # answer-key note for MANIFEST.md
    skills_as_table: bool = False  # DOCX only
    page_break_before_extras: bool = False  # PDF only

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def email(self) -> str:
        return f"{self.first.lower()}.{self.last.lower()}@example.com"

    @property
    def contact(self) -> str:
        return f"{self.location} | {self.email} | {self.phone}"

    @property
    def filename(self) -> str:
        return f"resume_{self.first.lower()}_{self.last.lower()}.{self.fmt}"

    @property
    def sections(self) -> tuple[Section, ...]:
        return (*self.extras, Section("Education", self.education))


RESUMES: tuple[Resume, ...] = (
    Resume(
        first="John",
        last="Doe",
        fmt="pdf",
        title="Senior Backend Python Developer",
        location="Austin, TX, USA",
        phone="+1 555-0101",
        years=8,
        summary=(
            "Senior backend engineer with 8 years of experience building Python services and "
            "APIs. Specialises in Python web frameworks, PostgreSQL performance tuning and "
            "event-driven architectures."
        ),
        skills=(
            ("Languages", "Python, SQL, Bash"),
            ("Frameworks", "Django, FastAPI, Flask, Celery"),
            ("Data", "PostgreSQL, Redis"),
            ("Cloud & Tools", "Docker, AWS Lambda, AWS SQS, GitHub Actions"),
        ),
        jobs=(
            Job(
                "Senior Backend Engineer",
                "Lonestar Payments",
                "Austin, TX",
                "2023 - Present",
                (
                    "Led migration of a monolithic Django app to 12 FastAPI microservices, "
                    "cutting p95 latency by 40%.",
                    "Designed an idempotent payment-webhook pipeline on Celery and Redis "
                    "processing 3M events per day.",
                    "Mentored 4 engineers and introduced mypy type checking across all Python "
                    "repositories.",
                ),
            ),
            Job(
                "Backend Developer",
                "Bluebonnet Health",
                "Austin, TX",
                "2020 - 2023",
                (
                    "Built HIPAA-compliant REST APIs with Django REST Framework.",
                    "Optimised PostgreSQL queries and indexes, reducing report generation from "
                    "9 minutes to 45 seconds.",
                    "Automated container deployments with Docker and GitHub Actions.",
                ),
            ),
            Job(
                "Junior Software Engineer",
                "Prairie Logistics",
                "Dallas, TX",
                "2018 - 2020",
                (
                    "Wrote Python ETL scripts consolidating shipment data from 5 carriers.",
                    "Maintained internal Flask dashboards for the operations team.",
                ),
            ),
        ),
        education=("B.S. Computer Science, Hill Country University, 2018",),
        extras=(
            Section(
                "Projects",
                (
                    "Maintainer of django-audit-lite, an open-source Python package for model "
                    "change history.",
                    "Speaker: 'Async Python in Production' at the Austin Python Meetup.",
                ),
            ),
            Section("Certifications", ("AWS Certified Developer - Associate",)),
        ),
        python_placement="Multiple times (title, summary, skills, bullets, page 2)",
        page_break_before_extras=True,
    ),
    Resume(
        first="Priya",
        last="Sharma",
        fmt="pdf",
        title="Data Scientist",
        location="Bengaluru, India",
        phone="+91 555-0102",
        years=5,
        summary=(
            "Data scientist with 5 years of experience in demand forecasting, experimentation "
            "and customer analytics for retail and fintech."
        ),
        skills=(
            ("Programming", "Python, SQL"),
            ("Libraries", "pandas, scikit-learn, XGBoost, statsmodels"),
            ("Methods", "Time-series forecasting, A/B testing, Bayesian statistics"),
            ("Tools", "Airflow, Tableau, Jupyter"),
        ),
        jobs=(
            Job(
                "Data Scientist",
                "Monsoon Retail",
                "Bengaluru",
                "2023 - Present",
                (
                    "Built XGBoost demand-forecasting models that reduced stock-outs by 18% "
                    "across 300 stores.",
                    "Designed and analysed 25+ A/B tests on the checkout funnel.",
                    "Productionised feature pipelines with Airflow.",
                ),
            ),
            Job(
                "Associate Data Scientist",
                "Ganga Fintech",
                "Pune",
                "2021 - 2023",
                (
                    "Developed a credit-risk scorecard using logistic regression and gradient "
                    "boosting.",
                    "Automated weekly KPI reporting with Jupyter notebooks and Tableau.",
                ),
            ),
        ),
        education=("M.Sc. Statistics, Deccan Institute of Science, 2021",),
        python_placement="Skills list only",
    ),
    Resume(
        first="Marcus",
        last="Chen",
        fmt="pdf",
        title="DevOps / Platform Engineer",
        location="Seattle, WA, USA",
        phone="+1 555-0103",
        years=7,
        summary=(
            "Platform engineer with 7 years of experience running Kubernetes at scale and "
            "writing infrastructure tooling in Go."
        ),
        skills=(
            ("Languages", "Go, Bash, HCL"),
            ("Platforms", "Kubernetes, Helm, ArgoCD, AWS EKS"),
            ("Infrastructure", "Terraform, Linux, Nginx"),
            ("Observability", "Prometheus, Grafana, Loki"),
        ),
        jobs=(
            Job(
                "Senior Platform Engineer",
                "Cascade Cloudworks",
                "Seattle, WA",
                "2022 - Present",
                (
                    "Operated 40+ Kubernetes clusters on EKS serving 600 microservices.",
                    "Wrote a Go Kubernetes operator that automates certificate rotation, "
                    "eliminating 95% of manual tickets.",
                    "Introduced GitOps with ArgoCD and Helm for all production deployments.",
                ),
            ),
            Job(
                "DevOps Engineer",
                "Rainier Media",
                "Tacoma, WA",
                "2019 - 2022",
                (
                    "Migrated on-prem virtual machines to AWS using reusable Terraform modules.",
                    "Built Prometheus and Grafana alerting, cutting MTTR from 2 hours to "
                    "25 minutes.",
                ),
            ),
        ),
        education=("B.S. Computer Engineering, Puget Sound Tech, 2019",),
        extras=(Section("Certifications", ("Certified Kubernetes Administrator (CKA)",)),),
    ),
    Resume(
        first="Elena",
        last="Rodriguez",
        fmt="docx",
        title="Java / Spring Developer",
        location="Madrid, Spain",
        phone="+34 555-0104",
        years=6,
        summary=(
            "Backend developer with 6 years of experience designing event-driven Java "
            "microservices with Spring Boot and Kafka for logistics and banking."
        ),
        skills=(
            ("Languages", "Java 17, Kotlin, SQL"),
            ("Frameworks", "Spring Boot, Spring Cloud, Hibernate"),
            ("Messaging", "Apache Kafka, RabbitMQ"),
            ("Tooling", "Maven, Gradle, Docker, Jenkins, JUnit 5"),
        ),
        jobs=(
            Job(
                "Senior Java Developer",
                "Meseta Logistics",
                "Madrid",
                "2023 - Present",
                (
                    "Designed event-driven order services with Spring Boot and Kafka handling "
                    "1,200 messages per second.",
                    "Reduced JVM memory footprint by 30% through garbage-collector tuning.",
                    "Led the upgrade of 15 services from Java 11 to Java 17.",
                ),
            ),
            Job(
                "Java Developer",
                "Castilla Bank Systems",
                "Valladolid",
                "2020 - 2023",
                (
                    "Built Spring Cloud microservices for card transaction processing.",
                    "Wrote integration tests with JUnit 5 and Testcontainers.",
                ),
            ),
        ),
        education=("B.Eng. Software Engineering, Universidad del Norte, 2020",),
        skills_as_table=True,
    ),
    Resume(
        first="Aisha",
        last="Okafor",
        fmt="docx",
        title="Frontend Engineer (TypeScript / React)",
        location="London, UK",
        phone="+44 555-0105",
        years=3,
        summary=(
            "Frontend engineer with 3 years of experience building accessible, fast web apps "
            "in TypeScript and React."
        ),
        skills=(
            ("Languages", "TypeScript, JavaScript, HTML, CSS"),
            ("Frameworks", "React, Next.js, Redux Toolkit, Tailwind CSS"),
            ("Testing", "Jest, Testing Library, Storybook"),
            ("Practices", "Accessibility (WCAG 2.2), Web performance"),
        ),
        jobs=(
            Job(
                "Frontend Engineer",
                "Thames Digital",
                "London",
                "2024 - Present",
                (
                    "Rebuilt the customer dashboard in React and TypeScript, raising the "
                    "Lighthouse performance score from 62 to 94.",
                    "Created a shared component library used by 5 product teams.",
                ),
            ),
            Job(
                "Junior Frontend Developer",
                "Brixton Studio",
                "London",
                "2023 - 2024",
                (
                    "Implemented responsive marketing pages with Next.js.",
                    "Raised unit-test coverage from 35% to 80% with Jest.",
                ),
            ),
        ),
        education=("B.Sc. Computer Science, Southbank University, 2023",),
    ),
    Resume(
        first="Liam",
        last="Walsh",
        fmt="docx",
        title="QA Automation Engineer",
        location="Dublin, Ireland",
        phone="+353 555-0106",
        years=4,
        summary=(
            "QA automation engineer with 4 years of experience building end-to-end and API "
            "test suites and wiring them into CI."
        ),
        skills=(
            ("Automation", "Selenium, Playwright, Cypress"),
            ("API testing", "Postman, REST Assured"),
            ("CI & Tools", "Jenkins, Git, JIRA, TestRail"),
            ("Practices", "Test planning, Risk-based testing"),
        ),
        jobs=(
            Job(
                "QA Automation Engineer",
                "Liffey Software",
                "Dublin",
                "2024 - Present",
                (
                    "Built an end-to-end regression suite in Python with pytest and Playwright, "
                    "reducing release testing from 3 days to 4 hours.",
                    "Integrated automated tests into Jenkins pipelines with parallel execution.",
                ),
            ),
            Job(
                "QA Engineer",
                "Emerald Insurance",
                "Cork",
                "2022 - 2024",
                (
                    "Wrote Cypress tests for the customer policy portal.",
                    "Defined test plans and tracked defects in JIRA and TestRail.",
                ),
            ),
        ),
        education=("B.Sc. Software Development, Leinster Institute of Technology, 2022",),
        python_placement="One experience bullet only",
    ),
    Resume(
        first="Sofia",
        last="Rossi",
        fmt="txt",
        title="Data Analyst (SQL / R)",
        location="Milan, Italy",
        phone="+39 555-0107",
        years=2,
        summary=(
            "Data analyst with 2 years of experience turning sales and marketing data into "
            "dashboards and recommendations."
        ),
        skills=(
            ("Languages", "SQL, R"),
            ("R packages", "tidyverse, ggplot2, dplyr"),
            ("BI & Tools", "BigQuery, Looker, Power BI, Excel"),
        ),
        jobs=(
            Job(
                "Data Analyst",
                "Navigli Commerce",
                "Milan",
                "2025 - Present",
                (
                    "Built SQL models in BigQuery feeding weekly sales dashboards in Looker.",
                    "Ran cohort analyses in R that informed a loyalty programme redesign.",
                ),
            ),
            Job(
                "Junior Analyst",
                "Duomo Marketing",
                "Milan",
                "2024 - 2025",
                (
                    "Cleaned and merged campaign data from 6 advertising platforms.",
                    "Produced monthly ggplot2 reports for clients.",
                ),
            ),
        ),
        education=("B.Sc. Statistics and Economics, Universita di Lombardia, 2024",),
    ),
    Resume(
        first="Kenji",
        last="Tanaka",
        fmt="txt",
        title="Staff Machine Learning Engineer",
        location="Tokyo, Japan",
        phone="+81 555-0108",
        years=10,
        summary=(
            "ML engineer with 10 years of experience taking computer-vision and ranking models "
            "from research to low-latency production on Kubernetes."
        ),
        skills=(
            ("Languages", "python, C++, SQL"),
            ("ML", "PyTorch, TensorFlow, ONNX, CUDA"),
            ("MLOps", "Kubernetes, Kubeflow, MLflow, Triton Inference Server"),
            ("Data", "Apache Spark, Parquet"),
        ),
        jobs=(
            Job(
                "Staff ML Engineer",
                "Sakura Vision",
                "Tokyo",
                "2022 - Present",
                (
                    "Lead a team of 6 deploying real-time image models on Kubernetes with "
                    "Triton Inference Server.",
                    "Cut GPU serving cost by 35% through ONNX export and INT8 quantisation.",
                ),
            ),
            Job(
                "ML Engineer",
                "Shibuya Ads",
                "Tokyo",
                "2018 - 2022",
                (
                    "Built a CTR-prediction training pipeline on Spark and Kubeflow.",
                    "Wrote internal python libraries for feature validation.",
                ),
            ),
            Job(
                "Software Engineer",
                "Kanda Systems",
                "Yokohama",
                "2016 - 2018",
                ("Developed C++ image-processing modules for factory inspection cameras.",),
            ),
        ),
        education=("M.Eng. Information Science, Kanto Institute of Technology, 2016",),
        python_placement="Lowercase 'python' only (skills + bullet)",
    ),
)

# Short resume in cp1252: e-acute, right single quote and em dash are not valid UTF-8 here.
CP1252_RESUME = (
    "Ren\u00e9 Lef\u00e8vre \u2014 Product Manager\n"
    "Lyon, France | rene.lefevre@example.com | +33 555-0109\n"
    "\n"
    "SUMMARY\n"
    "Product manager with 4 years\u2019 experience in B2B SaaS and caf\u00e9 POS systems.\n"
    "\n"
    "SKILLS\n"
    "Roadmapping, user research, SQL, Figma\n"
    "\n"
    "EXPERIENCE\n"
    "Product Manager \u2014 Rh\u00f4ne Software (2022 - Present)\n"
    "  - Launched a self-serve onboarding flow that cut time-to-value by 30%.\n"
)


def render_text(resume: Resume) -> str:
    """Plain-text rendering, used for TXT resumes and for the manifest's content checks."""
    lines = [resume.name.upper(), resume.title, resume.contact, "", "SUMMARY", resume.summary]
    lines += ["", "SKILLS", *(f"{cat}: {items}" for cat, items in resume.skills), ""]
    lines.append("EXPERIENCE")
    for job in resume.jobs:
        lines.append(f"{job.title} - {job.company}, {job.location} ({job.dates})")
        lines += [f"  - {bullet}" for bullet in job.bullets]
        lines.append("")
    for section in resume.sections:
        lines += [section.heading.upper(), *(f"- {item}" for item in section.items), ""]
    return "\n".join(lines).rstrip() + "\n"


def write_txt(resume: Resume, path: Path) -> None:
    path.write_text(render_text(resume), encoding="utf-8", newline="\n")


def write_pdf(resume: Resume, path: Path) -> None:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    heading = styles["Heading2"]
    bullet = ParagraphStyle("ResumeBullet", parent=body, leftIndent=14, bulletIndent=4)

    story = [
        Paragraph(escape(resume.name), styles["Title"]),
        Paragraph(escape(resume.title), styles["Heading3"]),
        Paragraph(escape(resume.contact), body),
        Paragraph("Summary", heading),
        Paragraph(escape(resume.summary), body),
        Paragraph("Skills", heading),
    ]
    story += [
        Paragraph(f"<b>{escape(cat)}:</b> {escape(items)}", bullet, bulletText="\u2022")
        for cat, items in resume.skills
    ]
    story.append(Paragraph("Experience", heading))
    for job in resume.jobs:
        header = f"<b>{escape(job.title)} - {escape(job.company)}</b>, {escape(job.location)}"
        story.append(Paragraph(f"{header} ({escape(job.dates)})", body))
        story += [Paragraph(escape(b), bullet, bulletText="\u2022") for b in job.bullets]
    if resume.page_break_before_extras:
        story.append(PageBreak())
    for section in resume.sections:
        story.append(Paragraph(escape(section.heading), heading))
        story += [Paragraph(escape(i), bullet, bulletText="\u2022") for i in section.items]

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Resume - {resume.name}",
        author=resume.name,
        invariant=1,  # fixed creation date and document ID
    )
    doc.build(story)


def write_reproducible_zip(data: bytes, path: Path) -> None:
    """Re-pack a zip with fixed entry timestamps so the bytes don't depend on wall-clock time."""
    with (
        zipfile.ZipFile(io.BytesIO(data)) as src,
        zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            entry = zipfile.ZipInfo(info.filename, date_time=FIXED_ZIP_TIME)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            dst.writestr(entry, src.read(info.filename))


def write_docx(resume: Resume, path: Path) -> None:
    doc = Document()
    doc.add_heading(resume.name, level=0)
    doc.add_paragraph().add_run(resume.title).bold = True
    doc.add_paragraph(resume.contact)

    doc.add_heading("Summary", level=1)
    doc.add_paragraph(resume.summary)

    doc.add_heading("Skills", level=1)
    if resume.skills_as_table:
        table = doc.add_table(rows=0, cols=2)
        table.style = "Table Grid"
        for cat, items in resume.skills:
            cells = table.add_row().cells
            cells[0].text = cat
            cells[1].text = items
    else:
        for cat, items in resume.skills:
            doc.add_paragraph(f"{cat}: {items}", style="List Bullet")

    doc.add_heading("Experience", level=1)
    for job in resume.jobs:
        para = doc.add_paragraph()
        para.add_run(f"{job.title} - {job.company}").bold = True
        para.add_run(f", {job.location} ({job.dates})")
        for bullet in job.bullets:
            doc.add_paragraph(bullet, style="List Bullet")

    for section in resume.sections:
        doc.add_heading(section.heading, level=1)
        for item in section.items:
            doc.add_paragraph(item, style="List Bullet")

    props = doc.core_properties
    props.title = f"Resume - {resume.name}"
    props.author = resume.name
    props.last_modified_by = "generate_samples.py"
    props.revision = 1
    props.created = FIXED_TIMESTAMP
    props.modified = FIXED_TIMESTAMP
    props.last_printed = FIXED_TIMESTAMP

    buffer = io.BytesIO()
    doc.save(buffer)
    write_reproducible_zip(buffer.getvalue(), path)


def write_shapes_pdf(path: Path) -> None:
    """A valid one-page PDF with vector shapes only - no text layer."""
    pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    pdf.setFillColor(HexColor("#4A6FA5"))
    pdf.rect(60, 650, 200, 120, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#E07A5F"))
    pdf.circle(400, 700, 60, fill=1, stroke=0)
    pdf.setStrokeColor(HexColor("#3D405B"))
    pdf.setLineWidth(3)
    for offset in range(0, 300, 30):
        pdf.line(60, 550 - offset, 535, 550 - offset)
    pdf.showPage()
    pdf.save()


def write_edge_cases() -> None:
    garbage = bytes((i * 37 + 11) % 256 for i in range(512))
    (EDGE_DIR / "corrupt.pdf").write_bytes(b"%PDF-1.7\n" + garbage)
    (EDGE_DIR / "empty.txt").write_bytes(b"")
    write_shapes_pdf(EDGE_DIR / "no_text_layer.pdf")
    (EDGE_DIR / "cp1252_resume.txt").write_bytes(CP1252_RESUME.encode("cp1252"))
    (EDGE_DIR / "unsupported.xlsx").write_bytes(b"PK\x03\x04" + bytes(range(64)))


EDGE_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "corrupt.pdf",
        "`%PDF-1.7` header followed by 512 garbage bytes",
        "`read_file` / `search_in_file` fail with `EXTRACTION_FAILED`",
    ),
    ("empty.txt", "Zero bytes", "Success with empty `content`"),
    (
        "no_text_layer.pdf",
        "Valid 1-page PDF containing only drawn shapes",
        "Success; `content` empty/whitespace, `pages` = 1",
    ),
    (
        "cp1252_resume.txt",
        "Short resume encoded as cp1252 (\u00e9, \u2019, \u2014) - invalid UTF-8",
        "Success via encoding fallback; metadata `encoding` = `cp1252`",
    ),
    (
        "unsupported.xlsx",
        "68 raw bytes with an unsupported extension",
        "Fails with `UNSUPPORTED_FORMAT`",
    ),
)


def write_manifest() -> None:
    def mentions(resume: Resume, term: str) -> bool:
        return term in render_text(resume).casefold()

    def names(predicate) -> str:
        return ", ".join(r.name for r in RESUMES if predicate(r))

    lines = [
        "# Sample data manifest",
        "",
        "Answer key for demos and tests. Generated by `scripts/generate_samples.py` - edit the",
        "script, not this file. All people, companies and contact details are fictional.",
        "",
        "## Resumes (`resumes/`)",
        "",
        "| File | Format | Name | Role | Years | Location | Key skills | Python? "
        "| Python placement |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in RESUMES:
        skills = [s for _, items in r.skills for s in items.split(", ")][:6]
        lines.append(
            f"| `{r.filename}` | {r.fmt.upper()} | {r.name} | {r.title} | {r.years} "
            f"| {r.location} | {', '.join(skills)} | {'Yes' if mentions(r, 'python') else 'No'} "
            f"| {r.python_placement} |"
        )
    lines += [
        "",
        "Notes: `resume_john_doe.pdf` spans 2 pages (Projects, Certifications and Education are",
        "on page 2). `resume_elena_rodriguez.docx` lists its skills in a table.",
        "",
        "### Quick answers",
        "",
        f"- **Mention Python (case-insensitive):** {names(lambda r: mentions(r, 'python'))}",
        f"- **Do not mention Python:** {names(lambda r: not mentions(r, 'python'))}",
        f"- **5+ years of experience:** {names(lambda r: r.years >= 5)}",
        f"- **Mention Kubernetes:** {names(lambda r: mentions(r, 'kubernetes'))}",
        "",
        "## Edge cases (`edge_cases/`)",
        "",
        "| File | Contents | Expected tool behaviour |",
        "|---|---|---|",
        *(f"| `{name}` | {contents} | {expected} |" for name, contents, expected in EDGE_CASES),
        "",
    ]
    (SAMPLE_DIR / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    writers = {"pdf": write_pdf, "docx": write_docx, "txt": write_txt}
    for resume in RESUMES:
        path = RESUME_DIR / resume.filename
        writers[resume.fmt](resume, path)
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    write_edge_cases()
    print(f"wrote {len(EDGE_CASES)} files in {EDGE_DIR.relative_to(ROOT).as_posix()}/")
    write_manifest()
    print("wrote sample_data/MANIFEST.md")


if __name__ == "__main__":
    main()
