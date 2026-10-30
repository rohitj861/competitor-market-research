"""Render the Markdown research report as a clean, print-ready PDF (fpdf2).

The Markdown is parsed into blocks (headings, paragraphs, lists, tables, code,
quotes, references) and drawn with fpdf2's layout primitives, which gives much
tighter control than fpdf2's generic HTML renderer.
"""

import logging
import os
import re
from datetime import date

from fpdf import FPDF, FontFace

# fontTools warns about tables it can't subset in some Windows fonts (harmless).
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

WIN_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
LINUX_FONTS = "/usr/share/fonts/truetype"  # Debian packages installed by the Dockerfile
# Unicode TTF fonts (₹, €, smart quotes, …): the first candidate whose files all exist is
# used (Windows, then Linux/Docker). Without one, core fonts are used (Latin-1 only).
BODY_TTF = [
    {s: os.path.join(WIN_FONTS, f) for s, f in
     {"": "arial.ttf", "B": "arialbd.ttf", "I": "ariali.ttf", "BI": "arialbi.ttf"}.items()},
    {s: os.path.join(LINUX_FONTS, "dejavu", f) for s, f in
     {"": "DejaVuSans.ttf", "B": "DejaVuSans-Bold.ttf", "I": "DejaVuSans-Oblique.ttf",
      "BI": "DejaVuSans-BoldOblique.ttf"}.items()},
]
MONO_TTF = [
    {"": os.path.join(WIN_FONTS, "consola.ttf")},
    {"": os.path.join(LINUX_FONTS, "dejavu", "DejaVuSansMono.ttf")},
]
# Fallbacks for glyphs the body font lacks (symbols/emoji, Indic scripts, Chinese, Korean).
FALLBACK_TTF = {
    "Sym": os.path.join(WIN_FONTS, "seguisym.ttf"),
    "Indic": os.path.join(WIN_FONTS, "Nirmala.ttc"),
    "Cjk": os.path.join(WIN_FONTS, "msyh.ttc"),
    "Kor": os.path.join(WIN_FONTS, "malgun.ttf"),
    "IndicLinux": os.path.join(LINUX_FONTS, "lohit-devanagari", "Lohit-Devanagari.ttf"),
    "CjkLinux": os.path.join(LINUX_FONTS, "wqy", "wqy-microhei.ttc"),
}

# Palette & sizes
INK = (33, 37, 41)
MUTED = (108, 117, 125)
ACCENT = (13, 71, 161)
RULE = (206, 212, 218)
HEADER_FILL = (13, 71, 161)
ZEBRA_FILL = (244, 247, 251)
CODE_FILL = (246, 248, 250)

BODY_SIZE = 10
LINE_H = 5.2  # mm per body line
HEADING_SIZES = {1: 18, 2: 14, 3: 12, 4: 11, 5: 10, 6: 10}


# ---------------------------------------------------------------- inline markup
_URL = re.compile(r"https?://[^\s)\]]+")


def _inline(text: str) -> str:
    """Convert common Markdown inline syntax to fpdf2's markdown dialect.

    fpdf2 understands **bold**, __italic__ and --underline--. Links are written as plain
    text, not clickable annotations: antivirus tools (e.g. McAfee "Phishing.JD") flag
    PDFs full of outbound links.
    """
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # images -> alt text
    # [text](url) -> "text (url)", or just the url when they are the same
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
                  lambda m: m.group(2) if m.group(1) == m.group(2) else f"{m.group(1)} ({m.group(2)})", text)
    # set URLs aside so the rewrites below can't mangle them
    urls: list[str] = []
    text = _URL.sub(lambda m: urls.append(m.group(0)) or f"\x00{len(urls) - 1}\x00", text)
    text = text.replace("--", "–")  # avoid accidental fpdf2 underline markers
    text = re.sub(r"`([^`]+)`", r"\1", text)  # inline code -> plain
    # italics: *x* or _x_ (but not **x** / __x__)
    text = re.sub(r"(?<![*\w])\*(?!\*)([^*\n]+?)\*(?!\*)", r"__\1__", text)
    text = re.sub(r"(?<![_\w])_(?!_)([^_\n]+?)_(?![_\w])", r"__\1__", text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"**__\1__**", text)
    # restore URLs, escaping anything fpdf2 would read as a style marker
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: re.sub(r"(\*\*|__|--|~~)", r"\\\1", urls[int(m.group(1))]), text)


def _plain(text: str) -> str:
    """Strip Markdown markup (used for width estimates)."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[*_`]", "", text)


# ---------------------------------------------------------------- block parsing
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_UL = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OL = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_REF = re.compile(r"^\s*\[\d+\]\s+")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


def parse_blocks(md: str) -> list[tuple]:
    lines = md.replace("\r\n", "\n").split("\n")
    blocks: list[tuple] = []
    para: list[str] = []
    i = 0

    def flush_para():
        if para:
            blocks.append(("p", " ".join(s.strip() for s in para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_para()
            i += 1
        elif stripped.startswith("```"):
            flush_para()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code)))
            i += 1
        elif m := re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", stripped):
            flush_para()
            blocks.append(("h", len(m.group(1)), m.group(2)))
            i += 1
        elif "|" in stripped and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush_para()
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            width = len(rows[0])
            rows = [(r + [""] * width)[:width] for r in rows]
            blocks.append(("table", rows))
        elif _HR.match(stripped):
            flush_para()
            blocks.append(("hr",))
            i += 1
        elif stripped.startswith(">"):
            flush_para()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(("quote", " ".join(quote)))
        elif _UL.match(line) or _OL.match(line):
            flush_para()
            ordered = bool(_OL.match(line))
            items: list[list] = []  # [level, marker, text]
            while i < len(lines):
                cur = lines[i]
                if mu := _UL.match(cur):
                    items.append([len(mu.group(1).expandtabs(4)) // 2, "•", mu.group(2)])
                elif mo := _OL.match(cur):
                    items.append([len(mo.group(1).expandtabs(4)) // 2, f"{mo.group(2)}.", mo.group(3)])
                elif cur.strip() and cur.startswith((" ", "\t")) and items:
                    items[-1][2] += " " + cur.strip()  # wrapped continuation line
                else:
                    break
                i += 1
            blocks.append(("list", ordered, [(min(lvl, 3), mk, txt) for lvl, mk, txt in items]))
        elif _REF.match(stripped):
            flush_para()
            blocks.append(("ref", stripped))
            i += 1
        else:
            para.append(line)
            if line.endswith("  "):  # Markdown hard line break
                flush_para()
            i += 1
    flush_para()
    return blocks


# ---------------------------------------------------------------- PDF document
class ReportPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.report_title = title
        self.body_font, self.mono_font = self._register_fonts()
        self.set_margins(18, 20, 18)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_title(title)
        self.alias_nb_pages()

    def _register_fonts(self) -> tuple[str, str]:
        def register(family: str, candidates: list[dict[str, str]]) -> bool:
            for paths in candidates:
                if all(os.path.exists(p) for p in paths.values()):
                    for style, path in paths.items():
                        self.add_font(family, style, path)
                    return True
            return False

        body = "Body" if register("Body", BODY_TTF) else "helvetica"
        mono = "Mono" if register("Mono", MONO_TTF) else "courier"
        self.unicode = body == "Body"
        if self.unicode:
            fallbacks = [name for name, f in FALLBACK_TTF.items() if register(name, [{"": f}])]
            self.set_fallback_fonts(fallbacks, exact_match=False)
            # Characters no font can draw are dropped: fpdf2 still measures them, which
            # shifts link underlines and table layout.
            self.glyphs = set().union(*(f.cmap for f in self.fonts.values() if hasattr(f, "cmap")))
        return body, mono

    def safe(self, text: str) -> str:
        if not self.unicode:
            return text.encode("latin-1", "replace").decode("latin-1")
        return "".join(ch for ch in text if ch == "\n" or ord(ch) in self.glyphs)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(self.body_font, "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, self.safe(self.report_title), align="L")
        self.ln(4)
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-12)
        self.set_font(self.body_font, "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, self.safe(f"Generated {date.today():%d %b %Y}"), align="L")
        self.set_x(self.l_margin)
        self.cell(0, 5, f"Page {self.page_no()} of {{nb}}", align="R")

    # -- blocks --------------------------------------------------------------
    def text_block(self, text: str, indent: float = 0, size: float = BODY_SIZE,
                   style: str = "", color=INK, after: float = 2.5):
        self.set_font(self.body_font, style, size)
        self.set_text_color(*color)
        self.set_x(self.l_margin + indent)
        self.multi_cell(self.epw - indent, LINE_H * size / BODY_SIZE, self.safe(_inline(text)),
                        markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(after)

    def heading(self, level: int, text: str, first: bool):
        size = HEADING_SIZES[level]
        if not first:
            self.ln(5 if level <= 2 else 3)
        # keep a heading with at least a few lines of what follows
        if self.get_y() + size * 0.6 + 4 * LINE_H > self.page_break_trigger:
            self.add_page()
        self.set_font(self.body_font, "B", size)
        self.set_text_color(*(ACCENT if level <= 3 else INK))
        self.multi_cell(0, size * 0.5, self.safe(_plain(text)), align="L",
                        new_x="LMARGIN", new_y="NEXT")
        if level <= 2:
            self.ln(1)
            self.set_draw_color(*(ACCENT if level == 1 else RULE))
            self.set_line_width(0.6 if level == 1 else 0.3)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.set_line_width(0.2)
        self.ln(3)

    def bullet_list(self, items):
        self.set_font(self.body_font, "", BODY_SIZE)
        for level, marker, text in items:
            indent = 4 + level * 6
            marker_w = 7 if marker != "•" else 4.5
            self.set_text_color(*(ACCENT if marker != "•" else MUTED))
            self.set_font(self.body_font, "B" if marker != "•" else "", BODY_SIZE)
            self.set_x(self.l_margin + indent)
            self.cell(marker_w, LINE_H, self.safe(marker))
            self.set_font(self.body_font, "", BODY_SIZE)
            self.set_text_color(*INK)
            self.multi_cell(self.epw - indent - marker_w, LINE_H, self.safe(_inline(text)),
                            markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
            self.ln(1.2)
        self.ln(1.5)

    def table_block(self, rows: list[list[str]]):
        ncols = len(rows[0])
        font_size = 9 if ncols <= 4 else 8
        # Each column gets room for its longest word (no mid-word breaks); the remaining
        # width is shared in proportion to how much text the column holds.
        self.set_font(self.body_font, "B", font_size)
        min_w, weights = [], []
        for c in range(ncols):
            cells = [self.safe(_plain(r[c])) for r in rows]
            words = [w for cell in cells for w in cell.split()] or [""]
            min_w.append(min(max(self.get_string_width(w) for w in words) + 4.5, self.epw / 2))
            weights.append(max(sum(len(t) for t in cells[1:]) / max(len(cells) - 1, 1), 4))
        spare = self.epw - sum(min_w)
        if spare > 0:
            col_widths = [m + spare * w / sum(weights) for m, w in zip(min_w, weights)]
        else:
            col_widths = [m * self.epw / sum(min_w) for m in min_w]

        self.set_font(self.body_font, "", font_size)
        self.set_text_color(*INK)
        self.set_draw_color(*RULE)
        with self.table(
            col_widths=col_widths,
            width=self.epw,
            text_align="LEFT",
            v_align="TOP",
            line_height=font_size * 0.48,
            padding=(1.8, 2),
            markdown=True,
            borders_layout="HORIZONTAL_LINES",
            cell_fill_mode="ROWS",
            cell_fill_color=ZEBRA_FILL,
            headings_style=FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=HEADER_FILL),
            repeat_headings=1,
        ) as table:
            for r, row in enumerate(rows):
                tr = table.row()
                for c, cell in enumerate(row):
                    text = _plain(cell) if r == 0 else _inline(cell)
                    if r > 0 and c == 0 and not cell.startswith("**"):
                        text = f"**{text}**"  # emphasise the row label
                    tr.cell(self.safe(text))
        self.ln(4)

    def code_block(self, code: str):
        self.set_font(self.mono_font, "", 8.5)
        self.set_fill_color(*CODE_FILL)
        self.set_text_color(*INK)
        self.multi_cell(0, 4.4, self.safe(code), fill=True, border=0, padding=2.5,
                        new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def quote_block(self, text: str):
        y0 = self.get_y()
        self.text_block(text, indent=5, style="I", color=MUTED, after=0)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.8)
        self.line(self.l_margin + 1.5, y0, self.l_margin + 1.5, self.get_y())
        self.set_line_width(0.2)
        self.ln(3)

    def reference(self, text: str):
        m = re.match(r"^\s*(\[\d+\])\s+(.*)$", text)
        num, body = m.group(1), m.group(2)
        self.set_font(self.body_font, "B", 8.5)
        self.set_text_color(*ACCENT)
        self.cell(9, 4.4, num)
        self.set_font(self.body_font, "", 8.5)
        self.set_text_color(*INK)
        self.multi_cell(self.epw - 9, 4.4, self.safe(_inline(body)), markdown=True, align="L",
                        new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def hr(self):
        self.ln(2)
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)


def markdown_to_pdf(md_text: str, title: str = "Market Research Report") -> bytes:
    pdf = ReportPDF(title)
    pdf.add_page()

    blocks = parse_blocks(md_text)
    for n, block in enumerate(blocks):
        kind = block[0]
        if kind == "h":
            pdf.heading(block[1], block[2], first=n == 0)
        elif kind == "p":
            pdf.text_block(block[1])
        elif kind == "list":
            pdf.bullet_list(block[2])
        elif kind == "table":
            pdf.table_block(block[1])
        elif kind == "code":
            pdf.code_block(block[1])
        elif kind == "quote":
            pdf.quote_block(block[1])
        elif kind == "ref":
            pdf.reference(block[1])
        elif kind == "hr":
            pdf.hr()
    return bytes(pdf.output())
