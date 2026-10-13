"""Full-meeting Race Edge Prediction Worksheet; informational claims only."""
from io import BytesIO
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm

NAVY = colors.HexColor('#10213B')
GOLD = colors.HexColor('#C7A35A')
PALE = colors.HexColor('#F1F3F6')
METHODS = [('Recent Best', 'Recent Best Projection'), ('Latest Form', 'Latest Projection'), ('Peak Ability', 'Peak Projection')]


def _number(v):
    try:
        return str(int(float(v))) if pd.notna(v) else '-'
    except (ValueError, TypeError):
        return '-'


def _claim(v):
    try:
        n = float(v)
        return f' (a{n:g})' if n > 0 else ''
    except (ValueError, TypeError):
        return ''


def build_worksheet_pdf(meeting, options, card_loader, runner_frame, db_counts_loader,
                        prediction_builder, history_loader):
    """Build all races in a single PDF; always calculate with claims OFF."""
    counts = db_counts_loader()
    title_style = ParagraphStyle('worksheet-title', fontName='Helvetica-Bold', fontSize=17, leading=21, textColor=GOLD, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('worksheet-sub', fontName='Helvetica', fontSize=9, leading=13, textColor=colors.white, alignment=TA_CENTER)
    heading_style = ParagraphStyle('race-heading', fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=NAVY)
    cell_style = ParagraphStyle('cell', fontName='Helvetica', fontSize=7.2, leading=9.5)
    bold_style = ParagraphStyle('boldcell', parent=cell_style, fontName='Helvetica-Bold')
    small_style = ParagraphStyle('small', fontName='Helvetica', fontSize=7, leading=10, textColor=colors.HexColor('#4D5666'))
    usable = landscape(A4)[0] - 24*mm
    gap = 3*mm
    colwidth = (usable-2*gap)/3
    inner_widths = [12*mm, colwidth-12*mm-19*mm, 19*mm]
    content = []
    header = Table([[Paragraph('RACE EDGE', title_style)], [Paragraph('PREDICTION WORKSHEET', subtitle_style)]], colWidths=[usable])
    header.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),NAVY),('TOPPADDING',(0,0),(-1,0),9),('BOTTOMPADDING',(0,-1),(-1,-1),10)]))
    content += [header, Spacer(1,4*mm), Paragraph('Full-meeting predictions | Apprentice claims shown but NOT applied | Margins cumulative behind leader', small_style), Spacer(1,5*mm)]
    warnings = []
    for race_key, _ in options:
        try:
            card = card_loader(meeting, str(race_key))
            frame = runner_frame(card, counts)
            if frame.empty:
                warnings.append(f'Race {race_key}: no runners')
                continue
            active = frame[frame['Status'] != 'Scratched'].copy()
            distance = card.get('distance')
            prediction = prediction_builder(active, race_date=card.get('date') or card.get('dateFormat'),
                                            distance_m=float(distance), history_loader=history_loader,
                                            apply_apprentice_claims=False)
            scenarios = (prediction or {}).get('scenarios', {})
            sections = []
            for name, projection_col in METHODS:
                scenario = scenarios.get(name)
                if scenario is None:
                    scenario = pd.DataFrame()
                rows = [[Paragraph('Pred', bold_style), Paragraph('Runner', bold_style), Paragraph('Behind', bold_style)]]
                rated_horses = set()
                if not scenario.empty:
                    scenario = scenario.sort_values('Rank')
                    leader = pd.to_numeric(scenario[projection_col], errors='coerce').max()
                    for _, r in scenario.iterrows():
                        horse = str(r['Horse'])
                        rated_horses.add(horse)
                        delta = max(0., (leader - float(r[projection_col])) * .5)
                        margin = 'Leader' if delta < 0.005 else f'{delta:.1f}L'
                        rows.append([Paragraph(_number(r['Rank']), cell_style),
                                     Paragraph(escape(f"#{_number(r.get('No.'))} {horse}{_claim(r.get('Apprentice Claim'))}"), cell_style),
                                     Paragraph(margin, cell_style)])
                for _, r in active.iterrows():
                    horse = str(r['Horse'])
                    if horse in rated_horses:
                        continue
                    has_runs = int(r.get('Race Edge Runs') or 0) > 0
                    status = 'INSUFFICIENT DATA' if has_runs else 'NO DATA'
                    rows.append([Paragraph('-', cell_style),
                                 Paragraph(escape(f"#{_number(r.get('No.'))} {horse}{_claim(r.get('Claim'))}"), cell_style),
                                 Paragraph(status, small_style)])
                table = Table(rows, colWidths=inner_widths, repeatRows=1, hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('LINEBELOW',(0,0),(-1,0),.5,GOLD),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#F8F9FA')]),
                    ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
                    ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
                sections.append([Paragraph(escape(name.upper()), bold_style), Spacer(1,2*mm), table])
            # Place each method in its own cell; keep the complete race together when possible.
            columns = Table([[sections[0], sections[1], sections[2]]], colWidths=[colwidth]*3, hAlign='LEFT')
            columns.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),
                 ('RIGHTPADDING',(0,0),(-1,-1),gap),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
            heading = f"RACE {_number(card.get('race') or race_key)}  |  {_number(distance)}m  |  {card.get('track') or card.get('clubName') or ''}  |  {card.get('surfaceDescr') or card.get('surface') or ''}"
            notes = Paragraph('ANALYST NOTES: __________________________________________________________________________________________________________________________', small_style)
            content.append(KeepTogether([Paragraph(escape(heading), heading_style), Spacer(1,2*mm), columns, Spacer(1,3*mm), notes, Spacer(1,6*mm)]))
        except Exception as exc:
            warnings.append(f'Race {race_key}: {exc}')
    if warnings:
        content.append(Paragraph('Export warnings: '+escape('; '.join(warnings)), small_style))
    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(A4), leftMargin=12*mm, rightMargin=12*mm,
                            topMargin=12*mm, bottomMargin=13*mm, title='Race Edge Prediction Worksheet')
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(NAVY)
        canvas.drawCentredString(landscape(A4)[0]/2, 7*mm, f'RACE EDGE  |  PREDICTION WORKSHEET  |  Page {doc.page}')
    doc.build(content, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue(), warnings
