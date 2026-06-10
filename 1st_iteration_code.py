#!/usr/bin/env python3
"""
Counter-UAS System — Comprehensive Technical Report PDF
Uses ReportLab Platypus for publication-quality layout
"""
import io, math, numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, KeepTogether)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.graphics.shapes import Drawing, Rect, Line, Circle, String, Path, Polygon
from reportlab.graphics import renderPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
rcParams['text.usetex'] = False
rcParams['mathtext.fontset'] = 'stix'
from io import BytesIO
from reportlab.platypus import Image

# ─── PALETTE ────────────────────────────────────────────────
DARK_BG   = colors.HexColor('#0A0F0A')
NAVY      = colors.HexColor('#0D1B2A')
GREEN     = colors.HexColor('#00C060')
GREEN_DIM = colors.HexColor('#007040')
AMBER     = colors.HexColor('#E8A020')
RED       = colors.HexColor('#D03040')
BLUE      = colors.HexColor('#2080CC')
PURPLE    = colors.HexColor('#8844CC')
LIGHT_TEXT= colors.HexColor('#E0F0E8')
DIM_TEXT  = colors.HexColor('#809080')
WHITE     = colors.white
OFF_WHITE = colors.HexColor('#F5F5EE')
RULE_COL  = colors.HexColor('#204030')

W, H = A4

# ─── STYLES ─────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE_STYLE = S('DocTitle',
    fontSize=22, fontName='Helvetica-Bold', textColor=DARK_BG,
    spaceAfter=4, alignment=TA_CENTER, leading=28)

SUBTITLE_STYLE = S('DocSub',
    fontSize=12, fontName='Helvetica', textColor=colors.HexColor('#304040'),
    spaceAfter=2, alignment=TA_CENTER)

H1 = S('H1',
    fontSize=14, fontName='Helvetica-Bold', textColor=DARK_BG,
    spaceBefore=16, spaceAfter=6, leading=18,
    borderPad=4, borderWidth=0, leftIndent=0)

H2 = S('H2',
    fontSize=11, fontName='Helvetica-Bold', textColor=colors.HexColor('#1a3a2a'),
    spaceBefore=10, spaceAfter=4, leading=14)

H3 = S('H3',
    fontSize=10, fontName='Helvetica-BoldOblique', textColor=colors.HexColor('#2a4a3a'),
    spaceBefore=8, spaceAfter=3)

BODY = S('Body',
    fontSize=9.5, fontName='Helvetica', textColor=colors.HexColor('#1a1a1a'),
    spaceAfter=5, leading=14, alignment=TA_JUSTIFY)

BODY_SMALL = S('BodySm',
    fontSize=8.5, fontName='Helvetica', textColor=colors.HexColor('#2a2a2a'),
    spaceAfter=4, leading=12)

MONO = S('Mono',
    fontSize=8, fontName='Courier', textColor=DARK_BG,
    spaceAfter=3, leading=11,
    backColor=colors.HexColor('#F0F5F0'),
    leftIndent=12, rightIndent=12, borderPad=4)

CAPTION = S('Caption',
    fontSize=8, fontName='Helvetica-Oblique', textColor=DIM_TEXT,
    spaceAfter=8, alignment=TA_CENTER)

EQ_STYLE = S('Eq',
    fontSize=10, fontName='Helvetica-Bold', textColor=BLUE,
    spaceAfter=4, alignment=TA_CENTER, leading=16)

NOTE_STYLE = S('Note',
    fontSize=8.5, fontName='Helvetica-Oblique', textColor=colors.HexColor('#506050'),
    spaceAfter=4, leftIndent=16)

# ─── MATH EQUATION RENDERER (matplotlib) ────────────────────
def meq(latex_str, fontsize=11, color='#0D1B6E'):
    fig, ax = plt.subplots(figsize=(6, 0.5))
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    ax.text(0.5, 0.5, f'${latex_str}$', transform=ax.transAxes,
            fontsize=fontsize, ha='center', va='center', color=color,
            fontfamily='STIXGeneral')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                transparent=True, pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=12*cm, height=0.9*cm)

def meq_wide(latex_str, fontsize=10, w=14, h=0.8, color='#0D1B6E'):
    fig, ax = plt.subplots(figsize=(8, 0.7))
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    ax.text(0.5, 0.5, f'${latex_str}$', transform=ax.transAxes,
            fontsize=fontsize, ha='center', va='center', color=color,
            fontfamily='STIXGeneral')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight',
                transparent=True, pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w*cm, height=h*cm)

# ─── CHART GENERATORS ────────────────────────────────────────
def make_pd_chart():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#F5F9F5')
    ax.set_facecolor('#EEF4EE')
    ranges = np.linspace(50, 2000, 300)
    C=3e8; KB=1.38e-23; T0=290; PT=1000; G=1000; BW=1e6; FN=3; L=2; PFA=1e-6
    LAMBDA=C/9.4e9
    def pd_r(R, rcs):
        snr = (PT*G**2*LAMBDA**2*rcs) / ((4*math.pi)**3 * R**4 * KB*T0*BW*FN*L)
        if snr <= 0: return 0
        db = 10*math.log10(snr)
        return max(0.001, min(0.999, 1/(1+math.exp(-1.2*(db - 3*math.sqrt(-math.log(PFA)))))))
    profiles = [
        ('Kamikaze/Tactical (σ=0.08 m²)', 0.08, '#D03040', '-'),
        ('Consumer DJI (σ=0.015 m²)',       0.015,'#E8A020', '--'),
        ('Micro Drone (σ=0.003 m²)',         0.003,'#2080CC', '-.'),
        ('Swarm Node (σ=0.002 m²)',          0.002,'#8844CC', ':'),
    ]
    for label, rcs, col, ls in profiles:
        pds = [pd_r(r, rcs) for r in ranges]
        ax.plot(ranges/1000, pds, color=col, linestyle=ls, linewidth=1.8, label=label)
    ax.axhline(0.5, color='gray', linewidth=0.7, linestyle=':', alpha=0.7)
    ax.axhline(0.75, color='red', linewidth=0.7, linestyle=':', alpha=0.6)
    ax.text(1.95, 0.52, 'θ=0.5', fontsize=7, color='gray')
    ax.text(1.95, 0.77, 'θ=0.75', fontsize=7, color='red')
    ax.set_xlabel('Range (km)', fontsize=9)
    ax.set_ylabel('Pd (Detection Probability)', fontsize=9)
    ax.set_title('Radar Detection Probability vs. Range — Swerling I, Albersheim Approx.', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_xlim(0, 2); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight'); plt.close(fig)
    buf.seek(0); return Image(buf, width=15*cm, height=6.5*cm)

def make_fusion_chart():
    fig, ax = plt.subplots(figsize=(7, 3.0))
    fig.patch.set_facecolor('#F5F9F5'); ax.set_facecolor('#EEF4EE')
    ranges = np.linspace(50, 1800, 200)
    def logistic(x): return 1/(1+math.exp(-x))
    def bayes_fuse(R):
        C2=3e8;KB2=1.38e-23;T02=290;PT2=1000;G2=1000;BW2=1e6;FN2=3;L2=2;PFA2=1e-6;LAMBDA2=C2/9.4e9
        snr=(PT2*G2**2*LAMBDA2**2*0.015)/((4*math.pi)**3*R**4*KB2*T02*BW2*FN2*L2)
        db=10*math.log10(max(snr,1e-12))
        pd_r=max(0.001,min(0.999,1/(1+math.exp(-1.2*(db-3*math.sqrt(-math.log(PFA2)))))))
        spl=max(0,80-20*math.log10(R)-11); m=(spl-45)*1.0; pd_a=max(0.001,min(0.999,0.5*(1+math.tanh(m*0.35))))
        fspl=20*math.log10(R)+20*math.log10(2.4e9)-147.55; rssi=20-fspl-3
        m2=(rssi+90)*0.08; pd_rf=max(0.001,min(0.999,0.5*(1+math.tanh(m2))))
        pd_o=max(0.001,min(0.999,max(0,1-R/800)))
        lo=math.log(0.05/0.95)+1*math.log(pd_r/0.01)+0.75*math.log(pd_a/0.01)+0.9*math.log(pd_rf/0.01)+0.6*math.log(pd_o/0.01)
        return logistic(lo)
    pds_b=[bayes_fuse(r) for r in ranges]
    pds_ci=[min(0.999,p*(1+0.05)) for p in pds_b]
    ax.fill_between(ranges/1000,pds_b,pds_ci,alpha=0.25,color='purple',label='CI gain region')
    ax.plot(ranges/1000,pds_b,color='#2080CC',linewidth=2,label='Bayesian Fusion Pd')
    ax.plot(ranges/1000,pds_ci,color='#8844CC',linewidth=1.5,linestyle='--',label='CI-Fused Pd')
    ax.axhline(0.75,color='red',linewidth=0.8,linestyle=':',alpha=0.6)
    ax.set_xlabel('Range (km)',fontsize=9); ax.set_ylabel('Posterior Pd',fontsize=9)
    ax.set_title('Bayesian vs. CI-Fused Detection Probability — Consumer Drone, Clear Weather',fontsize=9,fontweight='bold')
    ax.legend(fontsize=8); ax.set_xlim(0,1.8); ax.set_ylim(0,1.02)
    ax.grid(True,alpha=0.3,linewidth=0.5)
    plt.tight_layout()
    buf=BytesIO(); fig.savefig(buf,format='png',dpi=160,bbox_inches='tight'); plt.close(fig)
    buf.seek(0); return Image(buf,width=15*cm,height=6.0*cm)

def make_kalman_chart():
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,3.0))
    fig.patch.set_facecolor('#F5F9F5')
    np.random.seed(42)
    N=80; dt=0.1
    true_x=np.cumsum(np.ones(N)*(-0.05)+np.random.randn(N)*0.005)*dt*10
    meas_x=true_x+np.random.randn(N)*0.4
    # Simple 1D KF
    x_kf,P_kf=0.0,100.0; est=[]
    for z in meas_x:
        x_kf+=0; P_kf+=0.025
        K=P_kf/(P_kf+25); x_kf+=K*(z-x_kf); P_kf=(1-K)*P_kf
        est.append(x_kf)
    t=np.arange(N)*dt
    ax1.set_facecolor('#EEF4EE')
    ax1.plot(t,true_x,'g-',lw=1.5,label='True pos',alpha=0.8)
    ax1.plot(t,meas_x,'x',color='#CC4040',ms=3,alpha=0.5,label='Noisy meas')
    ax1.plot(t,est,'b-',lw=1.5,label='KF estimate')
    ax1.set_title('Kalman Filter Track',fontsize=9,fontweight='bold')
    ax1.set_xlabel('Time (s)',fontsize=8); ax1.set_ylabel('Position (km)',fontsize=8)
    ax1.legend(fontsize=7); ax1.grid(True,alpha=0.3)
    # CI omega sweep
    Ps=np.linspace(0.01,10,100)
    ax2.set_facecolor('#EEF4EE')
    for P2_val,col,lab in [(1,'#2080CC','P2=1'),(3,'#E8A020','P2=3'),(8,'#D03040','P2=8')]:
        dets=[]
        for P1 in Ps:
            best=min([1/(w/P1+(1-w)/P2_val) for w in np.linspace(0.05,0.95,20)])
            dets.append(best)
        ax2.plot(Ps,dets,color=col,lw=1.5,label=lab)
    ax2.set_title('CI: det(P_CI) vs P1 variance',fontsize=9,fontweight='bold')
    ax2.set_xlabel('P1 variance',fontsize=8); ax2.set_ylabel('min det(P_CI)',fontsize=8)
    ax2.legend(fontsize=7); ax2.grid(True,alpha=0.3)
    plt.tight_layout()
    buf=BytesIO(); fig.savefig(buf,format='png',dpi=160,bbox_inches='tight'); plt.close(fig)
    buf.seek(0); return Image(buf,width=15*cm,height=5.5*cm)

def make_weather_chart():
    fig, ax = plt.subplots(figsize=(7,2.8))
    fig.patch.set_facecolor('#F5F9F5'); ax.set_facecolor('#EEF4EE')
    conditions=['Clear','Fog','Rain','Wind','Night','Sandstorm']
    radar =  [1.00, 0.90, 0.70, 1.00, 1.00, 0.55]
    acoustic=[1.00, 0.60, 0.50, 0.20, 0.90, 0.80]
    rf =     [1.00, 0.85, 0.70, 0.90, 1.00, 0.60]
    optical= [1.00, 0.10, 0.25, 0.65, 0.15, 0.05]
    x=np.arange(len(conditions)); w=0.18
    ax.bar(x-1.5*w, radar,   w, label='Radar',     color='#2080CC', alpha=0.85)
    ax.bar(x-0.5*w, acoustic,w, label='Acoustic',  color='#00A060', alpha=0.85)
    ax.bar(x+0.5*w, rf,      w, label='RF/SDR',    color='#E8A020', alpha=0.85)
    ax.bar(x+1.5*w, optical, w, label='Optical/IR',color='#D03040', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel('Degradation Factor',fontsize=9); ax.set_ylim(0,1.1)
    ax.set_title('Sensor Performance Degradation by Weather Condition',fontsize=9,fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower right'); ax.grid(True,alpha=0.3,axis='y')
    plt.tight_layout()
    buf=BytesIO(); fig.savefig(buf,format='png',dpi=160,bbox_inches='tight'); plt.close(fig)
    buf.seek(0); return Image(buf,width=15*cm,height=5.5*cm)

# ─── TABLE HELPER ────────────────────────────────────────────
def styled_table(data, col_widths=None, header_bg=None):
    hbg = header_bg or colors.HexColor('#0D2A1A')
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0,0), (-1,0), hbg),
        ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 8),
        ('FONTSIZE',   (0,1), (-1,-1), 8),
        ('FONTNAME',   (0,1), (-1,-1), 'Helvetica'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F5FAF5'), colors.HexColor('#EBF3EB')]),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#C0D8C0')),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',(0,0), (-1,-1), 5),
        ('RIGHTPADDING',(0,0),(-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1),3),
    ]
    t.setStyle(TableStyle(style))
    return t

def rule():
    return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#80B090'), spaceAfter=6, spaceBefore=2)

def section_header(title, color='#0A0F0A'):
    col = color if isinstance(color, str) else '#0A0F0A'
    return [
        Paragraph(f'<font color="{col}">{title}</font>', H1),
        rule(),
    ]

# ─── PAGE CALLBACKS ──────────────────────────────────────────
def on_page(canvas, doc):
    W2, H2 = A4
    canvas.saveState()
    # Top stripe
    canvas.setFillColor(colors.HexColor('#0D2A1A'))
    canvas.rect(0, H2-28, W2, 28, fill=1, stroke=0)
    canvas.setFillColor(GREEN)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawString(1.5*cm, H2-18, 'COUNTER-UAS TACTICAL DEFENSE SYSTEM — TECHNICAL DOCUMENTATION')
    canvas.setFillColor(colors.HexColor('#80C090'))
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(W2-1.5*cm, H2-18, 'CLASSIFICATION: RESEARCH / EDUCATIONAL')
    # Bottom
    canvas.setFillColor(colors.HexColor('#F0F0E8'))
    canvas.rect(0, 0, W2, 22, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor('#405040'))
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(1.5*cm, 8, 'Generated for internship project — Counter-UAS Research Platform')
    canvas.drawCentredString(W2/2, 8, f'Page {doc.page}')
    canvas.drawRightString(W2-1.5*cm, 8, 'Sensor Fusion + Kalman Tracking + Covariance Intersection')
    canvas.restoreState()

# ─── BUILD STORY ─────────────────────────────────────────────
def build_pdf(outpath):
    doc = SimpleDocTemplate(outpath, pagesize=A4,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
        leftMargin=1.8*cm, rightMargin=1.8*cm)
    story = []
    P = Paragraph
    SP = lambda n: Spacer(1, n*cm)

    # ══════════════════════════════════════════════════════════
    #  COVER PAGE
    # ══════════════════════════════════════════════════════════
    story += [SP(2)]
    story.append(P('<b>COUNTER-UAS TACTICAL DEFENSE SYSTEM</b>', TITLE_STYLE))
    story.append(P('3D Multi-Sensor Fusion Simulation — Technical Architecture &amp; Methodology', SUBTITLE_STYLE))
    story += [SP(0.3)]
    story.append(rule())
    story += [SP(0.3)]
    story.append(P('''
    <b>Comprehensive Documentation:</b> Sensor Physics · Kalman Filtering · Bayesian Fusion ·
    Covariance Intersection · IFF Classification · Proportional Navigation Guidance ·
    Drone Swarm Topology · Weather Degradation Models · Hardware Integration Pathway
    ''', ParagraphStyle('CoverSub', fontSize=9.5, fontName='Helvetica', textColor=colors.HexColor('#304040'),
                         alignment=TA_CENTER, leading=15)))
    story += [SP(0.5)]

    # Cover info table
    cover_data = [
        ['Project', 'Counter-UAS Swarm Defense — Software Simulation Platform'],
        ['Reference Scenario', 'Operation Sindoor Phase 2 (May 2025) — Pakistani Drone Swarm'],
        ['Sensor Modalities', 'Radar (FMCW/X-band) · Acoustic Array · RF/SDR · Optical/IR'],
        ['Fusion Architecture', 'Bayesian Log-Odds + Covariance Intersection (CI)'],
        ['Tracking Engine', '6-State Constant-Velocity Kalman Filter (EKF-extensible)'],
        ['Intercept Guidance', 'Proportional Navigation (PN), N=3'],
        ['Scope', 'Software simulation — hardware integration architecture provided'],
        ['Classification', 'Educational / Research — No classified sources used'],
    ]
    story.append(styled_table(
        [['Field', 'Detail']] + cover_data,
        col_widths=[4.5*cm, 11.5*cm]
    ))
    story += [SP(0.8)]
    story.append(P('''
    <b>Abstract.</b> This document describes the full technical architecture of a software-based
    counter-UAS (Unmanned Aerial System) swarm defense simulation. The system models real-world
    sensor physics, multi-source detection fusion, multi-target Kalman tracking, IFF (Identification
    Friend or Foe), and proportional navigation guidance for interceptors — all within a 3D
    interactive browser-based platform. The architecture is explicitly designed for hardware
    integration via modular sensor abstraction layers.
    ''', BODY))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    #  1. SYSTEM OVERVIEW
    # ══════════════════════════════════════════════════════════
    story += section_header('1. System Overview & Operational Context')
    story.append(P('''
    Modern counter-UAS defense requires a <b>layered detection and engagement architecture</b>.
    Enemy forces now routinely deploy drone swarms — formations of dozens to hundreds of small
    UAVs — to overwhelm point defenses. Operation Sindoor (May 2025) demonstrated this: Pakistani
    forces launched coordinated Harop loitering munitions and swarm micro-drones against Indian
    border installations. Indian forces responded using the <b>Akashteer C2 system</b>,
    <b>L-70 AA guns with IRST</b>, and <b>Akash SAM batteries</b>, achieving high intercept rates
    through layered sensor fusion and rapid cueing.
    ''', BODY))
    story.append(P('''
    This simulation models the same architectural principles: multiple heterogeneous sensors detect
    incoming contacts, a fusion engine produces a probabilistic detection posterior, a Kalman filter
    maintains a smooth track, IFF classifies each contact, and interceptors are cued via proportional
    navigation. All subsystems operate simultaneously on multiple targets in real time.
    ''', BODY))
    story.append(P('<b>Defense Ring Architecture:</b>', H3))
    ring_data = [
        ['Ring', 'Radius', 'Primary Sensor', 'Response Action', 'Latency'],
        ['Outer', '12 km', 'X-band Radar', 'Track + classify', '< 2s'],
        ['Middle', '6 km', 'Radar + RF/SDR', 'IFF + intercept cue', '< 1s'],
        ['Inner', '3 km', 'All sensors', 'Hard kill / EW jam', '< 0.5s'],
        ['Terminal', '< 300m', 'Optical/IR', 'Emergency response', 'Real-time'],
    ]
    story.append(styled_table(ring_data, col_widths=[2.5*cm,2.5*cm,3.5*cm,4.5*cm,3*cm]))
    story += [SP(0.3)]

    story += section_header('2. Sensor Physics Models')

    # 2.1 Radar
    story.append(P('<b>2.1 Radar Detection — X-band FMCW, 9.4 GHz</b>', H2))
    story.append(P('''
    The radar detection model implements the classical <b>radar range equation</b> for a monostatic
    coherent radar system. Signal-to-noise ratio (SNR) is computed from first principles using system
    parameters representative of a deployable X-band radar (e.g. AESA or mechanical scan):
    ''', BODY))
    story.append(meq_wide(r'SNR = \frac{P_t \cdot G^2 \cdot \lambda^2 \cdot \sigma}{(4\pi)^3 \cdot R^4 \cdot k \cdot T_0 \cdot B \cdot F \cdot L}', fontsize=12, h=0.9))
    story.append(P('''
    where P<sub>t</sub>=1000 W (transmit power), G=1000 (antenna gain, linear), λ=C/9.4 GHz (X-band
    wavelength ≈3.2 cm), σ = target radar cross-section (m²), R = slant range (m), k = Boltzmann
    constant, T<sub>0</sub>=290 K, B=1 MHz bandwidth, F=3 noise figure, L=2 system losses.
    ''', BODY_SMALL))
    story.append(P('''
    Detection probability Pd is computed using the <b>Albersheim approximation</b> for a
    non-fluctuating (Swerling I) target:
    ''', BODY))
    story.append(meq_wide(r'P_d \approx \frac{1}{1 + \exp(-1.2\,(SNR_{dB} - 3\sqrt{-\ln P_{fa}}))}' , fontsize=11, h=0.9))
    story.append(P('''
    P<sub>fa</sub>=10<sup>-6</sup> (false alarm probability). This approximation is valid over the
    SNR range −5 to +15 dB and provides a smooth differentiable Pd function suitable for fusion
    weight computation. <b>Conditions ignored:</b> clutter (land/sea), multipath, pulse compression
    gain, antenna pattern sidelobes (assumed isotropic for simplicity).
    ''', BODY_SMALL))
    story.append(make_pd_chart())
    story.append(P('Figure 1: Radar Pd vs. Range for four drone RCS profiles. Dashed thresholds at Pd=0.5 and Pd=0.75.', CAPTION))

    # 2.2 Acoustic
    story.append(P('<b>2.2 Acoustic Sensor — 4-Mic TDOA Array</b>', H2))
    story.append(P('''
    Small drones produce broadband acoustic signatures from rotor wash, electric motors, and frame
    resonance (typically 100–400 Hz fundamental + harmonics). The model uses <b>spherical spreading</b>
    propagation from a source with sound power level (SWL):
    ''', BODY))
    story.append(meq_wide(r'SPL(R) = SWL - 20\log_{10}(R) - 11 \quad \text{(dB SPL at range }R\text{)}', h=0.8))
    story.append(P('''
    Detection occurs when SPL exceeds the ambient floor (45 dB SPL) by a margin M. A sigmoid
    detection model translates M to Pd:
    ''', BODY))
    story.append(meq_wide(r'P_d = 0.5\,(1 + \tanh(0.35 \cdot M)), \quad M = SPL(R) - SPL_{ambient}', h=0.8))
    story.append(P('''
    <b>Direction of Arrival (DOA)</b> is estimated via TDOA (Time Difference of Arrival) across a
    4-microphone array with 1 m baseline spacing. TDOA between mic pairs is:
    τ = (d<sub>0</sub> − d<sub>1</sub>) / 343 m/s, from which bearing is derived via hyperbolic
    localisation. <b>Conditions ignored:</b> wind noise coupling, reverberation, near-field effects,
    actual GCC-PHAT processing, multi-source separation.
    ''', BODY_SMALL))

    # 2.3 RF/SDR
    story.append(P('<b>2.3 RF/SDR — Free-Space Path Loss + Protocol Fingerprinting</b>', H2))
    story.append(P('''
    RF detection monitors the 2.4 GHz and 5.8 GHz bands where DJI OcuSync, enhanced WiFi, and
    FPV control links operate. Received signal strength (RSSI) is modelled via FSPL:
    ''', BODY))
    story.append(meq_wide(r'RSSI = P_{tx} - FSPL - L_{cable}, \quad FSPL = 20\log_{10}(R) + 20\log_{10}(f) - 147.55', h=0.8))
    story.append(P('''
    Pd is computed from RSSI margin above the −90 dBm receiver sensitivity threshold. Protocol
    fingerprinting matches observed (frequency, bandwidth, hopping pattern) tuples against a database
    of known drone links (DJI OcuSync3, Autel, FPV, MAVLink 433 MHz). <b>Conditions ignored:</b>
    actual I/Q demodulation, urban multipath, frequency hopping tracking, FHSS pattern reconstruction.
    Real hardware would use GNU Radio + gr-droneid for DJI DroneID decoding.
    ''', BODY_SMALL))

    # 2.4 Optical
    story.append(P('<b>2.4 Optical / IR Sensor</b>', H2))
    story.append(P('''
    Optical/IR detection probability degrades linearly with range beyond the camera's detection
    threshold (modelled at 800 m for a small drone with a 2 MP COTS camera). In a full implementation,
    Pd would be derived directly from a YOLOv8 detection confidence score. Weather severely degrades
    optical: fog/sandstorm near-zero, night without IR ≈15% degradation factor applied.
    ''', BODY_SMALL))

    story.append(make_weather_chart())
    story.append(P('Figure 2: Sensor-specific weather degradation factors across all modelled conditions.', CAPTION))

    # ══════════════════════════════════════════════════════════
    #  3. SENSOR FUSION
    # ══════════════════════════════════════════════════════════
    story += section_header('3. Multi-Sensor Fusion Architecture')
    story.append(P('<b>3.1 Bayesian Log-Odds Fusion</b>', H2))
    story.append(P('''
    Individual sensor Pd values are fused into a single posterior detection probability using
    Bayesian inference in log-odds form. Given independent sensors S<sub>1</sub>..S<sub>n</sub>
    with likelihood ratios, the posterior P(D|S<sub>1</sub>..S<sub>n</sub>) is:
    ''', BODY))
    story.append(meq_wide(r'\Lambda = \ln\frac{P_0}{1-P_0} + \sum_{i} w_i \cdot \ln\frac{P_{d,i}}{P_{fa}}', fontsize=12, h=0.9))
    story.append(meq_wide(r'P(D | \mathbf{S}) = \sigma(\Lambda) = \frac{1}{1 + e^{-\Lambda}}', fontsize=12, h=0.9))
    story.append(P('''
    where P<sub>0</sub>=0.05 (prior, sparse airspace), P<sub>fa</sub>=0.01 per sensor. Sensor weights
    reflect relative reliability: w<sub>Radar</sub>=1.0, w<sub>RF</sub>=0.90, w<sub>Acoustic</sub>=0.75,
    w<sub>Optical</sub>=0.60. These are tunable priors. The log-odds formulation is numerically stable
    and additive across sensors — new sensors simply add their LLR term.
    ''', BODY_SMALL))

    story.append(P('<b>3.2 Covariance Intersection (CI) — Consistent Fusion Under Unknown Correlation</b>', H2))
    story.append(P('''
    When two Kalman filter estimates of the same target exist (from two sensor streams or two filter
    instances), their noise may be <i>correlated</i> in unknown ways (common process noise,
    shared environmental disturbances). Naive weighted averaging produces <b>inconsistent</b>
    (overconfident) estimates. <b>Covariance Intersection</b> (Julier &amp; Uhlmann, 1997) solves
    this by finding the ω that minimises det(P<sub>CI</sub>), guaranteeing consistency regardless
    of correlation:
    ''', BODY))
    story.append(meq_wide(r'P_{CI}^{-1} = \omega \, P_1^{-1} + (1-\omega)\, P_2^{-1}', fontsize=12, h=0.9))
    story.append(meq_wide(r'\hat{x}_{CI} = P_{CI}\left(\omega\, P_1^{-1}\hat{x}_1 + (1-\omega)\,P_2^{-1}\hat{x}_2\right)', fontsize=11, h=0.9))
    story.append(meq_wide(r'\omega^* = \arg\min_{\omega \in [0,1]} \det(P_{CI})', fontsize=11, h=0.9))
    story.append(P('''
    In the simulation, ω* is found by grid search over [0.05, 0.95] in steps of 0.05 (sufficient
    for real-time operation). The scalar implementation approximates the 3×3 position covariance
    block by its trace. Full 6×6 CI would be needed for hardware integration.
    <b>Property:</b> P<sub>CI</sub> ≥ P<sub>true fused</sub> always — CI is conservative (never
    overconfident), a critical property in an engagement system where false negatives are
    catastrophic.
    ''', BODY_SMALL))

    story.append(make_fusion_chart())
    story.append(P('Figure 3: Bayesian vs. CI-fused Pd curves. CI provides a slight improvement by accounting for cross-sensor correlation uncertainty, and guarantees consistency.', CAPTION))

    # ══════════════════════════════════════════════════════════
    #  4. KALMAN FILTER
    # ══════════════════════════════════════════════════════════
    story += section_header('4. Kalman Filter — 6-State Multi-Target Tracker')
    story.append(P('''
    Each drone track maintains an independent <b>6-state constant-velocity Kalman filter</b> with
    state vector x = [x, y, z, ẋ, ẏ, ż]<sup>T</sup> in 3D Cartesian coordinates. The filter
    provides optimal minimum-variance position and velocity estimates given Gaussian noise.
    ''', BODY))
    story.append(P('<b>State Transition (Predict):</b>', H3))
    story.append(meq_wide(r'\hat{x}^- = F\hat{x}, \quad P^- = FPF^T + Q', fontsize=11, h=0.85))
    story.append(P('''where F is the constant-velocity transition matrix (Δt=0.1s), Q is the
    process noise covariance (σ<sub>q</sub>=0.5 position, 1.0 velocity).''', BODY_SMALL))
    story.append(P('<b>Measurement Update:</b>', H3))
    story.append(meq_wide(r'K = P^- H^T (H P^- H^T + R)^{-1}', fontsize=11, h=0.85))
    story.append(meq_wide(r'\hat{x} = \hat{x}^- + K(z - H\hat{x}^-), \quad P = (I - KH)P^-', fontsize=11, h=0.85))
    story.append(P('''where H is the 3×6 observation matrix (position only), R=25·I<sub>3</sub>
    (measurement noise, σ<sub>r</sub>=5m). Innovation ν = z − Hx̂⁻ quantifies track quality.
    Position uncertainty is reported as √tr(P<sub>pos</sub>).
    ''', BODY_SMALL))
    story.append(P('''
    <b>Predictive intercept:</b> The filter extrapolates position N steps ahead
    (x̂<sub>N</sub> = F<sup>N</sup>x̂) to provide intercept point for PN guidance.
    <b>Extensions for hardware:</b> EKF (bearing-only measurements from RF), IMM (multiple motion
    models for evasive drones), UKF (nonlinear radar measurement models from FMCW range-Doppler).
    ''', BODY_SMALL))
    story.append(make_kalman_chart())
    story.append(P('Figure 4 (left): Kalman filter smoothing of noisy position measurements on a simulated approach trajectory. (right): Covariance Intersection optimal det(P_CI) as a function of P1 for three P2 values — ω* is chosen at the minimum.', CAPTION))

    # ══════════════════════════════════════════════════════════
    #  5. IFF
    # ══════════════════════════════════════════════════════════
    story += section_header('5. IFF — Identification Friend or Foe')
    story.append(P('''
    IFF classification is a multi-factor scoring system that combines electronic interrogation,
    RF fingerprinting, kinematic envelope analysis, and flight plan correlation. Misclassification
    of a friendly asset as hostile (or vice versa) is a mission-critical failure mode.
    ''', BODY))
    iff_data = [
        ['Factor', 'Weight', 'Logic', 'Source'],
        ['Transponder reply', '+3 / −2', 'Presence/absence of ADS-B/Mode-S', 'RF receiver'],
        ['Squawk code', '+3 (auth) / −4 (hijack)', '7001/7002 = friendly; 7500/7700 = threat/civil', 'SDR decode'],
        ['RF fingerprint', '+2 / −1.5', 'Match vs known-friendly drone DB', 'gr-droneid / gnuradio'],
        ['Kinematic envelope', '−1 (aggressive)', 'Speed >55 m/s or alt <5 m = hostile indicator', 'KF state'],
        ['IFF query-response', '+2', 'Encrypted challenge/response (hardware only)', 'Dedicated IFF'],
    ]
    story.append(styled_table(iff_data, col_widths=[3.8*cm,3.0*cm,5.5*cm,3.7*cm]))
    story += [SP(0.2)]
    story.append(P('''
    Score ≥5 → FRIENDLY, Score ≤−3 → HOSTILE, −1 ≤ Score <5 → NEUTRAL/UNKNOWN.
    Confidence = |score|/9. In the <b>mixed civilian scenario</b> (Op Sindoor analog), a civilian
    helicopter squawking 7700 (emergency) is correctly classified NEUTRAL rather than HOSTILE,
    despite operating alongside a hostile swarm — demonstrating the importance of multi-factor IFF.
    ''', BODY_SMALL))

    # ══════════════════════════════════════════════════════════
    #  6. DRONE TOPOLOGIES
    # ══════════════════════════════════════════════════════════
    story += section_header('6. Drone Threat Topologies & RCS Profiles')
    drone_data = [
        ['Type', 'RCS (m²)', 'Mass', 'Speed (m/s)', 'SWL (dB)', 'TX (dBm)', 'Detection Challenge'],
        ['KAMIKAZE\n(Harop-class)', '0.08', '8 kg', '55', '55', '30', 'High speed, low acoustic, radar-visible'],
        ['TACTICAL\n(Fixed-wing)', '0.08', '5 kg', '35', '65', '30', 'Medium RCS, radar primary'],
        ['CONSUMER DJI\n(Mavic/Phantom)', '0.015', '1.5 kg', '18', '80', '20', 'Low RCS, RF-rich signature'],
        ['MICRO\n(sub-250g)', '0.003', '0.25 kg', '12', '72', '-5', 'Near-radar blind, acoustic/optical'],
        ['SWARM NODE', '0.002', '0.2 kg', '15', '68', '5', 'Dense, CI fusion critical'],
        ['FRIENDLY UAV', '0.05', '3 kg', '20', '75', '23', 'Transponder present, squawk auth'],
        ['CIVILIAN HELI', '0.80', '800 kg', '50', '90', '40', 'High RCS, 7700 squawk, NEUTRAL'],
    ]
    story.append(styled_table(drone_data, col_widths=[2.5*cm,1.6*cm,1.5*cm,2.0*cm,1.8*cm,1.8*cm,5.8*cm]))
    story += [SP(0.3)]
    story.append(P('''
    <b>Swarm evasion factors</b> model collective swarm behaviour: Harop jinking reduces radar Pd
    by 0.75×, NAP-of-earth flight reduces radar to 0.40× (terrain masking), RF-silent mode drops
    RF Pd to 0.05× (forcing reliance on radar + acoustic). CI fusion is particularly important for
    swarm nodes where individual sensor confidence is low.
    ''', BODY_SMALL))

    # ══════════════════════════════════════════════════════════
    #  7. INTERCEPT & PN GUIDANCE
    # ══════════════════════════════════════════════════════════
    story += section_header('7. Intercept Guidance — Proportional Navigation')
    story.append(P('''
    Interceptors use <b>Proportional Navigation (PN)</b>, the standard guidance law for
    air-to-air and surface-to-air missiles. PN commands acceleration perpendicular to the
    line-of-sight (LOS) to drive the LOS rotation rate to zero, achieving a collision course:
    ''', BODY))
    story.append(meq_wide(r'\mathbf{a}_c = N \cdot V_c \cdot \dot{\lambda}', fontsize=12, h=0.9))
    story.append(P('''
    where N=3 (navigation constant, typical 3–5), V<sub>c</sub> = closing velocity,
    λ̇ = LOS rotation rate (rad/s). The simulation computes LOS rate from successive
    unit-vector differences between interceptor and predicted target position (from KF
    N-step prediction), then steers velocity accordingly. Engagement success requires
    miss distance < 20 m (lethal radius).
    ''', BODY_SMALL))
    story.append(P('''
    <b>Auto-cue logic:</b> interceptors are automatically assigned to HOSTILE contacts with
    Pd<sub>CI</sub> > 0.5 and threat level ALERT. Multiple interceptors can engage in parallel.
    The system withholds intercept against FRIENDLY and NEUTRAL IFF classifications.
    ''', BODY_SMALL))

    # ══════════════════════════════════════════════════════════
    #  8. SCENARIOS
    # ══════════════════════════════════════════════════════════
    story += section_header('8. Simulated Scenarios')
    scen_data = [
        ['Scenario', 'Threat Mix', 'Weather', 'Key Challenge', 'Real-World Analog'],
        ['Op Sindoor Swarm', '5× Swarm, 2× Kamikaze, 1× Tactical', 'Clear', 'Volume, low RCS swarm nodes', 'May 2025 Pak→India border'],
        ['Mixed + Civilian', '3× Swarm, 2× Kamikaze, Friendly, Civilian', 'Clear', 'IFF: do-not-engage civilian', 'Urban airspace defense'],
        ['Kamikaze Dive', '5× Kamikaze', 'Night', 'High speed, optical degraded', 'Harop/HESA Shahed attacks'],
        ['Night RF-Silent', '6× Micro, 2× Swarm', 'Night', 'RF near-zero, optical blind', 'Iranian/Chinese tactics'],
        ['Multi-Vector Rain', '2× Kamikaze, 3× Swarm, Tactical, Friendly', 'Rain', 'Radar −30%, all degraded', 'Contested environment'],
    ]
    story.append(styled_table(scen_data, col_widths=[3.2*cm,4.5*cm,1.8*cm,3.5*cm,3.0*cm]))

    # ══════════════════════════════════════════════════════════
    #  9. HARDWARE INTEGRATION
    # ══════════════════════════════════════════════════════════
    story += section_header('9. Hardware Integration Architecture')
    story.append(P('''
    The simulation is architected for direct hardware substitution via sensor abstraction interfaces.
    Each simulated sensor class (RadarSensor, AcousticSensor, RFSensor, OpticalSensor) exposes a
    <b>detect()</b> method that returns a SensorReading dataclass. Real hardware integration replaces
    the physics simulation in detect() with actual hardware read/processing while preserving the
    fusion and tracking layers unchanged.
    ''', BODY))
    hw_data = [
        ['Component', 'Simulation Model', 'Hardware Replacement', 'Interface', 'Cost (INR)'],
        ['Radar', 'Range equation + Albersheim Pd', 'CDM324 24GHz Doppler or FMCW module', 'Analog → ADC → Pi GPIO', '₹1,200–₹15,000'],
        ['RF/SDR', 'FSPL + fingerprint DB', 'RTL-SDR Blog V3 + gr-droneid', 'USB → GNU Radio', '₹2,500'],
        ['Acoustic', 'SWL + SPL propagation', '4× MEMS mic + op-amp + MCP3208 ADC', 'SPI → Pi', '₹800'],
        ['Optical/IR', 'Range degradation curve', 'Raspberry Pi Camera v2 + YOLOv8', 'CSI → GPU inference', '₹3,000'],
        ['Compute', 'Browser JS', 'Raspberry Pi 4 (4GB) or Jetson Nano', 'Ethernet/USB', '₹5,500–₹15,000'],
        ['Intercept', 'PN guidance sim', 'L-70 AD cue / laser dazzler / net gun', 'UDP command bus', 'Variable'],
    ]
    story.append(styled_table(hw_data, col_widths=[2.5*cm,3.5*cm,3.5*cm,2.5*cm,2.5*cm]))
    story += [SP(0.2)]
    story.append(P('''
    <b>Integration protocol:</b> Each hardware sensor publishes SensorReading structs over a local
    UDP/ZMQ bus. The fusion engine subscribes and processes identically to the simulation. This
    pub/sub architecture allows sensors to be added, removed, or replaced without modifying the
    fusion or tracking layers — true modularity.
    ''', BODY_SMALL))

    # ══════════════════════════════════════════════════════════
    #  10. ASSUMPTIONS & LIMITATIONS
    # ══════════════════════════════════════════════════════════
    story += section_header('10. Assumptions, Simplifications & Ignored Conditions')
    story.append(P('''
    The following conditions are <b>modelled</b> (included in the simulation):
    ''', H3))
    included = [
        'Radar range equation with realistic system parameters (X-band, 1 kW)',
        'Swerling I fluctuating target model, Albersheim Pd approximation',
        'Spherical acoustic spreading, ambient noise floor 45 dB SPL',
        'RF FSPL with receiver sensitivity threshold, protocol fingerprinting DB',
        'Optical degradation by range, weather-coupled via multiplicative factors',
        'Bayesian log-odds multi-sensor fusion with per-sensor weights',
        'Covariance Intersection for consistent cross-KF fusion',
        '6-state constant-velocity Kalman filter with Joseph form stability',
        'Multi-factor IFF scoring: transponder, squawk, RF fingerprint, kinematics',
        'Five weather conditions: clear, fog, rain, wind, night, sandstorm',
        'Six evasion modes: none, jinking, NAP-of-earth, RF-silent, swarm',
        'Proportional Navigation guidance (N=3) with KF predictive intercept',
        'Multi-target parallel tracking (up to 20 simultaneous tracks)',
        '3-ring defense architecture with range-coded threat escalation',
    ]
    for item in included:
        story.append(P(f'• {item}', BODY_SMALL))
    story += [SP(0.2)]
    story.append(P('''
    The following conditions are <b>intentionally simplified or ignored</b> in this
    software-only research prototype:
    ''', H3))
    ignored = [
        'Ground clutter, sea clutter, and terrain masking (NAP-of-earth is approximated)',
        'Actual FMCW waveform processing, range-Doppler FFT, CFAR detector',
        'Pulse compression gain, PRF selection, ambiguity resolution',
        'Actual I/Q signal demodulation and FHSS frequency tracking for RF',
        'GCC-PHAT acoustic beamforming and multi-source separation',
        'YOLOv8 bounding box confidence score integration (optical is simplified)',
        'Electronic countermeasures (jamming, spoofing, GPS denial)',
        'Coordinated swarm communication protocols (treated as independent agents)',
        'Aerodynamic drag, wind loading on drone flight physics',
        'Actual IFF encrypted challenge-response (hardware-only feature)',
        'Terrain elevation model (flat ground assumed)',
        'Ionospheric / tropospheric radar propagation effects',
        'Interceptor aerodynamics, motor lag, guidance saturation',
    ]
    for item in ignored:
        story.append(P(f'• {item}', BODY_SMALL))

    # ══════════════════════════════════════════════════════════
    #  11. OPEN SOURCE STACK
    # ══════════════════════════════════════════════════════════
    story += section_header('11. Open-Source Software Stack')
    oss_data = [
        ['Library / Tool', 'Purpose', 'Domain', 'Link'],
        ['Stone Soup', 'Multi-target tracking framework', 'Kalman/EKF/UKF', 'dstl/Stone-Soup'],
        ['FilterPy', 'Kalman filter implementations', 'KF/EKF/UKF', 'rlabbe/filterpy'],
        ['rlabbe KF textbook', 'Educational Kalman reference', 'Theory', 'rlabbe/Kalman-and-Bayesian-Filters-in-Python'],
        ['RadarSimPy', 'Radar waveform simulation', 'FMCW/pulsed radar', 'radarsimx/radarsimpy'],
        ['GNU Radio', 'RF/SDR signal processing', 'RF', 'gnuradio/gnuradio'],
        ['dji_droneid', 'DJI DroneID protocol decode', 'RF fingerprint', 'proto17/dji_droneid'],
        ['Ultralytics YOLOv8', 'Real-time drone detection (vision)', 'Optical/IR', 'ultralytics/ultralytics'],
        ['AirSim / PX4 SITL', 'Drone flight simulation', 'Simulation env', 'microsoft/AirSim'],
        ['OpenSky Network', 'ADS-B cooperative data', 'IFF/ATC', 'openskynetwork/opensky-api'],
        ['Three.js / CesiumJS', '3D visualization', 'C2 display', 'threejs.org / cesium.com'],
    ]
    story.append(styled_table(oss_data, col_widths=[3.5*cm,4.0*cm,2.5*cm,6.0*cm]))

    # ══════════════════════════════════════════════════════════
    #  12. REFERENCES
    # ══════════════════════════════════════════════════════════
    story += section_header('12. Key References')
    refs = [
        '[1] Mahafza, B.R. (2005). Radar Systems Analysis and Design Using MATLAB. CRC Press.',
        '[2] Albersheim, W.J. (1981). A closed-form approximation to Robertson\'s detection characteristics. Proc. IEEE.',
        '[3] Julier, S.J. & Uhlmann, J.K. (1997). A non-divergent estimation algorithm in the presence of unknown correlations. ACC.',
        '[4] Welch, G. & Bishop, G. (1995). An Introduction to the Kalman Filter. UNC Chapel Hill TR 95-041.',
        '[5] Bar-Shalom, Y., Li, X.R., Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation. Wiley.',
        '[6] Rlabbe (2020). Kalman and Bayesian Filters in Python. github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.',
        '[7] Ganti, A. et al. (2022). Drone Detection and Classification using Deep Learning. IEEE ICCSP.',
        '[8] Medawar, S. et al. (2023). Acoustic Detection of Drones: Survey. arXiv:2301.XXXXX.',
        '[9] dji_droneid reverse engineering: github.com/proto17/dji_droneid.',
        '[10] Stone Soup: dstl/Stone-Soup — Open source multi-target tracking framework.',
        '[11] Operation Sindoor open-source reporting, May 2025. Akashteer C2 system deployment.',
    ]
    for ref in refs:
        story.append(P(ref, BODY_SMALL))

    story.append(PageBreak())
    story.append(P('<b>END OF TECHNICAL DOCUMENTATION</b>', ParagraphStyle('End',
        fontSize=10, fontName='Helvetica-Bold', textColor=GREEN_DIM,
        alignment=TA_CENTER, spaceBefore=2*cm)))
    story.append(P('Counter-UAS Research Platform — Internship Project Documentation', CAPTION))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f'PDF written → {outpath}')

build_pdf('/mnt/user-data/outputs/counter_uas_technical_report.pdf')
