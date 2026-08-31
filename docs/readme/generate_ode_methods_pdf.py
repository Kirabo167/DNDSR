# -*- coding: utf-8 -*-
"""ODE.hpp 时间推进方法推导 PDF 生成脚本"""
import os, sys

# 字体注册
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.styles import ParagraphStyle

FONT_REG = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
pdfmetrics.registerFont(TTFont("WQY", FONT_REG, subfontIndex=0))
pdfmetrics.registerFont(TTFont("WQY-Bold", FONT_REG, subfontIndex=0))

OUT_DIR = "/mnt/ssd-SATARAID5/home/mrz/projects/DNDSR/docs/readme"
os.makedirs(OUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUT_DIR, "ODE_Time_Stepping_Methods.pdf")

M = 1.8 * cm
PAGE_W = A4[0]
AVAIL_W = PAGE_W - 2 * M

# 样式
S_H1 = ParagraphStyle('H1', fontName='WQY-Bold', fontSize=16, leading=22, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor('#522aca'))
S_H2 = ParagraphStyle('H2', fontName='WQY-Bold', fontSize=13, leading=18, spaceBefore=12, spaceAfter=6)
S_BODY = ParagraphStyle('Body', fontName='WQY', fontSize=10.5, leading=17, alignment=TA_LEFT, wordWrap='CJK', firstLineIndent=21)
S_BODY0 = ParagraphStyle('Body0', fontName='WQY', fontSize=10.5, leading=17, alignment=TA_LEFT, wordWrap='CJK')
S_CODE = ParagraphStyle('Code', fontName='WQY', fontSize=9, leading=13, firstLineIndent=0, spaceAfter=0)
S_CAP = ParagraphStyle('Cap', fontName='WQY', fontSize=8.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor('#8f8a83'))

def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def math(s):
    """简单数学公式渲染（用 Unicode + 粗体）"""
    return Paragraph(esc(s).replace('**', '<b>').replace('//', '</b>'), S_BODY0)

def body(s):
    return Paragraph(esc(s), S_BODY)

def body0(s):
    return Paragraph(esc(s), S_BODY0)

def h1(s):
    return Paragraph(esc(s), S_H1)

def h2(s):
    return Paragraph(esc(s), S_H2)

def code_block(lines):
    txt = '<br/>'.join(esc(line) for line in lines)
    t = Table([[Paragraph(txt, S_CODE)]], colWidths=[AVAIL_W], hAlign='CENTER')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7f6f5')),
        ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor('#522aca')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t

def data_table(headers, rows, ratios, caption=None):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    S_HEAD = ParagraphStyle('THead', fontName='WQY-Bold', fontSize=9, leading=13, alignment=TA_CENTER, textColor=colors.white)
    S_CELL = ParagraphStyle('TCell', fontName='WQY', fontSize=9, leading=13, alignment=TA_CENTER)
    data = [[Paragraph(esc(h), S_HEAD) for h in headers]]
    for r in rows:
        data.append([Paragraph(esc(c), S_CELL) for c in r])
    widths = [r * AVAIL_W for r in ratios]
    t = Table(data, colWidths=widths, hAlign='CENTER', repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#522aca')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#8f8a83')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    out = [Spacer(1, 10), t]
    if caption:
        out += [Spacer(1, 4), Paragraph(caption, S_CAP)]
    out += [Spacer(1, 10)]
    return out

# ======================== 开始构建文档 ========================
doc = SimpleDocTemplate(PDF_PATH, pagesize=A4,
                        leftMargin=M, rightMargin=M, topMargin=1.6*cm, bottomMargin=1.7*cm,
                        title='ODE.hpp 时间推进方法推导', author='DNDSR')
story = []

# 封面
story.append(Spacer(1, 150))
story.append(Paragraph('<b>ODE.hpp 时间推进方法推导</b>', ParagraphStyle('Title', fontName='WQY-Bold', fontSize=26, leading=32, alignment=TA_CENTER)))
story.append(Spacer(1, 30))
story.append(Paragraph('DNDSR 项目 · Solver/ODE.hpp', ParagraphStyle('Sub', fontName='WQY', fontSize=14, leading=20, alignment=TA_CENTER, textColor=colors.HexColor('#8f8a83'))))
story.append(Spacer(1, 100))
story.append(Paragraph('2026年8月', ParagraphStyle('Date', fontName='WQY', fontSize=12, leading=16, alignment=TA_CENTER)))
story.append(PageBreak())

# ======================== 1. 隐式向后 Euler ========================
story.append(h1('1. 隐式向后 Euler（ImplicitEulerDualTimeStep）'))
story.append(body('隐式向后 Euler 是最简单的隐式时间推进方法，一阶精度，无条件 L-稳定。'))
story.append(h2('1.1 数学公式'))
story.append(body0('对于常微分方程：'))
story.append(code_block(['dx/dt = R(x)']))
story.append(body0('隐式向后 Euler 的离散形式为：'))
story.append(code_block(['(x^(n+1) - x^n) / Δt = R(x^(n+1))']))
story.append(body0('等价于：'))
story.append(code_block(['x^(n+1) = x^n + Δt · R(x^(n+1))']))
story.append(body0('引入 Newton 迭代求解非线性方程组。定义残差：'))
story.append(code_block(['F(x) = x - x^n - Δt · R(x) = 0']))
story.append(body0('Newton 迭代（第 k 步）：'))
story.append(code_block(['(I - Δt·J) Δx = -F(x^k)']))
story.append(body0('其中 J = ∂R/∂x 是 Jacobian 矩阵。每次迭代：'))
story.append(code_block(['x^(k+1) = x^k + Δx']))
story.append(h2('1.2 算法流程'))
story += data_table(['步骤', '操作'],
    [['1', '保存旧解 xLast = x^n'],
     ['2', '计算局部时间步长 dTau（CFL 限制）'],
     ['3', '计算空间残差 R(x^k)'],
     ['4', '构建 RHS：rhs = (xLast - x)/Δt + R(x)'],
     ['5', '求解线性系统 (I/Δt - J) Δx = rhs'],
     ['6', '更新解 x += Δx'],
     ['7', '检查收敛，未收敛返回步骤 3']], [0.15, 0.85])

# ======================== 2. SDIRK ========================
story.append(h1('2. SDIRK（Singly Diagonal Implicit Runge-Kutta）'))
story.append(body('SDIRK 是一种隐式 Runge-Kutta 方法，对角线上所有元素相等（单对角），便于实现。'))
story.append(h2('2.1 Butcher 表'))
story.append(body0('通用 SDIRK 的 Butcher 表（s 阶段）：'))
story += data_table(['c1', 'a11', '0', '...', '0'],
    [['c2', 'a21', 'a22', '...', '0'],
     ['...', '...', '...', '...', '...'],
     ['cs', 'as1', 'as2', '...', 'ass'],
     ['', 'b1', 'b2', '...', 'bs']], [0.2, 0.2, 0.2, 0.2, 0.2],
    caption='表 2-1 SDIRK Butcher 表')
story.append(h2('2.2 阶段方程'))
story.append(body0('对于每个内阶段 i：'))
story.append(code_block(['x^(i) = x^n + Δt · Σ(a_ij · R(x^(j)))']))
story.append(body0('特别地，对角线元素 a_ii 相等（SDIRK 特性），使得线性系统结构相同。'))
story.append(h2('2.3 ODE.hpp 中实现的方案'))
story += data_table(['方案', '阶段数', '精度', '特点'],
    [['0', '3', '4', 'L-稳定，默认方案'],
     ['1', '6', '高阶', '含显式第一级'],
     ['2', '4', '3', 'ESDIRK3，显式第一级'],
     ['3', '2', '2', '梯形法则，A-稳定'],
     ['4', '3', '2', 'ESDIRK2，显式第一级']], [0.2, 0.2, 0.2, 0.4])

# ======================== 3. BDF ========================
story.append(h1('3. BDF（Backward Differentiation Formula）'))
story.append(body('BDF 是多步隐式方法，利用前几步的解构造高阶时间导数。'))
story.append(h2('3.1 通用公式'))
story.append(body0('BDF-k（k 步方法）：'))
story.append(code_block(['Σ(α_j · x^(n+1-j)) = Δt · R(x^(n+1))']))
story.append(body0('其中 j = 0, 1, ..., k。展开：'))
story.append(code_block(['α0·x^(n+1) + α1·x^n + α2·x^(n-1) + ... = Δt · R(x^(n+1))']))
story.append(h2('3.2 ODE.hpp 中的 BDF 系数'))
story += data_table(['阶数', 'α0', 'α1', 'α2', 'α3', 'α4'],
    [['BDF1', '1', '1', 'NaN', 'NaN', 'NaN'],
     ['BDF2', '2/3', '4/3', '-1/3', 'NaN', 'NaN'],
     ['BDF3', '6/11', '18/11', '-9/11', '2/11', 'NaN'],
     ['BDF4', '12/25', '48/25', '-36/25', '16/25', '-3/25']], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
story.append(h2('3.3 算法流程'))
story.append(body0('BDF 算法需要存储前 k-1 步的解（xPrevs）和时间步长（dtPrevs）：'))
story.append(code_block(['rhs = Σ(αj · x^(n+1-j)) / Δt + R(x^(n+1))']))
story.append(code_block(['求解：(α0·I/Δt - J) Δx = rhs']))
story.append(body0('历史步更新（环形缓冲区）：'))
story.append(code_block(['xPrevs[prevStart] = x^n']))
story.append(code_block(['prevStart = (prevStart - 1) % prevSiz']))

# ======================== 4. V-BDF ========================
story.append(h1('4. 可变步长 BDF（V-BDF）'))
story.append(body('V-BDF 是 BDF 的扩展，允许时间步长变化，动态计算系数。'))
story.append(h2('4.1 核心思想'))
story.append(body0('当时间步长从 dt_prev 变为 dt 时，重新定义 BDF 系数。对于 k=2：'))
story.append(code_block(['φ = dt / (dt_prev + dt)']))
story.append(code_block(['R_t = dt / dt_prev']))
story.append(body0('动态系数：'))
story.append(code_block(['α0 = 1/(1+φ),  α1 = 1 + R_t·φ/(1+φ),  α2 = -R_t·φ/(1+φ)']))
story.append(h2('4.2 步长限制器'))
story.append(body0('防止步长变化过大导致非物理振荡：'))
story.append(code_block(['if (limitingV < 1) dt_new = BisectSolve(f, dt_prev, dt_prev * maxIncrease)']))
story.append(body0('其中 f 是步长限制的隐式函数。'))

# ======================== 5. Hermite-3 ========================
story.append(h1('5. Hermite-3（HM3）'))
story.append(body('Hermite-3 是一种高阶方法，利用时间导数信息，通过 Hermite 插值构造。'))
story.append(h2('5.1 核心参数'))
story.append(body0('参数 α（alphaHM3）控制中点位置，典型值 0.55。插值系数：'))
story += data_table(['系数', 'U2R2 (mask=0)', 'U2R1 (mask=1)', 'U3R1 (mask=2)'],
    [['cInter[0]', '-3α²+2α³+1', '-2α+α²+1', '动态计算'],
     ['cInter[1]', '3α²-2α³', '2α-α²', '动态计算'],
     ['cInter[2]', 'α-2α²+α³', '0', '动态计算'],
     ['cInter[3]', '-α²+α³', '-α+α²', '动态计算']], [0.2, 0.27, 0.27, 0.27])
story.append(h2('5.2 Simpson 积分权重'))
story.append(code_block(['wInteg[0] = -1/(6α) + 1/2']))
story.append(code_block(['wInteg[1] = -1/(6α(α-1))']))
story.append(code_block(['wInteg[2] = 1/(6α-6) + 1/2']))
story.append(h2('5.3 pMG 加速'))
story.append(body0('伪多网格（pMG）使用低阶空间离散作为 smoother，加速 Newton 收敛：'))
story.append(code_block(['for iMG = 1 to nMG:']))
story.append(code_block(['    frhs(rhs, xMG, ..., uPos=2)  // 低阶残差']))
story.append(code_block(['    fsolve(xMG, rhs, ..., uPos=2)']))
story.append(code_block(['    fincrement(xMG, xinc, ..., uPos=2)']))

# ======================== 6. SSP-RK3 ========================
story.append(h1('6. SSP-RK3（显式强稳定保持）'))
story.append(body('SSP-RK3 是三阶显式方法，每个阶段都是前一阶段的凸组合，保持总变差不增。'))
story.append(h2('6.1 算法'))
story.append(code_block(['// Stage 1:']))
story.append(code_block(['x^(1) = x^n + Δt·R(x^n)']))
story.append(code_block(['// Stage 2:']))
story.append(code_block(['x^(2) = 0.75·x^n + 0.25·x^(1) + 0.25·Δt·R(x^(1))']))
story.append(code_block(['// Stage 3:']))
story.append(code_block(['x^(n+1) = 1/3·x^n + 2/3·x^(2) + 2/3·Δt·R(x^(2))']))
story.append(h2('6.2 SSP 性质'))
story.append(body0('每个阶段的系数都是非负的，且和为 1，因此是凸组合。这保证了：'))
story.append(code_block(['TV(x^(n+1)) ≤ TV(x^n)']))
story.append(body0('即总变差不会增加，避免数值振荡。'))

# ======================== 总结表 ========================
story.append(h1('7. 方法对比总结'))
story += data_table(['方法', '类型', '阶数', '稳定性', '特点'],
    [['向后 Euler', '隐式', '1', 'L-稳定', '最稳定，最简单'],
     ['SDIRK', '隐式', '2~4', 'L-稳定', '多阶段，高精度'],
     ['BDF', '隐式', '1~4', 'A-稳定', '多步法，历史复用'],
     ['V-BDF', '隐式', '2', 'A-稳定', '自适应步长'],
     ['Hermite-3', '隐式', '高阶', '条件稳定', 'Hermite 插值 + pMG'],
     ['SSP-RK3', '显式', '3', 'TVD', 'SSP 性质，显式']], [0.2, 0.15, 0.15, 0.2, 0.3])

# 构建文档
doc.build(story)
print(f"PDF 生成完成: {PDF_PATH}")
