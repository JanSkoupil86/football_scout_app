"""Single-page 16:9 vector recruitment poster, with optional PNG preview/export."""
import base64, io, math
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth

COLORS = ['#29b99b','#5ba9f5','#d8a75a','#b28af0','#e0788a','#6bc1bb']
VERDICT_COLORS = {'Sign Now':'#087e61','Strong Follow-up':'#235b9b','Monitor':'#99631e','Decline':'#a43c46','Not Assessed':'#34455c'}

def poster_pdf(d, metric_rows, groups=None):
    """Render one vector PDF page, 960x540 pt (16:9)."""
    W,H=960,540; buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=(W,H),pageCompression=1)
    bg='#0d1520'; panel='#192532'; panel2='#202e3c'; ink='#f3f6fa'; muted='#a6b4c2'; line='#334352'
    def color(h): return HexColor(h)
    def rect(x,y,w,h,fill=panel,r=7):
        c.setFillColor(color(fill));c.roundRect(x,y,w,h,r,fill=1,stroke=0)
    def txt(x,y,s,size=9,bold=False,fill=ink,maxw=None):
        s=str(s or '').replace('\n',' ').encode('latin-1','replace').decode('latin-1')
        font='Helvetica-Bold' if bold else 'Helvetica'
        if maxw:
            while s and stringWidth(s,font,size)>maxw: s=s[:-1]
            if s!=str(s or '') and len(s)>3:s=s[:-2]+'..'
        c.setFont(font,size);c.setFillColor(color(fill));c.drawString(x,y,s)
    def lines(value,width,size=8,limit=4):
        out=[]
        for para in str(value or '').replace('\r','').split('\n'):
            words=para.split(); cur=''
            for word in words:
                nxt=(cur+' '+word).strip()
                if stringWidth(nxt,'Helvetica',size)<=width:cur=nxt
                else:
                    if cur:out.append(cur)
                    cur=word
            if cur:out.append(cur)
        if len(out)>limit:
            out=out[:limit];out[-1]=out[-1][:max(0,len(out[-1])-3)]+'...'
        return out
    def paragraph(x,top,content,width,size=8,leading=11,maxlines=4,fill=ink):
        for i,t in enumerate(lines(content,width,size,maxlines)):txt(x,top-i*leading,t,size,fill=fill)
    def bullets(x,top,content,width,maxlines=4,accent='#29b99b'):
        parts=[t.strip(' -•\t') for t in str(content or '').splitlines() if t.strip()]
        if not parts:txt(x,top,'Not assessed',7,fill=muted);return
        y=top
        for part in parts[:maxlines]:
            wrapped=lines(part,width-15,7.3,2)
            c.setFillColor(color(accent));c.circle(x+3,y+2,2,stroke=0,fill=1)
            for t in wrapped:
                txt(x+11,y,t,7.3,maxw=width-13);y-=11
            y-=3
    def photo(key,x,y,w,h):
        if not d.get(key):return False
        try:
            img=ImageReader(io.BytesIO(base64.b64decode(d[key])))
            c.drawImage(img,x,y,w,h,mask='auto',preserveAspectRatio=True,anchor='c')
            return True
        except Exception:return False
    c.setFillColor(color(bg));c.rect(0,0,W,H,stroke=0,fill=1)
    # Header
    rect(16,447,928,77)
    rect(25,456,64,59,panel2,5)
    if not photo('photo_b64',27,458,60,55):
        initials=''.join(n[:1] for n in str(d.get('name','')).split()[:2]).upper() or 'P'
        txt(40,480,initials,21,True,fill='#a6c6c0')
    txt(101,496,d.get('name','Player'),22,True,maxw=390)
    txt(102,476,f"{d.get('club','')}  |  {d.get('position','')}  |  {d.get('secondary_positions','')}",9,fill=muted,maxw=425)
    txt(102,459,f"{d.get('league','')}  ·  {d.get('season','')}",8,fill=muted,maxw=425)
    info=[('AGE',d.get('age')),('NATIONALITY',d.get('nationality')),('FOOT',d.get('foot')),('HEIGHT',d.get('height')),('CONTRACT',d.get('contract')),('VALUE',d.get('market_value'))]
    for i,(label,val) in enumerate(info):
        col=i%3; row=i//3; xx=555+col*119; yy=500-row*32
        txt(xx,yy,label,6.6,True,muted)
        txt(xx,yy-13,val or '—',9,True,maxw=110)
    # left decision
    verdict=d.get('verdict','Not Assessed');accent=VERDICT_COLORS.get(verdict,VERDICT_COLORS['Not Assessed'])
    rect(16,304,521,134,accent)
    txt(30,416,'RECRUITMENT RECOMMENDATION',8,True,fill='#eafaf7')
    txt(30,385,verdict,25,True,fill='#ffffff',maxw=285)
    txt(338,413,'CURRENT',7,True,fill='#e4f1ef');txt(424,413,'POTENTIAL',7,True,fill='#e4f1ef')
    def ability(key):
        try:
            v=float(d.get(key));return f'{v:g}' if 0<=v<=20 else '—'
        except (ValueError,TypeError):return '—'
    txt(337,385,ability('current_ability'),25,True,'#ffffff');txt(383,389,'/20',9,fill='#e4f1ef')
    txt(423,385,ability('potential'),25,True,'#ffffff');txt(469,389,'/20',9,fill='#e4f1ef')
    txt(30,359,'PROJECTED LEVEL',7,True,fill='#e4f1ef')
    txt(30,345,d.get('projected_level') or 'Not assessed',9,True,maxw=472)
    paragraph(30,326,d.get('fit') or d.get('projection') or d.get('next_action'),476,7.8,11,2,fill='#ffffff')
    # Summary
    rect(16,211,521,84)
    txt(28,277,'SCOUTING SUMMARY',9,True)
    paragraph(28,260,d.get('summary') or 'Add your video scouting summary in the editor.',495,8.5,12,4)
    # Bottom three columns
    cols=[(16,172,'STRENGTHS','strengths','#29b99b'),(193,172,'AREAS TO DEVELOP','weaknesses','#e6a856'),(370,167,'RECRUITMENT RISKS','risks','#e0788a')]
    for xx,ww,title,key,col in cols:
        rect(xx,58,ww,144)
        txt(xx+10,184,title,8,True,fill=col)
        bullets(xx+10,168,d.get(key) if key!='risks' or d.get('show_risks',True) else '',ww-20,5,col)
    # Right chart panel
    rect(546,58,398,380)
    txt(561,416,'DATA PROFILE',11,True)
    txt(561,401,f"{d.get('role','')}  ·  {d.get('depth','')}",8,fill=muted,maxw=355)
    cx,cy=745,245; r0=27; r1=122
    rows=list(metric_rows[:15]) if d.get('show_chart',True) else []
    groupmap={}
    for gi,(gname,metrics) in enumerate((groups or {}).items()):
        for metric in metrics:groupmap[metric]=(gname,gi)
    legend=[]
    if rows:
        n=len(rows);gap=0.045 if n>7 else 0.07
        for i,(metric,pct,raw) in enumerate(rows):
            gname,gi=groupmap.get(metric,('Other',5));shade=COLORS[gi%len(COLORS)]
            if gname not in [z[0] for z in legend]:legend.append((gname,shade))
            angle0=math.pi/2+2*math.pi*i/n+gap/2;angle1=math.pi/2+2*math.pi*(i+1)/n-gap/2
            def wedge(rad1,rad2,fill):
                p=c.beginPath();p.moveTo(cx+rad1*math.cos(angle0),cy+rad1*math.sin(angle0))
                steps=7
                for j in range(steps+1):
                    a=angle0+(angle1-angle0)*j/steps;p.lineTo(cx+rad2*math.cos(a),cy+rad2*math.sin(a))
                for j in range(steps,-1,-1):
                    a=angle0+(angle1-angle0)*j/steps;p.lineTo(cx+rad1*math.cos(a),cy+rad1*math.sin(a))
                p.close();c.setFillColor(color(fill));c.drawPath(p,stroke=0,fill=1)
            wedge(r0,r1,'#2d3c4a')
            pct=max(0,min(100,float(pct)))
            wedge(r0,r0+(r1-r0)*pct/100,shade)
            a=(angle0+angle1)/2; rr=r0+(r1-r0)*pct/100
            tx=cx+(min(r1-10,max(r0+15,rr-11)))*math.cos(a)
            ty=cy+(min(r1-10,max(r0+15,rr-11)))*math.sin(a)
            c.setFillColor(color('#0b1520'));c.circle(tx,ty,10,stroke=0,fill=1)
            c.setFillColor(color('#ffffff'));c.setFont('Helvetica-Bold',7)
            c.drawCentredString(tx,ty-2.4,str(round(pct)))
            # short metric labels outside perimeter, positioned to avoid overly long text
            label=' '.join(metric.replace(' per 90','/90').split())
            if len(label)>23:label=label[:21]+'..'
            lr=r1+12;lx=cx+lr*math.cos(a);ly=cy+lr*math.sin(a)
            # keep all labels inside panel, with constrained line width
            lx=max(558,min(925,lx))
            c.setFillColor(color(muted));c.setFont('Helvetica',5.7)
            if math.cos(a)<-0.1:c.drawRightString(lx,ly,label)
            elif math.cos(a)>0.1:c.drawString(lx,ly,label)
            else:c.drawCentredString(lx,ly,label)
        c.setFillColor(color(bg));c.circle(cx,cy,r0-3,stroke=0,fill=1)
        txt(cx-17,cy+3,'KPI',10,True)
        txt(cx-16,cy-11,'0-100',6,fill=muted)
    else:
        txt(632,249,'No percentile metrics available',10,fill=muted)
    # Legend / role score
    legend=legend[:5]
    for i,(name,col) in enumerate(legend):
        xx=560+(i%3)*125; yy=115-(i//3)*14
        c.setFillColor(color(col));c.circle(xx,yy+2,3,stroke=0,fill=1)
        txt(xx+8,yy,name[:19],6.6,maxw=108)
    c.setStrokeColor(color(line));c.line(559,87,931,87)
    txt(561,73,'ROLE SCORE',7,True,muted)
    txt(628,72,str(d.get('display_score') or 'N/A')+('*' if d.get('score_is_manual') else ''),12,True,fill='#55d7ae')
    txt(738,73,'PERCENTILE',7,True,muted)
    txt(815,72,str(d.get('display_percentile') or 'N/A')+('*' if d.get('percentile_is_manual') else ''),12,True,fill='#55d7ae')
    # footer
    txt(17,37,f"SCOUT  {d.get('scout') or '—'}    |    {d.get('organization') or 'Recruitment'}",7,fill=muted,maxw=370)
    txt(17,23,f"REPORT  {d.get('report_date','')}    |    SOURCE  Wyscout / scout observations",7,fill=muted,maxw=540)
    txt(562,37,f"BENCHMARK  n={d.get('benchmark_n',0)}    |    ROLE  {d.get('role','')}",7,fill=muted,maxw=375)
    if d.get('score_is_manual') or d.get('percentile_is_manual'):
        txt(562,23,'* Displayed analytical value manually overridden',6.5,fill='#e6a856')
    else:txt(562,23,'CONFIDENTIAL  ·  INTERNAL RECRUITMENT USE',6.5,fill=muted)
    c.showPage();c.save();return buf.getvalue()

def poster_png(pdf_bytes,scale=2):
    """Rasterize vector PDF for preview and PNG export; PyMuPDF optional."""
    import fitz
    doc=fitz.open(stream=pdf_bytes,filetype='pdf')
    try:return doc[0].get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).tobytes('png')
    finally:doc.close()
