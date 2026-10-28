#!/usr/bin/env python3
"""Optional construction of portable formula, code and SVG artwork. Never runs source code."""
import io, json, os, re, shutil, subprocess, sys, tempfile, tokenize, keyword
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'framemorph-mpl-'+str(os.getuid())))
from PIL import Image, ImageDraw, ImageFont

def checked_image(image):
    if image.width < 1 or image.height < 1 or image.width > 8192 or image.height > 8192 or image.width*image.height > 16000000:
        raise ValueError('Artwork exceeds the 16 megapixel / 8192 pixel limit. Reduce font size or content.')
    return image.convert('RGBA')

def normalize_math(text):
    text=text.strip()
    for left,right in [('$$','$$'),('$','$'),(r'\[',r'\]'),(r'\(',r'\)')]:
        if text.startswith(left) and text.endswith(right): text=text[len(left):-len(right)].strip();break
    return text

def equation(request):
    text=normalize_math(request.get('content',request.get('expression','')))
    if not text or len(text)>16000: raise ValueError('Enter a formula of 1 to 16000 characters.')
    backend=request.get('backend','auto'); size=float(request.get('font_size',request.get('size',48)))
    color=request.get('color','#1d293d')
    if not re.fullmatch(r'#[0-9a-fA-F]{6}',color): raise ValueError('Use a color like #386fe8.')
    if not 6<=size<=600: raise ValueError('Font size must be 6 to 600 pixels.')
    if backend in ('auto','mathtext'):
        try:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['savefig.transparent']=True
            from matplotlib.mathtext import math_to_image
            from matplotlib.font_manager import FontProperties
            buffer=io.BytesIO()
            math_to_image('$'+text+'$',buffer,prop=FontProperties(size=size*72/96),dpi=96,color=color,format='png')
            return checked_image(Image.open(buffer))
        except Exception:
            if backend=='mathtext': raise ValueError('This formula needs Full LaTeX, or has invalid math notation. Select Auto or Full LaTeX.')
    elif backend!='latex': raise ValueError('Unknown equation backend.')
    import fitz
    binary=shutil.which('pdflatex')
    if not binary:
        candidate=Path.home()/'.TinyTeX/bin/x86_64-linux/pdflatex'
        if candidate.is_file(): binary=str(candidate)
    if not binary: raise ValueError('Full LaTeX needs pdflatex and PyMuPDF. Fast Mathtext remains available.')
    # A formula fragment, not a document or arbitrary TeX program.
    blocked=r'\\(?:input|include|write|read|openin|openout|documentclass|usepackage|catcode|csname|def|edef|gdef|let|loop|repeat|directlua|special|immediate|shipout|pdfobj|pdffile|every|newread|newwrite)\b'
    if re.search(blocked,text,re.I): raise ValueError('Enter a mathematical fragment only; file commands, macros and document preambles are not supported.')
    text=re.sub(r'\\begin\{align\*?\}',r'\\begin{aligned}',text)
    text=re.sub(r'\\end\{align\*?\}',r'\\end{aligned}',text)
    document=(r'\documentclass[border=2pt]{standalone}'+'\n'+r'\usepackage{amsmath,amssymb,xcolor}'+'\n'+r'\begin{document}'+'\n'+
              r'\definecolor{fmcolor}{HTML}{'+color[1:]+'}\n'+r'\color{fmcolor}\fontsize{'+str(size*72/96)+'}{'+str(size)+'}\\selectfont\n'+
              r'$\displaystyle '+text+'$\n'+r'\end{document}')
    with tempfile.TemporaryDirectory(prefix='tex-',dir=str(Path(sys.argv[2]).resolve().parent)) as folder:
        root=Path(folder); (root/'formula.tex').write_text(document)
        env=dict(os.environ);env['openout_any']='p';env['shell_escape']='f';env['max_print_line']='200'
        try:
            result=subprocess.run([binary,'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','formula.tex'],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
        except subprocess.TimeoutExpired: raise ValueError('LaTeX took longer than 20 seconds. Simplify the formula.')
        if result.returncode or not (root/'formula.pdf').is_file():
            output=result.stdout.decode('utf-8','replace'); errors=[line for line in output.splitlines() if line.startswith('!') or line.startswith('l.')]
            raise ValueError('LaTeX: '+' '.join(errors[-4:])[:800])
        with fitz.open(root/'formula.pdf') as doc:
            page=doc[0]
            if page.rect.width*page.rect.height*(96/72)**2>16000000: raise ValueError('Formula is too large. Reduce font size.')
            pix=page.get_pixmap(dpi=96,alpha=True)
            return checked_image(Image.frombytes('RGBa',(pix.width,pix.height),pix.samples).convert('RGBA'))

def code(request):
    text=request.get('content','').replace('\t','    ')
    if not text or len(text)>32000 or len(text.splitlines())>200: raise ValueError('Code blocks need 1 to 32000 characters and at most 200 lines.')
    size=int(request.get('font_size',28))
    if not 8<=size<=150: raise ValueError('Code font size must be 8 to 150 pixels.')
    theme=request.get('theme','dark'); dark=theme!='light'
    language=request.get('language','python')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',size)
    advance=font.getlength('M'); row=int(size*1.5); pad=int(size*.8)
    lines=text.split('\n'); numbers=request.get('line_numbers',True)
    gutter=int(advance*(len(str(len(lines)))+2)) if numbers else 0
    width=int(max(map(len,lines))*advance)+2*pad+gutter; height=len(lines)*row+2*pad
    if width>8192 or height>8192 or width*height>16000000: raise ValueError('Code block is too large. Reduce font size or line length.')
    image=Image.new('RGBA',(max(100,width),max(40,height)), '#111c2e' if dark else '#f1f5fb'); draw=ImageDraw.Draw(image)
    palette={'normal':'#dce5f5' if dark else '#25344d','keyword':'#b399ff' if dark else '#7741bd','string':'#8ed6a6' if dark else '#16824b','number':'#f2bd74' if dark else '#b05d0c','comment':'#8293ad' if dark else '#708096'}
    if numbers:
        for i in range(len(lines)): draw.text((pad,pad+i*row),str(i+1),font=font,fill=palette['comment'])
    for i,line in enumerate(lines): draw.text((pad+gutter,pad+i*row),line,font=font,fill=palette['normal'])
    if language=='python':
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                kind='keyword' if token.type==tokenize.NAME and keyword.iskeyword(token.string) else {tokenize.STRING:'string',tokenize.NUMBER:'number',tokenize.COMMENT:'comment'}.get(token.type)
                if not kind: continue
                for index,fragment in enumerate(token.string.split('\n')):
                    line=token.start[0]-1+index; col=token.start[1] if index==0 else 0
                    draw.text((pad+gutter+col*advance,pad+line*row),fragment,font=font,fill=palette[kind])
        except (tokenize.TokenError,IndentationError): pass # Incomplete snippets still preview.
    elif language!='plain':
        pattern=r'(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(?P<number>\b\d+(?:\.\d+)?\b)|(?P<keyword>\b(?:auto|int|float|double|return|if|else|for|while|class|struct|const|void|bool|true|false|let|var|function|import|export|from|new|public|private|include|template|typename|namespace|using)\b)'
        for token in re.finditer(pattern,text):
            line=text.count('\n',0,token.start()); col=token.start()-(text.rfind('\n',0,token.start())+1)
            for i,fragment in enumerate(token.group().split('\n')):
                draw.text((pad+gutter+(col if i==0 else 0)*advance,pad+(line+i)*row),fragment,font=font,fill=palette[token.lastgroup])
    return checked_image(image)

def svg(request):
    import fitz
    import xml.etree.ElementTree as ET
    text=request.get('content','')
    if not text or len(text)>2000000: raise ValueError('SVG must contain 1 byte to 2 MB.')
    root=ET.fromstring(text)
    if root.tag.split('}')[-1]!='svg': raise ValueError('The document is not SVG.')
    for node in root.iter():
        if node.tag.split('}')[-1] in ('script','foreignObject'): raise ValueError('Scripts and foreignObject are not supported in SVG artwork.')
        for key,value in node.attrib.items():
            if key.split('}')[-1]=='href' and not value.startswith(('#','data:')): raise ValueError('Embed SVG images; external file and network references are not supported.')
    with fitz.open(stream=text.encode(),filetype='svg') as source:
        data=source.convert_to_pdf()
    with fitz.open('pdf',data) as doc:
        page=doc[0]; target=float(request.get('svg_size',1200))
        if not 64<=target<=4000: raise ValueError('SVG detail must be 64 to 4000 pixels.')
        scale=target/max(page.rect.width,page.rect.height)
        pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=True)
        return checked_image(Image.frombytes('RGBa',(pix.width,pix.height),pix.samples).convert('RGBA'))

def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
    image={'equation':equation,'code':code,'svg':svg}[request.get('kind','equation')](request)
    image.save(sys.argv[2])
if __name__=='__main__':
    try: main()
    except Exception as exc:
        Path(sys.argv[2]+'.error').write_text(str(exc)[:3000],encoding='utf-8');sys.exit(1)
