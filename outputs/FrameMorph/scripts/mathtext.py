#!/usr/bin/env python3
"""Rasterize math notation only. The animation engine is C++/CUDA."""
import json,sys,os,tempfile
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'framemorph-mpl-'+str(os.getuid())))
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.transparent']=True
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
request=json.load(open(sys.argv[1],encoding='utf-8'))
expression=request['expression'].strip()
if len(expression)>4096:raise ValueError('Equation is too long')
if not (expression.startswith('$') and expression.endswith('$')):expression='$'+expression+'$'
math_to_image(expression,sys.argv[2],prop=FontProperties(size=request.get('size',48)*72/96),dpi=96,color=request.get('color','#4ed5b8'),format='png')
