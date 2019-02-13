#python code for generating flowchart using pydot
#by Ayan, June 2018

import pydot
import os
HOME = os.getenv('HOME')+'/'
from cStringIO import StringIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

outpath = HOME + 'Dropbox/papers/enzo_paper/Figs/'
intermediate_color = 'chartreuse'
input_color = 'coral'
step_color = 'deepskyblue'
final_color = 'yellow'
fs = 20
fn = 'Tines-Roman'

graph = pydot.Dot(graph_type='digraph', ratio=0.6) # initialise graph

node_sim = pydot.Node('Simulated galaxy', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=intermediate_color) # define nodes
node_mod = pydot.Node('Model HII regions around young\nstar cluster, Assume Z gradient', fixedsize=True, width=3.2, height=0.8, style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=intermediate_color)
node_grid = pydot.Node('Creating 4D MAPPINGS HII region model grid', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=input_color)
node_parm = pydot.Node('HII region parameters - age,\nionisation parameter, density, metallicity', fixedsize=True, width=3.8, height=0.8, style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=intermediate_color) # define nodes
node_look = pydot.Node('Lookup MAPPINGS grid', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=step_color)
node_cont = pydot.Node('Stellar continuum', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=input_color)
node_fl = pydot.Node('Emission line fluxes', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=intermediate_color)
node_spec = pydot.Node('Make spectra along each spaxel', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=step_color)
node_vres = pydot.Node('Spectral resolution', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=input_color)
node_bin = pydot.Node('Spectral binning', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=step_color)
node_res = pydot.Node('Spatial resolution', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=input_color)
node_conv = pydot.Node('Spatial (PSF) convolution', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=step_color)
node_snr = pydot.Node('Noise model', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=input_color)
node_noise = pydot.Node('Add noise', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=step_color)
node_fitspec = pydot.Node('Fit spectrum along each pixel', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=step_color)
node_diag = pydot.Node('Metallicity diagnostic', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=input_color)
node_emap = pydot.Node('Emission line maps', style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=intermediate_color)
node_mmap = pydot.Node('FINAL OUTPUT- Metallicity maps/gradients', fixedsize=True, width=5.2, style='"rounded,filled"', shape='box', fontsize=fs, fontname=fn, fillcolor=final_color)

graph.add_node(node_sim) # add nodes
graph.add_node(node_mod)
graph.add_node(node_grid)
graph.add_node(node_parm)
graph.add_node(node_look)
graph.add_node(node_cont)
graph.add_node(node_fl)
graph.add_node(node_spec)
graph.add_node(node_vres)
graph.add_node(node_bin)
graph.add_node(node_res)
graph.add_node(node_conv)
graph.add_node(node_snr)
graph.add_node(node_noise)
graph.add_node(node_fitspec)
graph.add_node(node_diag)
graph.add_node(node_emap)
graph.add_node(node_mmap)

graph.add_edge(pydot.Edge(node_sim, node_mod)) # add connections
graph.add_edge(pydot.Edge(node_mod, node_parm))
graph.add_edge(pydot.Edge(node_grid, node_look))
graph.add_edge(pydot.Edge(node_parm, node_look))
graph.add_edge(pydot.Edge(node_look, node_fl))
graph.add_edge(pydot.Edge(node_cont, node_spec))
graph.add_edge(pydot.Edge(node_fl, node_spec))
graph.add_edge(pydot.Edge(node_spec, node_bin))
graph.add_edge(pydot.Edge(node_vres, node_bin))
graph.add_edge(pydot.Edge(node_bin, node_conv))
graph.add_edge(pydot.Edge(node_res, node_conv))
graph.add_edge(pydot.Edge(node_conv, node_noise))
graph.add_edge(pydot.Edge(node_snr, node_noise))
graph.add_edge(pydot.Edge(node_noise, node_fitspec))
graph.add_edge(pydot.Edge(node_fitspec, node_emap))
graph.add_edge(pydot.Edge(node_diag, node_mmap))
graph.add_edge(pydot.Edge(node_emap, node_mmap))

graph.write(outpath + 'flowchart.eps', format='eps') # save graph
print 'Saved file', outpath + 'flowchart.eps'

png_str = graph.create_png(prog='dot')
sio = StringIO()
sio.write(png_str)
sio.seek(0)
plt.close('all')
fig = plt.figure(figsize=(10,8))
img = mpimg.imread(sio)
imgplot = plt.imshow(img)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.show(block=False)