# PyMOL script for antibody-antigen visualization
# Generated automatically for PDB: 1ADQ

# Load structure
load output/1ADQ.pdb, 1ADQ

# Remove water molecules
remove resn HOH

# Create selections
select heavy_chain, chain H
select light_chain, chain L
select antigen, chain A

# Set up visualization
hide everything
show cartoon

# Color scheme
color blue, heavy_chain
color cyan, light_chain
color red, antigen

# Show important structural features
show sticks, name CA and (heavy_chain or light_chain) and resi 1-30
show surface, antigen
set transparency, 0.3, antigen

# Set up high-quality rendering
set antialias, 2
set ray_trace_mode, 1
set ray_shadows, 1
set specular, 1
set shininess, 10
set depth_cue, 1
set ray_opaque_background, 0

# Optimize view
orient
zoom
turn x, -30
turn y, 15

# Set high resolution
viewport 1920, 1440

# Render and save high-quality image
ray 2400, 1800
png output/1ADQ_final_visualization.png, dpi=300

# Clean up
quit
