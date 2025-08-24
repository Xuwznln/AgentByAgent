# Load trajectory
load /tmp/aspirin_md/trajectory.xyz, aspirin_md

# Set up visualization
hide everything
show sticks, aspirin_md
color atomic, aspirin_md
set stick_radius, 0.12
set sphere_scale, 0.25

# Set up camera and lighting  
center aspirin_md
zoom aspirin_md, 4
set antialias, 2
set ray_shadows, 1
set ambient, 0.3
set direct, 0.7

# Create animation settings
set movie_fps, 8

# Generate frames and save as PNG
python
import pymol
from pymol import cmd
import os

# Get number of states
num_states = cmd.count_states("aspirin_md")
print(f"Number of frames: {num_states}")

if num_states > 1:
    # Save individual PNG frames (every 2nd frame to reduce file count)
    for i in range(1, min(num_states + 1, 51), 2):  # Limit to 25 frames max
        cmd.frame(i)
        frame_file = f"/tmp/aspirin_md/frame_{i:04d}.png"
        cmd.png(frame_file, width=600, height=400, dpi=150, ray=1)
        print(f"Saved frame {i}")
    
    # Also save a single high-quality image
    cmd.frame(1)
    cmd.png(f"/tmp/aspirin_md/aspirin_snapshot.png", width=1200, height=800, dpi=300, ray=1)
    print("Saved high-quality snapshot")
else:
    print("Only one frame found, saving single image")
    cmd.png(f"/tmp/aspirin_md/aspirin_structure.png", width=800, height=600, dpi=150, ray=1)

python end

# Exit PyMOL
quit
