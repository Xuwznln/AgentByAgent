#!/usr/bin/env python3
"""
Final step: Create PyMOL visualization from xTB trajectory
"""

import os
import sys
import subprocess
import glob

# Add the tool directory to Python path
sys.path.insert(0, '/Users/yunhong/Desktop/AgentByAgent/tools/molecular_dynamics_xtb/venv/lib/python3.12/site-packages')
import imageio


def convert_trj_to_xyz(trj_file, output_dir):
    """Convert xTB trajectory file to XYZ format for PyMOL"""
    print("=" * 50)
    print("Step 1: Converting trajectory to XYZ format...")
    
    xyz_file = os.path.join(output_dir, "trajectory.xyz")
    
    try:
        with open(trj_file, 'r') as f_in, open(xyz_file, 'w') as f_out:
            lines = f_in.readlines()
            
            i = 0
            frame_count = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for number of atoms (first line of each frame)
                if line.isdigit():
                    num_atoms = int(line)
                    
                    # Write XYZ header
                    f_out.write(f"{num_atoms}\n")
                    f_out.write(f"Frame {frame_count} - time = {frame_count * 1.0} fs\n")
                    
                    # Write atomic coordinates (skip comment line)
                    i += 1
                    if i < len(lines) and not lines[i].strip().isdigit():
                        i += 1  # Skip comment line if present
                    
                    # Write coordinate lines
                    coord_count = 0
                    while coord_count < num_atoms and i < len(lines):
                        coord_line = lines[i].strip()
                        if coord_line and not coord_line.isdigit():
                            f_out.write(lines[i])
                            coord_count += 1
                        i += 1
                    
                    frame_count += 1
                else:
                    i += 1
            
            print(f"Converted {frame_count} frames to XYZ format")
            
    except Exception as e:
        print(f"Conversion failed: {e}")
        # Alternative approach - treat as direct XYZ
        print("Trying direct copy approach...")
        with open(trj_file, 'r') as f_in:
            content = f_in.read()
        
        with open(xyz_file, 'w') as f_out:
            f_out.write(content)
    
    return xyz_file


def create_pymol_script(xyz_file, output_dir):
    """Create PyMOL script to generate animated GIF from trajectory"""
    print("=" * 50)
    print("Step 2: Creating PyMOL visualization script...")
    
    script_file = os.path.join(output_dir, "make_animation.pml")
    
    # Use absolute paths
    abs_xyz_file = os.path.abspath(xyz_file)
    abs_output_dir = os.path.abspath(output_dir)
    
    pymol_script = f"""# Load trajectory
load {abs_xyz_file}, aspirin_md

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
print(f"Number of frames: {{num_states}}")

if num_states > 1:
    # Save individual PNG frames (every 2nd frame to reduce file count)
    for i in range(1, min(num_states + 1, 51), 2):  # Limit to 25 frames max
        cmd.frame(i)
        frame_file = f"{abs_output_dir}/frame_{{i:04d}}.png"
        cmd.png(frame_file, width=600, height=400, dpi=150, ray=1)
        print(f"Saved frame {{i}}")
    
    # Also save a single high-quality image
    cmd.frame(1)
    cmd.png(f"{abs_output_dir}/aspirin_snapshot.png", width=1200, height=800, dpi=300, ray=1)
    print("Saved high-quality snapshot")
else:
    print("Only one frame found, saving single image")
    cmd.png(f"{abs_output_dir}/aspirin_structure.png", width=800, height=600, dpi=150, ray=1)

python end

# Exit PyMOL
quit
"""
    
    with open(script_file, 'w') as f:
        f.write(pymol_script)
    
    print(f"PyMOL script created: {script_file}")
    return script_file


def run_pymol_script(script_file):
    """Run PyMOL script to generate animation frames"""
    print("=" * 50)
    print("Step 3: Running PyMOL to generate animation frames...")
    
    try:
        # Try different PyMOL executable names
        pymol_commands = ['pymol', 'PyMOL', '/Applications/PyMOL.app/Contents/bin/pymol']
        
        for pymol_cmd in pymol_commands:
            try:
                print(f"Trying PyMOL command: {pymol_cmd}")
                result = subprocess.run(
                    [pymol_cmd, '-c', script_file],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                print(f"PyMOL return code: {result.returncode}")
                if result.stdout:
                    print("PyMOL stdout:")
                    print(result.stdout)
                
                if result.stderr:
                    print("PyMOL stderr:")
                    print(result.stderr)
                
                if result.returncode == 0 or "Saved frame" in result.stdout:
                    print("PyMOL script executed successfully!")
                    return True
                    
            except FileNotFoundError:
                print(f"PyMOL command '{pymol_cmd}' not found")
                continue
            except subprocess.TimeoutExpired:
                print(f"PyMOL command '{pymol_cmd}' timed out")
                continue
        
        print("Warning: Could not execute PyMOL")
        return False
        
    except Exception as e:
        print(f"PyMOL execution failed: {str(e)}")
        return False


def create_gif_from_frames(output_dir, gif_name="aspirin_md_animation.gif"):
    """Create GIF from PNG frames using imageio"""
    print("=" * 50)
    print("Step 4: Creating animated GIF...")
    
    try:
        frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
        if not frame_files:
            print("No frame files found, checking for single structure image...")
            single_image = os.path.join(output_dir, "aspirin_structure.png")
            if os.path.exists(single_image):
                print(f"Found single structure image: {single_image}")
                return single_image
            else:
                print("No PNG files found")
                return None
        
        print(f"Creating GIF from {len(frame_files)} frames...")
        
        gif_file = os.path.join(output_dir, gif_name)
        with imageio.get_writer(gif_file, mode='I', duration=0.25) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)
        
        print(f"GIF created successfully: {gif_file}")
        return gif_file
        
    except Exception as e:
        print(f"Failed to create GIF: {str(e)}")
        return None


def main():
    """Main visualization workflow"""
    output_dir = "/tmp/aspirin_md"
    trj_file = os.path.join(output_dir, "xtb.trj")
    
    print("Starting Aspirin MD Visualization")
    print(f"Output directory: {output_dir}")
    print(f"Trajectory file: {trj_file}")
    
    if not os.path.exists(trj_file):
        print(f"Error: Trajectory file not found: {trj_file}")
        return
    
    try:
        # Step 1: Convert trajectory to XYZ
        xyz_file = convert_trj_to_xyz(trj_file, output_dir)
        
        # Step 2: Create PyMOL script
        script_file = create_pymol_script(xyz_file, output_dir)
        
        # Step 3: Run PyMOL to generate frames
        pymol_success = run_pymol_script(script_file)
        
        # Step 4: Create GIF
        gif_file = create_gif_from_frames(output_dir)
        
        print("=" * 50)
        print("VISUALIZATION COMPLETED!")
        print(f"Output directory: {output_dir}")
        print(f"XYZ trajectory: {xyz_file}")
        print(f"PyMOL script: {script_file}")
        if gif_file:
            print(f"Animation: {gif_file}")
        
        # List all PNG and GIF files
        print("\nGenerated files:")
        for f in sorted(os.listdir(output_dir)):
            if f.endswith(('.png', '.gif', '.xyz')):
                file_path = os.path.join(output_dir, f)
                size = os.path.getsize(file_path)
                print(f"  {f} ({size} bytes)")
        
        return {
            'xyz_trajectory': xyz_file,
            'pymol_script': script_file,
            'animation_file': gif_file
        }
        
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()