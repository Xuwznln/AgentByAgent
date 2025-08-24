"""
Molecular Dynamics Simulation Tools for xTB
Generates aspirin structure, runs xTB MD simulation, and creates PyMOL visualization
"""

import os
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np


def generate_aspirin_structure():
    """
    Generate aspirin molecule structure using RDKit
    Returns: RDKit molecule object and SDF content
    """
    # Aspirin SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(aspirin_smiles)
    if mol is None:
        raise ValueError("Failed to create aspirin molecule from SMILES")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Convert to SDF format
    sdf_content = Chem.MolToMolBlock(mol)
    
    print(f"Generated aspirin molecule with {mol.GetNumAtoms()} atoms")
    print(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(mol)}")
    
    return mol, sdf_content


def create_xtb_input_files(sdf_content, output_dir):
    """
    Create input files for xTB molecular dynamics simulation
    """
    # Write SDF structure file
    struct_file = os.path.join(output_dir, "aspirin.sdf")
    with open(struct_file, 'w') as f:
        f.write(sdf_content)
    
    # Create xTB input file for MD simulation
    input_file = os.path.join(output_dir, "md.inp")
    md_input = """$md
   temp=300.0
   time=10.0
   step=1.0
   shake=2
   nvt=true
$end
"""
    
    with open(input_file, 'w') as f:
        f.write(md_input)
    
    return struct_file, input_file


def run_xtb_md_simulation(xtb_path, struct_file, input_file, output_dir, num_threads=4):
    """
    Run xTB molecular dynamics simulation
    """
    print(f"Starting xTB MD simulation with {num_threads} threads...")
    
    # Set environment variables
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(num_threads)
    
    # Run xTB command
    cmd = [
        xtb_path,
        struct_file,
        "--input", input_file,
        "--md",
        "--gfn", "0",
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=output_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        print("xTB Output:")
        print(result.stdout)
        
        if result.stderr:
            print("xTB Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"xTB failed with return code {result.returncode}")
        
        # Check for trajectory file
        trj_file = os.path.join(output_dir, "xtb.trj")
        if not os.path.exists(trj_file):
            raise FileNotFoundError("Trajectory file xtb.trj not found")
        
        print(f"MD simulation completed successfully!")
        print(f"Trajectory file: {trj_file}")
        
        return trj_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("xTB simulation timed out")
    except Exception as e:
        raise RuntimeError(f"xTB simulation failed: {str(e)}")


def convert_trj_to_xyz(trj_file, output_dir):
    """
    Convert xTB trajectory file to XYZ format for PyMOL
    """
    print("Converting trajectory to XYZ format...")
    
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
                    f_out.write(f"Frame {frame_count}\n")
                    
                    # Write atomic coordinates
                    for j in range(i + 2, i + 2 + num_atoms):
                        if j < len(lines):
                            f_out.write(lines[j])
                    
                    frame_count += 1
                    i += num_atoms + 2
                else:
                    i += 1
            
            print(f"Converted {frame_count} frames to XYZ format")
            
    except Exception as e:
        # If direct conversion fails, try alternative approach
        print("Direct conversion failed, trying alternative method...")
        
        # Use a simple approach assuming TRJ is in XYZ-like format
        with open(trj_file, 'r') as f_in:
            content = f_in.read()
            
        with open(xyz_file, 'w') as f_out:
            f_out.write(content)
    
    return xyz_file


def create_pymol_script(xyz_file, output_dir, gif_name="md_animation.gif"):
    """
    Create PyMOL script to generate animated GIF from trajectory
    """
    gif_file = os.path.join(output_dir, gif_name)
    script_file = os.path.join(output_dir, "make_gif.pml")
    
    pymol_script = f"""
# Load trajectory
load {xyz_file}, aspirin_traj

# Set up visualization
hide everything
show sticks, aspirin_traj
color atomic
set stick_radius, 0.1
set sphere_scale, 0.3

# Set up camera and lighting
center aspirin_traj
zoom aspirin_traj, 5
set antialias, 2
set ray_shadows, 0

# Create animation settings
set movie_fps, 10
set movie_rock, 0

# Generate frames and save as GIF
python
import pymol
from pymol import cmd

# Get number of states
num_states = cmd.count_states("aspirin_traj")
print(f"Number of frames: {{num_states}}")

# Save individual PNG frames
for i in range(1, num_states + 1):
    cmd.frame(i)
    cmd.png(f"{output_dir}/frame_{{i:04d}}.png", width=800, height=600, dpi=150, ray=1)

python end

# Exit PyMOL
quit
"""
    
    with open(script_file, 'w') as f:
        f.write(pymol_script)
    
    return script_file, gif_file


def run_pymol_script(script_file):
    """
    Run PyMOL script to generate animation frames
    """
    print("Running PyMOL script...")
    
    try:
        # Try different PyMOL executable names
        pymol_commands = ['pymol', 'PyMOL', '/Applications/PyMOL.app/Contents/bin/pymol']
        
        for pymol_cmd in pymol_commands:
            try:
                result = subprocess.run(
                    [pymol_cmd, '-c', script_file],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print("PyMOL script executed successfully!")
                    print(result.stdout)
                    return True
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        raise RuntimeError("Could not find or execute PyMOL")
        
    except Exception as e:
        raise RuntimeError(f"PyMOL execution failed: {str(e)}")


def create_gif_from_frames(output_dir, gif_name="md_animation.gif"):
    """
    Create GIF from PNG frames using ImageIO or PIL
    """
    import glob
    
    try:
        # Try to use imageio first
        try:
            import imageio
            
            frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
            if not frame_files:
                raise FileNotFoundError("No frame files found")
            
            print(f"Creating GIF from {len(frame_files)} frames...")
            
            gif_file = os.path.join(output_dir, gif_name)
            with imageio.get_writer(gif_file, mode='I', duration=0.1) as writer:
                for frame_file in frame_files:
                    image = imageio.imread(frame_file)
                    writer.append_data(image)
            
            print(f"GIF created successfully: {gif_file}")
            return gif_file
            
        except ImportError:
            # Fallback to PIL
            from PIL import Image
            
            frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
            if not frame_files:
                raise FileNotFoundError("No frame files found")
            
            frames = []
            for frame_file in frame_files:
                img = Image.open(frame_file)
                frames.append(img)
            
            gif_file = os.path.join(output_dir, gif_name)
            frames[0].save(
                gif_file,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
            
            print(f"GIF created successfully: {gif_file}")
            return gif_file
            
    except Exception as e:
        print(f"Failed to create GIF: {str(e)}")
        return None


def run_complete_md_simulation(xtb_path="/Users/yunhong/miniconda3/envs/matdata/bin/xtb", 
                               num_threads=4, 
                               output_dir=None):
    """
    Complete workflow: Generate aspirin, run xTB MD, create PyMOL animation
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="aspirin_md_")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Working directory: {output_dir}")
    
    try:
        # Step 1: Generate aspirin structure
        print("=" * 50)
        print("Step 1: Generating aspirin structure...")
        mol, sdf_content = generate_aspirin_structure()
        
        # Step 2: Create xTB input files
        print("=" * 50)
        print("Step 2: Creating xTB input files...")
        struct_file, input_file = create_xtb_input_files(sdf_content, output_dir)
        
        # Step 3: Run xTB MD simulation
        print("=" * 50)
        print("Step 3: Running xTB molecular dynamics simulation...")
        trj_file = run_xtb_md_simulation(xtb_path, struct_file, input_file, output_dir, num_threads)
        
        # Step 4: Convert trajectory to XYZ
        print("=" * 50)
        print("Step 4: Converting trajectory to XYZ format...")
        xyz_file = convert_trj_to_xyz(trj_file, output_dir)
        
        # Step 5: Create PyMOL script
        print("=" * 50)
        print("Step 5: Creating PyMOL visualization script...")
        script_file, gif_file = create_pymol_script(xyz_file, output_dir)
        
        # Step 6: Run PyMOL to generate frames
        print("=" * 50)
        print("Step 6: Running PyMOL to generate animation frames...")
        run_pymol_script(script_file)
        
        # Step 7: Create GIF
        print("=" * 50)
        print("Step 7: Creating animated GIF...")
        final_gif = create_gif_from_frames(output_dir)
        
        print("=" * 50)
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print(f"Output directory: {output_dir}")
        print(f"Trajectory file: {trj_file}")
        print(f"XYZ trajectory: {xyz_file}")
        print(f"PyMOL script: {script_file}")
        if final_gif:
            print(f"Animated GIF: {final_gif}")
        
        return {
            'output_dir': output_dir,
            'trajectory_file': trj_file,
            'xyz_file': xyz_file,
            'pymol_script': script_file,
            'gif_file': final_gif,
            'structure_file': struct_file
        }
        
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        print(f"Output directory (for debugging): {output_dir}")
        raise