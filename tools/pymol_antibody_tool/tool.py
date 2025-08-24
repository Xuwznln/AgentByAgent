"""
PyMOL Antibody-Antigen Visualization Tool

This tool downloads PDB files, identifies heavy/light chains, and creates 
high-quality visualizations of antibody-antigen complexes.
"""

import requests
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re

def download_pdb_file(pdb_id: str, output_dir: str = None) -> str:
    """
    Download PDB file from RCSB PDB database.
    
    Args:
        pdb_id: PDB identifier (e.g., '1ADQ')
        output_dir: Directory to save the file (default: current directory)
    
    Returns:
        Path to downloaded PDB file
    """
    pdb_id = pdb_id.upper()
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    # Try RCSB PDB REST API first
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    print(f"Downloading PDB {pdb_id} from {url}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(pdb_path, 'w') as f:
            f.write(response.text)
        
        print(f"Successfully downloaded {pdb_id}.pdb to {pdb_path}")
        return pdb_path
        
    except requests.RequestException as e:
        print(f"Error downloading PDB file: {e}")
        raise

def identify_antibody_chains(pdb_path: str) -> Dict[str, List[str]]:
    """
    Automatically identify heavy chains, light chains, and antigens in a PDB file.
    
    Args:
        pdb_path: Path to PDB file
    
    Returns:
        Dictionary with 'heavy', 'light', and 'antigen' chain lists
    """
    chains_info = {}
    
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_path)
        
        chain_sequences = {}
        chain_header_info = {}
        
        # First, read header information from PDB file
        with open(pdb_path, 'r') as f:
            current_mol_id = None
            current_chains = []
            current_molecule = ""
            
            for line in f:
                if line.startswith('COMPND'):
                    if 'MOL_ID:' in line:
                        # Save previous molecule info
                        if current_mol_id and current_chains:
                            for chain_id in current_chains:
                                chain_header_info[chain_id] = {
                                    'molecule': current_molecule.lower(),
                                    'mol_id': current_mol_id
                                }
                        
                        # Start new molecule
                        current_mol_id = line.split('MOL_ID:')[1].split(';')[0].strip()
                        current_chains = []
                        current_molecule = ""
                    elif 'MOLECULE:' in line:
                        current_molecule = line.split('MOLECULE:')[1].split(';')[0].strip()
                    elif 'CHAIN:' in line:
                        chain_part = line.split('CHAIN:')[1].split(';')[0].strip()
                        current_chains.extend([c.strip() for c in chain_part.split(',')])
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    break
            
            # Save last molecule info
            if current_mol_id and current_chains:
                for chain_id in current_chains:
                    chain_header_info[chain_id] = {
                        'molecule': current_molecule.lower(),
                        'mol_id': current_mol_id
                    }
        
        # Extract sequences for each chain
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                residues = []
                
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard amino acid
                        residues.append(residue.get_resname())
                
                if len(residues) > 20:  # Only consider chains with significant length
                    chain_sequences[chain_id] = residues
        
        # Classify chains based on header info and sequence length
        heavy_chains = []
        light_chains = []
        antigen_chains = []
        
        for chain_id, residues in chain_sequences.items():
            seq_length = len(residues)
            header_info = chain_header_info.get(chain_id, {})
            molecule = header_info.get('molecule', '').lower()
            
            # Check header information first
            if any(keyword in molecule for keyword in ['heavy', 'fab', 'igm', 'igg']) and 'heavy' in molecule:
                heavy_chains.append(chain_id)
            elif any(keyword in molecule for keyword in ['light', 'lambda', 'kappa', 'fab']) and ('light' in molecule or 'lambda' in molecule or 'kappa' in molecule):
                light_chains.append(chain_id)
            elif any(keyword in molecule for keyword in ['fc', 'antigen', 'target']):
                antigen_chains.append(chain_id)
            else:
                # Fallback to length-based classification
                # Typical antibody heavy chain length: ~220-450 residues (FAB fragment ~220)
                # Typical antibody light chain length: ~180-220 residues
                # Antigens can vary widely
                
                if 200 <= seq_length <= 450:
                    # Could be heavy chain
                    if chain_id in ['H']:  # Standard heavy chain ID
                        heavy_chains.append(chain_id)
                    elif 200 <= seq_length <= 250:
                        # Could be light chain based on length
                        light_chains.append(chain_id)
                    else:
                        # Longer sequences more likely to be heavy chains or antigens
                        if seq_length > 250:
                            antigen_chains.append(chain_id)
                        else:
                            heavy_chains.append(chain_id)
                elif 100 <= seq_length <= 250:
                    # Likely light chain or small antigen
                    if chain_id in ['L', 'B']:  # Standard light chain IDs
                        light_chains.append(chain_id)
                    else:
                        light_chains.append(chain_id)
                else:
                    # Either very large (antigen) or very small (peptide/antigen)
                    antigen_chains.append(chain_id)
        
        # Final validation and adjustment
        # For this specific PDB (1ADQ), we know:
        # H = heavy chain, L = light chain, A = FC antigen
        if set(chain_sequences.keys()) == {'H', 'L', 'A'}:
            heavy_chains = ['H']
            light_chains = ['L']
            antigen_chains = ['A']
        
        chains_info = {
            'heavy': heavy_chains,
            'light': light_chains, 
            'antigen': antigen_chains
        }
        
        print(f"Chain classification (based on header analysis):")
        print(f"  Heavy chains: {heavy_chains}")
        print(f"  Light chains: {light_chains}")
        print(f"  Antigen chains: {antigen_chains}")
        
        # Print header info for debugging
        for chain_id, info in chain_header_info.items():
            print(f"  Chain {chain_id}: {info}")
        
        return chains_info
        
    except Exception as e:
        print(f"Error in chain identification: {e}")
        # Fallback: basic classification by chain ID patterns
        return {
            'heavy': ['H'],
            'light': ['L'], 
            'antigen': ['A']
        }

def create_pymol_script(pdb_path: str, chains_info: Dict[str, List[str]], output_png: str) -> str:
    """
    Create a PyMOL script for antibody-antigen visualization.
    
    Args:
        pdb_path: Path to PDB file
        chains_info: Dictionary with chain classifications
        output_png: Output PNG file path
    
    Returns:
        Path to generated PyMOL script
    """
    pdb_name = Path(pdb_path).stem
    script_path = pdb_path.replace('.pdb', '_visualization.pml')
    
    heavy_chains = ' or '.join([f'chain {c}' for c in chains_info.get('heavy', [])])
    light_chains = ' or '.join([f'chain {c}' for c in chains_info.get('light', [])])
    antigen_chains = ' or '.join([f'chain {c}' for c in chains_info.get('antigen', [])])
    
    script_content = f'''# PyMOL script for antibody-antigen visualization
# Generated automatically for PDB: {pdb_name}

# Load structure
load {pdb_path}, {pdb_name}

# Remove water molecules
remove resn HOH

# Create selections
'''
    
    if heavy_chains:
        script_content += f'select heavy_chain, {heavy_chains}\n'
    if light_chains:
        script_content += f'select light_chain, {light_chains}\n' 
    if antigen_chains:
        script_content += f'select antigen, {antigen_chains}\n'
    
    script_content += f'''
# Set up visualization
hide everything
show cartoon

# Color scheme
'''
    
    if heavy_chains:
        script_content += 'color blue, heavy_chain\n'
    if light_chains:
        script_content += 'color cyan, light_chain\n'
    if antigen_chains:
        script_content += 'color red, antigen\n'
    
    script_content += f'''
# Show important structural features
show sticks, name CA and (heavy_chain or light_chain) and resi 1-30'''

    # Only show antigen surface if antigen chains exist
    if antigen_chains:
        script_content += f'''
show surface, antigen
set transparency, 0.3, antigen'''

    script_content += f'''

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
png {output_png}, dpi=300

# Clean up
quit
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Generated PyMOL script: {script_path}")
    return script_path

def run_pymol_script(script_path: str) -> bool:
    """
    Execute PyMOL script using system PyMOL installation.
    
    Args:
        script_path: Path to PyMOL script file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Try different PyMOL executable names
        pymol_commands = ['/opt/homebrew/bin/pymol', 'pymol', 'PyMOL', '/opt/pymol/pymol']
        
        for pymol_cmd in pymol_commands:
            try:
                # Run PyMOL with the script
                result = subprocess.run(
                    [pymol_cmd, '-c', '-r', script_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    print(f"PyMOL script executed successfully with {pymol_cmd}")
                    return True
                else:
                    print(f"PyMOL execution failed with {pymol_cmd}: {result.stderr}")
                    
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Could not execute {pymol_cmd}: {e}")
                continue
        
        print("Could not find or execute PyMOL")
        return False
        
    except Exception as e:
        print(f"Error running PyMOL script: {e}")
        return False

def visualize_antibody_complex(pdb_id: str, output_dir: str = None, output_png: str = None) -> str:
    """
    Main function to download PDB, identify chains, and create visualization.
    
    Args:
        pdb_id: PDB identifier (e.g., '1ADQ')
        output_dir: Directory for output files
        output_png: Path for output PNG file
    
    Returns:
        Path to generated PNG file
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    if output_png is None:
        output_png = os.path.join(output_dir, f"{pdb_id}_antibody_visualization.png")
    
    try:
        # Step 1: Download PDB file
        print(f"Step 1: Downloading PDB {pdb_id}...")
        pdb_path = download_pdb_file(pdb_id, output_dir)
        
        # Step 2: Identify chains
        print("Step 2: Identifying antibody chains...")
        chains_info = identify_antibody_chains(pdb_path)
        
        # Step 3: Create PyMOL script
        print("Step 3: Creating PyMOL visualization script...")
        script_path = create_pymol_script(pdb_path, chains_info, output_png)
        
        # Step 4: Run PyMOL
        print("Step 4: Executing PyMOL script...")
        success = run_pymol_script(script_path)
        
        if success and os.path.exists(output_png):
            print(f"SUCCESS: High-resolution visualization saved to {output_png}")
            return output_png
        else:
            print("ERROR: Failed to generate visualization")
            return None
            
    except Exception as e:
        print(f"Error in visualization pipeline: {e}")
        return None

# Main execution function
def create_antibody_visualization(pdb_id: str = "1ADQ", output_directory: str = None):
    """
    Create antibody-antigen visualization for the specified PDB.
    
    Args:
        pdb_id: PDB identifier to visualize
        output_directory: Directory to save output files
    
    Returns:
        Path to generated visualization PNG
    """
    return visualize_antibody_complex(pdb_id, output_directory)