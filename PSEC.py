#!/usr/bin/env python3
# Author: Krishnendu Bera, Ph.D.
# Email: krishnendu39@gmil.com
# Date: 2025-04-03
# Description: This script extracts the protein sequence from a PDB file (either fetched from the PDB database or provided locally),
#              respects chain information, handles post-translational modifications (PTMs), cleans the PDB by removing non-standard residues,
#              calculates the total charge at a specified pH, generates a plot showing the charge dependence on pH, 
#              and saves the results to a single FASTA file and a charge information file.

import os
import requests
from Bio import PDB
import matplotlib.pyplot as plt
import numpy as np

# pKa values of amino acids for calculating charge at pH 7
default_pKa_values = {
    'D': 3.9,  # Aspartic acid (Asp)
    'E': 4.2,  # Glutamic acid (Glu)
    'C': 8.3,  # Cysteine (Cys)
    'Y': 10.1, # Tyrosine (Tyr)
    'K': 10.5, # Lysine (Lys)
    'R': 12.5, # Arginine (Arg)
    'H': 6.0   # Histidine (His)
}

# Set of standard amino acid residue codes (three-letter code)
standard_amino_acids = {
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
}

# Mapping three-letter residue codes to one-letter codes
three_to_one_letter = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# PTM modifications dictionary (this can be extended as needed)
ptm_charge_map = {
    'pS': -1,  # Phosphorylated Serine
    'pT': -1,  # Phosphorylated Threonine
    'pY': -1,  # Phosphorylated Tyrosine
    'Ac': 0,   # Acetylated (neutral charge)
    'Me': 0    # Methylation (neutral charge)
}

def extract_sequence(pdb_file):
    """
    Extracts the protein sequence from a PDB file, cleaning non-standard residues, and accounting for PTMs.

    Args:
    pdb_file (str): The path to the PDB file.

    Returns:
    dict: A dictionary with chain IDs as keys and their corresponding sequences as values.
    dict: Dictionary mapping PTM-modified residues to their one-letter codes.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('Protein', pdb_file)
    
    sequences = {}
    ptm_residues = {}

    # Parsing the MODRES section to get PTM information
    with open(pdb_file, 'r') as pdb_f:
        for line in pdb_f:
            if line.startswith('MODRES'):
                resname = line[12:15].strip()  # Modified residue name
                reschain = line[18:19].strip()  # Chain ID
                resseq = line[22:26].strip()    # Residue sequence number
                ptm_residues[(reschain, resseq)] = resname  # Map PTM residue to (chain, seq number)

    # Extracting sequence and handling PTMs
    for model in structure:
        for chain in model:
            chain_sequence = []
            for residue in chain:
                # Only add residues that are standard amino acids (ignoring non-standard residues like water, ligands)
                if residue.id[0] == ' ' and residue.get_resname() in standard_amino_acids:
                    three_letter = residue.get_resname()
                    if three_letter in three_to_one_letter:
                        # Get one-letter code for the residue
                        chain_sequence.append(three_to_one_letter[three_letter])
                        
                        # Check if this residue is PTM-modified
                        if (chain.get_id(), residue.get_id()[1]) in ptm_residues:
                            modified_residue = ptm_residues[(chain.get_id(), residue.get_id()[1])]
                            chain_sequence[-1] += modified_residue  # Append modification to residue

            sequences[chain.get_id()] = ''.join(chain_sequence)
    
    return sequences, ptm_residues

def save_fasta(sequences, output_dir):
    """
    Saves all chain sequences in one FASTA file.

    Args:
    sequences (dict): A dictionary with chain IDs as keys and their corresponding sequences as values.
    output_dir (str): The directory to save the FASTA file.
    """
    fasta_filename = os.path.join(output_dir, "all_chains_sequence.fasta")
    with open(fasta_filename, 'w') as fasta_file:
        for chain_id, seq in sequences.items():
            fasta_file.write(f">Chain_{chain_id}\n")
            fasta_file.write(seq + "\n")
    print(f"All sequences saved to: {fasta_filename}")

def save_charge_info(sequences, pKa_values, ptm_residues, pH, output_dir):
    """
    Saves the charge information for each chain in a text file.

    Args:
    sequences (dict): A dictionary with chain IDs as keys and their corresponding sequences as values.
    pKa_values (dict): The pKa values for the amino acids.
    ptm_residues (dict): PTM residues and their modifications.
    pH (float): The pH at which the charge is calculated.
    output_dir (str): The directory to save the charge information file.
    """
    charge_info_filename = os.path.join(output_dir, "charge_info.txt")
    with open(charge_info_filename, 'w') as charge_file:
        for chain_id, seq in sequences.items():
            total_charge = calculate_charge(seq, pH, pKa_values, ptm_residues)
            charge_file.write(f"Chain {chain_id} Charge at pH {pH}: {total_charge}\n")
    print(f"Charge information saved to: {charge_info_filename}")

def calculate_charge(sequence, pH, pKa_values, ptm_residues):
    """
    Calculates the total charge of the protein at a given pH, accounting for PTMs.

    Args:
    sequence (str): The amino acid sequence of the protein.
    pH (float): The pH at which to calculate the charge.
    pKa_values (dict): The pKa values for the amino acids.
    ptm_residues (dict): PTM residues and their modifications.

    Returns:
    float: The total charge of the protein at the given pH.
    """
    charge = 0.0
    
    for amino_acid in sequence:
        if amino_acid in pKa_values:
            pKa = pKa_values[amino_acid]
            if pH < pKa:
                charge += 1  # Protonated state (positive charge)
            elif pH > pKa:
                charge -= 1  # Deprotonated state (negative charge)
            else:
                charge += 0.0  # Neutral state
        
        # Check for PTM-modified residues and adjust the charge accordingly
        if amino_acid[-2:] in ptm_charge_map:
            charge += ptm_charge_map[amino_acid[-2:]]  # Modify charge based on PTM

    return charge

def download_pdb(pdb_id):
    """
    Downloads a PDB file from the RCSB PDB website.

    Args:
    pdb_id (str): The PDB ID of the protein.

    Returns:
    str: The filename of the downloaded PDB file.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    
    if response.status_code == 200:
        filename = f"{pdb_id}.pdb"
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {pdb_id}.pdb")
        return filename
    else:
        print(f"Error: Unable to fetch PDB file for {pdb_id}")
        return None

def plot_charge_vs_pH(sequence, pKa_values, ptm_residues, pH_range, output_dir):
    """
    Plots the total charge of the protein as a function of pH and saves the plot to a file.

    Args:
    sequence (str): The amino acid sequence of the protein.
    pKa_values (dict): The pKa values for the amino acids.
    ptm_residues (dict): PTM residues and their modifications.
    pH_range (list): A list of pH values to compute the charge over.
    output_dir (str): The directory to save the plot.
    """
    charges = []
    
    for pH in pH_range:
        charge = calculate_charge(sequence, pH, pKa_values, ptm_residues)
        charges.append(charge)

    plt.figure(figsize=(8, 6))
    plt.plot(pH_range, charges, label="Total Charge")
    plt.title("Protein Charge vs pH")
    plt.xlabel("pH")
    plt.ylabel("Total Charge")
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(x=7, color='red', linestyle='--', label="pH 7")
    plt.legend()
    
    # Save the plot as a PNG file
    plot_filename = os.path.join(output_dir, "charge_vs_pH.png")
    plt.savefig(plot_filename)
    print(f"Charge vs pH plot saved to: {plot_filename}")
    plt.close()

def main():
    choice = input("Do you want to (1) download a PDB file by its ID or (2) provide your own local PDB file? (Enter 1 or 2): ")
    
    pdb_file = None
    
    if choice == '1':
        pdb_id = input("Enter the PDB ID (e.g., 1A2B): ")
        pdb_file = download_pdb(pdb_id)
    elif choice == '2':
        pdb_file = input("Enter the path to your local PDB file: ")
        if not os.path.exists(pdb_file):
            print(f"Error: File {pdb_file} does not exist.")
            return
    else:
        print("Invalid choice.")
        return
    
    if pdb_file:
        # Extract sequence from the cleaned PDB file, including PTMs
        sequences, ptm_residues = extract_sequence(pdb_file)
        if not sequences:
            print("No standard amino acids found in the PDB file. The sequence could not be extracted.")
            return
        
        # Ask the user for charge values for PTMs if any are found
        if ptm_residues:
            print(f"Detected PTM-modified residues: {', '.join([f'{mod} at chain {chain} residue {seq}' for (chain, seq), mod in ptm_residues.items()])}")
            ptm_charge_values = {}
            for ptm_residue in ptm_residues.values():
                if ptm_residue not in ptm_charge_map:
                    ptm_charge_values[ptm_residue] = float(input(f"Enter the charge for {ptm_residue}: "))
                else:
                    ptm_charge_values[ptm_residue] = ptm_charge_map[ptm_residue]
            print(f"PTM charge values: {ptm_charge_values}")
        
        # Save sequences to a single FASTA file
        output_dir = os.getcwd()
        save_fasta(sequences, output_dir)
        
        # Ask for pH and pKa values
        pH_choice = input("Do you want to use a pH other than 7? (yes/no): ").lower()
        if pH_choice == 'yes':
            pH = float(input("Enter the pH: "))
            pKa_values = {aa: float(input(f"Enter the pKa for {aa} (default {default_pKa_values[aa]}): ") or default_pKa_values[aa]) for aa in default_pKa_values}
        else:
            pH = 7
            pKa_values = default_pKa_values
        
        # Save charge information to a file
        save_charge_info(sequences, pKa_values, ptm_residues, pH, output_dir)

        # Generate plot of charge vs pH and save it
        pH_range = np.linspace(0, 14, 100)
        plot_charge_vs_pH("".join(sequences.values()), pKa_values, ptm_residues, pH_range, output_dir)

if __name__ == "__main__":
    main()

