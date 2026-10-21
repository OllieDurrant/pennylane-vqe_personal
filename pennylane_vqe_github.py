# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:21:46 2026

@author: ollie
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# to help with the spam of loading dataset
import os
from contextlib import redirect_stdout, redirect_stderr
from functools import lru_cache


@lru_cache(maxsize=None)
def setup_molecule(molecule_name, freeze_orbitals, num_frozen_core):
    """Loads dataset and prepares the Hamiltonian based on freezing preferences."""
    print(f"Loading {molecule_name} dataset...")

    # Mutes the PennyLane progress bar output completely
    with open(os.devnull, 'w', encoding = 'utf-8') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            dataset = qml.data.load('qchem', folder_path="/tmp", molname=molecule_name)[0]

    if freeze_orbitals == True:
        n_electrons = dataset.molecule.n_electrons
        n_orbitals = dataset.molecule.n_orbitals
        active_electrons = n_electrons - 2 * num_frozen_core
        active_orbitals = n_orbitals - num_frozen_core
        print(f'Freezing {num_frozen_core} inner electrons, {active_electrons} active electrons')

        H, qubits = qml.qchem.molecular_hamiltonian(
            dataset.molecule,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )
        electrons = active_electrons
        hf = qml.qchem.hf_state(electrons, qubits)
    else:
        H = dataset.hamiltonian
        qubits = len(H.wires)
        hf = dataset.hf_state
        electrons = int(np.sum(hf))
        
    singles, doubles = qml.qchem.excitations(electrons=electrons, orbitals=qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    num_params = len(singles) + len(doubles)
    
    return H, qubits, electrons, hf, singles, doubles, s_wires, d_wires, num_params, dataset.fci_energy
    
    
    
def hf_ansatz(params, ansatz, qubits, hf, singles, doubles, s_wires, d_wires):
    """Choose which ansatz to use."""
    if ansatz == 'AllSD': 
        qml.AllSinglesDoubles(params, wires=range(qubits), hf_state=hf,
                              singles=singles, doubles=doubles)
    elif ansatz == 'UCCSD':
        qml.UCCSD(params, range(qubits), s_wires, d_wires, hf)
    elif ansatz == 'SEL':
        qml.StronglyEntanglingLayers(params, range(qubits))
    elif ansatz == 'PCU2':
        qml.ParticleConservingU2(params, wires=range(qubits), init_state=hf)

def build_qnodes(qubits, shots, ansatz, hf, singles, doubles, s_wires, d_wires):
    """Dynamically builds devices and QNodes based on current molecule parameters."""
    # Only applies to these excitation gates, so noise is not equivalent across ansatz choices
    # Condition: fires on the gates AllSinglesDoubles actually uses
    noise_cond = qml.noise.op_in([qml.SingleExcitation, qml.DoubleExcitation])
    # What to insert: depolarizing channel on the same wires as the matched gate
    noise_op = qml.noise.partial_wires(qml.DepolarizingChannel, 0.01)
    noise_model = qml.NoiseModel({noise_cond: noise_op})

    dev = qml.device('lightning.qubit', wires=qubits)
    dev_n = qml.device('default.mixed', wires=qubits)
    dev_noisy = qml.add_noise(dev_n, noise_model=noise_model)

    @qml.qnode(dev_noisy, shots=shots, interface="autograd") #for noise
    def cost_hf(params, H):
        """Expectation value of the Hamiltonian under the HF ansatz, for noisy"""
        hf_ansatz(params, ansatz, qubits, hf, singles, doubles, s_wires, d_wires)
        return qml.expval(H)

    @qml.qnode(dev, diff_method='adjoint', interface="autograd") #for fast 
    def cost_hf_theory(params, H):
        '''for exact cost'''
        hf_ansatz(params, ansatz, qubits, hf, singles, doubles, s_wires, d_wires)
        return qml.expval(H)

    def choose_noise(choice, params, H):
        if choice == 'theory':
            return cost_hf_theory(params, H)
        elif choice =='noise':
            return cost_hf(params, H)
        else:
            raise ValueError(f"Unknown choice: {choice!r} — expected 'theory' or 'noise'")
            
    return choose_noise, cost_hf_theory, cost_hf

def get_param_shape(ansatz, num_params, n_layers, qubits):
    if ansatz in ('AllSD', 'UCCSD'):
        return num_params
    elif ansatz == 'SEL':
        return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=qubits)
    elif ansatz == 'PCU2':
        return qml.ParticleConservingU2.shape(n_layers=n_layers, n_wires=qubits)

def optimizer(choose_noise_fn, H, params, choice, max_iterations=1000, verbose_every=20, tol=1e-5):
    """Gradient-descend params to minimize <H> using Adam."""
    opt = qml.AdamOptimizer()
    energy_history = [choose_noise_fn(choice, params, H)] 
    for i in range(max_iterations):
        params = opt.step(lambda p: choose_noise_fn(choice, p, H), params) 
        energy = choose_noise_fn(choice, params, H) 
        energy_history.append(energy)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"Step {i + 1:3d}: Energy = {energy:.6f}")
        if abs(energy_history[-1] - energy_history[-2]) < tol:
            print(f"Converged at step {i + 1} (ΔE < {tol})")
            break
    # Plotting removed from here and returned as data instead
    return params, energy_history

def info(params, H, cost_hf_theory_fn):
    info_res = qml.specs(cost_hf_theory_fn, level='device')(params, H)
    res = info_res['resources']
    print(f"Depth: {res.depth}")
    print(f"Total gates: {res.num_gates}")
    print(f"Gate breakdown: {dict(res.gate_types)}")

def run_vqe(molecule, choice, freeze_orbitals, num_frozen_core, shots, ansatz, n_layers, max_iterations=1000, start_at_hf=False, seed=32):
    """Run VQE. start_at_hf=True uses zero parameters (not necessarily
    the Hartree-Fock state for every ansatz)."""
    np.random.seed(seed)
    
    H, qubits, electrons, hf, singles, doubles, s_wires, d_wires, num_params, fci_energy = setup_molecule(molecule, freeze_orbitals, num_frozen_core)
    
    print(f"{str(qubits)} qubits, {num_params} variational parameters "
          f"({len(singles)} singles, {len(doubles)} doubles)")
          
    choose_noise_fn, cost_hf_theory, cost_hf = build_qnodes(qubits, shots, ansatz, hf, singles, doubles, s_wires, d_wires)
    
    shape = get_param_shape(ansatz, num_params, n_layers, qubits)
    if start_at_hf:
        initial_params = np.zeros(shape, requires_grad=True)
    else:
        initial_params = 0.1 * np.random.random(shape, requires_grad=True)
    
    print('inital parameter stats') 
    info(initial_params, H, cost_hf_theory)

    print("Starting optimization...")
    opt_params, energy_history = optimizer(choose_noise_fn, H, initial_params, choice, max_iterations)
    ground_state_energy = choose_noise_fn(choice, opt_params, H)
    
    print('final parameter stats')
    info(opt_params, H, cost_hf_theory)
    
    # Return everything you need to plot and interact with later
    return opt_params, ground_state_energy, energy_history, fci_energy, H, num_params, cost_hf

def plot_convergence(results_dict, fci_energy=None, title="VQE Convergence Comparison"):
    """
    Plots multiple convergence histories on the same graph.
    results_dict: Dictionary mapping a label (str) to an energy_history (list)
                  e.g., {'UCCSD': history1, 'AllSD': history2}
    """
    plt.figure(figsize=(10, 8))
    
    # Loop through the dictionary and plot each line
    for label, history in results_dict.items():
        plt.plot(history, label=f'VQE ({label})', linestyle = '--')
        
    if fci_energy is not None:
        plt.axhline(fci_energy, color='g', linestyle='-', label='Reference FCI energy')
        
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
def energy_landscape(cost_hf_fn, H, opt_params, vary_indices=(0, 1), n_points=20, span=np.pi):
    """2D slice of the energy landscape: vary the two parameters in
    vary_indices, freeze every other parameter at its optimized value."""
    i0, i1 = vary_indices
    theta0 = np.linspace(-span, span, n_points)
    theta1 = np.linspace(-span, span, n_points)
    X, Y = np.meshgrid(theta0, theta1)
    Z = np.zeros_like(X)

    print(f"Calculating landscape grid over params {i0} and {i1} "
          f"({n_points}x{n_points})...")
    base = np.array(opt_params, requires_grad=False)
    for a in range(X.shape[0]):
        for b in range(X.shape[1]):
            params = base.copy()
            params[i0] = X[a, b]
            params[i1] = Y[a, b]
            Z[a, b] = cost_hf_fn(params, H)
    return X, Y, Z

def plot_landscape(X, Y, Z, molecule, vary_indices=(0, 1)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, label='Energy (Hartrees)')
    ax.set_xlabel(f'Parameter {vary_indices[0]}')
    ax.set_ylabel(f'Parameter {vary_indices[1]}')
    ax.set_zlabel('Energy')
    ax.set_title(f'{molecule} VQE Energy Landscape')
    plt.show()
    
    
    
#%% run the shit!

# define variables
MOLECULE = "LiH"                  # e.g. "LiH" for item 4 (bigger molecule)
choice = 'theory'             #choose whether want noisy or perfect simulation, 'noise' or 'theory' 
freeze_orbitals = True      #for larger molecules
NUM_FROZEN_CORE = 1          # e.g. Li's 1s core for LiH


SHOTS = 500                        # e.g. 1000 to add shot noise (item 2); None = analytic
ansatz_list = ['AllSD', 'UCCSD', 'SEL', 'PCU2']              #choose the type of ansatz used, UCCSD, AllSD or SEL
#ansatz_list = ['SEL']
N_layers = 2                         #choice for depth of ansatz depth, only for SEL choice in ansatz
MAX_ITERATIONS = 1000
SEED = 32

# Extra experiments are off unless I specifically want to run them
run_ibm = False
run_orbital_analysis = False
run_gate_tests = False
run_barren_plateau = False


# Create a dictionary to store the history of each run
all_histories = {}
reference_fci_energy = None

# Loop through the variables and run VQE for each
for current_ansatz in ansatz_list:
    print(f"RUNNING VQE WITH ANSATZ: {current_ansatz}")
    
    opt_params, final_energy, energy_history, fci_energy, H, num_params, cost_hf_fn = run_vqe(
        molecule=MOLECULE, 
        choice=choice, 
        freeze_orbitals=freeze_orbitals, 
        num_frozen_core=NUM_FROZEN_CORE, 
        shots=SHOTS, 
        ansatz=current_ansatz, 
        n_layers=N_layers, 
        max_iterations=MAX_ITERATIONS,
        start_at_hf=False,
        seed=SEED
    )
    
    # Save the history to the dictionary using the ansatz name as the label
    all_histories[current_ansatz] = energy_history
    reference_fci_energy = fci_energy  # Save the reference energy once

# Pass the whole dictionary to the updated plot function
plot_convergence(
    results_dict=all_histories, 
    fci_energy=reference_fci_energy, 
    title=f'{MOLECULE} VQE Convergence Comparison'
)
# Energy Landscape
run_landscape = False
if num_params >= 2 and run_landscape == True:
    X, Y, Z = energy_landscape(cost_hf_fn, H, opt_params, vary_indices=(0, 1))
    plot_landscape(X, Y, Z, MOLECULE, vary_indices=(0, 1))
elif run_landscape == True:
    print("Only one variational parameter — skipping 2D landscape plot.")



#%% IBM hardware (only run this when I actually want to use QPU time)

if run_ibm:
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Account needs to be saved locally beforehand, not in this file
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Connected to backend: {backend.name}")

    H, qubits, electrons, hf, singles, doubles, s_wires, d_wires, num_params, fci_energy = setup_molecule(
        MOLECULE, freeze_orbitals, NUM_FROZEN_CORE
    )

    num_qubits = len(H.wires)
    print(f"Hamiltonian acts on {num_qubits} qubits (wires: {list(H.wires)})")

    # Local test instead of real hardware:
    # dev_ibm = qml.device('lightning.qubit', wires=H.wires)
    dev_ibm = qml.device('qiskit.remote', wires=H.wires, backend=backend, shots=SHOTS)

    @qml.qnode(dev_ibm)
    def cost_ibm(params, H):
        hf_ansatz(params, current_ansatz, num_qubits, hf, singles, doubles, s_wires, d_wires)
        return qml.expval(H)

    # Uses the parameters from the last ansatz in the comparison above
    print("\nSubmitting circuit to IBM Quantum queue...")
    ibm_energy = cost_ibm(opt_params, H)

    print("\n" + "=" * 50)
    print("IBM HARDWARE RESULTS")
    print("=" * 50)
    print(f"Target FCI Energy          : {fci_energy:.6f} Hartree")
    print(f"Simulated Ground State     : {final_energy:.6f} Hartree")
    print(f"Real IBM Hardware Energy   : {ibm_energy:.6f} Hartree")
    print(f"Hardware Error vs FCI      : {abs(ibm_energy - fci_energy):.6f} Hartree")
    print("=" * 50)

#%% a little test for if an electron is core or not 

def analyze_active_space(molecule_name):
    """
    Computes Hartree-Fock orbital energies and prints active space 
    recommendations for a given PennyLane qchem dataset molecule.
    """
    dataset = qml.data.load('qchem', folder_path="/tmp", molname=molecule_name)[0]
    mol = dataset.molecule
    
    n_electrons = mol.n_electrons
    n_orbitals = mol.n_orbitals
    
    # Run Hartree-Fock to get molecular orbital energies
    print(f"Running SCF calculation for {molecule_name}...")
    scf_func = qml.qchem.scf(mol)
    scf_results = scf_func()
    
    # Extract mo_energies from SCF output
    if isinstance(scf_results, tuple):
        # Depending on PennyLane version, scf returns (energy, mo_coeff, mo_energies) or similar
        energies = None
        for res in scf_results:
            if isinstance(res, (np.ndarray, list)) and len(res) == n_orbitals:
                energies = res
                break
        if energies is None:
            energies = scf_results[-1] # default to last element if structure varies
    else:
        energies = scf_results

    # Occupied spatial orbitals count
    n_occ = n_electrons // 2
    homo_idx = n_occ - 1
    lumo_idx = n_occ
    
    homo_energy = energies[homo_idx]
    lumo_energy = energies[lumo_idx]
    gap = lumo_energy - homo_energy
    
    print("=" * 70)
    print(f"  MOLECULAR ORBITAL ANALYSIS: {molecule_name}")
    print("=" * 70)
    print(f"Total Electrons        : {n_electrons}")
    print(f"Total Spatial Orbitals : {n_orbitals} ({n_orbitals * 2} qubits without freezing)")
    print(f"HOMO Energy            : {homo_energy:.4f} Hartree (Orbital {homo_idx})")
    print(f"LUMO Energy            : {lumo_energy:.4f} Hartree (Orbital {lumo_idx})")
    print(f"HOMO-LUMO Gap          : {gap:.4f} Hartree ({gap * 27.2114:.2f} eV)")
    print("-" * 70)
    print(f"{'Orbital #':<10} | {'Energy (Hartree)':<18} | {'Type':<18} | {'Recommendation'}")
    print("-" * 70)
    
    frozen_core_count = 0
    active_orbitals_count = 0
    
    for idx, E in enumerate(energies):
        if idx < homo_idx:
            # Orbitals >1.0 Hartree below HOMO are core orbitals
            if (homo_energy - E) > 1.0:
                orb_type = "Deep Core"
                rec = "FREEZE"
                frozen_core_count += 1
            else:
                orb_type = "Valence Occupied"
                rec = "KEEP (Active)"
                active_orbitals_count += 1
        elif idx == homo_idx:
            orb_type = "HOMO"
            rec = "KEEP (Active)"
            active_orbitals_count += 1
        elif idx == lumo_idx:
            orb_type = "LUMO"
            rec = "KEEP (Active)"
            active_orbitals_count += 1
        else:
            if (E - lumo_energy) > 1.5:
                orb_type = "High Virtual"
                rec = "TRUNCATE (Optional)"
            else:
                orb_type = "Virtual Unoccupied"
                rec = "KEEP (Active)"
                active_orbitals_count += 1
                
        print(f"{idx:<10} | {E:<18.4f} | {orb_type:<18} | {rec}")
        
    print("-" * 70)
    print(f"Recommended `NUM_FROZEN_CORE` setting : {frozen_core_count}")
    print(f"Resulting Active Qubits needed        : {active_orbitals_count * 2} qubits")
    print("=" * 70)


# Run analysis on molecule, H2O, LiH, O2 etc etc 
if run_orbital_analysis:
    analyze_active_space("LiH")





#%% ADDITIONAL TESTS
if run_gate_tests:
    gen, coeff = qml.generator(qml.SingleExcitation(0.5, wires=[0, 1]))
    print(gen)
    print(coeff)

    op = qml.SingleExcitation(0.5, wires=[0, 1])
    for gate in op.decomposition():
        print(gate)

    op2 = qml.DoubleExcitation(0.5, wires=[0, 1, 2, 3])
    for gate in op2.decomposition():
        print(gate)

#%% Barren plateau test (Uses local dev_bp since global dev was removed)
if run_barren_plateau:
    n_samples = 20
    layer_range = [1, 2, 4, 6, 8, 10, 12]
    variances = []
    num_qubits = len(H.wires)

    dev_bp = qml.device('lightning.qubit', wires=num_qubits)
    @qml.qnode(dev_bp, diff_method='adjoint', interface="autograd")
    def cost_bp(params):
        qml.StronglyEntanglingLayers(params, range(num_qubits))
        return qml.expval(H)

    for n_layers_test in layer_range:
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers_test, n_wires=num_qubits)
        grads_first_param = []
        for _ in range(n_samples):
            params = np.random.uniform(0, 2*np.pi, shape, requires_grad=True)
            grad = qml.grad(cost_bp)(params)
            grads_first_param.append(grad[0, 0, 0])  # track one parameter's gradient
        variances.append(np.var(grads_first_param))

    plt.figure()
    plt.plot(layer_range, variances, 'o-')
    plt.yscale('log')
    plt.xlabel('N_layers')
    plt.ylabel('Gradient variance (log scale)')
    plt.title('Barren plateau: gradient variance vs. depth')
    plt.show()




