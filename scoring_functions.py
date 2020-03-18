#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from subprocess import check_output as run
import pandas as pd
import time
import pickle
import re
import threading
import pexpect
import fileinput
rdBase.DisableLog('rdApp.error')

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.

   Passing *args and **kwargs through a subprocess call is slightly tricky because we need to know
   their types - everything will be a string once we have passed it. Therefor, we instead use class
   attributes which we can modify in place before any subprocess is created. Any **kwarg left over in
   the call to get_scoring_function will be checked against a list of (allowed) kwargs for the class
   and if a match is found the value of the item will be the new value for the class.

   If num_processes == 0, the scoring function will be run in the main process. Depending on how
   demanding the scoring function is and how well the OS handles the multiprocessing, this might
   be faster than multiprocessing in some cases."""

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
nms=[x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
from tensorflow.keras.models import load_model

from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol

from rdkit.Chem import Descriptors as des
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters, CalcNumAmideBonds
from desfun import Similarity, User_Fragment

descrittori =   {
                 'SA': (lambda x: SA(x)),
                 'QED': (lambda x: QED(x)),
                 ### rdkit ###
                 'MolWt': (lambda x: des.MolWt(x)),
                 'LogP': (lambda x: des.MolLogP(x)),
                 'MolMR': (lambda x: des.MolMR(x)),
                 'NumHDonors': (lambda x: des.NumHDonors(x)),
                 'NumHAcceptors': (lambda x: des.NumHAcceptors(x)),
                 'HeavyAtomCount': (lambda x: des.HeavyAtomCount(x)),
                 'NHOHCount':(lambda x: des.NHOHCount(x)),
                 'NOCount':(lambda x: des.NOCount(x)),
                 'CalcNumAtomStereoCenters': (lambda x: CalcNumAtomStereoCenters(x)),
                 'CalcNumAmideBonds': (lambda x: CalcNumAmideBonds(x)), 
                 'NumAliphaticCarbocycles':(lambda x: des.NumAliphaticCarbocycles(x)),
                 'NumAliphaticHeterocycles':(lambda x: des.NumAliphaticHeterocycles(x)),
                 'NumAliphaticRings':(lambda x: des.NumAliphaticRings(x)),
                 'NumAromaticCarbocycles':(lambda x: des.NumAromaticCarbocycles(x)),
                 'NumAromaticHeterocycles':(lambda x: des.NumAromaticHeterocycles(x)),
                 'NumAromaticRings':(lambda x: des.NumAromaticRings(x)),
                 'NumHeteroatoms': (lambda x: des.NumHeteroatoms(x)),
                 'NumRotatableBonds':(lambda x: des.NumRotatableBonds(x)),
                 'NumSaturatedCarbocycles':(lambda x: des.NumSaturatedCarbocycles(x)),
                 'NumSaturatedHeterocycles': (lambda x: des.NumSaturatedHeterocycles(x)),
                 'NumSaturatedRings':(lambda x: des.NumSaturatedRings(x)),
                 'RingCount':(lambda x: des.RingCount(x)),
                 'FractionCSP3':(lambda x: des.FractionCSP3(x)),
                 'fr_Al_COO':(lambda x: des.fr_Al_COO(x)),
                 'fr_Al_OH':(lambda x: des.fr_Al_OH(x)),
                 'fr_Al_OH_noTert':(lambda x: des.fr_Al_OH_noTert(x)),
                 'fr_ArN':(lambda x: des.fr_ArN(x)),
                 'fr_Ar_COO':(lambda x: des.fr_Ar_COO(x)),
                 'fr_Ar_N':(lambda x: des.fr_Ar_N(x)),
                 'fr_Ar_NH':(lambda x: des.fr_Ar_NH(x)),
                 'fr_Ar_OH':(lambda x: des.fr_Ar_OH(x)),
                 'fr_COO':(lambda x: des.fr_COO(x)),
                 'fr_COO2':(lambda x: des.fr_COO2(x)),
                 'fr_C_O':(lambda x: des.fr_C_O(x)),
                 'fr_C_O_noCOO':(lambda x: des.fr_C_O_noCOO(x)),
                 'fr_C_S':(lambda x: des.fr_C_S(x)),
                 'fr_HOCCN':(lambda x: des.fr_HOCCN(x)),
                 'fr_Imine':(lambda x: des.fr_Imine(x)),
                 'fr_NH0':(lambda x: des.fr_NH0(x)),
                 'fr_NH1':(lambda x: des.fr_NH1(x)),
                 'fr_NH2':(lambda x: des.fr_NH2(x)),
                 'fr_N_O':(lambda x: des.fr_N_O(x)),
                 'fr_Ndealkylation1':(lambda x: des.fr_Ndealkylation1(x)),
                 'fr_Ndealkylation2':(lambda x: des.fr_Ndealkylation2(x)),
                 'fr_Nhpyrrole':(lambda x: des.fr_Nhpyrrole(x)),
                 'fr_SH':(lambda x: des.fr_SH(x)),
                 'fr_aldehyde':(lambda x: des.fr_aldehyde(x)),
                 'fr_alkyl_carbamate':(lambda x: des.fr_alkyl_carbamate(x)),
                 'fr_alkyl_halide':(lambda x: des.fr_alkyl_halide(x)),
                 'fr_allylic_oxid':(lambda x: des.fr_allylic_oxid(x)),
                 'fr_amide':(lambda x: des.fr_amide(x)),
                 'fr_amidine':(lambda x: des.fr_amidine(x)),
                 'fr_aniline':(lambda x: des.fr_aniline(x)),
                 'fr_aryl_methyl':(lambda x: des.fr_aryl_methyl(x)),
                 'fr_azide':(lambda x: des.fr_azide(x)),
                 'fr_azo':(lambda x: des.fr_azo(x)),
                 'fr_barbitur':(lambda x: des.fr_barbitur(x)),
                 'fr_benzene':(lambda x: des.fr_benzene(x)),
                 'fr_benzodiazepine':(lambda x: des.fr_benzodiazepine(x)),
                 'fr_bicyclic':(lambda x: des.fr_bicyclic(x)),
                 'fr_diazo':(lambda x: des.fr_diazo(x)),
                 'fr_dihydropyridine':(lambda x: des.fr_dihydropyridine(x)),
                 'fr_epoxide':(lambda x: des.fr_epoxide(x)),
                 'fr_ester':(lambda x: des.fr_ester(x)),
                 'fr_ether':(lambda x: des.fr_ether(x)),
                 'fr_furan':(lambda x: des.fr_furan(x)),
                 'fr_guanido':(lambda x: des.fr_guanido(x)),
                 'fr_halogen':(lambda x: des.fr_halogen(x)),
                 'fr_hdrzine':(lambda x: des.fr_hdrzine(x)),
                 'fr_hdrzone':(lambda x: des.fr_hdrzone(x)),
                 'fr_imidazole':(lambda x: des.fr_imidazole(x)),
                 'fr_imide':(lambda x: des.fr_imide(x)),
                 'fr_isocyan':(lambda x: des.fr_isocyan(x)),
                 'fr_isothiocyan':(lambda x: des.fr_isothiocyan(x)),
                 'fr_ketone':(lambda x: des.fr_ketone(x)),
                 'fr_ketone_Topliss':(lambda x: des.fr_ketone_Topliss(x)),
                 'fr_lactam':(lambda x: des.fr_lactam(x)),
                 'fr_lactone':(lambda x: des.fr_lactone(x)),
                 'fr_methoxy':(lambda x: des.fr_methoxy(x)),
                 'fr_morpholine':(lambda x: des.fr_morpholine(x)),
                 'fr_nitrile':(lambda x: des.fr_nitrile(x)),
                 'fr_nitro':(lambda x: des.fr_nitro(x)),
                 'fr_nitro_arom':(lambda x: des.fr_nitro_arom(x)),
                 'fr_nitro_arom_nonortho':(lambda x: des.fr_nitro_arom_nonortho(x)),
                 'fr_nitroso':(lambda x: des.fr_nitroso(x)),
                 'fr_oxazole':(lambda x: des.fr_oxazole(x)),
                 'fr_oxime':(lambda x: des.fr_oxime(x)),
                 'fr_para_hydroxylation':(lambda x: des.fr_para_hydroxylation(x)),
                 'fr_phenol':(lambda x: des.fr_phenol(x)),
                 'fr_phenol_noOrthoHbond':(lambda x: des.fr_phenol_noOrthoHbond(x)),
                 'fr_phos_acid':(lambda x: des.fr_phos_acid(x)),
                 'fr_phos_ester':(lambda x: des.fr_phos_ester(x)),
                 'fr_piperdine':(lambda x: des.fr_piperdine(x)),
                 'fr_piperzine':(lambda x: des.fr_piperzine(x)),
                 'fr_priamide':(lambda x: des.fr_priamide(x)),
                 'fr_prisulfonamd':(lambda x: des.fr_prisulfonamd(x)),
                 'fr_pyridine':(lambda x: des.fr_pyridine(x)),
                 'fr_quatN':(lambda x: des.fr_quatN(x)),
                 'fr_sulfide':(lambda x: des.fr_sulfide(x)),
                 'fr_sulfonamd':(lambda x: des.fr_sulfonamd(x)),
                 'fr_sulfone':(lambda x: des.fr_sulfone(x)),
                 'fr_term_acetylene':(lambda x: des.fr_term_acetylene(x)),
                 'fr_tetrazole':(lambda x: des.fr_tetrazole(x)),
                 'fr_thiazole':(lambda x: des.fr_thiazole(x)),
                 'fr_thiocyan':(lambda x: des.fr_thiocyan(x)),
                 'fr_thiophene':(lambda x: des.fr_thiophene(x)),
                 'fr_unbrch_alkane':(lambda x: des.fr_unbrch_alkane(x)),
                 'fr_urea':(lambda x: des.fr_urea(x)),
                 'MaxEStateIndex':(lambda x: des.MaxEStateIndex(x)),
                 'MinEStateIndex':(lambda x: des.MinEStateIndex(x)),
                 'MaxAbsEStateIndex':(lambda x: des.MaxAbsEStateIndex(x)),
                 'MinAbsEStateIndex':(lambda x: des.MinAbsEStateIndex(x)),
                 'NumValenceElectrons':(lambda x: des.NumValenceElectrons(x)),
                 'NumRadicalElectrons':(lambda x: des.NumRadicalElectrons(x)),
                 'MaxPartialCharge':(lambda x: des.MaxPartialCharge(x)),
                 'MinPartialCharge':(lambda x: des.MinPartialCharge(x)),
                 'MaxAbsPartialCharge':(lambda x: des.MaxAbsPartialCharge(x)),
                 'MinAbsPartialCharge':(lambda x: des.MinAbsPartialCharge(x)),
                 'FpDensityMorgan1':(lambda x: des.FpDensityMorgan1(x)),
                 'FpDensityMorgan2':(lambda x: des.FpDensityMorgan2(x)),
                 'FpDensityMorgan3':(lambda x: des.FpDensityMorgan3(x)),
                 'BalabanJ':(lambda x: des.BalabanJ(x)),
                 'BertzCT':(lambda x: des.BertzCT(x)),
                 'Chi0':(lambda x: des.Chi0(x)),
                 'Chi0n':(lambda x: des.Chi0n(x)),
                 'Chi0v':(lambda x: des.Chi0v(x)),
                 'Chi1':(lambda x: des.Chi1(x)),
                 'Chi1n':(lambda x: des.Chi1n(x)),
                 'Chi1v':(lambda x: des.Chi1v(x)),
                 'Chi2n':(lambda x: des.Chi2n(x)),
                 'Chi2v':(lambda x: des.Chi2v(x)),
                 'Chi3n':(lambda x: des.Chi3n(x)),
                 'Chi3v':(lambda x: des.Chi3v(x)),
                 'Chi4n':(lambda x: des.Chi4n(x)),
                 'Chi4v':(lambda x: des.Chi4v(x)),
                 'HallKierAlpha':(lambda x: des.HallKierAlpha(x)),
                 'Ipc':(lambda x: des.Ipc(x)),
                 'Kappa1':(lambda x: des.Kappa1(x)),
                 'Kappa2':(lambda x: des.Kappa2(x)),
                 'Kappa3':(lambda x: des.Kappa3(x)),
                 'LabuteASA':(lambda x: des.LabuteASA(x)),
                 'TPSA':(lambda x: des.TPSA(x)),
                 'PEOE_VSA1':(lambda x: des.PEOE_VSA1(x)),
                 'PEOE_VSA10':(lambda x: des.PEOE_VSA10(x)),
                 'PEOE_VSA11':(lambda x: des.PEOE_VSA11(x)),
                 'PEOE_VSA12':(lambda x: des.PEOE_VSA12(x)),
                 'PEOE_VSA13':(lambda x: des.PEOE_VSA13(x)),
                 'PEOE_VSA14':(lambda x: des.PEOE_VSA14(x)),
                 'PEOE_VSA2':(lambda x: des.PEOE_VSA2(x)),
                 'PEOE_VSA3':(lambda x: des.PEOE_VSA3(x)),
                 'PEOE_VSA4':(lambda x: des.PEOE_VSA4(x)),
                 'PEOE_VSA5':(lambda x: des.PEOE_VSA5(x)),
                 'PEOE_VSA6':(lambda x: des.PEOE_VSA6(x)),
                 'PEOE_VSA7':(lambda x: des.PEOE_VSA7(x)),
                 'PEOE_VSA8':(lambda x: des.PEOE_VSA8(x)),
                 'PEOE_VSA9':(lambda x: des.PEOE_VSA9(x)),
                 'SMR_VSA1':(lambda x: des.SMR_VSA1(x)),
                 'SMR_VSA10':(lambda x: des.SMR_VSA10(x)),
                 'SMR_VSA2':(lambda x: des.SMR_VSA2(x)),
                 'SMR_VSA3':(lambda x: des.SMR_VSA3(x)),
                 'SMR_VSA4':(lambda x: des.SMR_VSA4(x)),
                 'SMR_VSA5':(lambda x: des.SMR_VSA5(x)),
                 'SMR_VSA6':(lambda x: des.SMR_VSA6(x)),
                 'SMR_VSA7':(lambda x: des.SMR_VSA7(x)),
                 'SMR_VSA8':(lambda x: des.SMR_VSA8(x)),
                 'SMR_VSA9':(lambda x: des.SMR_VSA9(x)),
                 'SlogP_VSA1':(lambda x: des.SlogP_VSA1(x)),
                 'SlogP_VSA10':(lambda x: des.SlogP_VSA10(x)),
                 'SlogP_VSA11':(lambda x: des.SlogP_VSA11(x)),
                 'SlogP_VSA12':(lambda x: des.SlogP_VSA12(x)),
                 'SlogP_VSA2':(lambda x: des.SlogP_VSA2(x)),
                 'SlogP_VSA3':(lambda x: des.SlogP_VSA3(x)),
                 'SlogP_VSA4':(lambda x: des.SlogP_VSA4(x)),
                 'SlogP_VSA5':(lambda x: des.SlogP_VSA5(x)),
                 'SlogP_VSA6':(lambda x: des.SlogP_VSA6(x)),
                 'SlogP_VSA7':(lambda x: des.SlogP_VSA7(x)),
                 'SlogP_VSA8':(lambda x: des.SlogP_VSA8(x)),
                 'SlogP_VSA9':(lambda x: des.SlogP_VSA9(x)),
                 'EState_VSA1':(lambda x: des.EState_VSA1(x)),
                 'EState_VSA10':(lambda x: des.EState_VSA10(x)),
                 'EState_VSA11':(lambda x: des.EState_VSA11(x)),
                 'EState_VSA2':(lambda x: des.EState_VSA2(x)),
                 'EState_VSA3':(lambda x: des.EState_VSA3(x)),
                 'EState_VSA4':(lambda x: des.EState_VSA4(x)),
                 'EState_VSA5':(lambda x: des.EState_VSA5(x)),
                 'EState_VSA6':(lambda x: des.EState_VSA6(x)),
                 'EState_VSA7':(lambda x: des.EState_VSA7(x)),
                 'EState_VSA8':(lambda x: des.EState_VSA8(x)),
                 'EState_VSA9':(lambda x: des.EState_VSA9(x)),
                 'VSA_EState1':(lambda x: des.VSA_EState1(x)),
                 'VSA_EState10':(lambda x: des.VSA_EState10(x)),
                 'VSA_EState2':(lambda x: des.VSA_EState2(x)),
                 'VSA_EState3':(lambda x: des.VSA_EState3(x)),
                 'VSA_EState4':(lambda x: des.VSA_EState4(x)),
                 'VSA_EState5':(lambda x: des.VSA_EState5(x)),
                 'VSA_EState6':(lambda x: des.VSA_EState6(x)),
                 'VSA_EState7':(lambda x: des.VSA_EState7(x)),
                 'VSA_EState8':(lambda x: des.VSA_EState8(x)),
                 'VSA_EState9':(lambda x: des.VSA_EState9(x)),
                 'Similarity':(lambda x,m: Similarity(x,m)),
                 'User_Fragment':(lambda x,m: User_Fragment(x,m)),

                 }

class logp_mw():

    kwargs = ['des_names','des_mins','des_maxs','des_rels','des_opts']
    des_names = ['MW']
    des_mins = [100]
    des_maxs = [300]
    des_rels = ['Minimize']
    des_opts = ['']
    
    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            try:
               vals = []
               scs = []
               for i, k in enumerate(self.des_names):
                   if (k=='Similarity' or k=='User_Fragment'):
                      vals.append(descrittori[k](mol,self.des_opts[i]))
                   else:
                      vals.append(descrittori[k](mol))
               for k in range(len(vals)):
                   dd = (self.des_maxs[k]-self.des_mins[k])/4.0
                   d_mins = self.des_mins[k]-vals[k]
                   d_maxs = self.des_maxs[k]-vals[k]
                   if (vals[k]>=self.des_mins[k] and vals[k]<=self.des_maxs[k]):
                      scs.append(1.0)
                      continue
                   if (dd==0):
                      if (vals[k]<self.des_mins[k]):
                         scs.append(np.exp(-(d_mins)**2))
                      if (vals[k]>self.des_maxs[k]):
                         scs.append(np.exp(-(d_maxs)**2))
                   else:
                      if (vals[k]<self.des_mins[k]):
                         scs.append(np.exp(-(d_mins/dd)**2)) 
                      if (vals[k]>self.des_maxs[k]):
                         scs.append(np.exp(-(d_maxs/dd)**2))
               pen = np.sum(np.array(scs))    
               vals.append(pen)
               return vals
            except:
               return np.zeros(len(self.des_names)+1)
        return np.zeros(len(self.des_names)+1)

class Worker():
    """A worker class for the Multiprocessing functionality. Spawns a subprocess
       that is listening for input SMILES and inserts the score into the given
       index in the given list."""
    def __init__(self, scoring_function=None):
        """The score_re is a regular expression that extracts the score from the
           stdout of the subprocess. This means only scoring functions with range
           0.0-1.0 will work, for other ranges this re has to be modified."""

        self.proc = pexpect.spawn('./multiprocess.py ' + scoring_function,
                                  encoding='utf-8')

        print(self.is_alive())

    def __call__(self, smile, index, result_list):
        self.proc.sendline(smile)
        output = self.proc.expect([re.escape(smile) + " 1\.0+|[0]\.[0-9]+", 'None', pexpect.TIMEOUT])
        if output is 0:
            score = float(self.proc.after.lstrip(smile + " "))
        elif output in [1, 2]:
            score = 0.0
        result_list[index] = score

    def is_alive(self):
        return self.proc.isalive()

class Multiprocessing():
    """Class for handling multiprocessing of scoring functions. OEtoolkits cant be used with
       native multiprocessing (cant be pickled), so instead we spawn threads that create
       subprocesses."""
    def __init__(self, num_processes=None, scoring_function=None):
        self.n = num_processes
        self.workers = [Worker(scoring_function=scoring_function) for _ in range(num_processes)]

    def alive_workers(self):
        return [i for i, worker in enumerate(self.workers) if worker.is_alive()]

    def __call__(self, smiles):
        scores = [0 for _ in range(len(smiles))]
        smiles_copy = [smile for smile in smiles]
        while smiles_copy:
            alive_procs = self.alive_workers()
            if not alive_procs:
               raise RuntimeError("All subprocesses are dead, exiting.")
            # As long as we still have SMILES to score
            used_threads = []
            # Threads name corresponds to the index of the worker, so here
            # we are actually checking which workers are busy
            for t in threading.enumerate():
                # Workers have numbers as names, while the main thread cant
                # be converted to an integer
                try:
                    n = int(t.name)
                    used_threads.append(n)
                except ValueError:
                    continue
            free_threads = [i for i in alive_procs if i not in used_threads]
            for n in free_threads:
                if smiles_copy:
                    # Send SMILES and what index in the result list the score should be inserted at
                    smile = smiles_copy.pop()
                    idx = len(smiles_copy)
                    t = threading.Thread(target=self.workers[n], name=str(n), args=(smile, idx, scores))
                    t.start()
            time.sleep(0.01)
        for t in threading.enumerate():
            try:
                n = int(t.name)
                t.join()
            except ValueError:
                continue
        return np.array(scores, dtype=np.float32)

class Singleprocessing():
    """Adds an option to not spawn new processes for the scoring functions, but rather
       run them in the main process."""
    def __init__(self, scoring_function=None):
        self.scoring_function = scoring_function()
    def __call__(self, smiles):
        scores = [self.scoring_function(smile) for smile in smiles]
        return np.array(scores, dtype=np.float32)

def get_scoring_function(scoring_function, num_processes=None, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    scoring_function_classes = [logp_mw]
    scoring_functions = [f.__name__ for f in scoring_function_classes]
    scoring_function_class = [f for f in scoring_function_classes if f.__name__ == scoring_function][0]

    if scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    for k, v in kwargs.items():
        if k in scoring_function_class.kwargs:
            setattr(scoring_function_class, k, v)

    if num_processes == 0:
        return Singleprocessing(scoring_function=scoring_function_class)
    return Multiprocessing(scoring_function=scoring_function, num_processes=num_processes)
