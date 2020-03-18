from rdkit import Chem
from rdkit.Chem import AllChem, FragmentMatcher
from rdkit import DataStructs

def Similarity(mol1,smi2):
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1,2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2,2)
    sim = DataStructs.FingerprintSimilarity(fp1,fp2)
    return sim
    
def User_Fragment(mol,frag):
    p = FragmentMatcher.FragmentMatcher()
    p.Init(frag)
    l = len(p.GetMatches(mol))
    return l
