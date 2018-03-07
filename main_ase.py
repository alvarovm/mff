from ase.io import iread, read

from m_ff.confs import Confs

testfiles = {
    'BIP_300': 'test/data/BIP_300/movie.xyz',
    'C_a': 'test/data/C_a/data_C.xyz',
    'Fe_vac': 'test/data/Fe_vad/vaca_iron500.xyz',
    'HNi': 'test/data/HNI/h_ase500.xyz'
}


if __name__ == '__main__':

    filename = 'test/data/Fe_vac/vaca_iron500.xyz'
    traj = read(filename, index=slice(None))




    # for atoms in iread(filename, index=slice(None)):
    #     print(atoms)
    #     atoms.arrays['force']




