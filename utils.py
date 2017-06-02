from IPython.display import display, HTML
from tempfile import NamedTemporaryFile
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from PyQuante.Constants import ang2bohr
import imolecule
from PyQuante.Ints import getbasis
import pickle
from math import trunc


def truncate(number, decimals=0):
    if decimals < 0:
        raise ValueError(
            'truncate received an invalid value of decimals ({})'.format(decimals))
    elif decimals == 0:
        return trunc(number)
    else:
        factor = float(10**decimals)
        return trunc(number * factor) / factor
# pyquante functions


def replace_allnew(astr, bstr, adict):
    with open(astr, 'r') as afile:
        data = afile.read()

    for key, value in adict.items():
        data = data.replace(key, str(value))

    with open(bstr, 'w') as bfile:
        bfile.write(data)
    return


def xyz_text(mol):
    xyz_str = ""
    for atom in mol.atoms:
        atom_type = atno2type(atom.atno)
        xyz_str += "%s           %1.1f" % (atom_type, atom.atno)
        xyz_str += "     {:12.12f}       {:12.12f}       {:12.12f}\n".format(
            atom.r[0], atom.r[1], atom.r[2])
    return xyz_str


def gamess_basisInfo(mol, basis_set):
    atom_types = [atno2type(atom.atno) for atom in mol.atoms]
    bfs = getbasis(mol, basis_set)
    currentId = -1
    basis_str = ""
    shell_id = 1
    gauss_id = 1
    for bfindx in range(len(bfs)):
        bf = bfs[bfindx]
        if bf.atid != currentId:
            currentId = bf.atid
            basis_str += '%s\n\n' % (atom_types[bf.atid])
        for prim in bf.prims:
            if prim.powers[1] == 0 and prim.powers[2] == 0:
                sym = power2sym(prim.powers)
                basis_str += "      {:d} {:s} {:d}".format(
                    shell_id, sym, gauss_id)
                basis_str += "      {:12.7f}    {:12.12f}\n".format(
                    prim.exp, prim.coef)
                gauss_id += 1
                count = True
        if count:
            shell_id += 1
            basis_str += '\n'
            count = False
    return basis_str,  shell_id - 1


def gamess_power2xyz(powers):
    xyz_str = ''
    xyz_types = ['X', 'Y', 'Z']
    for indx, i in enumerate(powers):
        for j in range(i):
            xyz_str += xyz_types[indx]
    if xyz_str == '':
        xyz_str = 'S'
    return xyz_str


def gamess_orbInfo(mol, basis_set):
    atom_types = [atno2type(atom.atno) for atom in mol.atoms]
    bfs = getbasis(mol, basis_set)
    bfs_atids = []
    bfs_atypes = []
    bfs_sym = []
    for bfindx in range(len(bfs)):
        bf = bfs[bfindx]
        bfs_atids.append(bf.atid + 1)
        bfs_atypes.append(atom_types[bf.atid])
        bfs_sym.append(gamess_power2xyz(bf.powers))
    return bfs_atids, bfs_atypes, bfs_sym


def gamess_orbStr(
    mol,
    basis_set,
    orbitals,
    orb_e,
):

    orb_str = ''
    (bfs_atids, bfs_atypes, bfs_sym) = gamess_orbInfo(mol, basis_set)

    for start_col in range(0, orbitals.shape[0], 5):
        end_col = min(start_col + 5, orbitals.shape[0])

        # number row

        orb_str += '                   '
        for acol in range(start_col, end_col):
            orb_str += '{:10d}'.format(acol + 1)
        orb_str += '\n'

        # eigen row

        orb_str += '                   '
        for acol in range(start_col, end_col):
            orb_str += '  {:8.4f} '.format(orb_e[acol])
        orb_str += '\n'

        # symmetry row

        orb_str += '                 {:10s}'.format('')
        for acol in range(start_col, end_col):
            orb_str += '{:10s}'.format('A')
        orb_str += '\n'

        # start printing

        for i in range(orbitals.shape[0]):
            orb_row = orbitals[i, start_col:end_col]
            orb_str += '    {:d}  {:s}  {:2d}  {:3s} '.format(i,
                                                              bfs_atypes[i], bfs_atids[i], bfs_sym[i])
            for acol in range(orb_row.shape[0]):
                orb_str += '  {:8.4f}'.format(orb_row[acol])
            orb_str += '\n'
        orb_str += '\n'

    # remove two last lines

    orb_str = orb_str[:-1]
    orb_str += ' ...... END OF RHF CALCULATION ......\n'
    return orb_str


def create_Orbital_file(
    filename,
    mol,
    basis_set,
    orbitals,
    orb_e,
):
    '''
    Creates a orbital file with extension .out
    that can be used with Avogadro
    for 3D orbital viewing.
    Based on a dummy GAMESS file.
    INPUTS:
     filename --> a string to be used for the name of the file, duh
      mol --> a PyQuante Molecule object
      basis_set -> a string indicating the basis_set used
      orbitals -> a matrix that containts the coefficients for the orbitals
      orb_e -> an array of orbital energies
    '''
    template_file = 'files/template_mo.out'
    new_file = '%s.out' % filename
    replaceDict = {}

    # general mol info

    replaceDict['{#Molname}'] = mol.name
    replaceDict['{#XYZ}'] = xyz_text(mol)
    replaceDict['{#Natoms}'] = len(mol.atoms)
    replaceDict['{#Nelectrons}'] = mol.get_nel()
    (nalpha, nbeta) = mol.get_alphabeta()
    replaceDict['{#Nalpha}'] = nalpha
    replaceDict['{#Nbeta}'] = nbeta
    replaceDict['{#Charge}'] = mol.charge
    replaceDict['{#Multiplicity}'] = mol.multiplicity

    # Basis set info
    basis_str, nshells = gamess_basisInfo(mol, basis_set)
    replaceDict['{#BasisInfo}'] = basis_str
    replaceDict['{#Nbfs}'] = len(getbasis(mol, basis_set))
    replaceDict['{#Nshells}'] = nshells

    # Orbital Info

    replaceDict['{#Orbitals}'] = gamess_orbStr(mol, basis_set,
                                               orbitals, orb_e)

    # create file

    replace_allnew(template_file, new_file, replaceDict)
    print('======================')
    print('Created %s successfully!' % new_file)
    print('View it now with Avogadro!')
    return


def visualize_Mol(molecule, angstroms=True):
    '''
    Returns a 3D representation of a molecule
    INPUTS:
     molecule --> a Pyquante molecule
     angstroms --> a True or False value indicating if it is in angstroms.
    '''

    mol = molecule.copy()
    # convert angstrom to bohr
    if angstroms:
        for atom in mol:
            coords = [a / ang2bohr for a in atom.pos()]
            atom.update_coords(coords)
    # create as xyz string
    xyz_str = mol.as_string()
    return imolecule.draw(xyz_str, format='xyz', shader="phong")


def visualize_Molecules(molecules, angstroms=False):
    '''
    Returns a 3D representation of a list of molecules
    INPUTS:
     molecules --> a list of Pyquante molecules
     angstroms --> a True or False value indicating if it is in angstroms.
    '''

    renders = []
    for mol in molecules:
        mol = mol.copy()
        # convert angstrom to bohr
        if angstroms:
            for atom in mol:
                coords = [a / ang2bohr for a in atom.pos()]
                atom.update_coords(coords)
        # create as xyz string
        xyz_str = mol.as_string()
        renders.append(imolecule.draw(
            xyz_str, size=(200, 150), format='xyz', shader="phong", display_html=False))
    columns = ('<div class="col-xs-6 col-sm-3">{}</div>'.format(r)
               for r in renders)
    return display(HTML('<div class="row">{}</div>'.format("".join(columns))))


def embedVideo(afile):
    '''This function returns a HTML embeded video of a file
    Input:
        -- afile : mp4 video file
    '''

    video = io.open(afile, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))


def embedAnimation(anim, plt, filename='', frames=20):
    '''This function returns a HTML embeded video of a maptlolib animation
    Input:
        -- anim : matplotlib animation
        -- plt, matplotlib module handle
    '''

    plt.close(anim._fig)

    if not hasattr(anim, '_encoded_video'):
        if filename == '':
            with NamedTemporaryFile(suffix='.mp4') as f:
                anim.save(f.name, fps=frames, extra_args=[
                          '-vcodec', 'libx264'])
                video = open(f.name, "rb").read()
        else:
            with open(filename, 'w') as f:
                anim.save(f, fps=frames, extra_args=['-vcodec', 'libx264'])
            video = open(filename, "rb").read()
        anim._encoded_video = video.encode("base64")

    VIDEO_TAG = """<video controls>
         <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
         Your browser does not support the video tag.
        </video>"""

    return HTML(VIDEO_TAG.format(anim._encoded_video))


def anim_to_html(anim):
    VIDEO_TAG = """<video controls>
    <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>"""

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)


def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))


def display_video(video_file):
    VIDEO_TAG = """<video controls>
    <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>"""

    return HTML(VIDEO_TAG.format(video_file))


def display_matrix(M):
    from IPython.display import Latex
    mat_str = "\\\\\n".join(
        [" & ".join(map('{0:.3f}'.format, line)) for line in M])
    Latex(r"""\[ \begin{bmatrix} %s \end{bmatrix} \] """ % mat_str)
    return Latex


def power2sym(powers):
    sym = 'Error'
    if powers in [(0, 0, 0)]:
        sym = 'S'
    if powers in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        sym = 'P'
    if powers in [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]:
        sym = 'D'
    if powers in [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0),
                  (1, 1, 1), (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)]:
        sym = 'F'

    return sym


def power2xyz(powers):
    xyz_str = ''
    xyz_types = ['x', 'y', 'z']
    for indx, i in enumerate(powers):
        if i > 0:
            if i == 1:
                xyz_str += xyz_types[indx]
            else:
                xyz_str += "%s^%d" % (xyz_types[indx], i)
    return xyz_str


def atno2type(atno):
    attype = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O',
              'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
    return attype[atno]


def get_orbitalInfo(mol, basis_set):

    if isinstance(basis_set, basestring):
        bfs = getbasis(mol, basis_set)
    else:
        bfs = basis_set

    g_basis = [[] for i in mol.atoms]

    for i_bf in range(len(bfs)):
        bf = bfs[i_bf]
        cgbf = (power2sym(bf.powers), power2xyz(bf.powers), [])
        for prim in bf.prims:
            cgbf[2].append((prim.exp, prim.coef))
        g_basis[bf.atid].append(cgbf)

    return g_basis, len(bfs)


def get_orbitalList(mol, basis_set):

    bfs = getbasis(mol, basis_set)
    g_basis = [[] for i in mol.atoms]
    # check for duplicates
    last_sym = 'N'
    last_pair = (0.0, 0.0)
    last_atid = -1
    # iterate bfs
    for i_bf in range(len(bfs)):
        bf = bfs[i_bf]
        sym = power2sym(bf.powers)
        cgbf = (sym,  [])

        for prim in bf.prims:
            pair = (prim.exp, prim.coef)
            cgbf[1].append(pair)

        if (last_sym != sym) or (pair != last_pair) or (last_atid != bf.atid):
            g_basis[bf.atid].append(cgbf)
            last_sym = sym
            last_pair = pair
            last_atid = bf.atid

    return g_basis


def print_mo_coeff():

    return


def print_orbitalInfo(mol, basis_set):
    """"" Print infomation on a basis set , like number of basis functions,
    symmetry type of the orbtials and gaussian decomposition of the orbitals.

    INPUTS:
    mol --> a Pyquante.Molecule object
    basis_set --> a string indicating the type of basis set
    """""
    g_basis, nbfs = get_orbitalInfo(mol, basis_set)
    orb_num = 1
    print("Molecule is using %d basis functions" % (nbfs))
    for i, atom in enumerate(mol.atoms):
        print("Atom %s, #%d" % (atno2type(atom.atno), i + 1))
        for one_basis in g_basis[i]:
            print("\t Orbital #%d, type %s %s, built with %d gaussians:" %
                  (orb_num, one_basis[0], one_basis[1], len(one_basis[2])))
            orb_num += 1
            for one_g in one_basis[2]:
                print("\t \t exp = %5.5f , coef = %5.5f" %
                      (one_g[0], one_g[1]))
    return


def orbital_index(mol, basis_set):
    """"" Print an index based on a basis set ,

    INPUTS:
    mol --> a Pyquante.Molecule object
    basis_set --> a string indicating the type of basis set
    """""
    g_basis, nbfs = get_orbitalInfo(mol, basis_set)
    orb_num = 1
    index = []
    for i, atom in enumerate(mol.atoms):
        for one_basis in g_basis[i]:
            atom_type = atno2type(atom.atno)
            sym_type = "%s_{%s}" % (one_basis[0], one_basis[1])
            sym_type = sym_type.replace("_{}", '')

           #index.append("%d-%s,$%s$"%(orb_num, atom_type, sym_type))
            index.append("$%s^{%d}_{%d},\;%s$" %
                         (atom_type, orb_num, i + 1, sym_type))

            orb_num += 1

    return index


def print_mo_overview(mol, orbe, orbs, basis_set):
    """"" Print infomation on a basis set , like number of basis functions,
    symmetry type of the orbtials and gaussian decomposition of the orbitals.

    INPUTS:
    mol --> a Pyquante.Molecule object
    basis_set --> a string indicating the type of basis set
    """""

    g_basis, nbfs = get_orbitalInfo(mol, basis_set)

    print("Molecule is using %d basis functions" % (nbfs))
    for i, atom in enumerate(mol.atoms):
        print("Atom %s, #%d" % (atno2type(atom.atno), i + 1))
        for one_basis in g_basis[i]:
            print("\t Orbital type %s %s, built with %d gaussians:" %
                  (one_basis[0], one_basis[1], len(one_basis[2])))
            for one_g in one_basis[2]:
                print("\t \t exp = %5.5f , coef = %5.5f" %
                      (one_g[0], one_g[1]))
    return


# def visualize_orbital(mol_xyz, atom_type, basis_set, orbs, iso_level=0.15):
#     """""  To be used for visualzing Molecular Orbitals.
#         Assumes you have chemview loaded.

#     INPUTS:
#     mol_xzy --> a list of xyz positions (e.g [[0,0,0],[0,0,1.4])
#     atom_type --> a list of atom types (e.g. 'C' or 'O')
#     basis_set --> list of basis function information.
#         orbitals --> a vector of mo coeficients, note this is not a matrix but a vector so you might want to use slicing notation (e.g. orbs[:,indx]) to retrieve a specific column for visualization.
#     iso_level -->  the coefficient to use for constructing the isosurface
#     """""
#     # make molecular orbitals
#     f = molecular_orbital(mol_xyz / 10.0, orbs, basis_set)
#     # make viewer
#     mv = MolecularViewer(mol_xyz / 10.0, {'atom_types': atom_type})
#     # normal view
#     mv.ball_and_sticks(ball_radius=0.02,)
#     # add isosurfaces
#     mv.add_isosurface(
# f, isolevel=iso_level, color=0xff0000, resolution=32, style='wireframe')

#     mv.add_isosurface(
# f, isolevel=-iso_level, color=0x0000ff, resolution=32,
# style='wireframe')

#     return mv


def save_hf_data(mol, basis_set, orbs):
    """"" Save the data for a specific molecule, to be used with load_hf_data.
        To be used for visualzing Molecular Orbitals.

    INPUTS:
    mol --> a Pyquante molecule
    basis_set -> string of the basis set being used
    orbs -> a matrix of MO orbitals
    """""
    # mol name
    mol_name = mol.name
    # xyz and atom types
    mol_xyz = []
    mol_type = []
    for atom in mol.atoms:
        mol_type.append(atno2type(atom.atno))
        mol_xyz.append(atom.r)
    mol_xyz = np.array(mol_xyz)

    # molecular guess bonds
    # atoms = [chemlab.core.Atoms(mol_type[i],r) for i,r in enumerate(mol_xyz)]
    # mol = chemlab.core.Molecule(atoms)
    # mol.guess_bonds()
    g_basis = get_orbitalList(mol, basis_set)

    # save data to a dict
    mol_dict = {}
    mol_dict['xyz'] = mol_xyz
    mol_dict['atom_type'] = mol_type
    mol_dict['basis_set'] = g_basis
    mol_dict['orbitals'] = orbs
    # pickle it
    pickle.dump(mol_dict, open(mol_name + ".pkl", "wb"))
    return


def load_hf_data(mol):
    """"" Load the data for a specific molecule, to be used with save_hf_data.
        Assumes there is a file named 'mol.pkl' in the same directory.

    INPUTS:
    mol --> a string with the name of the Pyquante molecule
    """""
    data = pickle.load(open("%s.pkl" % mol, "rb"))
    return data['xyz'], data['atom_type'], data['basis_set'], data['orbitals']


def symmetric_orthogonalization(A):
    """ Implements symmetric orthogonalization of matrix A and returns
     the transforming matrix X and the orthonormalized matrix Aorth
    """
    n = len(A)
    s, U = scipy.linalg.eigh(A)

    sm = np.diag(s)
    for i in range(n):
        sm[i, i] = 1.0 / np.sqrt(s[i])

    Xtmp = np.dot(U, np.dot(sm, np.transpose(U)))
    X = Xtmp
    AOrth = np.dot(np.transpose(X), np.dot(A, X))

    return X, AOrth


# def print_orbital_Matrix(mol, basis_set, C):

#     g_basis, nbfs = get_orbitalInfo(mol, basis_set)

#     basis_label=[]
#     count=1
#     for i, atom in enumerate(mol.atoms):
#         for one_basis in g_basis[i]:
#             basis_label.append("Basis %d, %1s%1s on %1s"%(count,one_basis[0], one_basis[1],atno2type(atom.atno)))
#             count+=1

#     table = [  [ format(format(truncate(i, 3), "0.5f"),'9s') for i in row] for indx,row in enumerate(C)]
#     for indx,row in enumerate(table):
#         table[indx] = [basis_label[indx]] + row

#     headers = [" "]+[ "MO %2d"%(indx+1) for indx,row in enumerate(C)]

#     print tabulate(table,headers=headers)
#     return

if __name__ == "__main__":
    print("Load me as a module please")
