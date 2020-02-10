import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

filename = 'results/sources.xml'

ex_params = ['Wex', 'Uex', 'Gex', 'Hex']
ft_params = ['Wft', 'Uft', 'Gft', 'Hft']

def readEye(node):
    mat = {}
    rows = int(node.findtext('rows'))
    cols = int(node.findtext('cols'))
    mat['data'] = np.eye(rows, cols)
    return mat


def readMixingParameter(node):
    A = {}
    for k in node.keys():
        A[k] = node.get(k)
    A['mixingType'] = A['mixing_type']
    del A['mixing_type']

    dim = [];
    for e in node.findall('dim'):
        dim.append(int(e.text))
    s = node.find('data').text.split()

    if node.find('type').text == 'real':
        A['data'] = np.reshape(np.array(list(map(float, s))), dim, order='F')
    else:
        buf = np.array(map(float, s))
        A['data'] = np.zeros(dim, dtype=complex)
        s = dim[0] * dim[1]
        d = [dim[0], dim[1]]
        for i in range(dim[-1]):
            inf = 2*s*i
            sup = s*(2*(i+1)-1)
            real_part = np.reshape(buf[inf:sup], d, order='F')

            inf = s*(2*(i+1)-1)
            sup = 2*s*(i+1)
            imag_part = np.reshape(buf[inf:sup], d, order='F')
            A['data'][:,:,i] = real_part + imag_part*1j
    return A

def readNonNegMatrix(node):
    mat = {}
    for k in node.keys():
        mat[k] = node.get(k)
    rows = int(node.findtext('rows'))
    cols = int(node.findtext('cols'))
    s = node.find('data').text

    mat['data'] = np.array(list(map(float, s.split()))).reshape(rows, cols, order='F')
    return mat

def readSource(node):
    source = {}
    if node.get('name') is not None:
        source['name'] = node.get('name')

    # read Mixing matrix
    source['A'] = readMixingParameter(node.find('A'))

    # read Excitation matrix
    for param in ex_params:
        if node.find(param).findtext('data').strip() != 'eye':
            source[param] = readNonNegMatrix(node.find(param));
        else:
            source[param] = readEye(node.find(param));

    # read Filter matrix
    if node.find('Wft') is not None:
        for param in ft_params:
            if node.find(param).findtext('data').strip() != 'eye':
                source[param] = readNonNegMatrix(node.find(param));
            else:
                source[param] = readEye(node.find(param))
    return source

def loadXML(fname):
    with open(fname, 'r') as f:
        root = ET.XML(f.read())

    data = {}
    # retrieve Window length
    data['wlen'] = int(root.findtext('wlen'))
    # retrieve all sources
    sources = root.findall('source')
    data['sources'] = []
    for sourceNode in sources:
        data['sources'].append(readSource(sourceNode))

    return data
#%%
data = loadXML(filename)

def plot_params(source, params, ex_ft):
    W_data = source[params[0]]['data']
    U_data = source[params[1]]['data']
    G_data = source[params[2]]['data']
    H_data = source[params[3]]['data']

    P_data = G_data@H_data
    E_data = W_data@U_data

    V_data = E_data@P_data

    fig, ((ax1, ax2, ax3, H), (ax5, ax6, G, P), (ax9, U, ax11, ax12), (W, E, ax15, V)) = plt.subplots(4,4,figsize=(20,20))
    fig.suptitle(source['name'] + ex_ft)

    H.imshow(H_data)
    G.imshow(G_data)
    P.imshow(P_data)
    U.imshow(U_data)
    W.imshow(W_data)
    E.imshow(E_data)
    V.imshow(V_data)

    H.set_title('H_'+ ex_ft)
    G.set_title('G_'+ ex_ft)
    P.set_title('P_'+ ex_ft)
    U.set_title('U_'+ ex_ft)
    W.set_title('W_'+ ex_ft)
    E.set_title('E_'+ ex_ft)
    V.set_title('V_'+ ex_ft)

    plt.show()
    return fig

for source in data['sources']:
    name = source['name']
    A = source['A']['data']

    plot_params(source, ex_params, 'ex')
    plot_params(source, ft_params, 'ft')
