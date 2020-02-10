import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

filename = 'audios/sources.xml'

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
    print(type(s[0]), s)
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
    print(type(rows), type(cols))
    mat['data'] = np.array(list(map(float, s.split()))).reshape(rows, cols, order='F')
    return mat

def readSource(node):
    source = {}
    if node.get('name') is not None:
        source['name'] = node.get('name')

    source['A'] = readMixingParameter(node.find('A'))
    for param in ['Wex', 'Uex', 'Gex', 'Hex']:
        if node.find(param).findtext('data').strip() != 'eye':
            source[param] = readNonNegMatrix(node.find(param));
    if node.find('Wft') is not None:
        for param in ['Wft', 'Uft', 'Gft', 'Hft']:
            if node.find(param).findtext('data').strip() != 'eye':
                source[param] = readNonNegMatrix(node.find(param));
    return source

def loadXML(fname):
    with open(fname, 'r') as f:
        root = ET.XML(f.read())

    data = {}
    data['wlen'] = int(root.findtext('wlen'))

    sources = root.findall('source')
    data['sources'] = []
    for sourceNode in sources:
        data['sources'].append(readSource(sourceNode))

    return data

data = loadXML(filename)

for source in data['sources']:
    name = source['name']
    A = source['A']['data']
    Hex = source['Hex']['data']
    Wex = source['Wex']['data']
    
    plt.imshow(Wex)
    plt.show()