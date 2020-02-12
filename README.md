# PAM4: Sound recording and Source Separation techniques
[ATIAM 2019-2020 PAM] Informed source separation scripts, using the FASST framework.

### Dependencies

The source separation algorithm uses the 2.2.2 version of [FASST](http://bass-db.gforge.inria.fr/fasst/) framework from A. Ozerov.

### Folder organization

The directories should be organized as follow:
```
PAM4
└──audios
    └── yourmixture.wav
└── results
    └── *_EstimatedSource.wav
└── temp
    └── sources.xml
```

### Run the source separation algorithm 
```
usage: separate.py [-h] [-J J] [-epoch EPOCH] [-v V] [-g G] [-p P] [-t T]
                   [-s S] [-d D] [-pan1 PAN1] [-pan2 PAN2] [-pan3 PAN3]
                   [-pan4 PAN4] [-pan5 PAN5] [-pan6 PAN6] [-pan7 PAN7]
                   [-pan8 PAN8]
                   filename

Informed audio source separation.

positional arguments:
  filename              Mixtures wave file.

optional arguments:
  -h, --help            show this help message and exit
  -J J, --nb_sources J  Number of sources.
  -epoch EPOCH, --nb_epoch EPOCH
                        Number of iteration of EM algorithm.
  -v V, --voice V       Number of voice tracks in the mixtures.
  -g G, --guitar G      Number of guitar tracks in the mixtures.
  -p P, --piano P       Number of piano tracks in the mixtures.
  -t T, --trumpet T     Number of trumpet tracks in the mixtures.
  -s S, --saxophone S   Number of saxophone tracks in the mixtures.
  -d D, --drums D       Number of drum tracks in the mixtures.
  -pan1 PAN1            Pan of 1st source.
  -pan2 PAN2            Pan of 2nd source.
  -pan3 PAN3            Pan of 3rd source.
  -pan4 PAN4            Pan of 4th source.
  -pan5 PAN5            Pan of 5th source.
  -pan6 PAN6            Pan of 6th source.
  -pan7 PAN7            Pan of 7th source.
  -pan8 PAN8            Pan of 8th source.
```

Source templates definition for voice, guitar, piano, trumpet, saxophone and drums can be found at `template.py` (WIP).

### View the estimated source model parameters
Plot the estimated source parameters from the .xml in the \temp folder.
```
python XMLReader.py
```

## Blind Source Separation (WIP)
Implementation of Cardoso's Blind Source Separation statistical principles.
```
python BSS.py
```

## Authors

* **A. Ozerov, E.Vincent, F. Bimbot** - *"A General Flexible Framework for the Handling of Prior
Information in Audio Source Separation" (2012)* - [IEEE](https://ieeexplore.ieee.org/document/6047568)

* **J.-F. Cardoso** - *"Blind signal separation: statistical principles" (1998)* - [ProcIEEE](http://www2.iap.fr/users/cardoso/papers/ProcIEEE.pdf)
