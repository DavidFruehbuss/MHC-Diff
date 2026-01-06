# A tool for identifiying the G-domain in PDB files that contain MHC.

## Dependencies
 - pymol==3.0.0
 - biopython==1.84

## Running the script

The script has its own reference pdb file. You only have to provide the pdb file that needs to be identified.
One can insert multiple pdb files:

```
scripts/find_gdomain pdb_path_1 pdb_path_2 ...
```

or:

```
scripts/find_gdomain /data/pdb_files/*.pdb
```

output:

```
pdb_path_1 M2-M180
pdb_path_2 A1-A179
...
```

Each row corresponds to an unknown PDB file.
The first column identifies the input file.
The second column indicates the residues of the G-domain. (chain ids + residue numbers)
