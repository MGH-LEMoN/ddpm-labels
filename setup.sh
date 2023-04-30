#!/bin/bash

git clone https://github.com/MGH-LEMoN/ddpm-labels.git && cd ddpm-labels

conda create --name ddpm-labels python=3.11 && conda activate ddpm-labels
pip install -r requirements.txt

python --version

export PYTHONPATH=$PWD

# copy/link voxelmorph data into the data folder
ln -s /cluster/vxmdata1/FS_Slim/proc/cleaned data/label-maps

# crop label maps and pad for equal size (and for ease of use with CNN)
python -c "import script.utils; max_size = create_compact_label_maps(); \
            pad_compact_label_maps(max_size)"

# write filenames to a text/csv file for use in dataloader
python -c "import script.utils; write_labelmap_names(LABEL_MAPS_PADDED, 'ddpm_files_padded.txt')"

make ddpm-test
