#!/bin/bash

CMD='cd /workspace/code/dlaa-pytorch-master; python3 -m apex.parallel.multiproc dlaa.py '
CMD+='--label=MyNewTraining --paths="[/dataset]" '
CMD+='--epochs=100          --log_interval=10       --save_interval=10  --disable_vgg '
CMD+='--batch_size=32       --train                 --tensorboard       --valperc=0 '
CMD+='--io_workers=4        --learning_rate=0.0001  --clr_min=0.0001    --clr_max=0.001 '
CMD+='--multigpu_mode="DDP" --tfx="BasicPQEncodingFunction"'

ngc batch run \
  --ace nv-us-west-2 \
  --team dt \
  --instance ngcv8 \
  --name "Testing DLSS training on NGC" \
  --image "nvidian/dt/dlaa-pytorch:pth040_cuda9010_py36_apex_gputil" \
  --workspace ericx-dlaa:/workspace \
  --datasetid 12525:/dataset \
  --result /result \
  --commandline "$CMD"