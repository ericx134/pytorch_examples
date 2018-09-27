

usage: mvcheck.py [-h] --color COLOR --mvec MVEC [--frame FRAME]
                  [--lookback LOOKBACK]

optional arguments:
  -h, --help           show this help message and exit
  --color COLOR        color images (use %d-like formatter for frame ID)
  --mvec MVEC          motion images (use %d-like formatter for frame ID)
  --frame FRAME        which frame ID to use
  --lookback LOOKBACK  how many frames to look back

Examples:


    python mvcheck.py 
        --color "\\sc-zfs-02\scratch.creo\people\shaveenk\DLAA\captured_data\RealisticRendering_path_3\ref\Room.%0.4d_hq.png" 
        --mvec "\\sc-zfs-02\scratch.creo\people\shaveenk\DLAA\captured_data\RealisticRendering_path_3\mvec\Room.%0.4d.pfm" 
        --lookback 5
        --frame 200

    python mvcheck.py
        --color "../../dl-data/SIGGRAPH/Classroom/Classroom_color_1spp_nojitter_%d.png"
        --mvec "../../dl-data/SIGGRAPH/Classroom/Classroom_mvec_nojitter_%d.pfm"
        --lookback 2


Output:

    Script will display the warped image against the target. 

    It will also save target.png and warped.png in the current directory