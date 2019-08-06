
                      SIR Tao-Leap Calibration Example


 - To run a calibration job, configure parameters in the 
   sir_tao_leap_calibration.py script and run: 

       python3  sir_tao_leap_calibration.py


 - For trajectory selection, configure parameters in the 
   sir_tao_leap_trajectorySelection.py script and run:   

       python3  sir_tao_leap_trajectorySelection.py


 - Additional information and example figures at:

https://wiki.idmod.org/display/EPI/Demo+2%3A+Selecting+targets+for+fitting+history+matching+emulators


    


Installation:
-------------

- Create (and activate) separate conda environment: 

    conda create --name hmdemo
    source activate history_matching_demos
    [to exit this environment: source deactivate]
    
    
- Install history matching dependencies

    sudo apt-get install nvidia-cuda-toolkit
    sudo apt-get install nvidia-cuda-dev
    sudo apt-get install python3-pycuda
    pip3 install pandas==0.24.2
    pip3 install scikit-cuda==0.5.2
    pip3 install scikit-learn==0.20.3
    pip3 install scipy==1.2.1   
    pip3 install Wand==0.5.4
    pip3 install jupyter-client==5.2.4
    pip3 install jupyter-console==6.0.0
    pip3 install jupyter-core==4.4.0
    pip3 install openpyxl==2.6.2
    pip3 install tables==3.5.1
    pip3 install xlrd==1.2.0
    pip3 install seaborn==0.9.0
    pip3 install statsmodels==0.9.0
    pip3 install matplotlib==3.0.3
    pip3 install patsy==0.5.1
    pip3 install pyDOE==0.3.8
    pip3 install numpy==1.16.3
    
    pip3 install minepy
    
    
    
- Download and install history_matching_demos

    mkdir [work dir]
    cd [work dir]
    git clone https://github.com/rnunez-IDM/history_matching_demos
    cd history_matching_demos
    python3 setup.py develop
    
    
    
- Run demos

    cd history_matching/examples/sir_tao_leap_3params
    python3 sir_tao_leap_calibration.py
