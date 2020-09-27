# Beam_auto_alignment
Autmatic alignment of beam into an optical cavity


##########################################################################################
# Run remotely via Jupyter-notebook
##########################################################################################

# 1. Check if a Jupyter-notebook already is running.

# Open ssh-tunnel to the remote machine (localPort:remotePort)
ssh -p 37 -N -f -L localhost:8890:localhost:8890 controls@150.203.48.25

# 2. Open a browser and type localhost:8890. If no notebook is running, go to step 3..

# 3. Start Jupyter-notebook from remote machine
ssh -p 37 controls@150.203.48.25 "/home/controls/Beam_auto_alignment/remote_notebook.sh"

# 4. Redo step 2. If tunnel has closed, redo step 1 too.
