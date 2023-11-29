
# Get eth0 IP address
HOST_ETH0_IP=`ip addr | grep eth0 | grep inet | awk '{print $2}' | cut -d"/" -f1`

# Run jupyter notebook for the localhost
jupyter lab --no-browser --config=/home/dbabbitt/.jupyter/jupyter_lab_config.py --notebook-dir=/mnt/c/Users/DaveBabbitt/Documents/GitHub --app-dir=/home/dbabbitt/anaconda3/share/jupyter/lab