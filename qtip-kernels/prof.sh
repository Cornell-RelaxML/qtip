export_name="/home/$USER/ncu_reps/qtip_$(date '+%Y%m%d_%H%M%S').ncu-rep"
make clean && make
sudo -E bash -c "PATH=/usr/local/cuda/bin/:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH ncu -f --set full -o \"$export_name\" --page details --target-processes all --import-source on test"
sudo chown $USER $export_name

