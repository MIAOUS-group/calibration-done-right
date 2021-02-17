# For citron vert or Cyber cobaye - run with sudo
# disable prefetchers
wrmsr -a 420 15

# performance cpu frequency governor
cpupower frequency-set -g performance

# No Turbo Boost
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
