# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# For citron vert or Cyber cobaye - run with sudo
# disable prefetchers
wrmsr -a 420 15

# performance cpu frequency governor
cpupower frequency-set -g performance

# No Turbo Boost
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
