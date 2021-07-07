# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

#/bin/bash

# Change this to the correct openssl installation path
export OPENSSL_DIR=$(readlink -f ~/openssl)
export X86_64_UNKNOWN_LINUX_GNU_OPENSSL_DIR=$OPENSSL_DIR
export PKG_CONFIG_PATH=$OPENSSL_DIR
export X86_64_UNKNOWN_LINUX_GNU_PKG_CONFIG_PATH=$OPENSSL_DIR
export LD_LIBRARY_PATH=$OPENSSL_DIR
cargo "$@"
