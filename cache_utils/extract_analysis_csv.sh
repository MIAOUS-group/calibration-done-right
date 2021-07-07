#!/bin/sh

# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

bzgrep '^Analysis:' "$1.txt.bz2" | cut -b 10- > "$1.csv"
bzgrep '^AVAnalysis:' "$1.txt.bz2" | cut -b 12- > "$1.AV.csv"
bzgrep '^AttackerAnalysis:' "$1.txt.bz2" | cut -b 18- > "$1.A.csv"
