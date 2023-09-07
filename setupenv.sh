#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Technology distributions
PKG_ARRAY=(stock intel)
PKG_ELEMENTS=${#PKG_ARRAY[@]}

function success {
echo "Environment $ENV_NAME created"
echo "Activate:"
echo -e "       \$ conda activate $ENV_NAME"
exit 0
}

function failed {
echo "Environment $ENV_NAME creation failed "
exit 1
}

ENV_NAME="intel_ai_disease_prediction"
echo Creating conda environment $ENV_NAME...
echo Setting up environment with packages : $PACKAGE...

# using intel technologies
conda env create -n $ENV_NAME -f env/intel/intel.yml
if [[ $? -ne 0 ]] ; then
failed
else
success
fi
