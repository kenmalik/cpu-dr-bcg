#!/usr/bin/env bash

COMMAND="../build/examples/example"
if ! [ -f ${COMMAND} ]; then
    echo "${COMMAND} not built"
    exit 1
fi

mat_dir=$1
if [ -z ${mat_dir} ]; then
    echo "Usage: test_matrices.sh [MAT_DIR]"
    exit 1
fi

readarray -d "" -t mats < <(find ${mat_dir} -maxdepth 1 -name "*.mat" -print0)

succeeded=()
failed=()

output() {
    echo "${#failed[@]} Failed:" 
    for f in ${failed[@]}; do
        echo $f
    done

    echo -e "\n${#succeeded[@]} Succeeded:" 
    for s in ${succeeded[@]}; do
        echo $s
    done
}

trap "output; exit" SIGINT
trap "output; exit" SIGTERM

for mat in ${mats[@]}; do
    mat_name=$(basename $mat)
    echo "Running against ${mat_name}" >&2

    "${COMMAND}" ${mat} 4 >&2
     
    if [[ $? != 0 ]]; then
        echo "${mat_name} failed" >&2
        failed+=(${mat_name})
    else
        echo "${mat_name} succeeded" >&2
        succeeded+=(${mat_name})
    fi
done

output
