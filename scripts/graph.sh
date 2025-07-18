subdirs=("coadread")

for subdir in "${subdirs[@]}"; do
    echo "Processing $subdir..."

    ff_path="PATH_TO_FF_H5_FILES"
    ffpe_path="PATH_TO_FF_H5_FILES"
    graph_save_path="SAVE_PATH"
    python extract_graph.py --ff_path "$ff_path" --ffpe_path "$ffpe_path" --graph_save_path "$graph_save_path"

    if [ $? -eq 0 ]; then
        echo "Successfully processed $subdir."
    else
        echo "Failed to process $subdir."
        exit 1
    fi

done

echo "All processing complete."