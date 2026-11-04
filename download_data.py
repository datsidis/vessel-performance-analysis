from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="krohnedigital/FuelCast",
    repo_type="dataset",
    local_dir="../data/raw",
)
print("Done: data saved in data/raw")