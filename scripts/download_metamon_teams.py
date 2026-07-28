import os
import tarfile
import shutil

from huggingface_hub import hf_hub_download

repo_id = 'jakegrigsby/metamon-teams'
filename = 'competitive/gen3ou.tar.gz'

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache_dir = os.path.join(root, 'data', 'teams_cache')
output_dir = os.path.join(root, 'data', 'teams')

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

tar_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    cache_dir=cache_dir,
    repo_type='dataset',
    revision='v5',
)
print('Downloaded:', tar_path)

extract_dir = os.path.join(cache_dir, 'gen3ou')
shutil.rmtree(extract_dir, ignore_errors=True)
with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(path=cache_dir)

team_files = sorted(
    [
        os.path.join(extract_dir, f)
        for f in os.listdir(extract_dir)
        if f.endswith('.gen3ou_team')
    ]
)
print('Found team files:', len(team_files))

result_path = os.path.join(output_dir, 'gen3ou.txt')
with open(result_path, 'w', encoding='utf-8') as out:
    for tf in team_files:
        with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip().replace('\r\n', '\n').replace('\n', ' / ')
        out.write(content + '\n')

print('Created:', result_path)
print('Sample:')
with open(result_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(line.strip())
