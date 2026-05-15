"""
Upload all best_*.pth checkpoints to WandB as a versioned artifact.

Usage (on remote server):
    python upload_checkpoints.py

Download on local machine:
    wandb artifact get <entity>/adl-crossouts-v3/checkpoints:latest --root ./checkpoints
"""
import os, glob
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.login(key=os.environ.get('WANDB_API_KEY'))

CHECKPOINT_DIR = './checkpoints'
WANDB_PROJECT  = 'adl-crossout-v5'
WANDB_ENTITY   = 'naggan-4'

files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'best_*.pth')))
if not files:
    print('No checkpoints found.')
    exit(1)

print(f'Found {len(files)} checkpoint(s):')
for f in files:
    size = os.path.getsize(f) / 1e6
    print(f'  {os.path.basename(f)}  ({size:.1f} MB)')

run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type='upload', name='checkpoint-upload')
artifact = wandb.Artifact('checkpoints', type='model',
                          description='Best model checkpoints from training')
for f in files:
    artifact.add_file(f, name=os.path.basename(f))

run.log_artifact(artifact)
run.finish()
print('\nDone. Download with:')
print(f'  wandb artifact get {WANDB_ENTITY}/{WANDB_PROJECT}/checkpoints:latest --root ./checkpoints')
