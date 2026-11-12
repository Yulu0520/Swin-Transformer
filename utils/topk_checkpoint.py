"""Keep validation top-5 weights and a portable, full-precision ranking index."""

import json
import math
import os
from pathlib import Path
import shutil

import torch


class TopKCheckpoints:
    def __init__(self, directory, k=5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.entries = []

    @staticmethod
    def sort_key(entry):
        return (entry['score'], entry['mae'], entry['rmse'],
                -1 if entry['epoch'] is None else entry['epoch'])

    @property
    def best_path(self):
        return str(self.directory / self.entries[0]['file']) if self.entries else None

    def state_dict(self):
        return [dict(entry) for entry in self.entries]

    def _write_manifest(self):
        target = self.directory / 'rankings.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(dict(version=1, criterion='2 * RMSE + MAE',
            top_k=self.k, entries=self.entries), indent=2, allow_nan=False) + '\n')
        os.replace(temporary, target)

    def consider(self, state_dict, mae, rmse, epoch):
        mae, rmse = float(mae), float(rmse)
        if not all(math.isfinite(x) and x >= 0 for x in (mae, rmse)):
            raise ValueError('Validation MAE/RMSE must be finite and nonnegative')
        entry = dict(mae=mae, rmse=rmse, score=mae + 2 * rmse, epoch=int(epoch),
                     file='MAE{:.2f}_epoch{}.pth'.format(mae, epoch))
        # Each epoch represents one validation result, including epoch 0 before fine-tuning.
        existing = next((item for item in self.entries if item['epoch'] == entry['epoch']), None)
        if existing:
            raise ValueError('Validation epoch {} is already ranked'.format(epoch))
        candidates = sorted(self.entries + [entry], key=self.sort_key)
        if entry not in candidates[:self.k]:
            return False
        path = self.directory / entry['file']
        temporary = path.with_suffix('.pth.tmp')
        torch.save(state_dict, temporary)
        os.replace(temporary, path)
        removed = candidates[self.k:]
        self.entries = candidates[:self.k]
        # Commit the new index before removing obsolete files.
        self._write_manifest()
        for old in removed:
            (self.directory / old['file']).unlink(missing_ok=True)
        return True

    def restore(self, entries, source_directory):
        """Restore a checkpoint's ranking snapshot into the new run directory."""
        source_directory = Path(source_directory)
        restored = sorted([dict(entry) for entry in entries], key=self.sort_key)[:self.k]
        for entry in restored:
            name = entry['file']
            if Path(name).name != name:
                raise ValueError('Ranking entries must contain a local filename')
            source, target = source_directory / name, self.directory / name
            if not source.is_file():
                raise FileNotFoundError('Ranked weights are missing: {}. Resume the latest checkpoint '
                                        'with its weight files, or use --init-checkpoint.'.format(source))
            if source.resolve() != target.resolve():
                if target.exists():
                    raise FileExistsError('Refusing to overwrite ranked weights: {}'.format(target))
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
        self.entries = restored
        self._write_manifest()

    def import_legacy_best(self, path, mae, rmse):
        """Older checkpoints only saved one best model, with no best epoch."""
        path = Path(path)
        entry = dict(file=path.name, mae=float(mae), rmse=float(rmse),
                     score=float(mae + 2 * rmse), epoch=None)
        self.restore([entry], path.parent)


def best_ranked_path(directory):
    directory = Path(directory)
    manifest = directory / 'rankings.json'
    if not manifest.is_file():
        raise FileNotFoundError('No rankings.json in {}; specify an exact checkpoint file'.format(directory))
    data = json.loads(manifest.read_text())
    if data.get('criterion') != '2 * RMSE + MAE':
        raise ValueError('Unsupported checkpoint ranking criterion')
    entries = sorted(data.get('entries', []), key=TopKCheckpoints.sort_key)
    if not entries:
        raise ValueError('No ranked weights in {}'.format(manifest))
    name = entries[0]['file']
    if Path(name).name != name:
        raise ValueError('Ranked checkpoint must be a local filename')
    result = directory / name
    if not result.is_file():
        raise FileNotFoundError('Top-ranked checkpoint is missing: {}'.format(result))
    return str(result)
