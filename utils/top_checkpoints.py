"""Bounded validation checkpoint ranking, independent of the training device."""

import json
import math
import os
import shutil
import tempfile


class TopCheckpoints:
    def __init__(self, directory, limit=5):
        if limit <= 0:
            raise ValueError('checkpoint limit must be positive')
        self.directory = os.path.abspath(directory)
        self.limit = limit
        self.records = []
        os.makedirs(self.directory, exist_ok=True)

    @staticmethod
    def filename(rank):
        return ('best_model.pth' if rank == 1 else
                'best_model_rank{}.pth'.format(rank))

    def path(self, rank=1):
        return os.path.join(self.directory, self.filename(rank))

    def _temporary(self):
        # PyTorch's zip writer rejects extensionless hidden filenames.
        fd, path = tempfile.mkstemp(prefix='checkpoint-', suffix='.tmp',
                                    dir=self.directory)
        os.close(fd)
        return path

    def _manifest(self):
        temporary = self._temporary()
        try:
            with open(temporary, 'w') as stream:
                json.dump(self.records, stream, indent=2, allow_nan=False)
                stream.write('\n')
            os.replace(temporary, os.path.join(self.directory, 'top_models.json'))
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def restore(self, records, source_directory):
        """Copy retained weights into this run without modifying the source run."""
        source_directory = os.path.abspath(source_directory)
        manifest = os.path.join(source_directory, 'top_models.json')
        if records and os.path.isfile(manifest):
            with open(manifest) as stream:
                current = json.load(stream)
            if current != records:
                raise ValueError('Checkpoint ranking differs from top_models.json; '
                                 'resume the latest .tar with its matching weights')
        selected = records[:self.limit]
        # Validate every source before copying anything.
        for record in selected:
            source = os.path.join(source_directory, record['filename'])
            if not os.path.isfile(source):
                raise FileNotFoundError('Retained checkpoint is missing: ' + source)
        restored = []
        for rank, record in enumerate(selected, 1):
            source = os.path.join(source_directory, record['filename'])
            target = self.path(rank)
            if os.path.abspath(source) != target:
                temporary = self._temporary()
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.remove(temporary)
            restored.append(dict(record, filename=self.filename(rank)))
        self.records = restored
        self._manifest()

    def consider(self, epoch, mae, rmse, save):
        """Call save(path) only on admission. Return the admitted 1-based rank."""
        score = float(mae) + 2.0 * float(rmse)
        if not all(math.isfinite(float(value)) for value in (mae, rmse, score)):
            return None
        position = sum(record['score'] <= score for record in self.records)
        if position >= self.limit:
            return None
        record = dict(epoch=int(epoch), mae=float(mae), rmse=float(rmse),
                      score=score)
        temporary = self._temporary()
        try:
            # Finish writing the new weights before touching the retained files.
            save(temporary)
            for rank in range(min(len(self.records), self.limit - 1), position, -1):
                os.replace(self.path(rank), self.path(rank + 1))
            os.replace(temporary, self.path(position + 1))
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        self.records.insert(position, record)
        self.records = self.records[:self.limit]
        for rank, item in enumerate(self.records, 1):
            item['filename'] = self.filename(rank)
        self._manifest()
        return position + 1
