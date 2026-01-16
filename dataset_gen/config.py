from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path

    @property
    def docs_dir(self) -> Path:
        return self.output_dir / "docs"

    @property
    def parsed_dir(self) -> Path:
        return self.output_dir / "parsed"

    @property
    def canonical_dir(self) -> Path:
        return self.output_dir / "canonical"

    @property
    def indexes_dir(self) -> Path:
        return self.output_dir / "indexes"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "doc_store.json"

