from pathlib import Path


START = "# BEGIN tubes_ka_disable_checkpoints"
END = "# END tubes_ka_disable_checkpoints"

CONFIG_BLOCK = f"""{START}
from datetime import datetime, timezone
from tornado.web import HTTPError
from jupyter_server.services.contents.checkpoints import Checkpoints, AsyncCheckpoints


class NoOpCheckpoints(Checkpoints):
    def create_checkpoint(self, contents_mgr, path):
        return {{"id": "checkpoint", "last_modified": datetime.now(timezone.utc)}}

    def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        raise HTTPError(404, "Checkpoint dinonaktifkan.")

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        return None

    def delete_checkpoint(self, checkpoint_id, path):
        return None

    def list_checkpoints(self, path):
        return []


class AsyncNoOpCheckpoints(AsyncCheckpoints):
    async def create_checkpoint(self, contents_mgr, path):
        return {{"id": "checkpoint", "last_modified": datetime.now(timezone.utc)}}

    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        raise HTTPError(404, "Checkpoint dinonaktifkan.")

    async def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        return None

    async def delete_checkpoint(self, checkpoint_id, path):
        return None

    async def list_checkpoints(self, path):
        return []


c.FileContentsManager.checkpoints_class = NoOpCheckpoints
c.AsyncFileContentsManager.checkpoints_class = AsyncNoOpCheckpoints
{END}
"""


def main():
    config_dir = Path.home() / ".jupyter"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "jupyter_server_config.py"

    if config_path.exists():
        current = config_path.read_text(encoding="utf-8")
    else:
        current = ""

    if START in current and END in current:
        before = current.split(START)[0].rstrip()
        after = current.split(END, 1)[1].lstrip()
        updated = f"{before}\n\n{CONFIG_BLOCK}\n\n{after}".strip() + "\n"
    else:
        updated = f"{current.rstrip()}\n\n{CONFIG_BLOCK}\n".lstrip()

    config_path.write_text(updated, encoding="utf-8")
    print(f"Konfigurasi Jupyter tersimpan di: {config_path}")
    print("Restart Jupyter agar konfigurasi aktif.")


if __name__ == "__main__":
    main()
