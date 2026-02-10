from pathlib import Path

# корень проекта
ROOT = Path(".")

# список директорий, которые надо пропустить (по имени папки)
EXCLUDED_DIRS = {"venv", ".git", "__pycache__", ".idea"}

OUTPUT_FILE = ROOT / "all_in_one.py"


def is_excluded(path: Path) -> bool:
    # если какая-то часть пути совпадает с именем исключённой директории — пропускаем
    return any(part in EXCLUDED_DIRS for part in path.parts)


def main():
    files = []

    for p in ROOT.rglob("*.py"):
        if p == OUTPUT_FILE:
            continue  # не включаем сам файл сборки
        if is_excluded(p):
            continue  # пропускаем файлы из исключённых директорий
        files.append(p)

    files.sort()  # фиксированный порядок

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for f in files:
            out.write(f"# === {f} ===\n")
            out.write(f.read_text(encoding="utf-8"))
            out.write("\n\n")


if __name__ == "__main__":
    main()
