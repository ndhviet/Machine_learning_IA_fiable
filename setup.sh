#!/usr/bin/env bash
# Chạy MỘT LẦN trên mỗi máy sau khi git clone / git pull:
#   bash setup.sh
set -e

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
PY="$PROJECT_DIR/.venv/bin/python"

# 1) Tạo venv nếu chưa có
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# 2) Cài thư viện
"$PY" -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  "$PY" -m pip install -r requirements.txt
fi
"$PY" -m pip install jupyter pyyaml ipykernel

# 3) Đăng ký kernel đúng tên trong _quarto.yml
"$PY" -m ipykernel install --user --name tp-ml-ia-fiable

# 4) Chỉ cho RStudio dùng Python của venv này (thay dòng cũ nếu có)
touch ~/.Renviron
grep -v '^QUARTO_PYTHON=' ~/.Renviron > ~/.Renviron.tmp || true
echo "QUARTO_PYTHON=$PY" >> ~/.Renviron.tmp
mv ~/.Renviron.tmp ~/.Renviron

echo ""
echo "Xong. Thoát hẳn RStudio (Cmd+Q), mở lại rồi bấm Render."
