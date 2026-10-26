# Machine_learning_IA_fiable

### 1. Créer le virtual environment

```bash
python3 -m venv .venv
```

### 2. Activer le virtual environment

```bash
source .venv/bin/activate
```

### 3. Vérifier la version de Python

```bash
python3 --version
```

### 4. Désactiver le virtual environment

```bash
deactivate
```

### 5. Supprimer le virtual environment

```bash
rm -rf .venv
```

### 6. Installer les dépendances

Après avoir activé le virtual environment :

```bash
pip install -r requirements.txt
```

### Thiết lập lần đầu trên mỗi máy

**Trên mỗi máy, sau `git pull` (chỉ thực hiện lần đầu):**

```bash
cd ~/Documents/github/Machine_learning_IA_fiable
bash setup.sh
```
Sau đó, thoát hẳn RStudio bằng `Cmd + Q`, mở lại RStudio, rồi bấm **Render** là chạy.

**Các lần sau:** chỉ cần `git pull` rồi **Render**, không phải làm gì thêm.